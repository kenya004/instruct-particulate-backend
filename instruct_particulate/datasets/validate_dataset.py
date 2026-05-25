import argparse
import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from instruct_particulate.datasets.dataset import InstructParticulateDataset


REQUIRED_KEYS = {
    "points",
    "normals",
    "link_ids",
    "sharp_point_flag",
    "links",
}
_OPTIONAL_QUERY_BOUNDARY_KEYS = {
    "query_boundary_distance_link_ids",
    "query_boundary_distance_to_raw_link",
}

DEFAULT_LOAD_CHECK_DATASET_KWARGS: Dict[str, Any] = {
    "num_shape_points": 1024,
    "num_query_points": 1024,
    "sharp_point_ratio": 0.5,
    "random_pose_prob": 0.5,
    "dropout_individual_link_prob": 0.5,
    "max_num_links": 8,
    "use_augmentation": True,
}


def _decode_json_field(value: Any, field_name: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(
                f"Expected scalar JSON field for '{field_name}', got shape {value.shape}"
            )
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise TypeError(
            f"Expected '{field_name}' to be stored as a JSON string, got {type(value).__name__}"
        )
    return json.loads(value)


def _iter_npz_paths(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix != ".npz":
            raise ValueError(f"Expected an .npz file, got {root}")
        yield root
        return

    yield from sorted(root.rglob("*.npz"))


def _validate_links(links: Any, data_path: Path) -> set[int]:
    if not isinstance(links, list):
        raise TypeError(
            f"'links' must decode to a list in {data_path}, got {type(links).__name__}"
        )

    valid_link_ids: set[int] = set()
    embedding_dim: int | None = None
    for link in links:
        if not isinstance(link, dict):
            raise TypeError(
                f"Each link record must be a dict in {data_path}, got {type(link).__name__}"
            )

        link_id = int(link["link_id"])
        if link_id < 0:
            raise ValueError(f"link_id values in {data_path} must be non-negative, got {link_id}")
        if link_id in valid_link_ids:
            raise ValueError(f"Duplicate link_id={link_id} in {data_path}")
        valid_link_ids.add(link_id)

        if "name" not in link:
            raise KeyError(f"Missing 'name' for link_id={link_id} in {data_path}")
        raw_name = link["name"]
        if isinstance(raw_name, str):
            name_variants = (raw_name,)
            name_is_list = False
        elif isinstance(raw_name, list) and all(isinstance(name, str) for name in raw_name):
            if not raw_name:
                raise ValueError(
                    f"'name' for link_id={link_id} in {data_path} must not be an empty list"
                )
            name_variants = tuple(raw_name)
            name_is_list = True
        else:
            raise TypeError(
                f"'name' for link_id={link_id} in {data_path} must be a string or a list of strings"
            )

        embedding = link.get("embedding")
        if embedding is None:
            raise ValueError(
                f"Missing cached embedding for link_id={link_id} in {data_path}"
            )

        embedding_array = np.asarray(embedding, dtype=np.float32)
        if name_is_list:
            if embedding_array.ndim != 2 or embedding_array.shape[0] != len(name_variants):
                raise ValueError(
                    f"Embedding for link_id={link_id} in {data_path} must have shape "
                    f"({len(name_variants)}, D) when 'name' is a list, got {embedding_array.shape}"
                )
            current_embedding_dim = int(embedding_array.shape[1])
        elif embedding_array.ndim != 1:
            raise ValueError(
                f"Embedding for link_id={link_id} in {data_path} must be 1D when 'name' is a string, "
                f"got shape {embedding_array.shape}"
            )
        else:
            current_embedding_dim = int(embedding_array.shape[0])
        if embedding_dim is None:
            embedding_dim = current_embedding_dim
        elif current_embedding_dim != embedding_dim:
            raise ValueError(
                f"All link embeddings in {data_path} must share one feature dimension, got "
                f"{embedding_dim} and {current_embedding_dim}"
            )

    return valid_link_ids


def _validate_joints(joints: Any, valid_link_ids: set[int], data_path: Path) -> None:
    """Validates decoded `joints` metadata for one cached sample.

    The dataset loader supports joint records whose motion type is known while
    the numeric axis or range supervision is absent. This validator therefore
    treats axis/range arrays as optional, while still checking shapes whenever
    those arrays are present.

    Args:
        joints: Decoded joint records.
        valid_link_ids: Valid link IDs declared in the sample.
        data_path: Source `.npz` path used for error messages.
    """
    if not isinstance(joints, list):
        raise TypeError(
            f"'joints' must decode to a list in {data_path}, got {type(joints).__name__}"
        )

    seen_joint_ids: set[int] = set()
    for joint in joints:
        if not isinstance(joint, dict):
            raise TypeError(
                f"Each joint record must be a dict in {data_path}, got {type(joint).__name__}"
            )

        joint_id = int(joint["joint_id"])
        if joint_id in seen_joint_ids:
            raise ValueError(f"Duplicate joint_id={joint_id} in {data_path}")
        seen_joint_ids.add(joint_id)

        parent_link_id = int(joint["parent_link_id"])
        child_link_id = int(joint["child_link_id"])
        if parent_link_id not in valid_link_ids:
            raise ValueError(
                f"parent_link_id={parent_link_id} does not exist in links metadata for {data_path}"
            )
        if child_link_id not in valid_link_ids:
            raise ValueError(
                f"child_link_id={child_link_id} does not exist in links metadata for {data_path}"
            )

        is_revolute = joint.get("is_revolute")
        is_prismatic = joint.get("is_prismatic")
        if not isinstance(is_revolute, bool):
            raise TypeError(
                f"is_revolute for joint_id={joint_id} in {data_path} must be a boolean"
            )
        if not isinstance(is_prismatic, bool):
            raise TypeError(
                f"is_prismatic for joint_id={joint_id} in {data_path} must be a boolean"
            )
        if not is_revolute and not is_prismatic:
            raise ValueError(
                f"Joint joint_id={joint_id} in {data_path} must have at least one motion component"
            )

        revolute_axis = joint.get("revolute_axis")
        if revolute_axis is not None:
            revolute_axis_array = np.asarray(revolute_axis, dtype=np.float32)
            if revolute_axis_array.shape != (6,):
                raise ValueError(
                    f"revolute_axis for joint_id={joint_id} in {data_path} must have shape (6,), got {revolute_axis_array.shape}"
                )

        prismatic_axis = joint.get("prismatic_axis")
        if prismatic_axis is not None:
            prismatic_axis_array = np.asarray(prismatic_axis, dtype=np.float32)
            if prismatic_axis_array.shape != (6,):
                raise ValueError(
                    f"prismatic_axis for joint_id={joint_id} in {data_path} must have shape (6,), got {prismatic_axis_array.shape}"
                )

        revolute_range = joint.get("revolute_range")
        if revolute_range is not None:
            revolute_range_array = np.asarray(revolute_range, dtype=np.float32)
            if revolute_range_array.shape != (2,):
                raise ValueError(
                    f"revolute_range for joint_id={joint_id} in {data_path} must have shape (2,), got {revolute_range_array.shape}"
                )
            if revolute_axis is None:
                raise ValueError(
                    f"Joint joint_id={joint_id} in {data_path} has revolute_range but no revolute_axis"
                )

        prismatic_range = joint.get("prismatic_range")
        if prismatic_range is not None:
            prismatic_range_array = np.asarray(prismatic_range, dtype=np.float32)
            if prismatic_range_array.shape != (2,):
                raise ValueError(
                    f"prismatic_range for joint_id={joint_id} in {data_path} must have shape (2,), got {prismatic_range_array.shape}"
                )
            if prismatic_axis is None:
                raise ValueError(
                    f"Joint joint_id={joint_id} in {data_path} has prismatic_range but no prismatic_axis"
                )

        if not is_revolute and (revolute_axis is not None or revolute_range is not None):
            raise ValueError(
                f"Joint joint_id={joint_id} in {data_path} has revolute metadata but is_revolute is false"
            )

        if not is_prismatic and (prismatic_axis is not None or prismatic_range is not None):
            raise ValueError(
                f"Joint joint_id={joint_id} in {data_path} has prismatic metadata but is_prismatic is false"
            )


def validate_sample(data_path: Path) -> bool:
    """Runs static validation on one cached `.npz` sample."""
    with np.load(data_path, allow_pickle=False) as data:
        missing_keys = REQUIRED_KEYS - set(data.files)
        if missing_keys:
            raise KeyError(
                f"Missing required keys in {data_path}: {sorted(missing_keys)}"
            )

        points = np.asarray(data["points"])
        normals = np.asarray(data["normals"])
        link_ids = np.asarray(data["link_ids"])
        sharp_point_flag = np.asarray(data["sharp_point_flag"])
        query_boundary_distance_link_ids = None
        if "query_boundary_distance_link_ids" in data.files:
            query_boundary_distance_link_ids = np.asarray(
                data["query_boundary_distance_link_ids"]
            )
        query_boundary_distance_to_raw_link = None
        if "query_boundary_distance_to_raw_link" in data.files:
            query_boundary_distance_to_raw_link = np.asarray(
                data["query_boundary_distance_to_raw_link"]
            )
        links = _decode_json_field(data["links"], "links")
        joints = []
        if "joints" in data.files:
            joints = _decode_json_field(data["joints"], "joints")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"'points' must have shape (N, 3), got {points.shape} in {data_path}")
    if normals.shape != points.shape:
        raise ValueError(
            f"'normals' must match 'points' shape, got {normals.shape} vs {points.shape} in {data_path}"
        )
    if link_ids.shape != (points.shape[0],):
        raise ValueError(
            f"'link_ids' must have shape ({points.shape[0]},), got {link_ids.shape} in {data_path}"
        )
    if sharp_point_flag.shape != (points.shape[0],):
        raise ValueError(
            f"'sharp_point_flag' must have shape ({points.shape[0]},), got {sharp_point_flag.shape} in {data_path}"
        )
    present_query_boundary_keys = {
        key
        for key, value in (
            ("query_boundary_distance_link_ids", query_boundary_distance_link_ids),
            ("query_boundary_distance_to_raw_link", query_boundary_distance_to_raw_link),
        )
        if value is not None
    }
    if present_query_boundary_keys and present_query_boundary_keys != _OPTIONAL_QUERY_BOUNDARY_KEYS:
        missing_query_boundary_keys = sorted(
            _OPTIONAL_QUERY_BOUNDARY_KEYS.difference(present_query_boundary_keys)
        )
        raise KeyError(
            "Cached query boundary fields must either both be present or both be absent, "
            f"missing {missing_query_boundary_keys} in {data_path}"
        )

    if not np.issubdtype(link_ids.dtype, np.integer):
        raise TypeError(f"'link_ids' must be an integer array in {data_path}, got {link_ids.dtype}")
    if not np.issubdtype(sharp_point_flag.dtype, np.bool_):
        raise TypeError(
            f"'sharp_point_flag' must be a boolean array in {data_path}, got {sharp_point_flag.dtype}"
        )
    if query_boundary_distance_link_ids is not None:
        if query_boundary_distance_link_ids.ndim != 1:
            raise ValueError(
                "'query_boundary_distance_link_ids' must have shape (K,), "
                f"got {query_boundary_distance_link_ids.shape} in {data_path}"
            )
        if not np.issubdtype(query_boundary_distance_link_ids.dtype, np.integer):
            raise TypeError(
                "'query_boundary_distance_link_ids' must be an integer array in "
                f"{data_path}, got {query_boundary_distance_link_ids.dtype}"
            )
        if query_boundary_distance_to_raw_link is None:
            raise KeyError(
                f"Missing 'query_boundary_distance_to_raw_link' in {data_path}"
            )
        if query_boundary_distance_to_raw_link.shape != (
            points.shape[0],
            query_boundary_distance_link_ids.shape[0],
        ):
            raise ValueError(
                "'query_boundary_distance_to_raw_link' must have shape "
                f"({points.shape[0]}, {query_boundary_distance_link_ids.shape[0]}), "
                f"got {query_boundary_distance_to_raw_link.shape} in {data_path}"
            )
        if not np.issubdtype(query_boundary_distance_to_raw_link.dtype, np.floating):
            raise TypeError(
                "'query_boundary_distance_to_raw_link' must be a floating array in "
                f"{data_path}, got {query_boundary_distance_to_raw_link.dtype}"
            )
        if np.any(~np.isfinite(query_boundary_distance_to_raw_link)):
            raise ValueError(
                f"'query_boundary_distance_to_raw_link' contains non-finite values in {data_path}"
            )
        if np.any(query_boundary_distance_to_raw_link < 0.0):
            raise ValueError(
                f"'query_boundary_distance_to_raw_link' must be non-negative in {data_path}"
            )
    invalid_negative_link_ids = np.unique(link_ids[link_ids < -1])
    if len(invalid_negative_link_ids) > 0:
        raise ValueError(
            f"'link_ids' may only use -1 as a sentinel for unlabeled points in {data_path}, "
            f"got {invalid_negative_link_ids.tolist()}"
        )

    valid_link_ids = _validate_links(links, data_path)

    annotated_link_ids = set(np.unique(link_ids[link_ids >= 0]).tolist())
    if not annotated_link_ids:
        raise ValueError(
            "No valid labeled query points are available; all link IDs are undefined (-1)"
        )
    if not annotated_link_ids.issubset(valid_link_ids):
        missing_link_ids = sorted(annotated_link_ids - valid_link_ids)
        raise ValueError(
            f"Point annotations reference missing link IDs in {data_path}: {missing_link_ids}"
        )
    if valid_link_ids != annotated_link_ids:
        missing_point_links = sorted(valid_link_ids - annotated_link_ids)
        raise ValueError(
            f"Links without any associated cached points in {data_path}: {missing_point_links}"
        )
    if query_boundary_distance_link_ids is not None:
        cached_boundary_link_ids = set(
            np.asarray(query_boundary_distance_link_ids, dtype=np.int64).tolist()
        )
        if cached_boundary_link_ids != valid_link_ids:
            raise ValueError(
                "Cached query boundary raw-link columns must match the represented link IDs "
                f"in {data_path}, got cached={sorted(cached_boundary_link_ids)} and "
                f"links={sorted(valid_link_ids)}"
            )
    if len(valid_link_ids) < 2:
        raise ValueError(
            f"Expected at least 2 links with cached points in {data_path}, got {len(valid_link_ids)}"
        )

    _validate_joints(joints, valid_link_ids, data_path)
    return True


def _compute_chunk_size(num_paths: int, workers: int) -> int:
    """Returns a reasonable chunk size for parallel validation jobs."""
    return max(32, int(np.ceil(num_paths / max(1, workers * 4))))


def _iter_path_chunks(paths: Sequence[Path], chunk_size: int) -> Iterable[list[Path]]:
    """Yields path chunks of size `chunk_size`."""
    for start_index in range(0, len(paths), chunk_size):
        yield list(paths[start_index : start_index + chunk_size])


def _build_load_check_dataset(paths: Sequence[Path]) -> InstructParticulateDataset:
    """Builds a lightweight dataset used only for `_getitem()` load checks."""
    return InstructParticulateDataset(
        root=paths[0].parent,
        data_paths=paths,
        **DEFAULT_LOAD_CHECK_DATASET_KWARGS,
    )


def _validate_path_chunk_job(
    chunk_paths: Sequence[str],
    run_load_check: bool,
) -> Dict[str, Any]:
    """Validates a chunk of `.npz` files inside one worker process."""
    paths = [Path(path_str) for path_str in chunk_paths]
    dataset: Optional[InstructParticulateDataset] = None
    if run_load_check:
        dataset = _build_load_check_dataset(paths)

    failures: list[str] = []
    invalid_paths: list[str] = []
    passed = 0
    for sample_index, data_path in enumerate(paths):
        try:
            validate_sample(data_path)
            if dataset is not None:
                dataset._getitem(sample_index)
        except Exception as error:
            failures.append(f"{data_path}: {error}")
            invalid_paths.append(str(data_path))
        else:
            passed += 1

    return {
        "total": len(paths),
        "passed": passed,
        "failures": failures,
        "invalid_paths": invalid_paths,
    }


def _build_process_pool_kwargs() -> Dict[str, Any]:
    """Returns optional kwargs for `ProcessPoolExecutor`.

    On POSIX systems, preferring `fork` keeps the validator compatible with
    large in-memory imports and matches the intended "use all cores" workload
    for local dataset validation. Other platforms fall back to the executor
    defaults.
    """
    if os.name != "posix":
        return {}

    start_methods = mp.get_all_start_methods()
    if "fork" not in start_methods:
        return {}
    return {"mp_context": mp.get_context("fork")}


def validate_paths(
    paths: Sequence[Path],
    *,
    workers: int,
    fail_fast: bool = False,
    run_load_check: bool = True,
    chunk_size: Optional[int] = None,
    progress_callback: Optional[
        Callable[[int, int, int, int, int, int], None]
    ] = None,
) -> Dict[str, Any]:
    """Validates cached dataset files and optionally load-checks them.

    Args:
        paths: Dataset files to validate.
        workers: Number of worker processes to use.
        fail_fast: Whether to stop after the first failing chunk.
        run_load_check: Whether to also construct dataset samples via
            `InstructParticulateDataset._getitem()`.
        chunk_size: Optional manual chunk size. When omitted, a chunk size is
            derived from the number of files and workers.
        progress_callback: Optional callback invoked after each completed chunk
            with `(done, total, passed, failed, completed_chunks, total_chunks)`.

    Returns:
        Summary dictionary with `total`, `passed`, `failed`, `failures`,
        `invalid_paths`, `chunk_size`, and `num_chunks`.
    """
    if not paths:
        raise ValueError("Expected at least one path to validate")
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")

    resolved_chunk_size = chunk_size
    if resolved_chunk_size is None:
        resolved_chunk_size = _compute_chunk_size(len(paths), workers)
    if resolved_chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {resolved_chunk_size}")

    path_chunks = list(_iter_path_chunks(paths, resolved_chunk_size))
    summary: Dict[str, Any] = {
        "total": len(paths),
        "passed": 0,
        "failed": 0,
        "failures": [],
        "invalid_paths": [],
        "chunk_size": resolved_chunk_size,
        "num_chunks": len(path_chunks),
        "run_load_check": run_load_check,
    }

    completed_chunks = 0

    def _update_summary(result: Dict[str, Any]) -> None:
        nonlocal completed_chunks
        summary["passed"] += int(result["passed"])
        summary["failed"] += int(result["total"]) - int(result["passed"])
        summary["failures"].extend(result["failures"])
        summary["invalid_paths"].extend(Path(path_str) for path_str in result["invalid_paths"])
        completed_chunks += 1
        if progress_callback is None:
            return
        done = int(summary["passed"]) + int(summary["failed"])
        progress_callback(
            done,
            int(summary["total"]),
            int(summary["passed"]),
            int(summary["failed"]),
            completed_chunks,
            int(summary["num_chunks"]),
        )

    if workers == 1 or len(path_chunks) == 1:
        for path_chunk in path_chunks:
            result = _validate_path_chunk_job(
                [str(path) for path in path_chunk],
                run_load_check,
            )
            _update_summary(result)
            if fail_fast and summary["failed"] > 0:
                break
        return summary

    with ProcessPoolExecutor(
        max_workers=workers,
        **_build_process_pool_kwargs(),
    ) as executor:
        future_to_chunk_index = {
            executor.submit(
                _validate_path_chunk_job,
                [str(path) for path in path_chunk],
                run_load_check,
            ): chunk_index
            for chunk_index, path_chunk in enumerate(path_chunks, start=1)
        }

        while future_to_chunk_index:
            done_futures, _ = wait(
                future_to_chunk_index,
                return_when=FIRST_COMPLETED,
            )
            for future in done_futures:
                future_to_chunk_index.pop(future, None)
                _update_summary(future.result())

            if fail_fast and summary["failed"] > 0:
                for pending_future in list(future_to_chunk_index):
                    pending_future.cancel()
                break

    return summary


def _confirm_delete_invalid_paths(invalid_paths: Sequence[Path]) -> bool:
    """Prompts before deleting invalid cached samples."""
    if not invalid_paths:
        return False

    file_label = "file" if len(invalid_paths) == 1 else "files"
    try:
        response = input(
            f"Delete the {len(invalid_paths)} invalid .npz {file_label} listed above? [y/N]: "
        )
    except EOFError:
        print("Skipping deletion because no confirmation was provided.", flush=True)
        return False

    return response.strip().lower() in {"y", "yes"}


def _delete_invalid_paths(invalid_paths: Sequence[Path]) -> list[str]:
    """Deletes invalid cached samples and returns deletion errors, if any."""
    deletion_errors: list[str] = []
    for invalid_path in invalid_paths:
        try:
            invalid_path.unlink()
        except FileNotFoundError:
            deletion_errors.append(f"{invalid_path}: file was already missing during deletion")
        except OSError as error:
            deletion_errors.append(f"{invalid_path}: failed to delete invalid file ({error})")
    return deletion_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate cached articulated dataset .npz files.")
    parser.add_argument("path", type=Path, help="Path to a dataset directory or a single .npz file.")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first validation error.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes to use. Defaults to all available CPU cores.",
    )
    parser.add_argument(
        "--skip-load-check",
        action="store_true",
        help="Only run static `.npz` validation without constructing dataset samples.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional number of files processed per worker job.",
    )
    args = parser.parse_args()

    paths = list(_iter_npz_paths(args.path))
    if not paths:
        raise ValueError(f"No .npz files found under {args.path}")

    def _progress_callback(
        done: int,
        total: int,
        passed: int,
        failed: int,
        completed_chunks: int,
        total_chunks: int,
    ) -> None:
        print(
            f"[{completed_chunks}/{total_chunks} chunks] "
            f"done={done}/{total} passed={passed} failed={failed}",
            flush=True,
        )

    summary = validate_paths(
        paths,
        workers=args.workers,
        fail_fast=args.fail_fast,
        run_load_check=not args.skip_load_check,
        chunk_size=args.chunk_size,
        progress_callback=_progress_callback,
    )

    failures = list(summary["failures"])
    if failures:
        for failure in failures:
            print(failure)
        invalid_paths = list(dict.fromkeys(summary["invalid_paths"]))
        if _confirm_delete_invalid_paths(invalid_paths):
            deletion_errors = _delete_invalid_paths(invalid_paths)
            deleted_count = len(invalid_paths) - len(deletion_errors)
            print(f"Deleted {deleted_count} invalid .npz file(s).")
            for deletion_error in deletion_errors:
                print(deletion_error)
        else:
            print("Kept invalid .npz file(s) on disk.")
        print(
            f"Validation failed: {summary['failed']} / {summary['total']} files had errors."
        )
        return 1

    print(f"Validation passed: {summary['passed']} / {summary['total']} files are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
