"""I/O helpers for particulate-compatible evaluation exports and GT caches."""

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover - optional for environments without mesh deps.
    trimesh = None

from instruct_particulate.utils.articulation_utils import (
    get_prismatic_articulation_range,
    get_revolute_articulation_range,
    plucker_to_axis_point,
)
from instruct_particulate.utils.data_utils import load_obj_raw_preserve
from instruct_particulate.utils.meta_root_utils import load_meta_root_annotation

JsonLikeDict = dict[str, Any]
MotionHierarchy = Sequence[tuple[int, int]]

UNREFINED_PREDICTION_SUFFIX = "_unrefined"
PART_ID_SCHEME_KEY = "part_id_scheme"
PROMPT_SLOT_PART_ID_SCHEME = "prompt_slot"
REVOLUTE_RANGE_SUPERVISION_VALID_KEY = "revolute_range_supervision_valid"
PRISMATIC_RANGE_SUPERVISION_VALID_KEY = "prismatic_range_supervision_valid"
CONTINUOUS_REVOLUTE_MASK_KEY = "continuous_revolute_mask"
MATCHING_MODE_SLOT_ALIGNED = "slot_aligned"
MATCHING_MODE_HUNGARIAN = "hungarian"
REFINED_VARIANT_KEY = "refined"
UNREFINED_VARIANT_KEY = "unrefined"


def _require_trimesh(caller_name: str) -> None:
    """Raises a clear error if `trimesh` is unavailable.

    Args:
        caller_name: Name of the public helper requiring `trimesh`.

    Raises:
        ImportError: If `trimesh` could not be imported.
    """
    if trimesh is None:
        raise ImportError(f"trimesh is required for {caller_name} but is not installed")


def decode_scalar_string(value: Any) -> str | None:
    """Decodes a scalar numpy/string payload into a Python string.

    Args:
        value: Value loaded from JSON or `.npz`.

    Returns:
        Decoded string, or `None` when the value cannot be interpreted as one.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_scalar_string(value.item())
    return None


def part_id_scheme_from_payload(payload: Mapping[str, Any]) -> str | None:
    """Returns the encoded part-ID scheme from a prediction or GT payload.

    Args:
        payload: Mapping loaded from an evaluation `.npz`.

    Returns:
        Encoded part-ID scheme, or `None` when absent.
    """
    return decode_scalar_string(payload.get(PART_ID_SCHEME_KEY))


def motion_arrays_from_kinematic_metadata(
    links: Sequence[Mapping[str, Any]],
    joints: Sequence[Mapping[str, Any]],
) -> JsonLikeDict:
    """Converts preprocessed link/joint metadata into evaluation motion arrays.

    Args:
        links: Link records from `kinematic.json`.
        joints: Joint records from `kinematic.json`.

    Returns:
        A particulate-compatible motion dictionary containing hierarchy, motion
        type masks, axes, and limits.
    """
    sorted_links = sorted(links, key=lambda item: int(item["link_id"]))
    sorted_joints = sorted(joints, key=lambda item: int(item["joint_id"]))
    num_links = max((int(link["link_id"]) for link in sorted_links), default=-1) + 1

    is_part_revolute = np.zeros((num_links,), dtype=np.bool_)
    is_part_prismatic = np.zeros((num_links,), dtype=np.bool_)
    revolute_plucker = np.zeros((num_links, 6), dtype=np.float32)
    revolute_range = np.zeros((num_links, 2), dtype=np.float32)
    prismatic_axis = np.zeros((num_links, 3), dtype=np.float32)
    prismatic_range = np.zeros((num_links, 2), dtype=np.float32)
    revolute_range_supervision_valid = np.zeros((num_links,), dtype=np.bool_)
    prismatic_range_supervision_valid = np.zeros((num_links,), dtype=np.bool_)
    continuous_revolute_mask = np.zeros((num_links,), dtype=np.bool_)
    motion_hierarchy: list[tuple[int, int]] = []

    for joint in sorted_joints:
        parent_link_id = int(joint["parent_link_id"])
        child_link_id = int(joint["child_link_id"])
        motion_hierarchy.append((parent_link_id, child_link_id))

        revolute_axis = joint.get("revolute_axis")
        if revolute_axis is not None:
            is_part_revolute[child_link_id] = True
            revolute_plucker[child_link_id] = np.asarray(revolute_axis, dtype=np.float32)

        revolute_limits = get_revolute_articulation_range(dict(joint))
        if revolute_limits is not None:
            revolute_range[child_link_id] = np.asarray(revolute_limits, dtype=np.float32)
        if joint.get("revolute_range") is not None:
            revolute_range_supervision_valid[child_link_id] = True
        elif revolute_axis is not None:
            continuous_revolute_mask[child_link_id] = True

        prismatic_axis_plucker = joint.get("prismatic_axis")
        if prismatic_axis_plucker is not None:
            is_part_prismatic[child_link_id] = True
            prismatic_axis[child_link_id] = plucker_to_axis_point(
                np.asarray(prismatic_axis_plucker, dtype=np.float32)
            )[0].astype(np.float32, copy=False)

        prismatic_limits = get_prismatic_articulation_range(dict(joint))
        if prismatic_limits is not None:
            prismatic_range[child_link_id] = np.asarray(prismatic_limits, dtype=np.float32)
        if joint.get("prismatic_range") is not None:
            prismatic_range_supervision_valid[child_link_id] = True

    return {
        "motion_hierarchy": motion_hierarchy,
        "is_part_revolute": is_part_revolute,
        "is_part_prismatic": is_part_prismatic,
        "revolute_plucker": revolute_plucker,
        "revolute_range": revolute_range,
        "prismatic_axis": prismatic_axis,
        "prismatic_range": prismatic_range,
        REVOLUTE_RANGE_SUPERVISION_VALID_KEY: revolute_range_supervision_valid,
        PRISMATIC_RANGE_SUPERVISION_VALID_KEY: prismatic_range_supervision_valid,
        CONTINUOUS_REVOLUTE_MASK_KEY: continuous_revolute_mask,
    }


def joint_records_from_motion_arrays(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Builds joint-like records aligned with one motion payload.

    Args:
        payload: Prediction or GT payload containing motion arrays.

    Returns:
        Joint records keyed by the payload's own part-ID scheme.
    """
    motion_hierarchy = np.asarray(payload["motion_hierarchy"], dtype=np.int32)
    is_part_revolute = np.asarray(payload["is_part_revolute"], dtype=np.bool_)
    is_part_prismatic = np.asarray(payload["is_part_prismatic"], dtype=np.bool_)
    revolute_plucker = np.asarray(payload["revolute_plucker"], dtype=np.float32)
    revolute_range = np.asarray(payload["revolute_range"], dtype=np.float32)
    prismatic_axis = np.asarray(payload["prismatic_axis"], dtype=np.float32)
    prismatic_range = np.asarray(payload["prismatic_range"], dtype=np.float32)
    prismatic_plucker = payload.get("prismatic_plucker")

    joint_records: list[dict[str, Any]] = []
    for joint_id, (parent_link_id, child_link_id) in enumerate(motion_hierarchy.tolist()):
        child_link_id = int(child_link_id)
        joint_records.append(
            {
                "joint_id": int(joint_id),
                "parent_link_id": int(parent_link_id),
                "child_link_id": child_link_id,
                "is_revolute": bool(is_part_revolute[child_link_id]),
                "is_prismatic": bool(is_part_prismatic[child_link_id]),
                "revolute_axis": (
                    revolute_plucker[child_link_id].astype(np.float32).tolist()
                    if bool(is_part_revolute[child_link_id])
                    else None
                ),
                "revolute_range": (
                    revolute_range[child_link_id].astype(np.float32).tolist()
                    if bool(is_part_revolute[child_link_id])
                    else None
                ),
                "prismatic_axis": (
                    np.asarray(prismatic_plucker[child_link_id], dtype=np.float32).tolist()
                    if prismatic_plucker is not None and bool(is_part_prismatic[child_link_id])
                    else (
                        prismatic_axis[child_link_id].astype(np.float32).tolist()
                        if bool(is_part_prismatic[child_link_id])
                        else None
                    )
                ),
                "prismatic_range": (
                    prismatic_range[child_link_id].astype(np.float32).tolist()
                    if bool(is_part_prismatic[child_link_id])
                    else None
                ),
            }
        )
    return joint_records


def remap_meta_root_to_prompt_slots(
    face_part_ids: np.ndarray,
    links: Sequence[Mapping[str, Any]],
    joints: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], dict[int, int]]:
    """Remaps raw meta-root link IDs into contiguous prompt-slot IDs.

    Args:
        face_part_ids: Raw face segmentation labels from the preprocessed asset.
        links: Raw link records from the preprocessed asset.
        joints: Raw joint records from the preprocessed asset.

    Returns:
        `(remapped_face_part_ids, remapped_links, remapped_joints, original_to_slot)`.

    Raises:
        ValueError: If the face segmentation or joints reference unknown links.
    """
    sorted_link_ids = sorted({int(link["link_id"]) for link in links})
    original_to_slot = {
        original_link_id: slot_id
        for slot_id, original_link_id in enumerate(sorted_link_ids)
    }
    if not original_to_slot:
        raise ValueError("Expected at least one link record when building eval payloads")

    face_part_ids = np.asarray(face_part_ids, dtype=np.int32)
    remapped_face_part_ids = face_part_ids.copy()
    unique_face_part_ids = set(np.unique(face_part_ids).tolist())
    unknown_face_part_ids = sorted(unique_face_part_ids - set(original_to_slot))
    if unknown_face_part_ids:
        raise ValueError(
            "Face segmentation references missing link records: "
            f"{unknown_face_part_ids}"
        )
    for original_link_id, slot_id in original_to_slot.items():
        remapped_face_part_ids[face_part_ids == int(original_link_id)] = int(slot_id)

    remapped_links: list[dict[str, Any]] = []
    for link in links:
        remapped_link = dict(link)
        remapped_link["link_id"] = int(original_to_slot[int(link["link_id"])])
        remapped_links.append(remapped_link)

    remapped_joints: list[dict[str, Any]] = []
    for joint in joints:
        parent_link_id = int(joint["parent_link_id"])
        child_link_id = int(joint["child_link_id"])
        if parent_link_id not in original_to_slot or child_link_id not in original_to_slot:
            raise ValueError(
                "Joint references missing link records after remapping: "
                f"parent={parent_link_id}, child={child_link_id}"
            )
        remapped_joint = dict(joint)
        remapped_joint["parent_link_id"] = int(original_to_slot[parent_link_id])
        remapped_joint["child_link_id"] = int(original_to_slot[child_link_id])
        remapped_joints.append(remapped_joint)
    return remapped_face_part_ids, remapped_links, remapped_joints, original_to_slot


def _sample_mesh_points(
    mesh_path: Path,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly samples points and face indices from a mesh file."""
    _require_trimesh("_sample_mesh_points")
    vertices, faces = load_obj_raw_preserve(mesh_path)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    points, face_indices = mesh.sample(int(num_points), return_index=True)
    return np.asarray(points, dtype=np.float32), np.asarray(face_indices, dtype=np.int32)


def _sample_trimesh_points(
    mesh: trimesh.Trimesh,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly samples points and face indices from one in-memory mesh."""
    _require_trimesh("_sample_trimesh_points")
    points, face_indices = mesh.sample(int(num_points), return_index=True)
    return np.asarray(points, dtype=np.float32), np.asarray(face_indices, dtype=np.int32)


def _build_sampled_payload(
    *,
    face_part_ids: np.ndarray,
    motion_payload: Mapping[str, Any],
    points: np.ndarray,
    face_indices: np.ndarray,
    part_id_scheme: str | None,
) -> JsonLikeDict:
    """Builds one evaluation payload view from face labels and motion arrays."""
    face_part_ids = np.asarray(face_part_ids, dtype=np.int32)
    if face_part_ids.ndim != 1:
        raise ValueError(f"Expected face_part_ids to be 1D, got shape {face_part_ids.shape}")
    if np.any(face_part_ids < 0):
        raise ValueError(
            "Evaluation export does not support negative face part IDs: "
            f"{np.unique(face_part_ids[face_part_ids < 0]).tolist()}"
        )
    sampled_part_ids = face_part_ids[np.asarray(face_indices, dtype=np.int32)]
    payload = {
        "points": np.asarray(points, dtype=np.float32),
        "part_ids": np.asarray(sampled_part_ids, dtype=np.int32),
        "face_part_ids": face_part_ids.astype(np.int32, copy=False),
        "motion_hierarchy": np.asarray(motion_payload["motion_hierarchy"], dtype=np.int32),
        "is_part_revolute": np.asarray(motion_payload["is_part_revolute"], dtype=np.bool_),
        "is_part_prismatic": np.asarray(motion_payload["is_part_prismatic"], dtype=np.bool_),
        "revolute_plucker": np.asarray(motion_payload["revolute_plucker"], dtype=np.float32),
        "revolute_range": np.asarray(motion_payload["revolute_range"], dtype=np.float32),
        "prismatic_axis": np.asarray(motion_payload["prismatic_axis"], dtype=np.float32),
        "prismatic_range": np.asarray(motion_payload["prismatic_range"], dtype=np.float32),
    }
    if "prismatic_plucker" in motion_payload:
        payload["prismatic_plucker"] = np.asarray(
            motion_payload["prismatic_plucker"],
            dtype=np.float32,
        )
    if part_id_scheme is not None:
        payload[PART_ID_SCHEME_KEY] = np.asarray(part_id_scheme)
    return payload


def _suffix_prediction_payload(payload: JsonLikeDict, suffix: str) -> JsonLikeDict:
    """Adds a suffix to every key in a prediction payload."""
    return {f"{key}{suffix}": value for key, value in payload.items()}


def _build_prediction_export_payload(
    *,
    face_part_ids: np.ndarray,
    motion_hierarchy: MotionHierarchy,
    is_part_revolute: np.ndarray,
    is_part_prismatic: np.ndarray,
    revolute_plucker: np.ndarray,
    revolute_range: np.ndarray,
    prismatic_axis: np.ndarray,
    prismatic_range: np.ndarray,
    prismatic_plucker: np.ndarray | None,
) -> JsonLikeDict:
    """Builds one prediction payload block written to `eval/pred.npz`."""
    payload: JsonLikeDict = {
        PART_ID_SCHEME_KEY: np.asarray(PROMPT_SLOT_PART_ID_SCHEME),
        "face_part_ids": np.asarray(face_part_ids, dtype=np.int32),
        "motion_hierarchy": np.asarray(motion_hierarchy, dtype=np.int32),
        "is_part_revolute": np.asarray(is_part_revolute, dtype=np.bool_),
        "is_part_prismatic": np.asarray(is_part_prismatic, dtype=np.bool_),
        "revolute_plucker": np.asarray(revolute_plucker, dtype=np.float32),
        "revolute_range": np.asarray(revolute_range, dtype=np.float32),
        "prismatic_axis": np.asarray(prismatic_axis, dtype=np.float32),
        "prismatic_range": np.asarray(prismatic_range, dtype=np.float32),
    }
    if prismatic_plucker is not None:
        payload["prismatic_plucker"] = np.asarray(prismatic_plucker, dtype=np.float32)
    return payload


def apply_joint_range_evaluation_policy(
    prediction: Mapping[str, Any],
    gt_cache: Mapping[str, Any],
) -> JsonLikeDict:
    """Adjusts predicted joint ranges for articulated-state evaluation.

    Policy:
    - Continuous revolute joints use `[0, 2π]` on the prediction side.
    - Joints with finite range supervision keep the predicted range.
    - All other joints are treated as static by zeroing their ranges.

    Args:
        prediction: Prediction payload in prompt-slot space.
        gt_cache: Prompt-slot GT cache payload carrying supervision masks.

    Returns:
        Copy of `prediction` with adjusted `revolute_range` and
        `prismatic_range`.
    """
    adjusted_prediction: JsonLikeDict = {}
    for key, value in prediction.items():
        if isinstance(value, np.ndarray):
            adjusted_prediction[key] = np.asarray(value).copy()
        else:
            adjusted_prediction[key] = value

    missing_gt_keys = [
        key
        for key in (
            REVOLUTE_RANGE_SUPERVISION_VALID_KEY,
            PRISMATIC_RANGE_SUPERVISION_VALID_KEY,
            CONTINUOUS_REVOLUTE_MASK_KEY,
        )
        if key not in gt_cache
    ]
    if missing_gt_keys:
        raise ValueError(
            "GT cache is missing required joint-range policy fields "
            f"{missing_gt_keys}. Rebuild the GT payload with the current evaluation helpers."
        )

    revolute_range = np.asarray(adjusted_prediction["revolute_range"], dtype=np.float32)
    prismatic_range = np.asarray(adjusted_prediction["prismatic_range"], dtype=np.float32)
    revolute_range_supervision_valid = np.asarray(
        gt_cache[REVOLUTE_RANGE_SUPERVISION_VALID_KEY],
        dtype=np.bool_,
    )
    prismatic_range_supervision_valid = np.asarray(
        gt_cache[PRISMATIC_RANGE_SUPERVISION_VALID_KEY],
        dtype=np.bool_,
    )
    continuous_revolute_mask = np.asarray(
        gt_cache[CONTINUOUS_REVOLUTE_MASK_KEY],
        dtype=np.bool_,
    )

    if revolute_range.shape[0] != revolute_range_supervision_valid.shape[0]:
        raise ValueError(
            "Prediction/GT part count mismatch for revolute range policy: "
            f"{revolute_range.shape[0]} vs {revolute_range_supervision_valid.shape[0]}"
        )
    if prismatic_range.shape[0] != prismatic_range_supervision_valid.shape[0]:
        raise ValueError(
            "Prediction/GT part count mismatch for prismatic range policy: "
            f"{prismatic_range.shape[0]} vs {prismatic_range_supervision_valid.shape[0]}"
        )

    revolute_range[~revolute_range_supervision_valid & ~continuous_revolute_mask] = 0.0
    prismatic_range[~prismatic_range_supervision_valid] = 0.0
    revolute_range[continuous_revolute_mask] = np.asarray(
        [0.0, 2.0 * np.pi],
        dtype=np.float32,
    )

    adjusted_prediction["revolute_range"] = revolute_range
    adjusted_prediction["prismatic_range"] = prismatic_range
    return adjusted_prediction


def save_evaluation_prediction(
    *,
    output_dir: Path,
    mesh: trimesh.Trimesh,
    face_part_ids: np.ndarray,
    face_part_ids_unrefined: np.ndarray | None = None,
    motion_hierarchy: MotionHierarchy,
    is_part_revolute: np.ndarray,
    is_part_prismatic: np.ndarray,
    revolute_plucker: np.ndarray,
    revolute_range: np.ndarray,
    prismatic_axis: np.ndarray,
    prismatic_range: np.ndarray,
    prismatic_plucker: np.ndarray | None = None,
) -> Path:
    """Writes evaluation `pred.obj` and `pred.npz` files in prompt-slot space."""
    _require_trimesh("save_evaluation_prediction")

    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(eval_dir / "pred.obj")

    prediction_payload = _build_prediction_export_payload(
        face_part_ids=face_part_ids,
        motion_hierarchy=motion_hierarchy,
        is_part_revolute=is_part_revolute,
        is_part_prismatic=is_part_prismatic,
        revolute_plucker=revolute_plucker,
        revolute_range=revolute_range,
        prismatic_axis=prismatic_axis,
        prismatic_range=prismatic_range,
        prismatic_plucker=prismatic_plucker,
    )
    if face_part_ids_unrefined is not None:
        unrefined_prediction_payload = _build_prediction_export_payload(
            face_part_ids=face_part_ids_unrefined,
            motion_hierarchy=motion_hierarchy,
            is_part_revolute=is_part_revolute,
            is_part_prismatic=is_part_prismatic,
            revolute_plucker=revolute_plucker,
            revolute_range=revolute_range,
            prismatic_axis=prismatic_axis,
            prismatic_range=prismatic_range,
            prismatic_plucker=prismatic_plucker,
        )
        prediction_payload.update(
            _suffix_prediction_payload(
                unrefined_prediction_payload,
                UNREFINED_PREDICTION_SUFFIX,
            )
        )
    np.savez(eval_dir / "pred.npz", **prediction_payload)
    return eval_dir


def build_prediction_payload_variants_from_mesh(
    *,
    mesh: trimesh.Trimesh,
    face_part_ids: np.ndarray,
    motion_hierarchy: MotionHierarchy,
    is_part_revolute: np.ndarray,
    is_part_prismatic: np.ndarray,
    revolute_plucker: np.ndarray,
    revolute_range: np.ndarray,
    prismatic_axis: np.ndarray,
    prismatic_range: np.ndarray,
    num_points: int = 100_000,
    face_part_ids_unrefined: np.ndarray | None = None,
    prismatic_plucker: np.ndarray | None = None,
) -> dict[str, JsonLikeDict]:
    """Builds refined and optional unrefined prediction payloads from one mesh.

    Args:
        mesh: Predicted mesh in normalized evaluation coordinates.
        face_part_ids: Refined face labels indexed in prompt-slot space.
        motion_hierarchy: Predicted motion tree edges.
        is_part_revolute: Revolute mask indexed by prompt slot.
        is_part_prismatic: Prismatic mask indexed by prompt slot.
        revolute_plucker: Predicted revolute axes in prompt-slot space.
        revolute_range: Predicted revolute limits in prompt-slot space.
        prismatic_axis: Predicted prismatic directions in prompt-slot space.
        prismatic_range: Predicted prismatic limits in prompt-slot space.
        num_points: Number of surface points sampled from ``mesh``.
        face_part_ids_unrefined: Optional face labels before connected-component
            refinement.
        prismatic_plucker: Optional predicted prismatic Plucker lines.

    Returns:
        Mapping from variant name to sampled prediction payload.
    """
    _require_trimesh("build_prediction_payload_variants_from_mesh")

    points, face_indices = _sample_trimesh_points(mesh, num_points)
    motion_payload: JsonLikeDict = {
        "motion_hierarchy": np.asarray(motion_hierarchy, dtype=np.int32),
        "is_part_revolute": np.asarray(is_part_revolute, dtype=np.bool_),
        "is_part_prismatic": np.asarray(is_part_prismatic, dtype=np.bool_),
        "revolute_plucker": np.asarray(revolute_plucker, dtype=np.float32),
        "revolute_range": np.asarray(revolute_range, dtype=np.float32),
        "prismatic_axis": np.asarray(prismatic_axis, dtype=np.float32),
        "prismatic_range": np.asarray(prismatic_range, dtype=np.float32),
    }
    if prismatic_plucker is not None:
        motion_payload["prismatic_plucker"] = np.asarray(
            prismatic_plucker,
            dtype=np.float32,
        )

    prediction_variants = {
        REFINED_VARIANT_KEY: _build_sampled_payload(
            face_part_ids=face_part_ids,
            motion_payload=motion_payload,
            points=points,
            face_indices=face_indices,
            part_id_scheme=PROMPT_SLOT_PART_ID_SCHEME,
        )
    }
    if face_part_ids_unrefined is not None:
        prediction_variants[UNREFINED_VARIANT_KEY] = _build_sampled_payload(
            face_part_ids=face_part_ids_unrefined,
            motion_payload=motion_payload,
            points=points,
            face_indices=face_indices,
            part_id_scheme=PROMPT_SLOT_PART_ID_SCHEME,
        )
    return prediction_variants


def build_gt_payload_from_meta_root(
    *,
    meta_root: Path,
    num_points: int = 100_000,
) -> JsonLikeDict:
    """Builds one prompt-slot-aligned GT payload from a preprocessed asset."""
    _require_trimesh("build_gt_payload_from_meta_root")

    original_obj_path, face_part_ids, links, joints, _ = load_meta_root_annotation(meta_root)
    (
        remapped_face_part_ids,
        remapped_links,
        remapped_joints,
        original_to_slot,
    ) = remap_meta_root_to_prompt_slots(face_part_ids, links, joints)
    points, face_indices = _sample_mesh_points(original_obj_path, num_points)
    sampled_part_ids = remapped_face_part_ids[np.asarray(face_indices, dtype=np.int32)]
    motion_arrays = motion_arrays_from_kinematic_metadata(remapped_links, remapped_joints)

    return {
        PART_ID_SCHEME_KEY: np.asarray(PROMPT_SLOT_PART_ID_SCHEME),
        "points": np.asarray(points, dtype=np.float32),
        "part_ids": np.asarray(sampled_part_ids, dtype=np.int32),
        "face_part_ids": remapped_face_part_ids.astype(np.int32, copy=False),
        "motion_hierarchy": np.asarray(motion_arrays["motion_hierarchy"], dtype=np.int32),
        "is_part_revolute": motion_arrays["is_part_revolute"],
        "is_part_prismatic": motion_arrays["is_part_prismatic"],
        "revolute_plucker": motion_arrays["revolute_plucker"],
        "revolute_range": motion_arrays["revolute_range"],
        "prismatic_axis": motion_arrays["prismatic_axis"],
        "prismatic_range": motion_arrays["prismatic_range"],
        REVOLUTE_RANGE_SUPERVISION_VALID_KEY: motion_arrays[REVOLUTE_RANGE_SUPERVISION_VALID_KEY],
        PRISMATIC_RANGE_SUPERVISION_VALID_KEY: motion_arrays[PRISMATIC_RANGE_SUPERVISION_VALID_KEY],
        CONTINUOUS_REVOLUTE_MASK_KEY: motion_arrays[CONTINUOUS_REVOLUTE_MASK_KEY],
        "original_link_ids": np.asarray(
            [
                original_link_id
                for original_link_id, _ in sorted(
                    original_to_slot.items(),
                    key=lambda item: item[1],
                )
            ],
            dtype=np.int32,
        ),
        "link_names": np.asarray(
            [
                str(link.get("name", f"link_{int(link['link_id'])}"))
                for link in sorted(remapped_links, key=lambda item: int(item["link_id"]))
            ],
            dtype=np.str_,
        ),
    }


def cache_gt_from_meta_root(
    *,
    meta_root: Path,
    output_path: Path,
    num_points: int = 100_000,
) -> JsonLikeDict:
    """Caches one preprocessed asset into prompt-slot-aligned GT `.npz` format."""
    payload = build_gt_payload_from_meta_root(
        meta_root=meta_root,
        num_points=num_points,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    return payload
