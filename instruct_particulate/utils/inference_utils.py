import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from instruct_particulate.utils.articulation_utils import (
    plucker_to_axis_point,
    transform_plucker,
)
from instruct_particulate.utils.data_utils import (
    sample_points as sample_mesh_points,
    sample_points_per_face as sample_mesh_points_per_face,
)


def write_json(path: Path, payload: Any) -> None:
    """Writes a JSON payload with stable indentation."""
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def axis_point_to_plucker_torch(axis: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """Converts an axis-point line representation to Plucker coordinates."""
    axis = F.normalize(axis, dim=-1, eps=1e-8)
    moment = torch.linalg.cross(axis, point, dim=-1)
    return torch.cat((axis, moment), dim=-1)


def fit_axis_to_closest_points_torch(
    query_points: torch.Tensor,
    closest_axis_points: torch.Tensor,
    *,
    direction_hint: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fits an axis from closest-point targets using a closed-form projection objective.

    The fitted line minimizes

        sum_i ||c_i - proj_L(q_i)||^2

    where `q_i` are `query_points`, `c_i` are `closest_axis_points`, and `L`
    is the recovered axis. This enforces both that the predicted closest points
    lie on the axis and that `(q_i - c_i)` is perpendicular to the axis.

    Returns `(axis_direction, axis_point)`, where `axis_point` is the point on
    the recovered line closest to the origin.
    """
    if query_points.shape != closest_axis_points.shape:
        raise ValueError(
            "query_points and closest_axis_points must share the same shape, "
            f"got {tuple(query_points.shape)} and {tuple(closest_axis_points.shape)}"
        )
    if query_points.ndim != 2 or query_points.shape[-1] != 3:
        raise ValueError(
            f"Expected query_points and closest_axis_points to have shape (N, 3), got {tuple(query_points.shape)}"
        )
    if query_points.numel() == 0:
        zero = query_points.new_zeros(3)
        return zero, zero

    result_dtype = (
        query_points.dtype
        if query_points.dtype in (torch.float32, torch.float64)
        else torch.float32
    )
    solve_dtype = torch.float32 if query_points.device.type == "mps" else torch.float64
    query_points = query_points.to(dtype=solve_dtype)
    closest_axis_points = closest_axis_points.to(dtype=solve_dtype)
    if direction_hint is None:
        direction_hint = query_points.new_zeros(3)
    else:
        direction_hint = direction_hint.to(
            device=query_points.device,
            dtype=solve_dtype,
        )

    closest_axis_mean = closest_axis_points.mean(dim=0)
    centered_closest_axis_points = closest_axis_points - closest_axis_mean.unsqueeze(0)
    query_residuals = query_points - closest_axis_points
    objective_matrix = (
        query_residuals.transpose(0, 1) @ query_residuals
        - centered_closest_axis_points.transpose(0, 1) @ centered_closest_axis_points
    )

    direction_hint_norm = torch.linalg.norm(direction_hint)
    normalized_direction_hint = direction_hint / direction_hint_norm.clamp_min(1e-12)
    eigenvalues, eigenvectors = torch.linalg.eigh(objective_matrix)
    axis_direction = eigenvectors[:, 0]
    if direction_hint_norm > 1e-8:
        eigenspace_scale = torch.maximum(
            eigenvalues.abs().max(),
            objective_matrix.abs().max(),
        ).clamp_min(1.0)
        eigenspace_tol = eigenspace_scale * (
            1e-8 if solve_dtype == torch.float64 else 1e-5
        )
        smallest_eigenspace = eigenvectors[
            :,
            eigenvalues <= (eigenvalues[0] + eigenspace_tol),
        ]
        projected_hint = smallest_eigenspace @ (
            smallest_eigenspace.transpose(0, 1) @ normalized_direction_hint
        )
        if torch.linalg.norm(projected_hint) > 1e-8:
            axis_direction = projected_hint
    if direction_hint_norm > 1e-8 and torch.dot(axis_direction, normalized_direction_hint) < 0:
        axis_direction = -axis_direction
    axis_direction = F.normalize(axis_direction, dim=-1, eps=1e-8)
    fallback_direction = torch.where(
        direction_hint_norm > 1e-8,
        normalized_direction_hint,
        query_points.new_tensor([1.0, 0.0, 0.0]),
    )
    if not torch.isfinite(axis_direction).all() or torch.linalg.norm(axis_direction) <= 1e-8:
        axis_direction = fallback_direction

    axis_point = closest_axis_mean - axis_direction * torch.dot(axis_direction, closest_axis_mean)
    if not torch.isfinite(axis_point).all():
        axis_point = closest_axis_mean
    return axis_direction.to(dtype=result_dtype), axis_point.to(dtype=result_dtype)


def estimate_prismatic_limit_torch(
    current_points: torch.Tensor,
    target_points: torch.Tensor,
    axis_direction: torch.Tensor,
) -> torch.Tensor:
    """Fits a shared prismatic displacement along `axis_direction`."""
    if current_points.numel() == 0:
        return current_points.new_zeros(())

    axis_direction = F.normalize(axis_direction, dim=-1, eps=1e-8)
    offsets = target_points - current_points
    return (offsets * axis_direction.unsqueeze(0)).sum(dim=-1).mean()


def estimate_revolute_limit_torch(
    current_points: torch.Tensor,
    target_points: torch.Tensor,
    axis_direction: torch.Tensor,
    axis_point: torch.Tensor,
) -> torch.Tensor:
    """Fits a shared revolute angle around the given axis via a global least-squares solve."""
    if current_points.numel() == 0:
        return current_points.new_zeros(())

    axis_direction = F.normalize(axis_direction, dim=-1, eps=1e-8)

    def _project_perpendicular(points: torch.Tensor) -> torch.Tensor:
        offsets = points - axis_point.unsqueeze(0)
        parallel = (offsets * axis_direction.unsqueeze(0)).sum(dim=-1, keepdim=True)
        return offsets - parallel * axis_direction.unsqueeze(0)

    current_perp = _project_perpendicular(current_points)
    target_perp = _project_perpendicular(target_points)
    cosine_term = (current_perp * target_perp).sum(dim=-1).sum()
    sine_term = torch.linalg.cross(current_perp, target_perp, dim=-1)
    sine_term = (sine_term * axis_direction.unsqueeze(0)).sum(dim=-1).sum()
    return torch.atan2(sine_term, cosine_term)


def strip_text_model_state(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("encoder.text_model.")
    }


_ALLOWED_MISSING_CHECKPOINT_KEYS = frozenset(
    {
        "encoder.no_text_conditioning_embedding",
    }
)


def load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.resolved.yaml"
    if not config_path.exists():
        config_path = run_dir / "config.source.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find config.resolved.yaml or config.source.yaml under {run_dir}"
        )
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}, got {type(config).__name__}")
    return config


def configure_runtime_environment(config: Mapping[str, Any]) -> None:
    runtime_config = config.get("runtime", {})
    if not isinstance(runtime_config, Mapping):
        return
    configured_hf_cache_dir = Path(
        runtime_config.get("hf_cache_dir", ".cache/huggingface")
    ).expanduser()
    fallback_hf_cache_dir = (Path.cwd() / ".cache" / "huggingface").resolve()
    candidate_hf_cache_dirs = [
        configured_hf_cache_dir.resolve(),
        fallback_hf_cache_dir,
    ]
    hf_cache_dir = None
    hf_hub_cache = None
    for candidate_hf_cache_dir in candidate_hf_cache_dirs:
        candidate_hf_hub_cache = candidate_hf_cache_dir / "hub"
        try:
            candidate_hf_hub_cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        hf_cache_dir = candidate_hf_cache_dir
        hf_hub_cache = candidate_hf_hub_cache
        break
    if hf_cache_dir is None or hf_hub_cache is None:
        raise OSError(
            "Could not create a writable Hugging Face cache directory. "
            f"Tried: {candidate_hf_cache_dirs}"
        )
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hf_hub_cache)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_model_checkpoint_for_inference(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing_keys, unexpected_keys = model.load_state_dict(
        strip_text_model_state(checkpoint["model"]),
        strict=False,
    )
    non_text_missing_keys = [
        key
        for key in missing_keys
        if not key.startswith("encoder.text_model.")
        and key not in _ALLOWED_MISSING_CHECKPOINT_KEYS
    ]
    if non_text_missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model state mismatch. "
            f"Missing keys: {non_text_missing_keys}; unexpected keys: {unexpected_keys}"
        )
    model.to(device)
    model.eval()
    return checkpoint


def resolve_inference_sampling_config(
    config: Mapping[str, Any],
) -> tuple[int, int | None, float]:
    datasets_config = config.get("datasets", {})
    if not isinstance(datasets_config, Mapping):
        raise ValueError("Config is missing a datasets mapping")
    shared_config = datasets_config.get("shared")
    if shared_config is not None and not isinstance(shared_config, Mapping):
        raise ValueError(
            "datasets.shared must be a mapping when provided, "
            f"got {type(shared_config).__name__}"
        )

    dataset_entries: list[Mapping[str, Any]] = []
    for split_name in ("train", "val"):
        split_entries = datasets_config.get(split_name, [])
        if isinstance(split_entries, Sequence) and not isinstance(split_entries, (str, bytes)):
            dataset_entries.extend(
                entry for entry in split_entries if isinstance(entry, Mapping)
            )

    if not dataset_entries:
        raise ValueError("Config does not contain any dataset entries under datasets.train or datasets.val")

    if shared_config is not None:
        missing_keys = [
            key for key in ("num_shape_points", "num_query_points") if key not in shared_config
        ]
        if missing_keys:
            raise KeyError(
                "datasets.shared must define num_shape_points and num_query_points, "
                f"missing {missing_keys}"
            )
        num_shape_points = int(shared_config["num_shape_points"])
        default_num_query_points = int(shared_config["num_query_points"])
    else:
        num_shape_points_values = {
            int(entry["num_shape_points"])
            for entry in dataset_entries
            if "num_shape_points" in entry
        }
        if len(num_shape_points_values) != 1:
            raise ValueError(
                "Inference requires a single num_shape_points value across saved dataset configs, "
                f"got {sorted(num_shape_points_values)}"
            )

        num_query_points_values = {
            int(entry["num_query_points"])
            for entry in dataset_entries
            if "num_query_points" in entry
        }
        default_num_query_points = None
        if len(num_query_points_values) == 1:
            default_num_query_points = next(iter(num_query_points_values))
        num_shape_points = next(iter(num_shape_points_values))

    sharp_point_ratio_values = {
        float(entry.get("sharp_point_ratio", 0.5))
        for entry in dataset_entries
    }
    if len(sharp_point_ratio_values) != 1:
        raise ValueError(
            "Inference requires a single sharp_point_ratio across saved dataset configs, "
            f"got {sorted(sharp_point_ratio_values)}"
        )

    return (
        num_shape_points,
        default_num_query_points,
        next(iter(sharp_point_ratio_values)),
    )


def validate_link_names(
    link_names: Sequence[str],
    *,
    require_unique: bool = True,
) -> list[str]:
    cleaned_link_names = [str(link_name).strip() for link_name in link_names]
    if not cleaned_link_names:
        raise ValueError("At least one link name is required")
    if any(not link_name for link_name in cleaned_link_names):
        raise ValueError("Link names must be non-empty strings")
    if require_unique and len(set(cleaned_link_names)) != len(cleaned_link_names):
        raise ValueError(f"Link names must be unique, got {cleaned_link_names}")
    return cleaned_link_names


def build_joint_tensors(
    num_links: int,
    joint_specs: Sequence[tuple[int, int, str]],
    *,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    if num_links <= 0:
        raise ValueError(f"num_links must be positive, got {num_links}")

    parent_by_child: dict[int, int] = {}
    for parent_link_id, child_link_id, joint_type in joint_specs:
        if not 0 <= parent_link_id < num_links:
            raise ValueError(
                f"Parent link ID {parent_link_id} is outside the valid range [0, {num_links - 1}]"
            )
        if not 0 <= child_link_id < num_links:
            raise ValueError(
                f"Child link ID {child_link_id} is outside the valid range [0, {num_links - 1}]"
            )
        if parent_link_id == child_link_id:
            raise ValueError("Joint parent and child link IDs must be different")
        if child_link_id in parent_by_child:
            raise ValueError(f"Child link {child_link_id} appears in more than one joint")
        parent_by_child[child_link_id] = parent_link_id

    for start_link_id in range(num_links):
        seen: set[int] = set()
        current_link_id = start_link_id
        while current_link_id in parent_by_child:
            if current_link_id in seen:
                raise ValueError("Joint specs must form an acyclic forest")
            seen.add(current_link_id)
            current_link_id = parent_by_child[current_link_id]

    num_joints = len(joint_specs)
    joint_connections = torch.full(
        (1, num_joints, 2),
        fill_value=-1,
        dtype=torch.long,
        device=device,
    )
    joint_valid_flag = torch.zeros((1, num_joints), dtype=torch.bool, device=device)
    is_revolute = torch.zeros((1, num_joints), dtype=torch.bool, device=device)
    is_prismatic = torch.zeros((1, num_joints), dtype=torch.bool, device=device)

    for joint_idx, (parent_link_id, child_link_id, joint_type) in enumerate(joint_specs):
        joint_connections[0, joint_idx] = torch.tensor(
            [parent_link_id, child_link_id],
            dtype=torch.long,
            device=device,
        )
        joint_valid_flag[0, joint_idx] = True
        if joint_type == "revolute":
            is_revolute[0, joint_idx] = True
        else:
            is_prismatic[0, joint_idx] = True

    return {
        "joint_connections": joint_connections,
        "joint_valid_flag": joint_valid_flag,
        "is_revolute": is_revolute,
        "is_prismatic": is_prismatic,
    }


def _to_batched_float_tensor(array: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array).float().unsqueeze(0).to(device)


def _build_inference_batch(
    *,
    shape_points: np.ndarray,
    shape_normals: np.ndarray,
    query_points: np.ndarray,
    query_normals: np.ndarray,
    num_links: int,
    joint_connections: torch.Tensor,
    joint_valid_flag: torch.Tensor,
    is_revolute: torch.Tensor,
    is_prismatic: torch.Tensor,
    device: torch.device,
    link_point_prompts: torch.Tensor | None = None,
    link_point_prompt_normals: torch.Tensor | None = None,
    link_point_prompt_dropout_eligible: torch.Tensor | None = None,
    link_text_prompts: Sequence[str] | None = None,
    link_text_embeddings: torch.Tensor | None = None,
) -> dict[str, Any]:
    batch = {
        "shape_points": _to_batched_float_tensor(shape_points, device=device),
        "shape_point_normals": _to_batched_float_tensor(shape_normals, device=device),
        "query_points": _to_batched_float_tensor(query_points, device=device),
        "query_point_normals": _to_batched_float_tensor(query_normals, device=device),
        "link_point_prompts": None,
        "link_point_prompt_normals": None,
        "link_valid_flag": torch.ones((1, num_links), dtype=torch.bool, device=device),
        "link_text_prompts": None if link_text_prompts is None else [list(link_text_prompts)],
        "link_text_embeddings": None,
        "link_ids": None,
        "joint_connections": joint_connections.unsqueeze(0).to(device),
        "joint_valid_flag": joint_valid_flag.unsqueeze(0).to(device),
        "is_revolute": is_revolute.unsqueeze(0).to(device),
        "is_prismatic": is_prismatic.unsqueeze(0).to(device),
    }
    if link_point_prompts is not None:
        batch["link_point_prompts"] = link_point_prompts.unsqueeze(0).to(device)
    if link_point_prompt_normals is not None:
        batch["link_point_prompt_normals"] = (
            link_point_prompt_normals.unsqueeze(0).to(device)
        )
    if link_point_prompt_dropout_eligible is not None:
        batch["link_point_prompt_dropout_eligible"] = (
            link_point_prompt_dropout_eligible.unsqueeze(0).to(device)
        )
    if link_text_embeddings is not None:
        batch["link_text_embeddings"] = link_text_embeddings.unsqueeze(0).to(device)
    return batch


def _link_point_prompt_dropout_eligibility_from_link_names(
    link_names: Sequence[str],
) -> torch.Tensor:
    """Marks links whose text prompt is unique within the sample."""
    cleaned_link_names = validate_link_names(link_names, require_unique=False)
    name_counts: dict[str, int] = {}
    for link_name in cleaned_link_names:
        name_counts[link_name] = name_counts.get(link_name, 0) + 1
    return torch.tensor(
        [name_counts[link_name] == 1 for link_name in cleaned_link_names],
        dtype=torch.bool,
    )


def prepare_inference_batch_from_mesh_with_prompts(
    mesh,
    *,
    num_shape_points: int,
    num_query_points: int,
    num_query_points_per_face_for_seg: int | None = None,
    sharp_point_ratio: float,
    link_names: Sequence[str],
    joint_specs: Sequence[tuple[int, int, str]],
    device: torch.device,
    link_point_prompts: torch.Tensor | None = None,
    link_point_prompt_normals: torch.Tensor | None = None,
    require_unique_link_names: bool = True,
) -> tuple[dict[str, Any], np.ndarray]:
    """Builds a mesh inference batch, optionally with explicit link point prompts."""
    shape_points, shape_normals, _, _ = sample_mesh_points(
        mesh,
        num_points=num_shape_points,
        sharp_point_ratio=sharp_point_ratio,
    )
    if num_query_points_per_face_for_seg is None:
        query_points, query_normals, _, query_face_indices = sample_mesh_points(
            mesh,
            num_points=num_query_points,
            sharp_point_ratio=0.0,
        )
    else:
        query_points, query_normals, _, query_face_indices = sample_mesh_points_per_face(
            mesh,
            num_points_per_face=int(num_query_points_per_face_for_seg),
        )

    cleaned_link_names = validate_link_names(
        link_names,
        require_unique=require_unique_link_names,
    )
    if (link_point_prompts is None) != (link_point_prompt_normals is None):
        raise ValueError(
            "link_point_prompts and link_point_prompt_normals must both be provided or both be None"
        )
    if link_point_prompts is not None:
        if link_point_prompts.ndim != 2 or link_point_prompts.shape != (
            len(cleaned_link_names),
            3,
        ):
            raise ValueError(
                "link_point_prompts must have shape (num_links, 3), "
                f"got {tuple(link_point_prompts.shape)} for {len(cleaned_link_names)} links"
            )
        assert link_point_prompt_normals is not None
        if link_point_prompt_normals.ndim != 2 or link_point_prompt_normals.shape != (
            len(cleaned_link_names),
            3,
        ):
            raise ValueError(
                "link_point_prompt_normals must have shape (num_links, 3), "
                f"got {tuple(link_point_prompt_normals.shape)} for {len(cleaned_link_names)} links"
            )

    joint_tensors = build_joint_tensors(len(cleaned_link_names), joint_specs, device=device)
    link_point_prompt_dropout_eligible = _link_point_prompt_dropout_eligibility_from_link_names(
        cleaned_link_names
    )
    batch = _build_inference_batch(
        shape_points=shape_points,
        shape_normals=shape_normals,
        query_points=query_points,
        query_normals=query_normals,
        num_links=len(cleaned_link_names),
        joint_connections=joint_tensors["joint_connections"][0],
        joint_valid_flag=joint_tensors["joint_valid_flag"][0],
        is_revolute=joint_tensors["is_revolute"][0],
        is_prismatic=joint_tensors["is_prismatic"][0],
        device=device,
        link_point_prompts=link_point_prompts,
        link_point_prompt_normals=link_point_prompt_normals,
        link_point_prompt_dropout_eligible=link_point_prompt_dropout_eligible,
        link_text_prompts=cleaned_link_names,
    )
    return batch, query_face_indices.astype(np.int64, copy=False)


def prepare_inference_batch_from_mesh(
    mesh,
    *,
    num_shape_points: int,
    num_query_points: int,
    num_query_points_per_face_for_seg: int | None = None,
    sharp_point_ratio: float,
    link_names: Sequence[str],
    joint_specs: Sequence[tuple[int, int, str]],
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    return prepare_inference_batch_from_mesh_with_prompts(
        mesh,
        num_shape_points=num_shape_points,
        num_query_points=num_query_points,
        num_query_points_per_face_for_seg=num_query_points_per_face_for_seg,
        sharp_point_ratio=sharp_point_ratio,
        link_names=link_names,
        joint_specs=joint_specs,
        device=device,
        link_point_prompts=None,
        link_point_prompt_normals=None,
        require_unique_link_names=True,
    )


def _sample_points_on_triangles(triangle_vertices: np.ndarray) -> np.ndarray:
    """Samples one point uniformly from each triangle in `triangle_vertices`.

    Args:
        triangle_vertices: Array with shape `(N, 3, 3)`.

    Returns:
        Sampled points with shape `(N, 3)`.
    """
    if triangle_vertices.ndim != 3 or triangle_vertices.shape[1:] != (3, 3):
        raise ValueError(
            "triangle_vertices must have shape (N, 3, 3), "
            f"got {triangle_vertices.shape}"
        )
    if triangle_vertices.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    r1 = np.random.random((triangle_vertices.shape[0], 1))
    r2 = np.random.random((triangle_vertices.shape[0], 1))
    sqrt_r1 = np.sqrt(r1)
    barycentric = np.concatenate(
        (
            1.0 - sqrt_r1,
            sqrt_r1 * (1.0 - r2),
            sqrt_r1 * r2,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    return (triangle_vertices * barycentric[:, :, None]).sum(axis=1).astype(
        np.float32,
        copy=False,
    )


def sample_balanced_query_points_from_face_seg(
    mesh,
    *,
    face_part_ids: np.ndarray,
    num_points: int,
) -> dict[str, np.ndarray]:
    """Samples query points with near-equal support per predicted part.

    The sampler draws points uniformly from triangle area within each predicted
    part, then balances the total sample budget across the remaining part IDs.

    Args:
        mesh: Mesh whose face ordering matches `face_part_ids`.
        face_part_ids: Predicted part ID for each face.
        num_points: Total number of query points to sample.

    Returns:
        Dictionary containing sampled query points, normals, face indices,
        assigned part IDs, and per-part query counts.
    """
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")

    mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
    face_part_ids = np.asarray(face_part_ids, dtype=np.int64)
    if face_part_ids.shape != (mesh_faces.shape[0],):
        raise ValueError(
            "face_part_ids must have one entry per face, "
            f"got {face_part_ids.shape} for {mesh_faces.shape[0]} faces"
        )

    unique_part_ids = np.unique(face_part_ids[face_part_ids >= 0]).astype(
        np.int64,
        copy=False,
    )
    if unique_part_ids.size == 0:
        raise ValueError("face_part_ids must contain at least one non-negative part ID")

    samples_per_part = np.full(
        unique_part_ids.shape,
        fill_value=num_points // unique_part_ids.size,
        dtype=np.int64,
    )
    samples_per_part[: num_points % unique_part_ids.size] += 1

    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    mesh_face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
    mesh_face_areas = np.asarray(mesh.area_faces, dtype=np.float64)

    point_chunks: list[np.ndarray] = []
    normal_chunks: list[np.ndarray] = []
    face_index_chunks: list[np.ndarray] = []
    part_id_chunks: list[np.ndarray] = []

    for part_id, part_query_count in zip(unique_part_ids, samples_per_part, strict=True):
        if int(part_query_count) <= 0:
            continue
        candidate_face_indices = np.flatnonzero(face_part_ids == int(part_id)).astype(
            np.int64,
            copy=False,
        )
        if candidate_face_indices.size == 0:
            raise ValueError(
                f"Cannot sample query points for part_id={int(part_id)} because it has no faces"
            )

        candidate_face_areas = mesh_face_areas[candidate_face_indices]
        if np.any(candidate_face_areas > 0.0):
            face_probabilities = candidate_face_areas / candidate_face_areas.sum()
            sampled_face_indices = np.random.choice(
                candidate_face_indices,
                size=int(part_query_count),
                replace=True,
                p=face_probabilities,
            ).astype(np.int64, copy=False)
        else:
            sampled_face_indices = np.random.choice(
                candidate_face_indices,
                size=int(part_query_count),
                replace=True,
            ).astype(np.int64, copy=False)

        sampled_triangles = mesh_vertices[mesh_faces[sampled_face_indices]]
        sampled_points = _sample_points_on_triangles(sampled_triangles)
        point_chunks.append(sampled_points)
        normal_chunks.append(
            mesh_face_normals[sampled_face_indices].astype(np.float32, copy=False)
        )
        face_index_chunks.append(sampled_face_indices)
        part_id_chunks.append(
            np.full((int(part_query_count),), int(part_id), dtype=np.int64)
        )

    sampled_points = np.concatenate(point_chunks, axis=0)
    sampled_normals = np.concatenate(normal_chunks, axis=0)
    sampled_face_indices = np.concatenate(face_index_chunks, axis=0)
    sampled_part_ids = np.concatenate(part_id_chunks, axis=0)
    permutation = np.random.permutation(len(sampled_points)).astype(np.int64, copy=False)
    sampled_points = sampled_points[permutation]
    sampled_normals = sampled_normals[permutation]
    sampled_face_indices = sampled_face_indices[permutation]
    sampled_part_ids = sampled_part_ids[permutation]

    return {
        "query_points": sampled_points.astype(np.float32, copy=False),
        "query_normals": sampled_normals.astype(np.float32, copy=False),
        "face_indices": sampled_face_indices.astype(np.int64, copy=False),
        "link_ids": sampled_part_ids.astype(np.int64, copy=False),
        "unique_part_ids": unique_part_ids.astype(np.int64, copy=False),
        "query_counts": samples_per_part.astype(np.int64, copy=False),
    }


def build_joint_refit_batch(
    batch: Mapping[str, Any],
    *,
    query_points: np.ndarray,
    query_normals: np.ndarray,
    assigned_link_ids: np.ndarray,
) -> dict[str, Any]:
    """Clones an inference batch with new balanced query samples and fixed link IDs."""
    device = batch["shape_points"].device
    refit_batch = dict(batch)
    refit_batch["query_points"] = _to_batched_float_tensor(
        np.asarray(query_points, dtype=np.float32),
        device=device,
    )
    refit_batch["query_point_normals"] = _to_batched_float_tensor(
        np.asarray(query_normals, dtype=np.float32),
        device=device,
    )
    refit_batch["link_ids"] = (
        torch.from_numpy(np.asarray(assigned_link_ids, dtype=np.int64))
        .long()
        .unsqueeze(0)
        .to(device)
    )
    return refit_batch


def run_joint_refit_from_face_seg(
    model: Any,
    *,
    batch: Mapping[str, Any],
    mesh: Any,
    face_part_ids: np.ndarray,
    num_query_points: int,
    query_batch_size: int = 8192,
    no_point_prompt_for_unique_text: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Runs the second inference pass from balanced query samples.

    Stage 1 segmentation stays unchanged. This helper performs a second forward
    pass that resamples query points evenly across the refined face
    segmentation, then decodes joint parameters while also measuring the
    segmentation confidence of the assigned part ID for each sampled query.
    """
    refit_sampling = sample_balanced_query_points_from_face_seg(
        mesh,
        face_part_ids=face_part_ids,
        num_points=int(num_query_points),
    )
    refit_batch = build_joint_refit_batch(
        batch,
        query_points=refit_sampling["query_points"],
        query_normals=refit_sampling["query_normals"],
        assigned_link_ids=refit_sampling["link_ids"],
    )
    refit_output = run_batched_model_inference(
        model,
        query_batch_size=int(query_batch_size),
        no_point_prompt_for_unique_text=bool(no_point_prompt_for_unique_text),
        **refit_batch,
    )
    return refit_output, refit_sampling


def run_batched_model_inference(
    model: Any,
    *,
    shape_points: torch.Tensor,
    shape_point_normals: torch.Tensor,
    query_points: torch.Tensor,
    query_point_normals: torch.Tensor,
    link_point_prompts: torch.Tensor | None,
    link_point_prompt_normals: torch.Tensor | None,
    link_valid_flag: torch.Tensor,
    joint_connections: torch.Tensor,
    joint_valid_flag: torch.Tensor,
    is_revolute: torch.Tensor,
    is_prismatic: torch.Tensor,
    link_point_prompt_dropout_eligible: torch.Tensor | None = None,
    link_text_prompts: Sequence[Sequence[str]] | None = None,
    link_text_embeddings: torch.Tensor | None = None,
    link_ids: torch.Tensor | None = None,
    query_batch_size: int = 8192,
    no_point_prompt_for_unique_text: bool = False,
    decode_joint_parameters: bool = True,
) -> dict[str, Any]:
    if query_batch_size <= 0:
        raise ValueError(f"query_batch_size must be positive, got {query_batch_size}")
    forced_no_point_prompt_mask: torch.Tensor | None = None
    if no_point_prompt_for_unique_text and link_point_prompts is not None:
        if link_point_prompt_dropout_eligible is None:
            raise ValueError(
                "no_point_prompt_for_unique_text requires link_point_prompt_dropout_eligible "
                "when link_point_prompts are provided"
            )
        if link_point_prompt_dropout_eligible.shape != link_valid_flag.shape:
            raise ValueError(
                "link_point_prompt_dropout_eligible must match link_valid_flag shape, "
                f"got {tuple(link_point_prompt_dropout_eligible.shape)} and "
                f"{tuple(link_valid_flag.shape)}"
            )
        forced_no_point_prompt_mask = (
            link_point_prompt_dropout_eligible.to(
                device=link_valid_flag.device,
                dtype=torch.bool,
            )
            & link_valid_flag
        )

    with torch.inference_mode():
        (
            shape_pretrained_features,
            query_pretrained_features,
            link_point_prompt_pretrained_features,
        ) = model.encoder._compute_pretrained_point_features(
            shape_points=shape_points,
            query_points=query_points,
            link_point_prompts=link_point_prompts,
        )
        shape_latents = model.encoder.encode_shape(
            shape_points,
            shape_point_normals,
            pretrained_features=shape_pretrained_features,
        )
        initial_link_latents = model.encoder.encode_links(
            link_point_prompts=link_point_prompts,
            link_point_prompt_normals=link_point_prompt_normals,
            link_valid_flag=link_valid_flag,
            link_point_prompt_dropout_eligible=link_point_prompt_dropout_eligible,
            forced_no_point_prompt_mask=forced_no_point_prompt_mask,
            link_point_prompt_pretrained_features=link_point_prompt_pretrained_features,
            link_text_prompts=link_text_prompts,
            link_text_embeddings=link_text_embeddings,
        )

        link_block_outputs: list[torch.Tensor] = []
        running_link_latents = initial_link_latents
        for link_to_shape_cross_attn, link_self_attn in zip(
            model.encoder.link_to_shape_cross_attn,
            model.encoder.link_self_attn,
            strict=True,
        ):
            running_link_latents = link_to_shape_cross_attn(
                running_link_latents,
                shape_latents,
                query_mask=link_valid_flag,
            )
            running_link_latents = link_self_attn(running_link_latents, mask=link_valid_flag)
            link_block_outputs.append(running_link_latents)
        link_latents = model.encoder.link_output_norm(running_link_latents)
        link_latents = link_latents.masked_fill(~link_valid_flag.unsqueeze(-1), 0)
        segmentation_link_latents = model.build_segmentation_link_latents(
            link_latents=link_latents,
            joint_connections=joint_connections,
            joint_valid_flag=joint_valid_flag,
        )

        all_query_latents: list[torch.Tensor] | None = (
            [] if decode_joint_parameters else None
        )
        all_segmentation_logits: list[torch.Tensor] = []
        all_joint_decoding_link_ids: list[torch.Tensor] = []
        all_joint_decoding_confidences: list[torch.Tensor] = []
        all_revolute_motion_points: list[torch.Tensor] = []
        all_prismatic_motion_points: list[torch.Tensor] = []

        num_queries = query_points.shape[1]
        for start_idx in range(0, num_queries, query_batch_size):
            end_idx = min(start_idx + query_batch_size, num_queries)
            query_points_chunk = query_points[:, start_idx:end_idx]
            query_point_normals_chunk = query_point_normals[:, start_idx:end_idx]

            query_latents_chunk = model.encoder._embed_point_tokens(
                query_points_chunk,
                query_point_normals_chunk,
                pretrained_features=(
                    None
                    if query_pretrained_features is None
                    else query_pretrained_features[:, start_idx:end_idx]
                ),
            )
            for block_link_latents, query_to_shape_cross_attn, query_to_link_cross_attn in zip(
                link_block_outputs,
                model.encoder.query_to_shape_cross_attn,
                model.encoder.query_to_link_cross_attn,
                strict=True,
            ):
                query_latents_chunk = query_to_shape_cross_attn(query_latents_chunk, shape_latents)
                query_latents_chunk = query_to_link_cross_attn(
                    query_latents_chunk,
                    block_link_latents,
                    context_mask=link_valid_flag,
                )
            query_latents_chunk = model.encoder.query_output_norm(query_latents_chunk)
            segmentation_logits_chunk = model.decode_segmentation(
                query_latents=query_latents_chunk,
                link_latents=segmentation_link_latents,
                link_valid_flag=link_valid_flag,
            )

            if all_query_latents is not None:
                all_query_latents.append(query_latents_chunk)
            all_segmentation_logits.append(segmentation_logits_chunk)

            joint_decoding_link_ids_chunk: torch.Tensor | None = None
            joint_decoding_confidences_chunk: torch.Tensor | None = None
            if link_ids is not None or (
                decode_joint_parameters
                and model.joint_decode_type in {
                    "overparametrized",
                    "overparam+dir",
                    "overparam+singledir",
                }
            ):
                segmentation_probabilities_chunk = segmentation_logits_chunk.softmax(dim=-1)
                if link_ids is not None:
                    joint_decoding_link_ids_chunk = link_ids[:, start_idx:end_idx]
                    joint_decoding_confidences_chunk = segmentation_probabilities_chunk.gather(
                        dim=-1,
                        index=joint_decoding_link_ids_chunk.clamp_min(0).unsqueeze(-1),
                    ).squeeze(-1)
                    joint_decoding_confidences_chunk = joint_decoding_confidences_chunk.masked_fill(
                        joint_decoding_link_ids_chunk < 0,
                        0.0,
                    )
                else:
                    (
                        joint_decoding_confidences_chunk,
                        joint_decoding_link_ids_chunk,
                    ) = segmentation_probabilities_chunk.max(dim=-1)
                all_joint_decoding_link_ids.append(joint_decoding_link_ids_chunk)
                all_joint_decoding_confidences.append(joint_decoding_confidences_chunk)

            if (
                decode_joint_parameters
                and model.joint_decode_type in {
                    "overparametrized",
                    "overparam+dir",
                    "overparam+singledir",
                }
            ):
                assert joint_decoding_link_ids_chunk is not None
                revolute_motion_points_chunk, prismatic_motion_points_chunk = (
                    model._decode_joint_motion_points(
                        query_latents=query_latents_chunk,
                        link_latents=link_latents,
                        assigned_link_ids=joint_decoding_link_ids_chunk,
                        joint_connections=joint_connections,
                    )
                )
                all_revolute_motion_points.append(revolute_motion_points_chunk)
                all_prismatic_motion_points.append(prismatic_motion_points_chunk)

        query_latents = (
            torch.cat(all_query_latents, dim=1)
            if all_query_latents is not None
            else None
        )
        segmentation_logits = torch.cat(all_segmentation_logits, dim=1)

        output: dict[str, Any] = {
            "shape_latents": shape_latents,
            "query_latents": query_latents,
            "query_points": query_points,
            "link_latents": link_latents,
            "segmentation_logits": segmentation_logits,
            "joint_connections": joint_connections,
            "joint_valid_flag": joint_valid_flag,
            "is_revolute": is_revolute,
            "is_prismatic": is_prismatic,
        }

        if not decode_joint_parameters:
            output.update(
                {
                    "revolute_axis": None,
                    "prismatic_axis": None,
                    "revolute_range": None,
                    "prismatic_range": None,
                    "revolute_closest_axis_points": None,
                    "revolute_closest_axis_points_decoder": None,
                    "revolute_low_points": None,
                    "revolute_high_points": None,
                    "revolute_axis_directions": None,
                    "prismatic_closest_axis_points": None,
                    "prismatic_closest_axis_points_decoder": None,
                    "prismatic_low_points": None,
                    "prismatic_high_points": None,
                    "prismatic_axis_directions": None,
                    "joint_decoding_link_ids": None,
                    "joint_decoding_confidences": None,
                    "joint_query_counts": None,
                    "revolute_parameter_valid": None,
                    "prismatic_parameter_valid": None,
                }
            )
            return output

        joint_decoding_link_ids = None
        joint_decoding_confidences = None
        joint_query_counts = None
        if all_joint_decoding_link_ids:
            joint_decoding_link_ids = torch.cat(all_joint_decoding_link_ids, dim=1)
            joint_decoding_confidences = torch.cat(all_joint_decoding_confidences, dim=1)
            joint_query_counts = _count_assigned_queries_per_joint(
                assigned_link_ids=joint_decoding_link_ids,
                joint_connections=joint_connections,
                joint_valid_flag=joint_valid_flag,
            )

        if model.joint_decode_type in {"plain", "plain+fm"}:
            if joint_query_counts is None:
                revolute_parameter_valid = joint_valid_flag & is_revolute
                prismatic_parameter_valid = joint_valid_flag & is_prismatic
            else:
                revolute_parameter_valid = (
                    joint_valid_flag & is_revolute & (joint_query_counts > 0)
                )
                prismatic_parameter_valid = (
                    joint_valid_flag & is_prismatic & (joint_query_counts > 0)
                )
            revolute_axis, prismatic_axis, revolute_range, prismatic_range = (
                model.decode_joint_parameters(
                    link_latents=link_latents,
                    joint_connections=joint_connections,
                    joint_valid_flag=joint_valid_flag,
                    is_revolute=is_revolute,
                    is_prismatic=is_prismatic,
                )
            )
            output.update(
                {
                    "revolute_axis": revolute_axis,
                    "prismatic_axis": prismatic_axis,
                    "revolute_range": revolute_range,
                    "prismatic_range": prismatic_range,
                    "revolute_closest_axis_points": None,
                    "revolute_low_points": None,
                    "revolute_high_points": None,
                    "revolute_axis_directions": None,
                    "prismatic_closest_axis_points": None,
                    "prismatic_low_points": None,
                    "prismatic_high_points": None,
                    "prismatic_axis_directions": None,
                    "joint_decoding_link_ids": joint_decoding_link_ids,
                    "joint_decoding_confidences": joint_decoding_confidences,
                    "joint_query_counts": joint_query_counts,
                    "revolute_parameter_valid": revolute_parameter_valid,
                    "prismatic_parameter_valid": prismatic_parameter_valid,
                }
            )
            return output

        assert joint_decoding_link_ids is not None
        assert joint_query_counts is not None
        revolute_parameter_valid = (
            joint_valid_flag & is_revolute & (joint_query_counts > 0)
        )
        prismatic_parameter_valid = (
            joint_valid_flag & is_prismatic & (joint_query_counts > 0)
        )
        revolute_motion_points_decoder_raw = torch.cat(all_revolute_motion_points, dim=1)
        prismatic_motion_points_decoder_raw = torch.cat(all_prismatic_motion_points, dim=1)
        revolute_joint_axis_directions = None
        prismatic_joint_axis_directions = None
        if model.joint_decode_type == "overparam+singledir":
            (
                revolute_joint_axis_directions,
                prismatic_joint_axis_directions,
            ) = model._decode_joint_axis_directions(
                link_latents=link_latents,
                joint_connections=joint_connections,
                joint_valid_flag=joint_valid_flag,
                is_revolute=is_revolute,
                is_prismatic=is_prismatic,
            )
        revolute_motion_points_decoder = revolute_motion_points_decoder_raw
        prismatic_motion_points_decoder = prismatic_motion_points_decoder_raw
        revolute_motion_points = model._convert_overparam_motion_points_to_world_coordinates(
            motion_points=revolute_motion_points_decoder,
            query_points=query_points,
            assigned_link_ids=joint_decoding_link_ids,
        )
        prismatic_motion_points = model._convert_overparam_motion_points_to_world_coordinates(
            motion_points=prismatic_motion_points_decoder,
            query_points=query_points,
            assigned_link_ids=joint_decoding_link_ids,
        )
        revolute_axis, prismatic_axis, revolute_range, prismatic_range = (
            model.decode_joint_parameters(
                link_latents=link_latents,
                joint_connections=joint_connections,
                joint_valid_flag=joint_valid_flag,
                is_revolute=is_revolute,
                is_prismatic=is_prismatic,
                query_points=query_points,
                assigned_link_ids=joint_decoding_link_ids,
                decoded_motion_points=(revolute_motion_points, prismatic_motion_points),
                decoded_axis_directions=(
                    None
                    if revolute_joint_axis_directions is None or prismatic_joint_axis_directions is None
                    else (
                        revolute_joint_axis_directions,
                        prismatic_joint_axis_directions,
                    )
                ),
                decoded_motion_points_are_world=True,
            )
        )
        output.update(
            {
                "revolute_axis": revolute_axis,
                "prismatic_axis": prismatic_axis,
                "revolute_range": revolute_range,
                "prismatic_range": prismatic_range,
                "revolute_closest_axis_points": revolute_motion_points[..., :3],
                "revolute_closest_axis_points_decoder": revolute_motion_points_decoder_raw[..., :3],
                "revolute_low_points": revolute_motion_points[..., 3:6],
                "revolute_high_points": revolute_motion_points[..., 6:9],
                "revolute_axis_directions": (
                    revolute_motion_points[..., 9:12]
                    if model.joint_decode_type == "overparam+dir"
                    else revolute_joint_axis_directions
                    if model.joint_decode_type == "overparam+singledir"
                    else None
                ),
                "prismatic_closest_axis_points": prismatic_motion_points[..., :3],
                "prismatic_closest_axis_points_decoder": prismatic_motion_points_decoder_raw[..., :3],
                "prismatic_low_points": prismatic_motion_points[..., 3:6],
                "prismatic_high_points": prismatic_motion_points[..., 6:9],
                "prismatic_axis_directions": (
                    prismatic_motion_points[..., 9:12]
                    if model.joint_decode_type == "overparam+dir"
                    else prismatic_joint_axis_directions
                    if model.joint_decode_type == "overparam+singledir"
                    else None
                ),
                "joint_decoding_link_ids": joint_decoding_link_ids,
                "joint_decoding_confidences": joint_decoding_confidences,
                "joint_query_counts": joint_query_counts,
                "revolute_parameter_valid": revolute_parameter_valid,
                "prismatic_parameter_valid": prismatic_parameter_valid,
            }
        )
        return output


def _count_assigned_queries_per_joint(
    *,
    assigned_link_ids: torch.Tensor,
    joint_connections: torch.Tensor,
    joint_valid_flag: torch.Tensor,
) -> torch.Tensor:
    child_link_ids = joint_connections[..., 1]
    query_counts = (assigned_link_ids.unsqueeze(-1) == child_link_ids.unsqueeze(1)).sum(dim=1)
    return query_counts.masked_fill(~joint_valid_flag, 0)


def motion_arrays_from_model_output(
    output: Mapping[str, Any],
    *,
    num_links: int,
) -> dict[str, np.ndarray]:
    revolute_parameter_valid = np.zeros(num_links, dtype=np.bool_)
    prismatic_parameter_valid = np.zeros(num_links, dtype=np.bool_)
    revolute_plucker = np.zeros((num_links, 6), dtype=np.float32)
    prismatic_plucker = np.zeros((num_links, 6), dtype=np.float32)
    revolute_range = np.zeros((num_links, 2), dtype=np.float32)
    prismatic_range = np.zeros((num_links, 2), dtype=np.float32)

    joint_connections = output["joint_connections"][0].detach().cpu().numpy()
    joint_valid_flag = output["joint_valid_flag"][0].detach().cpu().numpy().astype(bool, copy=False)
    is_revolute = output["is_revolute"][0].detach().cpu().numpy().astype(bool, copy=False)
    is_prismatic = output["is_prismatic"][0].detach().cpu().numpy().astype(bool, copy=False)
    joint_revolute_parameter_valid = output.get("revolute_parameter_valid")
    if joint_revolute_parameter_valid is None:
        joint_revolute_parameter_valid = output["joint_valid_flag"] & output["is_revolute"]
    joint_revolute_parameter_valid = (
        joint_revolute_parameter_valid[0].detach().cpu().numpy().astype(bool, copy=False)
    )
    joint_prismatic_parameter_valid = output.get("prismatic_parameter_valid")
    if joint_prismatic_parameter_valid is None:
        joint_prismatic_parameter_valid = output["joint_valid_flag"] & output["is_prismatic"]
    joint_prismatic_parameter_valid = (
        joint_prismatic_parameter_valid[0].detach().cpu().numpy().astype(bool, copy=False)
    )
    revolute_axis_tensor = output["revolute_axis"][0].detach()
    prismatic_axis_tensor = output["prismatic_axis"][0].detach()
    revolute_limits_tensor = output["revolute_range"][0].detach()
    prismatic_limits_tensor = output["prismatic_range"][0].detach()
    revolute_axis = revolute_axis_tensor.cpu().numpy()
    prismatic_axis = prismatic_axis_tensor.cpu().numpy()
    revolute_limits = revolute_limits_tensor.cpu().numpy()
    prismatic_limits = prismatic_limits_tensor.cpu().numpy()

    for joint_idx in np.flatnonzero(joint_valid_flag):
        child_link_id = int(joint_connections[joint_idx, 1])
        if is_revolute[joint_idx] and joint_revolute_parameter_valid[joint_idx]:
            revolute_parameter_valid[child_link_id] = True
            revolute_plucker[child_link_id] = revolute_axis[joint_idx].astype(np.float32, copy=False)
            revolute_range[child_link_id] = revolute_limits[joint_idx].astype(np.float32, copy=False)
        if is_prismatic[joint_idx] and joint_prismatic_parameter_valid[joint_idx]:
            prismatic_parameter_valid[child_link_id] = True
            prismatic_plucker[child_link_id] = prismatic_axis[joint_idx].astype(np.float32, copy=False)
            prismatic_range[child_link_id] = prismatic_limits[joint_idx].astype(np.float32, copy=False)

    return {
        "is_part_revolute": revolute_parameter_valid.copy(),
        "is_part_prismatic": prismatic_parameter_valid.copy(),
        "revolute_parameter_valid": revolute_parameter_valid,
        "prismatic_parameter_valid": prismatic_parameter_valid,
        "revolute_plucker": revolute_plucker,
        "prismatic_plucker": prismatic_plucker,
        "revolute_range": revolute_range,
        "prismatic_range": prismatic_range,
    }


def denormalize_motion_parameters(
    revolute_plucker: np.ndarray,
    prismatic_plucker: np.ndarray,
    revolute_range: np.ndarray,
    prismatic_range: np.ndarray,
    *,
    center: np.ndarray,
    scale: float,
) -> dict[str, np.ndarray]:
    inverse_transform = np.eye(4, dtype=np.float32)
    inverse_transform[:3, :3] *= np.float32(1.0 / scale)
    inverse_transform[:3, 3] = np.asarray(center, dtype=np.float32)

    denormalized_revolute = np.zeros_like(revolute_plucker, dtype=np.float32)
    denormalized_prismatic = np.zeros_like(prismatic_plucker, dtype=np.float32)
    for link_idx in range(revolute_plucker.shape[0]):
        if np.linalg.norm(revolute_plucker[link_idx, :3]) > 0.0:
            denormalized_revolute[link_idx] = transform_plucker(
                revolute_plucker[link_idx],
                inverse_transform,
            )
        if np.linalg.norm(prismatic_plucker[link_idx, :3]) > 0.0:
            denormalized_prismatic[link_idx] = transform_plucker(
                prismatic_plucker[link_idx],
                inverse_transform,
            )

    return {
        "revolute_plucker": denormalized_revolute,
        "prismatic_plucker": denormalized_prismatic,
        "revolute_range": np.asarray(revolute_range, dtype=np.float32).copy(),
        "prismatic_range": np.asarray(prismatic_range, dtype=np.float32) / np.float32(scale),
    }


def prismatic_directions_from_plucker(prismatic_plucker: np.ndarray) -> np.ndarray:
    prismatic_directions = np.zeros((prismatic_plucker.shape[0], 3), dtype=np.float32)
    for link_idx in range(prismatic_plucker.shape[0]):
        if np.linalg.norm(prismatic_plucker[link_idx, :3]) <= 0.0:
            continue
        axis_direction, _ = plucker_to_axis_point(prismatic_plucker[link_idx])
        prismatic_directions[link_idx] = axis_direction.astype(np.float32, copy=False)
    return prismatic_directions


def build_predicted_kinematic_records(
    link_names: Sequence[str],
    joint_specs: Sequence[tuple[int, int, str]],
    *,
    revolute_plucker: np.ndarray,
    prismatic_plucker: np.ndarray,
    prismatic_axis: np.ndarray | None = None,
    revolute_range: np.ndarray,
    prismatic_range: np.ndarray,
    revolute_parameter_valid: np.ndarray | None = None,
    prismatic_parameter_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    links = [
        {"link_id": int(link_id), "name": str(link_name)}
        for link_id, link_name in enumerate(link_names)
    ]
    joints: list[dict[str, Any]] = []
    for joint_id, (parent_link_id, child_link_id, joint_type) in enumerate(joint_specs):
        joints.append(
            {
                "joint_id": int(joint_id),
                "parent_link_id": int(parent_link_id),
                "child_link_id": int(child_link_id),
                "joint_description": (
                    f"{link_names[parent_link_id]} -> {link_names[child_link_id]} ({joint_type})"
                ),
                "is_revolute": joint_type == "revolute",
                "is_prismatic": joint_type == "prismatic",
            }
        )
    return build_predicted_kinematic_records_from_links_and_joints(
        links,
        joints,
        revolute_plucker=revolute_plucker,
        prismatic_plucker=prismatic_plucker,
        prismatic_axis=prismatic_axis,
        revolute_range=revolute_range,
        prismatic_range=prismatic_range,
        revolute_parameter_valid=revolute_parameter_valid,
        prismatic_parameter_valid=prismatic_parameter_valid,
    )


def _resolve_link_parameter_valid_mask(
    parameter_valid: np.ndarray | None,
    *,
    num_links: int,
) -> np.ndarray:
    if parameter_valid is None:
        return np.ones(num_links, dtype=np.bool_)
    parameter_valid = np.asarray(parameter_valid, dtype=np.bool_)
    if parameter_valid.shape != (num_links,):
        raise ValueError(
            f"Expected parameter-valid mask to have shape ({num_links},), "
            f"got {parameter_valid.shape}"
        )
    return parameter_valid


def build_predicted_joint_records_from_links_and_joints(
    links: Sequence[Mapping[str, Any]],
    joints: Sequence[Mapping[str, Any]],
    *,
    revolute_plucker: np.ndarray,
    prismatic_plucker: np.ndarray,
    prismatic_axis: np.ndarray | None = None,
    revolute_range: np.ndarray,
    prismatic_range: np.ndarray,
    revolute_parameter_valid: np.ndarray | None = None,
    prismatic_parameter_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Builds predicted kinematic records while preserving stored topology.

    Prismatic joints are exported as direction-only whenever possible because
    their line anchor is not physically meaningful. Callers may pass
    ``prismatic_axis`` explicitly; otherwise this derives directions from
    ``prismatic_plucker`` for backward compatibility.
    """
    predicted_links = [
        {
            "link_id": int(link["link_id"]),
            "name": str(link.get("name", f"link_{int(link['link_id'])}")),
        }
        for link in sorted(links, key=lambda item: int(item["link_id"]))
    ]
    num_links = len(predicted_links)
    revolute_parameter_valid_mask = _resolve_link_parameter_valid_mask(
        revolute_parameter_valid,
        num_links=num_links,
    )
    prismatic_parameter_valid_mask = _resolve_link_parameter_valid_mask(
        prismatic_parameter_valid,
        num_links=num_links,
    )
    if prismatic_axis is None:
        prismatic_axis = prismatic_directions_from_plucker(prismatic_plucker)
    else:
        prismatic_axis = np.asarray(prismatic_axis, dtype=np.float32)
        if prismatic_axis.shape != (num_links, 3):
            raise ValueError(
                f"Expected prismatic_axis to have shape ({num_links}, 3), "
                f"got {prismatic_axis.shape}"
            )

    predicted_joints: list[dict[str, Any]] = []
    for joint_idx, joint in enumerate(sorted(joints, key=lambda item: int(item["joint_id"]))):
        child_link_id = int(joint["child_link_id"])
        is_revolute = bool(joint.get("is_revolute", False))
        is_prismatic = bool(joint.get("is_prismatic", False))
        has_revolute_parameters = is_revolute and revolute_parameter_valid_mask[child_link_id]
        has_prismatic_parameters = is_prismatic and prismatic_parameter_valid_mask[child_link_id]
        predicted_joints.append(
            {
                "joint_id": int(joint.get("joint_id", joint_idx)),
                "parent_link_id": int(joint["parent_link_id"]),
                "child_link_id": child_link_id,
                "joint_description": str(joint.get("joint_description", "")),
                "is_revolute": is_revolute,
                "is_prismatic": is_prismatic,
                "revolute_axis": (
                    np.asarray(revolute_plucker[child_link_id], dtype=np.float32).copy()
                    if has_revolute_parameters
                    else None
                ),
                "prismatic_axis": (
                    np.asarray(prismatic_axis[child_link_id], dtype=np.float32).copy()
                    if has_prismatic_parameters
                    else None
                ),
                "revolute_range": (
                    np.asarray(revolute_range[child_link_id], dtype=np.float32).copy()
                    if has_revolute_parameters
                    else None
                ),
                "prismatic_range": (
                    np.asarray(prismatic_range[child_link_id], dtype=np.float32).copy()
                    if has_prismatic_parameters
                    else None
                ),
            }
        )

    return {"links": predicted_links, "joints": predicted_joints}


def build_predicted_kinematic_records_from_links_and_joints(
    links: Sequence[Mapping[str, Any]],
    joints: Sequence[Mapping[str, Any]],
    *,
    revolute_plucker: np.ndarray,
    prismatic_plucker: np.ndarray,
    prismatic_axis: np.ndarray | None = None,
    revolute_range: np.ndarray,
    prismatic_range: np.ndarray,
    revolute_parameter_valid: np.ndarray | None = None,
    prismatic_parameter_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Builds JSON-serializable predicted kinematic metadata."""
    predicted_records = build_predicted_joint_records_from_links_and_joints(
        links,
        joints,
        revolute_plucker=revolute_plucker,
        prismatic_plucker=prismatic_plucker,
        prismatic_axis=prismatic_axis,
        revolute_range=revolute_range,
        prismatic_range=prismatic_range,
        revolute_parameter_valid=revolute_parameter_valid,
        prismatic_parameter_valid=prismatic_parameter_valid,
    )
    predicted_joints: list[dict[str, Any]] = []
    for joint in predicted_records["joints"]:
        predicted_joint = dict(joint)
        for key in (
            "revolute_axis",
            "prismatic_axis",
            "revolute_range",
            "prismatic_range",
        ):
            value = predicted_joint.get(key)
            if value is not None:
                predicted_joint[key] = np.asarray(value, dtype=np.float32).tolist()
        predicted_joints.append(predicted_joint)
    return {
        "links": predicted_records["links"],
        "joints": predicted_joints,
    }
