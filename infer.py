import argparse
from dataclasses import dataclass
import gc
from pathlib import Path
import traceback
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from instruct_particulate.model import Particulate2ArticulationModel
from instruct_particulate.utils.data_utils import (
    canonicalize_up_dir,
    load_trimesh,
    normalize_mesh,
    up_dir_rotation_matrix,
    reorient_mesh_to_z_up,
)
from instruct_particulate.utils.auto_kinematics_utils import (
    DEFAULT_AUTO_KINEMATICS_AZIMUTHS,
    prepare_auto_kinematics_mesh_batch,
)
from instruct_particulate.utils.export_utils import export_urdf
from instruct_particulate.utils.inference_utils import (
    axis_point_to_plucker_torch,
    build_predicted_joint_records_from_links_and_joints,
    build_predicted_kinematic_records,
    build_predicted_kinematic_records_from_links_and_joints,
    configure_runtime_environment,
    denormalize_motion_parameters,
    estimate_prismatic_limit_torch,
    estimate_revolute_limit_torch,
    fit_axis_to_closest_points_torch,
    load_model_checkpoint_for_inference,
    load_run_config,
    motion_arrays_from_model_output,
    prepare_inference_batch_from_mesh,
    prepare_inference_batch_from_mesh_with_prompts,
    prepare_inference_batch_from_meta_root,
    prismatic_directions_from_plucker,
    prompt_for_link_names_and_joints,
    range_forward_masks_by_child_link,
    resolve_checkpoint_path,
    resolve_input_mode,
    resolve_inference_sampling_config,
    resolve_num_query_points,
    run_joint_refit_from_face_seg,
    run_batched_model_inference,
    suppress_unforwarded_joint_ranges,
    write_json,
    zero_unforwarded_link_ranges,
)
from instruct_particulate.utils.meta_root_utils import (
    is_supported_meta_root,
    load_meta_root_annotation,
)
from instruct_particulate.utils.saved_auto_kinematics_utils import (
    SavedAutoKinematicsRoot,
    is_saved_auto_kinematics_root,
    load_saved_auto_kinematics_root,
)
from instruct_particulate.utils.inference_visualization_utils import (
    compute_link_geometric_mean_confidences,
    print_inference_summary,
    save_joint_overparam_visualization_from_model_output,
    save_point_query_visualization,
    save_predicted_point_query_visualization,
    save_segmented_visualizations,
    select_visualized_link_point_prompts,
)
from instruct_particulate.utils.postprocessing_utils import (
    find_unrefined_part_ids_for_faces,
    refine_face_part_ids_for_inference,
)


_EVALUATION_IMPORT_ERROR: ImportError | None = None
try:
    from instruct_particulate.utils.evaluation_io import save_evaluation_prediction
    from instruct_particulate.utils.evaluation_pipeline import (
        DEFAULT_EVAL_NUM_POINTS,
        existing_result_path_for_input,
        export_and_evaluate_meta_root_prediction,
        load_cached_sample_result,
        per_object_result_path,
        refresh_output_root_summaries,
    )
except ImportError as exc:
    save_evaluation_prediction = None
    DEFAULT_EVAL_NUM_POINTS = None
    existing_result_path_for_input = None
    export_and_evaluate_meta_root_prediction = None
    load_cached_sample_result = None
    per_object_result_path = None
    refresh_output_root_summaries = None
    _EVALUATION_IMPORT_ERROR = exc


@dataclass(frozen=True)
class PreparedMeshGeometry:
    original_mesh: Any
    normalized_mesh: Any
    center: np.ndarray
    scale: float
    up_dir_rotation: np.ndarray
    render_mesh: Any
    render_to_model_rotation: np.ndarray


@dataclass(frozen=True)
class PreparedMeshBatch:
    batch: dict[str, Any]
    query_face_indices: np.ndarray
    link_names: list[str]
    joint_specs: list[tuple[int, int, str]]
    auto_kinematics_metadata: dict[str, Any]
    link_point_prompt_face_ids: np.ndarray | None = None


_INFER_OVERPARAM_AXIS_AABB_HALF_EXTENT_MIN = 1e-4


def _parse_num_query_points_per_face_for_seg(value: str) -> int:
    normalized_value = str(value).strip().lower()
    if normalized_value in {"true", "yes", "on"}:
        return 1
    try:
        parsed_value = int(normalized_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--num-query-points-per-face-for-seg expects a positive integer"
        ) from exc
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            "--num-query-points-per-face-for-seg expects a positive integer"
        )
    return parsed_value


def _require_evaluation_dependencies() -> None:
    if _EVALUATION_IMPORT_ERROR is not None:
        raise ImportError(
            "Evaluation dependencies are unavailable in this environment. "
            "Run without --eval or install the missing evaluation packages."
        ) from _EVALUATION_IMPORT_ERROR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run articulated-part inference for a mesh or preprocessed meta-root export using a "
            "saved training run."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help=(
            "Path to an input mesh, one preprocessed meta-root directory, one saved "
            "auto-kinematics folder, or a directory whose immediate children are "
            "preprocessed meta-root / saved auto-kinematics folders"
        ),
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Training run directory containing config and checkpoints",
    )
    parser.add_argument(
        "--ckpt-idx",
        type=str,
        default=None,
        help=(
            "Checkpoint step to load. Defaults to the step_*.pt file with the largest step count. "
            "Also accepts 'latest' or 'best'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./inference-results"),
        help="Root directory used to save inference outputs",
    )
    parser.add_argument(
        "--num-query-points",
        type=int,
        default=102400,
        help=(
            "Number of query points for the joint-refit pass. Also used for the "
            "segmentation pass unless --num-query-points-per-face-for-seg is set."
        ),
    )
    parser.add_argument(
        "--num-query-points-per-face-for-seg",
        type=_parse_num_query_points_per_face_for_seg,
        nargs="?",
        const=1,
        default=3,
        dest="num_query_points_per_face_for_seg",
        metavar="N",
        help=(
            "For the first segmentation pass only, sample N query points uniformly inside "
            "every mesh face instead of sampling --num-query-points uniformly over mesh "
            "area. Passing the flag without N uses N=1. The second joint-refit pass still "
            "uses --num-query-points."
        ),
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=8192,
        help="Number of query points to forward per batch.",
    )
    parser.add_argument(
        "--joint-decoding-confidence-temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature applied to segmentation softmax confidences before confidence-weighted "
            "overparameterized joint refitting. Values below 1.0 sharpen the weights; values "
            "above 1.0 flatten them."
        ),
    )
    parser.add_argument(
        "--animation-frames",
        type=int,
        default=50,
        help="Number of frames to bake into the animated GLB output for mesh inputs.",
    )
    parser.add_argument(
        "--strict-face-postprocess",
        action="store_true",
        help=(
            "Use strict face-label refinement. Without this flag, the base non-strict path "
            "keeps the unrefined face labels."
        ),
    )
    parser.add_argument(
        "--enforce-connectivity-per-part",
        action="store_true",
        help=(
            "After the base strict/non-strict face pass, also enforce that each part ID "
            "occupies a single connected face group using the distance-graph reassignment "
            "post-processing."
        ),
    )
    parser.add_argument(
        "--export-urdf",
        action="store_true",
        help="Also export a URDF asset under the output directory for mesh inputs.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help=(
            "For meta-root inputs, export particulate-compatible eval files, compute per-object "
            "evaluation metrics under output_dir/<asset_name>/eval, and refresh root-level "
            "overall/per-category summaries. Mesh inputs still export eval files only."
        ),
    )
    parser.add_argument(
        "--no-point-prompt",
        action="store_true",
        help=(
            "For links whose text/name signature is unique within the sample, remove the "
            "point prompt and rely on text conditioning only during inference."
        ),
    )
    parser.add_argument(
        "--auto-kinematics",
        type=str,
        default=None,
        help=(
            "Optional multimodal model ID used to infer mesh kinematics and point prompts "
            "automatically, for example 'gpt-5.4' or 'gemini-3.1-pro-preview'."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=("low", "medium", "high"),
        default="medium",
        help=(
            "Reasoning effort / thinking level for auto-kinematics models "
            "(OpenAI and Gemini)."
        ),
    )
    parser.add_argument(
        "--azimuths",
        type=float,
        nargs="+",
        default=list(DEFAULT_AUTO_KINEMATICS_AZIMUTHS),
        help=(
            "Azimuth angles in degrees used for auto-kinematics mesh rendering. "
            "Only used when --auto-kinematics is set."
        ),
    )
    parser.add_argument(
        "--blender-bin",
        type=Path,
        default=None,
        help=(
            "Optional Blender binary used to render auto-kinematics mesh views. "
            "Defaults to auto-detecting a Blender install under the home directory."
        ),
    )
    parser.add_argument(
        "--up_dir",
        "--up-dir",
        dest="up_dir",
        type=canonicalize_up_dir,
        default="+Z",
        help=(
            "Up direction of a direct mesh input. Supported values are +X, -X, +Y, -Y, +Z, "
            "and -Z (X/Y/Z shorthand is also accepted). Mesh inputs are rotated into +Z-up "
            "before normalization and inference."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip inputs whose deterministic outputs already exist. Without --eval, this uses "
            "output_dir/<input_stem>/metadata.json as the completion marker. For meta-root "
            "inputs with --eval, this reuses the per-object eval JSON when it follows the "
            "current schema."
        ),
    )
    args = parser.parse_args()
    if args.joint_decoding_confidence_temperature <= 0.0:
        parser.error("--joint-decoding-confidence-temperature must be positive")
    return args


def tensor_to_numpy(tensor: torch.Tensor, *, dtype: np.dtype[Any] | type) -> np.ndarray:
    """Converts a tensor to a CPU NumPy array with the requested dtype."""
    return tensor.detach().cpu().numpy().astype(dtype, copy=False)


def denormalize_points(
    points: np.ndarray,
    *,
    center: np.ndarray | tuple[float, float, float],
    scale: float,
) -> np.ndarray:
    """Converts normalized model-space points back to mesh-space coordinates."""
    return (
        np.asarray(points, dtype=np.float32) / np.float32(scale)
        + np.asarray(center, dtype=np.float32)
    ).astype(np.float32, copy=False)


def _confidence_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Returns a weighted mean over the leading dimension."""
    values = values.float()
    weights = weights.float()
    if values.shape[0] != weights.shape[0]:
        raise ValueError(
            "values and weights must agree on the leading dimension, "
            f"got {tuple(values.shape)} and {tuple(weights.shape)}"
        )
    if values.shape[0] == 0:
        return values.new_zeros(values.shape[1:])
    total_weight = weights.sum()
    if float(total_weight.item()) <= 0.0:
        return values.mean(dim=0)
    view_shape = (weights.shape[0],) + (1,) * (values.ndim - 1)
    return (values * weights.view(view_shape)).sum(dim=0) / total_weight


def _weighted_median_1d(
    values: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Returns the weighted median of a 1D tensor."""
    values = values.float().reshape(-1)
    weights = weights.float().reshape(-1)
    if values.shape != weights.shape:
        raise ValueError(
            "values and weights must have matching shapes, "
            f"got {tuple(values.shape)} and {tuple(weights.shape)}"
        )
    if values.numel() == 0:
        return values.new_zeros(())
    if float(weights.sum().item()) <= 0.0:
        return torch.quantile(values, 0.5)

    sorted_values, order = torch.sort(values)
    sorted_weights = weights[order]
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    cutoff = 0.5 * sorted_weights.sum()
    median_index = torch.searchsorted(cumulative_weights, cutoff, right=False)
    median_index = median_index.clamp_max(sorted_values.shape[0] - 1)
    return sorted_values[median_index]


def _confidence_weighted_coordinatewise_median(
    values: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Returns a coordinate-wise weighted median over the leading dimension."""
    values = values.float()
    weights = weights.float().reshape(-1)
    if values.ndim != 2:
        raise ValueError(f"Expected values with shape (N, D), got {tuple(values.shape)}")
    if values.shape[0] != weights.shape[0]:
        raise ValueError(
            "values and weights must agree on the leading dimension, "
            f"got {tuple(values.shape)} and {tuple(weights.shape)}"
        )
    if values.shape[0] == 0:
        return values.new_zeros((values.shape[-1],))
    if float(weights.sum().item()) <= 0.0:
        return torch.quantile(values, 0.5, dim=0)
    return torch.stack(
        [
            _weighted_median_1d(values[:, dim_idx], weights)
            for dim_idx in range(values.shape[1])
        ],
        dim=0,
    )


def _weighted_fit_axis_to_closest_points(
    query_points: torch.Tensor,
    closest_axis_points: torch.Tensor,
    weights: torch.Tensor,
    *,
    direction_hint: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fits a weighted least-squares axis from per-query closest-point targets."""
    if query_points.shape != closest_axis_points.shape:
        raise ValueError(
            "query_points and closest_axis_points must share the same shape, "
            f"got {tuple(query_points.shape)} and {tuple(closest_axis_points.shape)}"
        )
    if query_points.ndim != 2 or query_points.shape[-1] != 3:
        raise ValueError(
            f"Expected query_points and closest_axis_points to have shape (N, 3), got {tuple(query_points.shape)}"
        )
    if weights.ndim != 1 or weights.shape[0] != query_points.shape[0]:
        raise ValueError(
            "weights must have shape (N,), "
            f"got {tuple(weights.shape)} for query_points {tuple(query_points.shape)}"
        )
    if query_points.numel() == 0:
        zero = query_points.new_zeros(3)
        return zero, zero

    weights = weights.float()
    valid_mask = torch.isfinite(weights) & (weights > 0)
    if not bool(valid_mask.any().item()):
        return fit_axis_to_closest_points_torch(
            query_points,
            closest_axis_points,
            direction_hint=direction_hint,
        )

    query_points = query_points[valid_mask]
    closest_axis_points = closest_axis_points[valid_mask]
    weights = weights[valid_mask]
    normalized_weights = weights / weights.sum().clamp_min(1e-12)

    result_dtype = (
        query_points.dtype
        if query_points.dtype in (torch.float32, torch.float64)
        else torch.float32
    )
    solve_dtype = torch.float32 if query_points.device.type == "mps" else torch.float64
    query_points = query_points.to(dtype=solve_dtype)
    closest_axis_points = closest_axis_points.to(dtype=solve_dtype)
    normalized_weights = normalized_weights.to(dtype=solve_dtype)
    if direction_hint is None:
        direction_hint = query_points.new_zeros(3)
    else:
        direction_hint = direction_hint.to(
            device=query_points.device,
            dtype=solve_dtype,
        )

    closest_axis_mean = (closest_axis_points * normalized_weights.unsqueeze(-1)).sum(dim=0)
    centered_closest_axis_points = closest_axis_points - closest_axis_mean.unsqueeze(0)
    query_residuals = query_points - closest_axis_points
    objective_matrix = (
        query_residuals.transpose(0, 1)
        @ (query_residuals * normalized_weights.unsqueeze(-1))
        - centered_closest_axis_points.transpose(0, 1)
        @ (centered_closest_axis_points * normalized_weights.unsqueeze(-1))
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


def _confidence_weighted_estimate_prismatic_limit(
    current_points: torch.Tensor,
    target_points: torch.Tensor,
    axis_direction: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Fits a weighted shared prismatic displacement."""
    if current_points.numel() == 0:
        return current_points.new_zeros(())
    if weights.ndim != 1 or weights.shape[0] != current_points.shape[0]:
        raise ValueError(
            "weights must have shape (N,), "
            f"got {tuple(weights.shape)} for current_points {tuple(current_points.shape)}"
        )
    valid_mask = torch.isfinite(weights) & (weights > 0)
    if not bool(valid_mask.any().item()):
        return estimate_prismatic_limit_torch(
            current_points,
            target_points,
            axis_direction,
        )
    axis_direction = F.normalize(axis_direction.float(), dim=-1, eps=1e-8)
    weights = weights[valid_mask].float()
    projections = (
        (target_points[valid_mask].float() - current_points[valid_mask].float())
        * axis_direction.unsqueeze(0)
    ).sum(dim=-1)
    return (projections * weights).sum() / weights.sum().clamp_min(1e-12)


def _confidence_weighted_estimate_revolute_limit(
    current_points: torch.Tensor,
    target_points: torch.Tensor,
    axis_direction: torch.Tensor,
    axis_point: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Fits a weighted shared revolute angle around the given axis."""
    if current_points.numel() == 0:
        return current_points.new_zeros(())
    if weights.ndim != 1 or weights.shape[0] != current_points.shape[0]:
        raise ValueError(
            "weights must have shape (N,), "
            f"got {tuple(weights.shape)} for current_points {tuple(current_points.shape)}"
        )
    valid_mask = torch.isfinite(weights) & (weights > 0)
    if not bool(valid_mask.any().item()):
        return estimate_revolute_limit_torch(
            current_points,
            target_points,
            axis_direction,
            axis_point,
        )

    axis_direction = F.normalize(axis_direction.float(), dim=-1, eps=1e-8)
    axis_point = axis_point.float()
    current_points = current_points[valid_mask].float()
    target_points = target_points[valid_mask].float()
    weights = weights[valid_mask].float()

    def _project_perpendicular(points: torch.Tensor) -> torch.Tensor:
        offsets = points - axis_point.unsqueeze(0)
        parallel = (offsets * axis_direction.unsqueeze(0)).sum(dim=-1, keepdim=True)
        return offsets - parallel * axis_direction.unsqueeze(0)

    current_perp = _project_perpendicular(current_points)
    target_perp = _project_perpendicular(target_points)
    cosine_term = ((current_perp * target_perp).sum(dim=-1) * weights).sum()
    sine_term = torch.linalg.cross(current_perp, target_perp, dim=-1)
    sine_term = ((sine_term * axis_direction.unsqueeze(0)).sum(dim=-1) * weights).sum()
    return torch.atan2(sine_term, cosine_term)


def _confidence_weighted_axis_direction_vote(
    predicted_directions: torch.Tensor,
    weights: torch.Tensor,
    *,
    sign_hint: torch.Tensor | None = None,
) -> torch.Tensor:
    """Returns one weighted averaged unit direction while treating flips as equivalent."""
    predicted_directions = predicted_directions.float()
    weights = weights.float()
    if predicted_directions.ndim != 2 or predicted_directions.shape[-1] != 3:
        raise ValueError(
            f"Expected predicted_directions with shape (N, 3), got {tuple(predicted_directions.shape)}"
        )
    if weights.ndim != 1 or weights.shape[0] != predicted_directions.shape[0]:
        raise ValueError(
            "weights must have shape (N,), "
            f"got {tuple(weights.shape)} for predicted_directions {tuple(predicted_directions.shape)}"
        )
    if predicted_directions.numel() == 0:
        return predicted_directions.new_zeros(3)

    direction_norms = torch.linalg.vector_norm(predicted_directions, dim=-1)
    valid_mask = torch.isfinite(weights) & (weights > 0) & (direction_norms > 1e-8)
    if not bool(valid_mask.any().item()):
        return predicted_directions.new_zeros(3)

    unit_directions = predicted_directions[valid_mask] / direction_norms[valid_mask].unsqueeze(-1)
    weights = weights[valid_mask]
    direction_covariance = unit_directions.transpose(0, 1) @ (
        unit_directions * weights.unsqueeze(-1)
    )
    _, eigenvectors = torch.linalg.eigh(direction_covariance)
    anchor_direction = eigenvectors[:, -1]
    alignment = torch.sign(unit_directions @ anchor_direction)
    alignment = torch.where(
        alignment == 0,
        torch.ones_like(alignment),
        alignment,
    )
    aligned_mean_direction = (
        unit_directions
        * alignment.unsqueeze(-1)
        * weights.unsqueeze(-1)
    ).sum(dim=0) / weights.sum().clamp_min(1e-12)
    if float(torch.linalg.vector_norm(aligned_mean_direction).item()) <= 1e-8:
        axis_direction = F.normalize(anchor_direction, dim=0, eps=1e-8)
    else:
        axis_direction = F.normalize(aligned_mean_direction, dim=0, eps=1e-8)

    if sign_hint is not None:
        sign_hint = sign_hint.float()
        if float(torch.linalg.vector_norm(sign_hint).item()) > 1e-8:
            if float(torch.dot(axis_direction, sign_hint).item()) < 0.0:
                axis_direction = -axis_direction
    return axis_direction


def _fit_confidence_weighted_revolute_joint_parameters(
    query_points: torch.Tensor,
    closest_axis_points: torch.Tensor,
    low_points: torch.Tensor,
    high_points: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fits one revolute axis and limits from weighted per-query targets."""
    query_points = query_points.float()
    closest_axis_points = closest_axis_points.float()
    low_points = low_points.float()
    high_points = high_points.float()
    weights = weights.float()
    if query_points.numel() == 0:
        zero_axis = query_points.new_zeros(6)
        zero_range = query_points.new_zeros(2)
        return zero_axis, zero_range

    direction_terms = (
        torch.linalg.cross(
            query_points - closest_axis_points,
            low_points - closest_axis_points,
            dim=-1,
        )
        + torch.linalg.cross(
            query_points - closest_axis_points,
            high_points - closest_axis_points,
            dim=-1,
        )
        + torch.linalg.cross(
            low_points - closest_axis_points,
            high_points - closest_axis_points,
            dim=-1,
        )
    )
    direction_hint = _confidence_weighted_mean(direction_terms, weights)
    axis_direction, axis_point = _weighted_fit_axis_to_closest_points(
        query_points,
        closest_axis_points,
        weights,
        direction_hint=direction_hint,
    )
    revolute_axis = axis_point_to_plucker_torch(axis_direction, axis_point)
    low_limit = _confidence_weighted_estimate_revolute_limit(
        query_points,
        low_points,
        axis_direction,
        axis_point,
        weights,
    )
    high_limit = _confidence_weighted_estimate_revolute_limit(
        query_points,
        high_points,
        axis_direction,
        axis_point,
        weights,
    )
    return revolute_axis, torch.stack((low_limit, high_limit))


def _fit_confidence_weighted_revolute_joint_parameters_with_direction(
    query_points: torch.Tensor,
    closest_axis_points: torch.Tensor,
    low_points: torch.Tensor,
    high_points: torch.Tensor,
    predicted_axis_directions: torch.Tensor,
    weights: torch.Tensor,
    *,
    direction_vote_query_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fits one revolute axis and limits from weighted per-query directional targets."""
    query_points = query_points.float()
    closest_axis_points = closest_axis_points.float()
    low_points = low_points.float()
    high_points = high_points.float()
    predicted_axis_directions = predicted_axis_directions.float()
    weights = weights.float()
    if query_points.numel() == 0:
        zero_axis = query_points.new_zeros(6)
        zero_range = query_points.new_zeros(2)
        return zero_axis, zero_range

    direction_sign_terms = (
        torch.linalg.cross(
            query_points - closest_axis_points,
            low_points - closest_axis_points,
            dim=-1,
        )
        + torch.linalg.cross(
            query_points - closest_axis_points,
            high_points - closest_axis_points,
            dim=-1,
        )
        + torch.linalg.cross(
            low_points - closest_axis_points,
            high_points - closest_axis_points,
            dim=-1,
        )
    )
    vote_predicted_axis_directions = predicted_axis_directions
    vote_weights = weights
    vote_direction_sign_terms = direction_sign_terms
    if direction_vote_query_mask is not None:
        if bool(direction_vote_query_mask.any().item()):
            vote_predicted_axis_directions = predicted_axis_directions[direction_vote_query_mask]
            vote_weights = weights[direction_vote_query_mask]
            vote_direction_sign_terms = direction_sign_terms[direction_vote_query_mask]
    axis_direction = _confidence_weighted_axis_direction_vote(
        vote_predicted_axis_directions,
        vote_weights,
        sign_hint=_confidence_weighted_mean(vote_direction_sign_terms, vote_weights),
    )
    if float(torch.linalg.vector_norm(axis_direction).item()) <= 1e-8:
        return _fit_confidence_weighted_revolute_joint_parameters(
            query_points,
            closest_axis_points,
            low_points,
            high_points,
            weights,
        )

    axis_point = _confidence_weighted_coordinatewise_median(
        closest_axis_points,
        weights,
    )
    low_limit = _confidence_weighted_estimate_revolute_limit(
        query_points,
        low_points,
        axis_direction,
        axis_point,
        weights,
    )
    high_limit = _confidence_weighted_estimate_revolute_limit(
        query_points,
        high_points,
        axis_direction,
        axis_point,
        weights,
    )
    if float(low_limit.item()) > float(high_limit.item()):
        axis_direction = -axis_direction
        low_limit = _confidence_weighted_estimate_revolute_limit(
            query_points,
            low_points,
            axis_direction,
            axis_point,
            weights,
        )
        high_limit = _confidence_weighted_estimate_revolute_limit(
            query_points,
            high_points,
            axis_direction,
            axis_point,
            weights,
        )
    revolute_axis = axis_point_to_plucker_torch(axis_direction, axis_point)
    return revolute_axis, torch.stack((low_limit, high_limit))


def _fit_confidence_weighted_prismatic_joint_parameters(
    query_points: torch.Tensor,
    closest_axis_points: torch.Tensor,
    low_points: torch.Tensor,
    high_points: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fits one prismatic axis and limits from weighted per-query targets."""
    query_points = query_points.float()
    closest_axis_points = closest_axis_points.float()
    low_points = low_points.float()
    high_points = high_points.float()
    weights = weights.float()
    if query_points.numel() == 0:
        zero_axis = query_points.new_zeros(6)
        zero_range = query_points.new_zeros(2)
        return zero_axis, zero_range

    axis_direction, axis_point = _weighted_fit_axis_to_closest_points(
        query_points,
        closest_axis_points,
        weights,
        direction_hint=_confidence_weighted_mean(high_points - low_points, weights),
    )
    prismatic_axis = axis_point_to_plucker_torch(axis_direction, axis_point)
    low_limit = _confidence_weighted_estimate_prismatic_limit(
        query_points,
        low_points,
        axis_direction,
        weights,
    )
    high_limit = _confidence_weighted_estimate_prismatic_limit(
        query_points,
        high_points,
        axis_direction,
        weights,
    )
    return prismatic_axis, torch.stack((low_limit, high_limit))


def _fit_confidence_weighted_prismatic_joint_parameters_with_direction(
    query_points: torch.Tensor,
    closest_axis_points: torch.Tensor,
    low_points: torch.Tensor,
    high_points: torch.Tensor,
    predicted_axis_directions: torch.Tensor,
    weights: torch.Tensor,
    *,
    direction_vote_query_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fits one prismatic axis and limits from weighted per-query directional targets."""
    query_points = query_points.float()
    closest_axis_points = closest_axis_points.float()
    low_points = low_points.float()
    high_points = high_points.float()
    predicted_axis_directions = predicted_axis_directions.float()
    weights = weights.float()
    if query_points.numel() == 0:
        zero_axis = query_points.new_zeros(6)
        zero_range = query_points.new_zeros(2)
        return zero_axis, zero_range

    vote_predicted_axis_directions = predicted_axis_directions
    vote_weights = weights
    vote_sign_hint_terms = high_points - low_points
    if direction_vote_query_mask is not None:
        if bool(direction_vote_query_mask.any().item()):
            vote_predicted_axis_directions = predicted_axis_directions[direction_vote_query_mask]
            vote_weights = weights[direction_vote_query_mask]
            vote_sign_hint_terms = vote_sign_hint_terms[direction_vote_query_mask]
    axis_direction = _confidence_weighted_axis_direction_vote(
        vote_predicted_axis_directions,
        vote_weights,
        sign_hint=_confidence_weighted_mean(vote_sign_hint_terms, vote_weights),
    )
    if float(torch.linalg.vector_norm(axis_direction).item()) <= 1e-8:
        return _fit_confidence_weighted_prismatic_joint_parameters(
            query_points,
            closest_axis_points,
            low_points,
            high_points,
            weights,
        )

    axis_point = _confidence_weighted_coordinatewise_median(
        closest_axis_points,
        weights,
    )
    low_limit = _confidence_weighted_estimate_prismatic_limit(
        query_points,
        low_points,
        axis_direction,
        weights,
    )
    high_limit = _confidence_weighted_estimate_prismatic_limit(
        query_points,
        high_points,
        axis_direction,
        weights,
    )
    if float(low_limit.item()) > float(high_limit.item()):
        axis_direction = -axis_direction
        low_limit = _confidence_weighted_estimate_prismatic_limit(
            query_points,
            low_points,
            axis_direction,
            weights,
        )
        high_limit = _confidence_weighted_estimate_prismatic_limit(
            query_points,
            high_points,
            axis_direction,
            weights,
        )
    prismatic_axis = axis_point_to_plucker_torch(axis_direction, axis_point)
    return prismatic_axis, torch.stack((low_limit, high_limit))


def _fit_confidence_weighted_revolute_joint_parameters_with_single_direction(
    query_points: torch.Tensor,
    closest_axis_points: torch.Tensor,
    low_points: torch.Tensor,
    high_points: torch.Tensor,
    predicted_axis_direction: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fits one revolute axis and limits from weighted query targets plus one direction."""
    query_points = query_points.float()
    closest_axis_points = closest_axis_points.float()
    low_points = low_points.float()
    high_points = high_points.float()
    predicted_axis_direction = predicted_axis_direction.float()
    weights = weights.float()
    if query_points.numel() == 0:
        zero_axis = query_points.new_zeros(6)
        zero_range = query_points.new_zeros(2)
        return zero_axis, zero_range

    if float(torch.linalg.vector_norm(predicted_axis_direction).item()) <= 1e-8:
        return _fit_confidence_weighted_revolute_joint_parameters(
            query_points,
            closest_axis_points,
            low_points,
            high_points,
            weights,
        )
    axis_direction = F.normalize(predicted_axis_direction, dim=0, eps=1e-8)
    axis_point = _confidence_weighted_coordinatewise_median(
        closest_axis_points,
        weights,
    )
    low_limit = _confidence_weighted_estimate_revolute_limit(
        query_points,
        low_points,
        axis_direction,
        axis_point,
        weights,
    )
    high_limit = _confidence_weighted_estimate_revolute_limit(
        query_points,
        high_points,
        axis_direction,
        axis_point,
        weights,
    )
    if float(low_limit.item()) > float(high_limit.item()):
        axis_direction = -axis_direction
        low_limit = _confidence_weighted_estimate_revolute_limit(
            query_points,
            low_points,
            axis_direction,
            axis_point,
            weights,
        )
        high_limit = _confidence_weighted_estimate_revolute_limit(
            query_points,
            high_points,
            axis_direction,
            axis_point,
            weights,
        )
    revolute_axis = axis_point_to_plucker_torch(axis_direction, axis_point)
    return revolute_axis, torch.stack((low_limit, high_limit))


def _fit_confidence_weighted_prismatic_joint_parameters_with_single_direction(
    query_points: torch.Tensor,
    closest_axis_points: torch.Tensor,
    low_points: torch.Tensor,
    high_points: torch.Tensor,
    predicted_axis_direction: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fits one prismatic axis and limits from weighted query targets plus one direction."""
    query_points = query_points.float()
    closest_axis_points = closest_axis_points.float()
    low_points = low_points.float()
    high_points = high_points.float()
    predicted_axis_direction = predicted_axis_direction.float()
    weights = weights.float()
    if query_points.numel() == 0:
        zero_axis = query_points.new_zeros(6)
        zero_range = query_points.new_zeros(2)
        return zero_axis, zero_range

    if float(torch.linalg.vector_norm(predicted_axis_direction).item()) <= 1e-8:
        return _fit_confidence_weighted_prismatic_joint_parameters(
            query_points,
            closest_axis_points,
            low_points,
            high_points,
            weights,
        )
    axis_direction = F.normalize(predicted_axis_direction, dim=0, eps=1e-8)
    axis_point = _confidence_weighted_coordinatewise_median(
        closest_axis_points,
        weights,
    )
    low_limit = _confidence_weighted_estimate_prismatic_limit(
        query_points,
        low_points,
        axis_direction,
        weights,
    )
    high_limit = _confidence_weighted_estimate_prismatic_limit(
        query_points,
        high_points,
        axis_direction,
        weights,
    )
    if float(low_limit.item()) > float(high_limit.item()):
        axis_direction = -axis_direction
        low_limit = _confidence_weighted_estimate_prismatic_limit(
            query_points,
            low_points,
            axis_direction,
            weights,
        )
        high_limit = _confidence_weighted_estimate_prismatic_limit(
            query_points,
            high_points,
            axis_direction,
            weights,
        )
    prismatic_axis = axis_point_to_plucker_torch(axis_direction, axis_point)
    return prismatic_axis, torch.stack((low_limit, high_limit))


def _apply_joint_decoding_confidence_temperature(
    joint_decoding_confidences: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Applies a temperature transform to segmentation confidences used for joint refitting."""
    if temperature <= 0.0:
        raise ValueError("joint decoding confidence temperature must be positive")
    confidences = joint_decoding_confidences.to(dtype=torch.float32).clamp_min(0.0)
    if temperature == 1.0:
        return confidences
    return confidences.pow(1.0 / float(temperature))


def _resolve_second_pass_part_aabb_bbox_source_link_ids(
    *,
    assigned_link_ids: torch.Tensor,
    segmentation_logits: torch.Tensor,
) -> torch.Tensor:
    """Returns the second-pass query subset that contributes to each part AABB."""
    if segmentation_logits.ndim != 3:
        raise ValueError(
            "segmentation_logits must have shape (B, Q, num_parts), "
            f"got {tuple(segmentation_logits.shape)}"
        )
    if assigned_link_ids.shape != segmentation_logits.shape[:2]:
        raise ValueError(
            "assigned_link_ids must match segmentation_logits batch/query dims, "
            f"got {tuple(assigned_link_ids.shape)} and {tuple(segmentation_logits.shape)}"
        )

    predicted_link_ids = segmentation_logits.argmax(dim=-1)
    bbox_source_link_ids = torch.full_like(assigned_link_ids, -1)
    valid_assigned_mask = assigned_link_ids >= 0
    agreement_mask = valid_assigned_mask & (predicted_link_ids == assigned_link_ids)
    bbox_source_link_ids[agreement_mask] = assigned_link_ids[agreement_mask]
    return bbox_source_link_ids


def _compute_query_link_aabb_parameters_from_source_subset(
    *,
    query_points: torch.Tensor,
    assigned_link_ids: torch.Tensor,
    bbox_source_link_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes per-link AABB parameters from a filtered query subset.

    The AABB for one link is estimated from the subset encoded by
    `bbox_source_link_ids`. If that subset is empty for a link, this falls back
    to all queries assigned to that link. The resulting center / half-extent is
    broadcast to every query assigned to the link.
    """
    if assigned_link_ids.shape != query_points.shape[:2]:
        raise ValueError(
            "assigned_link_ids must match query_points batch/query dims, "
            f"got {tuple(assigned_link_ids.shape)} and {tuple(query_points.shape)}"
        )
    if bbox_source_link_ids.shape != assigned_link_ids.shape:
        raise ValueError(
            "bbox_source_link_ids must match assigned_link_ids, "
            f"got {tuple(bbox_source_link_ids.shape)} and {tuple(assigned_link_ids.shape)}"
        )

    query_points = query_points.float()
    centers = query_points.new_zeros(query_points.shape)
    half_extents = query_points.new_ones(query_points.shape)

    for batch_idx in range(query_points.shape[0]):
        batch_assigned_link_ids = assigned_link_ids[batch_idx]
        valid_assigned_mask = batch_assigned_link_ids >= 0
        if not bool(valid_assigned_mask.any().item()):
            continue

        unique_link_ids = torch.unique(batch_assigned_link_ids[valid_assigned_mask])
        for link_id in unique_link_ids.tolist():
            target_mask = batch_assigned_link_ids == int(link_id)
            source_mask = bbox_source_link_ids[batch_idx] == int(link_id)
            if not bool(source_mask.any().item()):
                source_mask = target_mask

            source_query_points = query_points[batch_idx][source_mask]
            min_corner = source_query_points.min(dim=0).values
            max_corner = source_query_points.max(dim=0).values
            centers[batch_idx][target_mask] = 0.5 * (min_corner + max_corner)
            half_extents[batch_idx][target_mask] = (
                0.5 * (max_corner - min_corner)
            ).clamp_min(_INFER_OVERPARAM_AXIS_AABB_HALF_EXTENT_MIN)

    return centers, half_extents


def _denormalize_overparam_axis_points_from_second_pass_intersection(
    *,
    axis_points: torch.Tensor,
    query_points: torch.Tensor,
    assigned_link_ids: torch.Tensor,
    segmentation_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Denormalizes part-AABB axis points from the first/second-pass query intersection."""
    bbox_source_link_ids = _resolve_second_pass_part_aabb_bbox_source_link_ids(
        assigned_link_ids=assigned_link_ids,
        segmentation_logits=segmentation_logits,
    )
    centers, half_extents = _compute_query_link_aabb_parameters_from_source_subset(
        query_points=query_points,
        assigned_link_ids=assigned_link_ids,
        bbox_source_link_ids=bbox_source_link_ids,
    )
    return axis_points.float() * half_extents + centers, bbox_source_link_ids


def _apply_confidence_weighted_overparam_joint_voting(
    model: Particulate2ArticulationModel,
    motion_output: dict[str, Any],
    *,
    query_points: torch.Tensor,
    confidence_temperature: float = 1.0,
) -> dict[str, Any]:
    """Recomputes overparameterized joint parameters with confidence-weighted voting."""
    if model.joint_decode_type not in {
        "overparametrized",
        "overparam+dir",
        "overparam+singledir",
    }:
        return motion_output

    joint_decoding_link_ids = motion_output.get("joint_decoding_link_ids")
    joint_decoding_confidences = motion_output.get("joint_decoding_confidences")
    revolute_closest_axis_points = motion_output.get("revolute_closest_axis_points")
    revolute_low_points = motion_output.get("revolute_low_points")
    revolute_high_points = motion_output.get("revolute_high_points")
    prismatic_closest_axis_points = motion_output.get("prismatic_closest_axis_points")
    prismatic_low_points = motion_output.get("prismatic_low_points")
    prismatic_high_points = motion_output.get("prismatic_high_points")
    segmentation_logits = motion_output.get("segmentation_logits")
    if (
        joint_decoding_link_ids is None
        or joint_decoding_confidences is None
        or revolute_closest_axis_points is None
        or revolute_low_points is None
        or revolute_high_points is None
        or prismatic_closest_axis_points is None
        or prismatic_low_points is None
        or prismatic_high_points is None
    ):
        return motion_output

    joint_connections = motion_output["joint_connections"]
    joint_valid_flag = motion_output["joint_valid_flag"]
    is_revolute = motion_output["is_revolute"]
    is_prismatic = motion_output["is_prismatic"]

    query_points = query_points.to(
        device=joint_decoding_link_ids.device,
        dtype=torch.float32,
    )
    bbox_source_link_ids: torch.Tensor | None = None
    if getattr(model, "overparam_closest_axis_uses_part_aabb", False) is True:
        if segmentation_logits is None:
            raise ValueError(
                "part_aabb inference-time refit requires second-pass segmentation_logits"
            )
        revolute_closest_axis_points_decoder = motion_output.get(
            "revolute_closest_axis_points_decoder"
        )
        prismatic_closest_axis_points_decoder = motion_output.get(
            "prismatic_closest_axis_points_decoder"
        )
        segmentation_logits = segmentation_logits.to(device=query_points.device)
        if revolute_closest_axis_points_decoder is not None:
            (
                revolute_closest_axis_points,
                bbox_source_link_ids,
            ) = _denormalize_overparam_axis_points_from_second_pass_intersection(
                axis_points=revolute_closest_axis_points_decoder.to(
                    device=query_points.device,
                    dtype=torch.float32,
                ),
                query_points=query_points,
                assigned_link_ids=joint_decoding_link_ids,
                segmentation_logits=segmentation_logits,
            )
        if prismatic_closest_axis_points_decoder is not None:
            (
                prismatic_closest_axis_points,
                bbox_source_link_ids,
            ) = _denormalize_overparam_axis_points_from_second_pass_intersection(
                axis_points=prismatic_closest_axis_points_decoder.to(
                    device=query_points.device,
                    dtype=torch.float32,
                ),
                query_points=query_points,
                assigned_link_ids=joint_decoding_link_ids,
                segmentation_logits=segmentation_logits,
            )
    joint_decoding_confidences = _apply_joint_decoding_confidence_temperature(
        joint_decoding_confidences,
        temperature=confidence_temperature,
    ).to(device=joint_decoding_link_ids.device)
    batch_size, max_joints = joint_connections.shape[:2]
    revolute_axis = query_points.new_zeros((batch_size, max_joints, 6))
    prismatic_axis = query_points.new_zeros((batch_size, max_joints, 6))
    revolute_range = query_points.new_zeros((batch_size, max_joints, 2))
    prismatic_range = query_points.new_zeros((batch_size, max_joints, 2))
    revolute_axis_directions = None
    prismatic_axis_directions = None
    if model.joint_decode_type in {"overparam+dir", "overparam+singledir"}:
        revolute_axis_directions = motion_output["revolute_axis_directions"]
        prismatic_axis_directions = motion_output["prismatic_axis_directions"]
        if revolute_axis_directions is None or prismatic_axis_directions is None:
            if model.joint_decode_type == "overparam+dir":
                raise ValueError(
                    "direction-assisted overparam inference requires per-query axis "
                    "direction predictions"
                )
            raise ValueError(
                "overparam+singledir inference requires per-joint axis direction predictions"
            )

    child_link_ids = joint_connections[..., 1]
    for batch_idx in range(batch_size):
        for joint_idx in range(max_joints):
            child_link_id = child_link_ids[batch_idx, joint_idx]
            query_mask = joint_decoding_link_ids[batch_idx] == child_link_id
            if not bool(query_mask.any().item()):
                continue

            joint_query_points = query_points[batch_idx][query_mask]
            joint_weights = joint_decoding_confidences[batch_idx][query_mask]
            joint_direction_vote_query_mask = None
            if bbox_source_link_ids is not None:
                joint_direction_vote_query_mask = (
                    bbox_source_link_ids[batch_idx][query_mask] == int(child_link_id)
                )
                if not bool(joint_direction_vote_query_mask.any().item()):
                    joint_direction_vote_query_mask = None

            if model.joint_decode_type == "overparam+dir":
                (
                    revolute_axis[batch_idx, joint_idx],
                    revolute_range[batch_idx, joint_idx],
                ) = _fit_confidence_weighted_revolute_joint_parameters_with_direction(
                    joint_query_points,
                    revolute_closest_axis_points[batch_idx][query_mask],
                    revolute_low_points[batch_idx][query_mask],
                    revolute_high_points[batch_idx][query_mask],
                    revolute_axis_directions[batch_idx][query_mask],
                    joint_weights,
                    direction_vote_query_mask=joint_direction_vote_query_mask,
                )
                (
                    prismatic_axis[batch_idx, joint_idx],
                    prismatic_range[batch_idx, joint_idx],
                ) = _fit_confidence_weighted_prismatic_joint_parameters_with_direction(
                    joint_query_points,
                    prismatic_closest_axis_points[batch_idx][query_mask],
                    prismatic_low_points[batch_idx][query_mask],
                    prismatic_high_points[batch_idx][query_mask],
                    prismatic_axis_directions[batch_idx][query_mask],
                    joint_weights,
                    direction_vote_query_mask=joint_direction_vote_query_mask,
                )
            elif model.joint_decode_type == "overparam+singledir":
                (
                    revolute_axis[batch_idx, joint_idx],
                    revolute_range[batch_idx, joint_idx],
                ) = _fit_confidence_weighted_revolute_joint_parameters_with_single_direction(
                    joint_query_points,
                    revolute_closest_axis_points[batch_idx][query_mask],
                    revolute_low_points[batch_idx][query_mask],
                    revolute_high_points[batch_idx][query_mask],
                    revolute_axis_directions[batch_idx, joint_idx],
                    joint_weights,
                )
                (
                    prismatic_axis[batch_idx, joint_idx],
                    prismatic_range[batch_idx, joint_idx],
                ) = _fit_confidence_weighted_prismatic_joint_parameters_with_single_direction(
                    joint_query_points,
                    prismatic_closest_axis_points[batch_idx][query_mask],
                    prismatic_low_points[batch_idx][query_mask],
                    prismatic_high_points[batch_idx][query_mask],
                    prismatic_axis_directions[batch_idx, joint_idx],
                    joint_weights,
                )
            else:
                (
                    revolute_axis[batch_idx, joint_idx],
                    revolute_range[batch_idx, joint_idx],
                ) = _fit_confidence_weighted_revolute_joint_parameters(
                    joint_query_points,
                    revolute_closest_axis_points[batch_idx][query_mask],
                    revolute_low_points[batch_idx][query_mask],
                    revolute_high_points[batch_idx][query_mask],
                    joint_weights,
                )
                (
                    prismatic_axis[batch_idx, joint_idx],
                    prismatic_range[batch_idx, joint_idx],
                ) = _fit_confidence_weighted_prismatic_joint_parameters(
                    joint_query_points,
                    prismatic_closest_axis_points[batch_idx][query_mask],
                    prismatic_low_points[batch_idx][query_mask],
                    prismatic_high_points[batch_idx][query_mask],
                    joint_weights,
                )

    revolute_mask = (joint_valid_flag & is_revolute).unsqueeze(-1)
    prismatic_mask = (joint_valid_flag & is_prismatic).unsqueeze(-1)
    weighted_motion_output = dict(motion_output)
    weighted_motion_output.update(
        {
            "revolute_closest_axis_points": revolute_closest_axis_points,
            "prismatic_closest_axis_points": prismatic_closest_axis_points,
            "revolute_axis": revolute_axis.masked_fill(~revolute_mask, 0),
            "prismatic_axis": prismatic_axis.masked_fill(~prismatic_mask, 0),
            "revolute_range": revolute_range.masked_fill(~revolute_mask[..., :1], 0),
            "prismatic_range": prismatic_range.masked_fill(~prismatic_mask[..., :1], 0),
        }
    )
    if bbox_source_link_ids is not None:
        weighted_motion_output["part_aabb_bbox_source_link_ids"] = bbox_source_link_ids
    return weighted_motion_output


def build_joint_refit_metadata(
    refit_sampling: dict[str, np.ndarray] | None,
    *,
    num_links: int,
) -> dict[str, Any]:
    """Returns metadata describing the balanced joint-refit sampling stage."""
    if refit_sampling is None:
        return {
            "joint_refit_applied": False,
        }

    per_link_query_counts = np.zeros((num_links,), dtype=np.int32)
    refit_link_ids = np.asarray(refit_sampling["link_ids"], dtype=np.int64)
    unique_link_ids, query_counts = np.unique(refit_link_ids, return_counts=True)
    per_link_query_counts[unique_link_ids.astype(np.int64, copy=False)] = query_counts.astype(
        np.int32,
        copy=False,
    )
    return {
        "joint_refit_applied": True,
        "joint_refit_strategy": "balanced_surface_queries_from_refined_face_seg",
        "joint_refit_unique_part_ids": np.asarray(
            refit_sampling["unique_part_ids"],
            dtype=np.int32,
        ).tolist(),
        "joint_refit_query_counts_per_part": np.asarray(
            refit_sampling["query_counts"],
            dtype=np.int32,
        ).tolist(),
        "joint_refit_query_counts_per_link": per_link_query_counts.tolist(),
    }


def _query_visualization_part_confidences_from_motion_output(
    motion_output: dict[str, Any],
) -> dict[int, float] | None:
    joint_decoding_link_ids = motion_output.get("joint_decoding_link_ids")
    joint_decoding_confidences = motion_output.get("joint_decoding_confidences")
    if joint_decoding_link_ids is None or joint_decoding_confidences is None:
        return None

    return compute_link_geometric_mean_confidences(
        query_link_ids=tensor_to_numpy(joint_decoding_link_ids[0], dtype=np.int64),
        query_confidences=tensor_to_numpy(joint_decoding_confidences[0], dtype=np.float32),
    )


def decode_face_part_ids(
    mesh: Any,
    *,
    point_part_ids: np.ndarray,
    point_part_probabilities: np.ndarray,
    query_face_indices: np.ndarray,
    input_part_ids: np.ndarray,
    strict: bool,
    enforce_connectivity_per_part: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Decodes unrefined and refined face labels from point-level predictions."""
    face_part_ids_unrefined = find_unrefined_part_ids_for_faces(
        mesh,
        point_part_ids,
        query_face_indices,
    )
    face_part_ids = refine_face_part_ids_for_inference(
        mesh,
        face_part_ids_unrefined,
        point_part_probabilities=point_part_probabilities,
        face_indices=query_face_indices,
        input_part_ids=input_part_ids,
        strict=bool(strict),
        enforce_connectivity_per_part=bool(enforce_connectivity_per_part),
    )
    return face_part_ids, face_part_ids_unrefined


def compute_motion_prediction_artifacts(
    args: argparse.Namespace,
    *,
    model: Particulate2ArticulationModel,
    batch: dict[str, Any],
    normalized_mesh: Any,
    face_part_ids: np.ndarray,
    joint_refit_num_query_points: int,
    num_links: int,
    center: np.ndarray | tuple[float, float, float],
    scale: float,
) -> dict[str, Any]:
    """Runs joint decoding/refit and returns normalized/world motion artifacts."""
    motion_output, joint_refit_sampling = run_joint_refit_from_face_seg(
        model=model,
        batch=batch,
        mesh=normalized_mesh,
        face_part_ids=face_part_ids,
        num_query_points=int(joint_refit_num_query_points),
        query_batch_size=int(args.query_batch_size),
        no_point_prompt_for_unique_text=bool(args.no_point_prompt),
    )
    if model.joint_decode_type in {
        "overparametrized",
        "overparam+dir",
        "overparam+singledir",
    }:
        motion_output = _apply_confidence_weighted_overparam_joint_voting(
            model,
            motion_output,
            query_points=(
                torch.from_numpy(np.asarray(joint_refit_sampling["query_points"], dtype=np.float32))
                .unsqueeze(0)
                .to(batch["query_points"].device)
            ),
            confidence_temperature=args.joint_decoding_confidence_temperature,
        )
    motion_arrays_normalized = motion_arrays_from_model_output(
        motion_output,
        num_links=int(num_links),
    )
    if model.joint_decode_type in {"plain", "plain+fm"}:
        motion_arrays_normalized = _canonicalize_plain_motion_arrays(
            motion_arrays_normalized
        )
    motion_arrays_world = denormalize_motion_parameters(
        motion_arrays_normalized["revolute_plucker"],
        motion_arrays_normalized["prismatic_plucker"],
        motion_arrays_normalized["revolute_range"],
        motion_arrays_normalized["prismatic_range"],
        center=center,
        scale=scale,
    )
    return {
        "motion_output": motion_output,
        "joint_refit_sampling": joint_refit_sampling,
        "motion_arrays_normalized": motion_arrays_normalized,
        "motion_arrays_world": motion_arrays_world,
        "prismatic_axis_normalized": prismatic_directions_from_plucker(
            motion_arrays_normalized["prismatic_plucker"]
        ),
        "prismatic_axis_world": prismatic_directions_from_plucker(
            motion_arrays_world["prismatic_plucker"]
        ),
    }


def _canonicalize_axis_range_pairs(
    axis_parameters: np.ndarray,
    ranges: np.ndarray,
    *,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flips valid axis/range pairs so exported limits use `low <= high`."""
    canonical_axis = np.asarray(axis_parameters, dtype=np.float32).copy()
    canonical_ranges = np.asarray(ranges, dtype=np.float32).copy()
    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    flip_mask = valid_mask & (canonical_ranges[:, 0] > canonical_ranges[:, 1])
    canonical_axis[flip_mask] *= np.float32(-1.0)
    canonical_ranges[flip_mask] *= np.float32(-1.0)
    return canonical_axis, canonical_ranges


def _canonicalize_plain_motion_arrays(
    motion_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Returns copies of plain-decoder motion arrays with a deterministic sign convention."""
    canonical_motion_arrays = {
        key: value.copy()
        for key, value in motion_arrays.items()
    }
    (
        canonical_motion_arrays["revolute_plucker"],
        canonical_motion_arrays["revolute_range"],
    ) = _canonicalize_axis_range_pairs(
        canonical_motion_arrays["revolute_plucker"],
        canonical_motion_arrays["revolute_range"],
        valid_mask=canonical_motion_arrays["revolute_parameter_valid"],
    )
    (
        canonical_motion_arrays["prismatic_plucker"],
        canonical_motion_arrays["prismatic_range"],
    ) = _canonicalize_axis_range_pairs(
        canonical_motion_arrays["prismatic_plucker"],
        canonical_motion_arrays["prismatic_range"],
        valid_mask=canonical_motion_arrays["prismatic_parameter_valid"],
    )
    return canonical_motion_arrays


def _save_query_predictions(
    output_path: Path,
    *,
    query_points: np.ndarray,
    face_indices: np.ndarray,
    part_ids: np.ndarray,
    center: np.ndarray,
    scale: float,
    joint_refit_sampling: dict[str, np.ndarray] | None,
    query_normals: np.ndarray | None = None,
    gt_part_ids: np.ndarray | None = None,
) -> None:
    """Writes first-pass query predictions plus optional joint-refit debug arrays."""
    payload: dict[str, np.ndarray] = {
        "query_points": denormalize_points(
            query_points,
            center=center,
            scale=scale,
        ),
        "face_indices": np.asarray(face_indices, dtype=np.int32),
        "part_ids": np.asarray(part_ids, dtype=np.int32),
    }
    if query_normals is not None:
        payload["query_normals"] = np.asarray(query_normals, dtype=np.float32)
    if gt_part_ids is not None:
        payload["gt_part_ids"] = np.asarray(gt_part_ids, dtype=np.int32)
    if joint_refit_sampling is not None:
        payload.update(
            {
                "joint_refit_query_points": denormalize_points(
                    joint_refit_sampling["query_points"],
                    center=center,
                    scale=scale,
                ),
                "joint_refit_query_normals": np.asarray(
                    joint_refit_sampling["query_normals"],
                    dtype=np.float32,
                ),
                "joint_refit_face_indices": np.asarray(
                    joint_refit_sampling["face_indices"],
                    dtype=np.int32,
                ),
                "joint_refit_part_ids": np.asarray(
                    joint_refit_sampling["link_ids"],
                    dtype=np.int32,
                ),
            }
        )
    np.savez(output_path, **payload)


def _missing_link_ids(num_links: int, unique_part_ids: np.ndarray) -> list[int]:
    """Returns input link IDs absent from the final face segmentation."""
    return sorted(
        set(range(int(num_links))) - set(np.asarray(unique_part_ids, dtype=np.int32).tolist())
    )


def resolve_visualized_batch_link_point_prompts(
    *,
    args: argparse.Namespace,
    batch: dict[str, Any],
    links: list[dict[str, Any]],
    center: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns denormalized prompt points filtered for visualization-only display."""
    link_point_prompts = batch.get("link_point_prompts")
    if link_point_prompts is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    link_point_prompts_world = denormalize_points(
        tensor_to_numpy(link_point_prompts[0], dtype=np.float32),
        center=center,
        scale=scale,
    )
    dropout_eligible = batch.get("link_point_prompt_dropout_eligible")
    dropout_eligible_np = (
        None
        if dropout_eligible is None
        else tensor_to_numpy(dropout_eligible[0], dtype=np.bool_)
    )
    return select_visualized_link_point_prompts(
        link_point_prompts=link_point_prompts_world,
        links=links,
        hide_unique_text_prompts=bool(args.no_point_prompt),
        link_point_prompt_dropout_eligible=dropout_eligible_np,
    )


def build_base_metadata(
    args: argparse.Namespace,
    *,
    mode: str,
    input_path: Path,
    run_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    num_shape_points: int,
    segmentation_num_query_points: int,
    joint_refit_num_query_points: int,
    sharp_point_ratio: float,
) -> dict[str, Any]:
    """Builds the metadata fields common to mesh and meta-root inference."""
    return {
        "mode": mode,
        "input_path": str(input_path),
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "num_shape_points": int(num_shape_points),
        "num_query_points": int(joint_refit_num_query_points),
        "segmentation_num_query_points": int(segmentation_num_query_points),
        "joint_refit_num_query_points": int(joint_refit_num_query_points),
        "num_query_points_per_face_for_seg": (
            None
            if args.num_query_points_per_face_for_seg is None
            else int(args.num_query_points_per_face_for_seg)
        ),
        "query_batch_size": int(args.query_batch_size),
        "no_point_prompt": bool(args.no_point_prompt),
        "enforce_connectivity_per_part": bool(args.enforce_connectivity_per_part),
        "joint_decoding_confidence_temperature": float(
            args.joint_decoding_confidence_temperature
        ),
        "sharp_point_ratio": float(sharp_point_ratio),
    }


def write_metadata_and_summary(
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    checkpoint_path: Path,
    unique_part_ids: np.ndarray,
    visualization_path: Path | None = None,
    overparam_visualization_path: Path | None = None,
    optional_paths: dict[str, Path | None] | None = None,
) -> None:
    """Writes metadata.json and prints the standard end-of-run summary."""
    if optional_paths is not None:
        for key, path in optional_paths.items():
            if path is not None:
                metadata[key] = str(path)
    write_json(output_dir / "metadata.json", metadata)
    print_inference_summary(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        unique_part_ids=unique_part_ids,
        visualization_path=visualization_path,
        overparam_visualization_path=overparam_visualization_path,
    )


def _resolve_output_dir(args: argparse.Namespace, input_path: Path) -> Path:
    output_root = args.output_dir.expanduser().resolve()
    output_dir = output_root / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_joint_refit_num_query_points(
    args: argparse.Namespace,
    *,
    default_num_query_points: int | None,
) -> int:
    return resolve_num_query_points(args.num_query_points, default_num_query_points)


def _resolve_segmentation_num_query_points(
    args: argparse.Namespace,
    *,
    mesh: Any,
    default_num_query_points: int | None,
) -> int:
    if args.num_query_points_per_face_for_seg is None:
        return _resolve_joint_refit_num_query_points(
            args,
            default_num_query_points=default_num_query_points,
        )

    num_faces = int(np.asarray(mesh.faces).shape[0])
    if num_faces <= 0:
        raise ValueError("Cannot use --num-query-points-per-face-for-seg for a mesh with no faces")
    return int(args.num_query_points_per_face_for_seg) * num_faces


def _resolve_pending_input_paths(
    args: argparse.Namespace,
    input_paths: list[Path],
) -> list[Path]:
    """Filters already-completed inputs when `--skip-existing` is enabled.

    Args:
        args: Parsed CLI arguments.
        input_paths: Resolved inputs to process.

    Returns:
        Input paths that still need inference.

    Non-eval inference writes deterministic output directories, so a completed
    run is identified by the final metadata file written at the end of the
    per-input pipeline.
    """
    if not args.skip_existing:
        return input_paths

    output_root = args.output_dir.expanduser().resolve()
    pending_input_paths: list[Path] = []
    for input_path in input_paths:
        input_mode = resolve_input_mode(input_path)
        if args.eval and input_mode == "meta_root":
            cached_result = load_cached_sample_result(output_root, input_path)
            if cached_result is not None:
                result_path = existing_result_path_for_input(output_root, input_path)
                print(f"Skipping {input_path}: found existing output {result_path}")
                continue
            stale_result_path = existing_result_path_for_input(output_root, input_path)
            if stale_result_path.exists():
                print(f"Recomputing {input_path}: existing eval JSON is outdated {stale_result_path}")
            pending_input_paths.append(input_path)
            continue

        if not args.eval:
            marker_path = output_root / input_path.stem / "metadata.json"
            if marker_path.exists():
                print(f"Skipping {input_path}: found existing output {marker_path}")
                continue
            pending_input_paths.append(input_path)
            continue

        marker_path = output_root / input_path.stem / "eval" / "pred.npz"
        if marker_path.exists():
            print(f"Skipping {input_path}: found existing output {marker_path}")
            continue
        pending_input_paths.append(input_path)
    return pending_input_paths


def _resolve_input_paths(input_path: Path) -> list[Path]:
    """Expand a directory of meta-roots into per-asset inputs for one-shot inference."""
    resolved_input_path = input_path.expanduser().resolve()
    if not resolved_input_path.is_dir():
        return [resolved_input_path]
    if is_supported_meta_root(resolved_input_path):
        return [resolved_input_path]
    if is_saved_auto_kinematics_root(resolved_input_path):
        return [resolved_input_path]

    child_inputs = [
        child_path.resolve()
        for child_path in sorted(resolved_input_path.iterdir())
        if is_supported_meta_root(child_path) or is_saved_auto_kinematics_root(child_path)
    ]
    if child_inputs:
        return child_inputs
    return [resolved_input_path]


def _prepare_saved_auto_kinematics_mesh_batch(
    *,
    auto_root: SavedAutoKinematicsRoot,
    mesh: Any,
    num_shape_points: int,
    num_query_points: int,
    num_query_points_per_face_for_seg: int | None,
    sharp_point_ratio: float,
    device: torch.device,
) -> PreparedMeshBatch:
    batch, query_face_indices = prepare_inference_batch_from_mesh_with_prompts(
        mesh,
        num_shape_points=num_shape_points,
        num_query_points=int(num_query_points),
        num_query_points_per_face_for_seg=num_query_points_per_face_for_seg,
        sharp_point_ratio=sharp_point_ratio,
        link_names=auto_root.link_names,
        joint_specs=auto_root.joint_specs,
        device=device,
        link_point_prompts=torch.from_numpy(auto_root.link_point_prompts).float(),
        link_point_prompt_normals=torch.from_numpy(auto_root.link_point_prompt_normals).float(),
        require_unique_link_names=False,
    )
    return PreparedMeshBatch(
        batch=batch,
        query_face_indices=np.asarray(query_face_indices, dtype=np.int64),
        link_names=list(auto_root.link_names),
        joint_specs=list(auto_root.joint_specs),
        auto_kinematics_metadata={
            "source": "saved_auto_kinematics_root",
            "auto_kinematics_root": str(auto_root.root_path),
            "mesh_path": str(auto_root.mesh_path),
            "mesh_up_dir": str(auto_root.mesh_up_dir),
            "geometry_metadata_path": str(auto_root.geometry_metadata_path),
            "parsed_response_path": str(auto_root.parsed_response_path),
            "lifted_prompts_path": str(auto_root.lifted_prompts_path),
        },
        link_point_prompt_face_ids=(
            None
            if auto_root.link_point_prompt_face_ids is None
            else np.asarray(auto_root.link_point_prompt_face_ids, dtype=np.int64)
        ),
    )


def prepare_mesh_geometry(
    *,
    input_path: Path,
    up_dir: str,
) -> PreparedMeshGeometry:
    """Loads one mesh input and prepares both model-space and Blender render-space copies."""
    raw_mesh = load_trimesh(input_path)
    original_mesh, up_dir_rotation = reorient_mesh_to_z_up(raw_mesh, up_dir)
    normalized_mesh, center, scale = normalize_mesh(original_mesh)

    render_export_rotation = up_dir_rotation_matrix("+Y", "+Z")
    blender_import_rotation = up_dir_rotation_matrix("+Y", "+Z")
    render_mesh = original_mesh.copy()
    render_transform = np.eye(4, dtype=np.float32)
    render_transform[:3, :3] = render_export_rotation
    render_mesh.apply_transform(render_transform)

    return PreparedMeshGeometry(
        original_mesh=original_mesh,
        normalized_mesh=normalized_mesh,
        center=np.asarray(center, dtype=np.float32),
        scale=float(scale),
        up_dir_rotation=np.asarray(up_dir_rotation, dtype=np.float32),
        render_mesh=render_mesh,
        render_to_model_rotation=(blender_import_rotation @ render_export_rotation).T,
    )


def _prepare_mesh_batch(
    args: argparse.Namespace,
    *,
    mesh_geometry: PreparedMeshGeometry,
    output_dir: Path,
    num_shape_points: int,
    num_query_points: int,
    num_query_points_per_face_for_seg: int | None,
    sharp_point_ratio: float,
    device: torch.device,
) -> PreparedMeshBatch:
    """Builds the direct-mesh batch, either from prompts or auto-kinematics."""
    if args.auto_kinematics:
        auto_kinematics = prepare_auto_kinematics_mesh_batch(
            mesh=mesh_geometry.normalized_mesh,
            original_mesh=mesh_geometry.original_mesh,
            render_mesh=mesh_geometry.render_mesh,
            output_dir=output_dir,
            auto_kinematics_model_id=str(args.auto_kinematics),
            azimuths_deg=args.azimuths,
            system_prompt_path=Path(__file__).with_name("kinematic-inference-system-prompt.md"),
            num_shape_points=num_shape_points,
            num_query_points=int(num_query_points),
            num_query_points_per_face_for_seg=num_query_points_per_face_for_seg,
            sharp_point_ratio=sharp_point_ratio,
            device=device,
            center=mesh_geometry.center,
            scale=mesh_geometry.scale,
            render_mesh_space="canonical_z_up_pre_blender_import",
            render_to_batch_rotation=mesh_geometry.render_to_model_rotation,
            reasoning_effort=str(args.reasoning_effort),
            blender_bin=args.blender_bin,
        )
        return PreparedMeshBatch(
            batch=auto_kinematics["batch"],
            query_face_indices=np.asarray(auto_kinematics["query_face_indices"], dtype=np.int64),
            link_names=list(auto_kinematics["link_names"]),
            joint_specs=list(auto_kinematics["joint_specs"]),
            auto_kinematics_metadata=dict(auto_kinematics["metadata"]),
            link_point_prompt_face_ids=np.asarray(
                auto_kinematics["link_point_prompt_face_ids"],
                dtype=np.int64,
            ),
        )

    link_names, joint_specs = prompt_for_link_names_and_joints()
    batch, query_face_indices = prepare_inference_batch_from_mesh(
        mesh_geometry.normalized_mesh,
        num_shape_points=num_shape_points,
        num_query_points=int(num_query_points),
        num_query_points_per_face_for_seg=num_query_points_per_face_for_seg,
        sharp_point_ratio=sharp_point_ratio,
        link_names=link_names,
        joint_specs=joint_specs,
        device=device,
    )
    return PreparedMeshBatch(
        batch=batch,
        query_face_indices=np.asarray(query_face_indices, dtype=np.int64),
        link_names=link_names,
        joint_specs=joint_specs,
        auto_kinematics_metadata={},
        link_point_prompt_face_ids=None,
    )


def _run_mesh_like_prediction(
    args: argparse.Namespace,
    *,
    model: Particulate2ArticulationModel,
    batch: dict[str, Any],
    original_mesh: Any,
    normalized_mesh: Any,
    query_face_indices: np.ndarray,
    joint_refit_num_query_points: int,
    num_links: int,
    center: np.ndarray | tuple[float, float, float],
    scale: float,
) -> dict[str, Any]:
    """Runs the shared mesh/meta-root prediction pipeline after batch preparation."""
    output = run_batched_model_inference(
        model,
        query_batch_size=int(args.query_batch_size),
        no_point_prompt_for_unique_text=bool(args.no_point_prompt),
        decode_joint_parameters=False,
        **batch,
    )
    point_part_probabilities = (
        F.softmax(output["segmentation_logits"], dim=-1)[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    point_part_ids = (
        output["segmentation_logits"].argmax(dim=-1)[0].detach().cpu().numpy().astype(np.int32)
    )
    face_part_ids, face_part_ids_unrefined = decode_face_part_ids(
        normalized_mesh,
        point_part_ids=point_part_ids,
        point_part_probabilities=point_part_probabilities,
        query_face_indices=query_face_indices,
        input_part_ids=np.arange(int(num_links), dtype=np.int32),
        strict=bool(args.strict_face_postprocess),
        enforce_connectivity_per_part=bool(
            getattr(args, "enforce_connectivity_per_part", False)
        ),
    )
    motion_artifacts = compute_motion_prediction_artifacts(
        args,
        model=model,
        batch=batch,
        normalized_mesh=normalized_mesh,
        face_part_ids=face_part_ids,
        joint_refit_num_query_points=int(joint_refit_num_query_points),
        num_links=int(num_links),
        center=center,
        scale=scale,
    )
    return {
        "query_points": tensor_to_numpy(batch["query_points"][0], dtype=np.float32),
        "query_normals": tensor_to_numpy(batch["query_point_normals"][0], dtype=np.float32),
        "point_part_ids": point_part_ids,
        "face_part_ids": face_part_ids,
        "face_part_ids_unrefined": face_part_ids_unrefined,
        **motion_artifacts,
    }


def write_mesh_like_prediction_files(
    output_dir: Path,
    *,
    face_part_ids: np.ndarray,
    query_points: np.ndarray,
    query_face_indices: np.ndarray,
    point_part_ids: np.ndarray,
    center: np.ndarray,
    scale: float,
    joint_refit_sampling: dict[str, np.ndarray] | None,
    query_normals: np.ndarray | None = None,
    gt_part_ids: np.ndarray | None = None,
) -> None:
    """Writes the common segmentation/query artifacts for mesh-like inputs."""
    np.save(output_dir / "seg.npy", np.asarray(face_part_ids, dtype=np.int32))
    _save_query_predictions(
        output_dir / "query_predictions.npz",
        query_points=query_points,
        query_normals=query_normals,
        face_indices=query_face_indices,
        part_ids=point_part_ids,
        gt_part_ids=gt_part_ids,
        center=np.asarray(center, dtype=np.float32),
        scale=scale,
        joint_refit_sampling=joint_refit_sampling,
    )


def write_kinematic_and_overparam_visualization(
    output_dir: Path,
    *,
    kinematic_records: dict[str, Any],
    visualization_records: dict[str, Any] | None,
    motion_output: dict[str, Any],
    query_points: np.ndarray,
    point_part_ids: np.ndarray,
    joint_refit_sampling: dict[str, np.ndarray] | None,
    center: np.ndarray,
    scale: float,
) -> Path | None:
    """Writes `kinematic.json` and the shared overparameterized joint visualization."""
    write_json(output_dir / "kinematic.json", kinematic_records)
    records_for_visualization = (
        kinematic_records if visualization_records is None else visualization_records
    )
    if joint_refit_sampling is None:
        overparam_visualization_query_points = denormalize_points(
            query_points,
            center=np.asarray(center, dtype=np.float32),
            scale=scale,
        )
        overparam_visualization_link_ids = np.asarray(point_part_ids, dtype=np.int32)
    else:
        overparam_visualization_query_points = denormalize_points(
            joint_refit_sampling["query_points"],
            center=np.asarray(center, dtype=np.float32),
            scale=scale,
        )
        overparam_visualization_link_ids = np.asarray(
            joint_refit_sampling["link_ids"],
            dtype=np.int32,
        )
    return save_joint_overparam_visualization_from_model_output(
        output_dir,
        output=motion_output,
        query_points=overparam_visualization_query_points,
        query_link_ids=overparam_visualization_link_ids,
        links=records_for_visualization["links"],
        joints=records_for_visualization["joints"],
        center=center,
        scale=scale,
    )


def save_articulated_mesh_outputs(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    original_mesh: Any,
    face_part_ids: np.ndarray,
    motion_hierarchy: list[tuple[int, int]],
    is_part_revolute: np.ndarray,
    is_part_prismatic: np.ndarray,
    revolute_plucker: np.ndarray,
    revolute_range: np.ndarray,
    prismatic_axis: np.ndarray,
    prismatic_range: np.ndarray,
    urdf_name: str,
    link_names: list[str],
    urdf_is_part_revolute: np.ndarray | None = None,
    urdf_is_part_prismatic: np.ndarray | None = None,
) -> np.ndarray:
    """Saves the common segmented mesh exports, plus optional URDF."""
    mesh_parts_original, unique_part_ids = save_segmented_visualizations(
        output_dir,
        original_mesh,
        face_part_ids,
        motion_hierarchy=motion_hierarchy,
        is_part_revolute=is_part_revolute,
        is_part_prismatic=is_part_prismatic,
        revolute_plucker=revolute_plucker,
        revolute_range=revolute_range,
        prismatic_axis=prismatic_axis,
        prismatic_range=prismatic_range,
        animation_frames=int(args.animation_frames),
    )

    if args.export_urdf:
        export_urdf(
            mesh_parts_original,
            unique_part_ids,
            motion_hierarchy,
            (
                np.asarray(is_part_revolute, dtype=np.bool_)
                if urdf_is_part_revolute is None
                else np.asarray(urdf_is_part_revolute, dtype=np.bool_)
            ),
            (
                np.asarray(is_part_prismatic, dtype=np.bool_)
                if urdf_is_part_prismatic is None
                else np.asarray(urdf_is_part_prismatic, dtype=np.bool_)
            ),
            revolute_plucker,
            revolute_range,
            prismatic_axis,
            prismatic_range,
            output_path=str(output_dir / "urdf" / "model.urdf"),
            name=urdf_name,
            link_names=link_names,
        )
    return unique_part_ids


def run_mesh_inference(
    args: argparse.Namespace,
    *,
    model: Particulate2ArticulationModel,
    checkpoint_path: Path,
    run_dir: Path,
    device: torch.device,
    num_shape_points: int,
    default_num_query_points: int | None,
    sharp_point_ratio: float,
) -> None:
    input_path = args.input_path.expanduser().resolve()
    input_mode = resolve_input_mode(input_path)
    output_dir = _resolve_output_dir(args, input_path)
    saved_auto_root: SavedAutoKinematicsRoot | None = None
    mesh_input_path = input_path
    mesh_input_up_dir = args.up_dir
    mode = "mesh"
    if input_mode == "saved_auto_kinematics_root":
        saved_auto_root = load_saved_auto_kinematics_root(input_path)
        mesh_input_path = saved_auto_root.mesh_path
        mesh_input_up_dir = saved_auto_root.mesh_up_dir
        mode = "saved_auto_kinematics_root"
    mesh_geometry = prepare_mesh_geometry(
        input_path=mesh_input_path,
        up_dir=mesh_input_up_dir,
    )
    joint_refit_num_query_points = _resolve_joint_refit_num_query_points(
        args,
        default_num_query_points=default_num_query_points,
    )
    segmentation_num_query_points = _resolve_segmentation_num_query_points(
        args,
        mesh=mesh_geometry.normalized_mesh,
        default_num_query_points=default_num_query_points,
    )
    if saved_auto_root is not None:
        mesh_batch = _prepare_saved_auto_kinematics_mesh_batch(
            auto_root=saved_auto_root,
            mesh=mesh_geometry.normalized_mesh,
            num_shape_points=num_shape_points,
            num_query_points=int(segmentation_num_query_points),
            num_query_points_per_face_for_seg=args.num_query_points_per_face_for_seg,
            sharp_point_ratio=sharp_point_ratio,
            device=device,
        )
    else:
        mesh_batch = _prepare_mesh_batch(
            args,
            mesh_geometry=mesh_geometry,
            output_dir=output_dir,
            num_shape_points=num_shape_points,
            num_query_points=int(segmentation_num_query_points),
            num_query_points_per_face_for_seg=args.num_query_points_per_face_for_seg,
            sharp_point_ratio=sharp_point_ratio,
            device=device,
        )
    prediction = _run_mesh_like_prediction(
        args,
        model=model,
        batch=mesh_batch.batch,
        original_mesh=mesh_geometry.original_mesh,
        normalized_mesh=mesh_geometry.normalized_mesh,
        query_face_indices=mesh_batch.query_face_indices,
        joint_refit_num_query_points=int(joint_refit_num_query_points),
        num_links=len(mesh_batch.link_names),
        center=mesh_geometry.center,
        scale=mesh_geometry.scale,
    )
    motion_output = prediction["motion_output"]
    joint_refit_sampling = prediction["joint_refit_sampling"]
    motion_arrays_normalized = prediction["motion_arrays_normalized"]
    motion_arrays_world = prediction["motion_arrays_world"]
    prismatic_axis_world = prediction["prismatic_axis_world"]
    prismatic_axis_normalized = prediction["prismatic_axis_normalized"]
    point_part_ids = prediction["point_part_ids"]
    face_part_ids = prediction["face_part_ids"]
    face_part_ids_unrefined = prediction["face_part_ids_unrefined"]
    query_points = prediction["query_points"]

    write_mesh_like_prediction_files(
        output_dir,
        face_part_ids=face_part_ids,
        query_points=query_points,
        query_face_indices=mesh_batch.query_face_indices,
        point_part_ids=point_part_ids,
        center=mesh_geometry.center,
        scale=mesh_geometry.scale,
        joint_refit_sampling=joint_refit_sampling,
    )

    motion_hierarchy = [
        (parent_link_id, child_link_id)
        for parent_link_id, child_link_id, _ in mesh_batch.joint_specs
    ]
    predicted_kinematic = build_predicted_kinematic_records(
        mesh_batch.link_names,
        mesh_batch.joint_specs,
        revolute_plucker=motion_arrays_world["revolute_plucker"],
        prismatic_plucker=motion_arrays_world["prismatic_plucker"],
        prismatic_axis=prismatic_axis_world,
        revolute_range=motion_arrays_world["revolute_range"],
        prismatic_range=motion_arrays_world["prismatic_range"],
        revolute_parameter_valid=motion_arrays_normalized["revolute_parameter_valid"],
        prismatic_parameter_valid=motion_arrays_normalized["prismatic_parameter_valid"],
    )
    visualization_link_point_prompts_world, visualization_link_point_prompt_ids = (
        resolve_visualized_batch_link_point_prompts(
            args=args,
            batch=mesh_batch.batch,
            links=predicted_kinematic["links"],
            center=mesh_geometry.center,
            scale=mesh_geometry.scale,
        )
    )
    visualization_path = save_predicted_point_query_visualization(
        output_dir,
        query_points=denormalize_points(
            query_points,
            center=mesh_geometry.center,
            scale=mesh_geometry.scale,
        ),
        query_normals=prediction["query_normals"],
        predicted_part_ids=point_part_ids,
        link_point_prompts=visualization_link_point_prompts_world,
        link_point_prompt_ids=visualization_link_point_prompt_ids,
        links=predicted_kinematic["links"],
        predicted_joints=predicted_kinematic["joints"],
        predicted_link_confidences=_query_visualization_part_confidences_from_motion_output(
            motion_output
        ),
    )
    overparam_visualization_path = write_kinematic_and_overparam_visualization(
        output_dir,
        kinematic_records=predicted_kinematic,
        visualization_records=None,
        motion_output=motion_output,
        query_points=query_points,
        point_part_ids=point_part_ids,
        joint_refit_sampling=joint_refit_sampling,
        center=mesh_geometry.center,
        scale=mesh_geometry.scale,
    )
    unique_part_ids = save_articulated_mesh_outputs(
        args,
        output_dir=output_dir,
        original_mesh=mesh_geometry.original_mesh,
        face_part_ids=face_part_ids,
        motion_hierarchy=motion_hierarchy,
        is_part_revolute=motion_arrays_normalized["is_part_revolute"],
        is_part_prismatic=motion_arrays_normalized["is_part_prismatic"],
        revolute_plucker=motion_arrays_world["revolute_plucker"],
        revolute_range=motion_arrays_world["revolute_range"],
        prismatic_axis=prismatic_axis_world,
        prismatic_range=motion_arrays_world["prismatic_range"],
        urdf_name=input_path.stem,
        link_names=mesh_batch.link_names,
    )

    eval_output_dir = None
    if args.eval:
        eval_output_dir = save_evaluation_prediction(
            output_dir=output_dir,
            mesh=mesh_geometry.normalized_mesh,
            face_part_ids=face_part_ids,
            face_part_ids_unrefined=face_part_ids_unrefined,
            motion_hierarchy=motion_hierarchy,
            is_part_revolute=motion_arrays_normalized["is_part_revolute"],
            is_part_prismatic=motion_arrays_normalized["is_part_prismatic"],
            revolute_plucker=motion_arrays_normalized["revolute_plucker"],
            revolute_range=motion_arrays_normalized["revolute_range"],
            prismatic_axis=prismatic_axis_normalized,
            prismatic_range=motion_arrays_normalized["prismatic_range"],
        )

    metadata = build_base_metadata(
        args,
        mode=mode,
        input_path=input_path,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        num_shape_points=num_shape_points,
        segmentation_num_query_points=segmentation_num_query_points,
        joint_refit_num_query_points=joint_refit_num_query_points,
        sharp_point_ratio=sharp_point_ratio,
    ) | {
        "link_names": mesh_batch.link_names,
        "num_input_parts": int(len(mesh_batch.link_names)),
        "joint_specs": [
            {
                "parent_link_id": int(parent_link_id),
                "child_link_id": int(child_link_id),
                "joint_type": joint_type,
            }
            for parent_link_id, child_link_id, joint_type in mesh_batch.joint_specs
        ],
        "unique_part_ids_in_segmentation": unique_part_ids.astype(int).tolist(),
        "missing_link_ids_in_segmentation": _missing_link_ids(
            len(mesh_batch.link_names),
            unique_part_ids,
        ),
        "visualization_path": str(visualization_path),
        "normalization": {
            "center": mesh_geometry.center.tolist(),
            "scale": mesh_geometry.scale,
        },
        "mesh_orientation": {
            "input_up_dir": str(mesh_input_up_dir),
            "canonical_up_dir": "+Z",
            "reoriented_to_canonical_z_up": bool(mesh_input_up_dir != "+Z"),
            "rotation_matrix": mesh_geometry.up_dir_rotation.tolist(),
        },
    } | build_joint_refit_metadata(
        joint_refit_sampling,
        num_links=len(mesh_batch.link_names),
    )
    if mesh_batch.auto_kinematics_metadata:
        metadata["auto_kinematics"] = mesh_batch.auto_kinematics_metadata
    write_metadata_and_summary(
        output_dir=output_dir,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        unique_part_ids=unique_part_ids,
        visualization_path=visualization_path,
        overparam_visualization_path=overparam_visualization_path,
        optional_paths={
            "overparam_visualization_path": overparam_visualization_path,
            "eval_output_dir": eval_output_dir,
        },
    )


def run_meta_root_inference(
    args: argparse.Namespace,
    *,
    model: Particulate2ArticulationModel,
    checkpoint_path: Path,
    run_dir: Path,
    device: torch.device,
    num_shape_points: int,
    default_num_query_points: int | None,
    sharp_point_ratio: float,
) -> None:
    meta_root = args.input_path.expanduser().resolve()
    original_obj_path, face_gt_link_ids, links, joints, meta_root_format = load_meta_root_annotation(
        meta_root,
    )
    original_mesh = load_trimesh(original_obj_path)
    normalized_mesh, center, scale = normalize_mesh(original_mesh)
    joint_refit_num_query_points = _resolve_joint_refit_num_query_points(
        args,
        default_num_query_points=default_num_query_points,
    )
    segmentation_num_query_points = _resolve_segmentation_num_query_points(
        args,
        mesh=normalized_mesh,
        default_num_query_points=default_num_query_points,
    )
    batch, meta_info = prepare_inference_batch_from_meta_root(
        normalized_mesh,
        face_link_ids=face_gt_link_ids,
        links=links,
        joints=joints,
        num_shape_points=num_shape_points,
        num_query_points=int(segmentation_num_query_points),
        num_query_points_per_face_for_seg=args.num_query_points_per_face_for_seg,
        sharp_point_ratio=sharp_point_ratio,
        device=device,
    )
    prediction = _run_mesh_like_prediction(
        args,
        model=model,
        batch=batch,
        original_mesh=original_mesh,
        normalized_mesh=normalized_mesh,
        query_face_indices=meta_info["query_face_indices"],
        joint_refit_num_query_points=int(joint_refit_num_query_points),
        num_links=len(meta_info["link_names"]),
        center=center,
        scale=scale,
    )
    motion_output = prediction["motion_output"]
    joint_refit_sampling = prediction["joint_refit_sampling"]
    motion_arrays_normalized = prediction["motion_arrays_normalized"]
    motion_arrays_world = prediction["motion_arrays_world"]
    prismatic_axis_world = prediction["prismatic_axis_world"]
    prismatic_axis_normalized = prediction["prismatic_axis_normalized"]
    point_part_ids = prediction["point_part_ids"]
    face_part_ids = prediction["face_part_ids"]
    face_part_ids_unrefined = prediction["face_part_ids_unrefined"]
    query_points = prediction["query_points"]
    query_normals = prediction["query_normals"]
    (
        gt_revolute_range_forward_mask,
        gt_prismatic_range_forward_mask,
    ) = range_forward_masks_by_child_link(
        meta_info["sorted_joints"],
        num_links=len(meta_info["link_names"]),
    )
    forwarded_revolute_range_world = zero_unforwarded_link_ranges(
        motion_arrays_world["revolute_range"],
        forward_mask=gt_revolute_range_forward_mask,
    )
    forwarded_prismatic_range_world = zero_unforwarded_link_ranges(
        motion_arrays_world["prismatic_range"],
        forward_mask=gt_prismatic_range_forward_mask,
    )
    output_dir = _resolve_output_dir(args, meta_root)
    write_mesh_like_prediction_files(
        output_dir,
        face_part_ids=face_part_ids,
        query_points=query_points,
        query_normals=query_normals,
        query_face_indices=meta_info["query_face_indices"],
        point_part_ids=point_part_ids,
        center=np.asarray(center, dtype=np.float32),
        scale=scale,
        joint_refit_sampling=joint_refit_sampling,
        gt_part_ids=meta_info["query_gt_link_ids"],
    )

    predicted_kinematic = build_predicted_kinematic_records_from_links_and_joints(
        meta_info["sorted_links"],
        meta_info["sorted_joints"],
        revolute_plucker=motion_arrays_world["revolute_plucker"],
        prismatic_plucker=motion_arrays_world["prismatic_plucker"],
        prismatic_axis=prismatic_axis_world,
        revolute_range=motion_arrays_world["revolute_range"],
        prismatic_range=motion_arrays_world["prismatic_range"],
        revolute_parameter_valid=motion_arrays_normalized["revolute_parameter_valid"],
        prismatic_parameter_valid=motion_arrays_normalized["prismatic_parameter_valid"],
    )
    predicted_kinematic["joints"] = suppress_unforwarded_joint_ranges(
        predicted_kinematic["joints"],
        revolute_range_forward_mask=gt_revolute_range_forward_mask,
        prismatic_range_forward_mask=gt_prismatic_range_forward_mask,
    )
    predicted_joint_records = build_predicted_joint_records_from_links_and_joints(
        meta_info["sorted_links"],
        meta_info["sorted_joints"],
        revolute_plucker=motion_arrays_world["revolute_plucker"],
        prismatic_plucker=motion_arrays_world["prismatic_plucker"],
        prismatic_axis=prismatic_axis_world,
        revolute_range=motion_arrays_world["revolute_range"],
        prismatic_range=motion_arrays_world["prismatic_range"],
        revolute_parameter_valid=motion_arrays_normalized["revolute_parameter_valid"],
        prismatic_parameter_valid=motion_arrays_normalized["prismatic_parameter_valid"],
    )
    predicted_joint_records["joints"] = suppress_unforwarded_joint_ranges(
        predicted_joint_records["joints"],
        revolute_range_forward_mask=gt_revolute_range_forward_mask,
        prismatic_range_forward_mask=gt_prismatic_range_forward_mask,
    )
    visualization_link_point_prompts_world, visualization_link_point_prompt_ids = (
        resolve_visualized_batch_link_point_prompts(
            args=args,
            batch=batch,
            links=predicted_joint_records["links"],
            center=np.asarray(center, dtype=np.float32),
            scale=scale,
        )
    )
    overparam_visualization_path = write_kinematic_and_overparam_visualization(
        output_dir,
        kinematic_records=predicted_kinematic,
        visualization_records=predicted_joint_records,
        motion_output=motion_output,
        query_points=query_points,
        point_part_ids=point_part_ids,
        joint_refit_sampling=joint_refit_sampling,
        center=np.asarray(center, dtype=np.float32),
        scale=scale,
    )

    motion_hierarchy = [
        (int(joint["parent_link_id"]), int(joint["child_link_id"]))
        for joint in meta_info["sorted_joints"]
    ]
    unique_part_ids = save_articulated_mesh_outputs(
        args,
        output_dir=output_dir,
        original_mesh=original_mesh,
        face_part_ids=face_part_ids,
        motion_hierarchy=motion_hierarchy,
        is_part_revolute=motion_arrays_normalized["is_part_revolute"],
        is_part_prismatic=motion_arrays_normalized["is_part_prismatic"],
        revolute_plucker=motion_arrays_world["revolute_plucker"],
        revolute_range=forwarded_revolute_range_world,
        prismatic_axis=prismatic_axis_world,
        prismatic_range=forwarded_prismatic_range_world,
        urdf_name=meta_root.stem,
        link_names=meta_info["link_names"],
        urdf_is_part_revolute=motion_arrays_normalized["is_part_revolute"]
        & gt_revolute_range_forward_mask,
        urdf_is_part_prismatic=motion_arrays_normalized["is_part_prismatic"]
        & gt_prismatic_range_forward_mask,
    )

    eval_output_dir = None
    eval_result_path = None
    if args.eval:
        eval_output_dir, _ = export_and_evaluate_meta_root_prediction(
            output_dir=output_dir,
            meta_root=meta_root,
            mesh=normalized_mesh,
            face_part_ids=face_part_ids,
            face_part_ids_unrefined=face_part_ids_unrefined,
            motion_hierarchy=motion_hierarchy,
            is_part_revolute=motion_arrays_normalized["is_part_revolute"],
            is_part_prismatic=motion_arrays_normalized["is_part_prismatic"],
            revolute_plucker=motion_arrays_normalized["revolute_plucker"],
            revolute_range=motion_arrays_normalized["revolute_range"],
            prismatic_axis=prismatic_axis_normalized,
            prismatic_range=motion_arrays_normalized["prismatic_range"],
            num_points=DEFAULT_EVAL_NUM_POINTS,
        )
        eval_result_path = per_object_result_path(eval_output_dir)

    visualization_path = save_point_query_visualization(
        output_dir,
        query_points=denormalize_points(
            query_points,
            center=np.asarray(center, dtype=np.float32),
            scale=scale,
        ),
        query_normals=query_normals,
        predicted_part_ids=point_part_ids,
        gt_part_ids=meta_info["query_gt_link_ids"].astype(np.int32, copy=False),
        link_point_prompts=visualization_link_point_prompts_world,
        link_point_prompt_ids=visualization_link_point_prompt_ids,
        links=predicted_joint_records["links"],
        predicted_link_confidences=_query_visualization_part_confidences_from_motion_output(
            motion_output
        ),
        predicted_joints=predicted_joint_records["joints"],
        gt_joints=meta_info["sorted_joints"],
    )

    metadata = build_base_metadata(
        args,
        mode="meta_root",
        input_path=meta_root,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        num_shape_points=num_shape_points,
        segmentation_num_query_points=segmentation_num_query_points,
        joint_refit_num_query_points=joint_refit_num_query_points,
        sharp_point_ratio=sharp_point_ratio,
    ) | {
        "original_obj_path": str(original_obj_path),
        "meta_root_format": meta_root_format,
        "link_names": meta_info["link_names"],
        "num_input_parts": int(len(meta_info["link_names"])),
        "joints": [
            {
                "joint_id": int(joint["joint_id"]),
                "parent_link_id": int(joint["parent_link_id"]),
                "child_link_id": int(joint["child_link_id"]),
                "is_revolute": bool(joint.get("is_revolute", False)),
                "is_prismatic": bool(joint.get("is_prismatic", False)),
            }
            for joint in meta_info["sorted_joints"]
        ],
        "unique_part_ids_in_segmentation": unique_part_ids.astype(int).tolist(),
        "missing_link_ids_in_segmentation": _missing_link_ids(
            len(meta_info["link_names"]),
            unique_part_ids,
        ),
        "visualization_path": str(visualization_path),
        "normalization": {
            "center": np.asarray(center, dtype=np.float32).tolist(),
            "scale": float(scale),
        },
    } | build_joint_refit_metadata(
        joint_refit_sampling,
        num_links=len(meta_info["link_names"]),
    )
    if meta_root_format == "kinematic_json":
        metadata["face_seg_path"] = str(meta_root / "seg.npy")
        metadata["kinematic_json_path"] = str(meta_root / "kinematic.json")
    else:
        metadata["legacy_meta_path"] = str(meta_root / "meta.npz")
        metadata["legacy_link_axes_plucker_path"] = str(meta_root / "link_axes_plucker.npz")
        metadata["legacy_link_range_path"] = str(meta_root / "link_range.npz")
    write_metadata_and_summary(
        output_dir=output_dir,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        unique_part_ids=unique_part_ids,
        visualization_path=visualization_path,
        overparam_visualization_path=overparam_visualization_path,
        optional_paths={
            "overparam_visualization_path": overparam_visualization_path,
            "eval_output_dir": eval_output_dir,
            "eval_result_path": eval_result_path,
        },
    )


def _run_inference_for_single_input(
    args: argparse.Namespace,
    *,
    model: Particulate2ArticulationModel,
    checkpoint_path: Path,
    run_dir: Path,
    device: torch.device,
    num_shape_points: int,
    default_num_query_points: int | None,
    sharp_point_ratio: float,
) -> None:
    input_mode = resolve_input_mode(args.input_path)
    if input_mode == "meta_root":
        run_meta_root_inference(
            args,
            model=model,
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
            device=device,
            num_shape_points=num_shape_points,
            default_num_query_points=default_num_query_points,
            sharp_point_ratio=sharp_point_ratio,
        )
        return

    run_mesh_inference(
        args,
        model=model,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        device=device,
        num_shape_points=num_shape_points,
        default_num_query_points=default_num_query_points,
        sharp_point_ratio=sharp_point_ratio,
    )


def _cleanup_after_single_input(device: torch.device) -> None:
    """Release per-object Python/CUDA allocations before the next asset."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.eval:
        _require_evaluation_dependencies()
    input_paths = _resolve_input_paths(args.input_path)
    pending_input_paths = _resolve_pending_input_paths(args, input_paths)
    output_root = args.output_dir.expanduser().resolve()
    if not pending_input_paths:
        print("Skipping inference: all requested inputs already have outputs.")
        if args.eval:
            sample_results = refresh_output_root_summaries(output_root)
            if sample_results:
                print(
                    f"Refreshed evaluation summaries for {len(sample_results)} samples under "
                    f"{output_root}"
                )
        return

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is not available.")
    run_dir = args.run_dir.expanduser().resolve()
    config = load_run_config(run_dir)
    configure_runtime_environment(config)

    checkpoint_path = resolve_checkpoint_path(run_dir, args.ckpt_idx)
    num_shape_points, default_num_query_points, sharp_point_ratio = (
        resolve_inference_sampling_config(config)
    )

    model = Particulate2ArticulationModel(**config["model"])
    load_model_checkpoint_for_inference(model, checkpoint_path, device=device)
    if args.no_point_prompt and not model.encoder.use_text_conditioning:
        raise ValueError("--no-point-prompt requires a text-conditioned model.")
    input_modes = {resolve_input_mode(input_path) for input_path in pending_input_paths}
    if args.auto_kinematics and input_modes != {"mesh"}:
        raise ValueError("--auto-kinematics is only supported for direct mesh inputs.")
    if "saved_auto_kinematics_root" in input_modes and args.up_dir != "+Z":
        raise ValueError(
            "--up_dir is not used for saved auto-kinematics roots; "
            "they load the stored visualization mesh in +Y-up space automatically."
        )
    if args.up_dir != "+Z" and input_modes - {"mesh"}:
        raise ValueError("--up_dir is only supported for direct mesh inputs.")
    if model.encoder.use_text_conditioning and input_modes & {
        "mesh",
        "meta_root",
        "saved_auto_kinematics_root",
    }:
        model.encoder.compute_link_text_embeddings_on_the_fly = True

    print(
        f"Running inference for {len(pending_input_paths)} inputs from "
        f"{args.input_path.expanduser().resolve()} "
        f"using checkpoint {checkpoint_path}"
    )
    num_failures = 0
    for input_idx, input_path in enumerate(pending_input_paths, start=1):
        print(f"[{input_idx}/{len(pending_input_paths)}] {input_path}")
        per_input_args = argparse.Namespace(**vars(args))
        per_input_args.input_path = input_path
        try:
            _run_inference_for_single_input(
                per_input_args,
                model=model,
                checkpoint_path=checkpoint_path,
                run_dir=run_dir,
                device=device,
                num_shape_points=num_shape_points,
                default_num_query_points=default_num_query_points,
                sharp_point_ratio=sharp_point_ratio,
            )
        except Exception as exc:
            num_failures += 1
            exc_message = str(exc)
            exc_detail = (
                f"{type(exc).__name__}: {exc_message}"
                if exc_message
                else type(exc).__name__
            )
            print(f"Error processing {input_path}: {exc_detail}")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        finally:
            del per_input_args
            _cleanup_after_single_input(device)
    print(
        f"Completed inference for {len(pending_input_paths) - num_failures} / "
        f"{len(pending_input_paths)} inputs."
    )
    if args.eval:
        sample_results = refresh_output_root_summaries(output_root)
        if sample_results:
            print(
                f"Wrote evaluation summaries for {len(sample_results)} samples under "
                f"{output_root}"
            )
    if num_failures and not bool(args.eval):
        raise RuntimeError(f"Inference failed for {num_failures} input(s).")


if __name__ == "__main__":
    main()
