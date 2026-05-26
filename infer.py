from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from instruct_particulate.model import Particulate2ArticulationModel
from instruct_particulate.utils.data_utils import (
    load_trimesh,
    normalize_mesh,
    up_dir_rotation_matrix,
    reorient_mesh_to_z_up,
)
from instruct_particulate.utils.export_utils import export_urdf
from instruct_particulate.utils.inference_utils import (
    axis_point_to_plucker_torch,
    denormalize_motion_parameters,
    estimate_prismatic_limit_torch,
    estimate_revolute_limit_torch,
    fit_axis_to_closest_points_torch,
    motion_arrays_from_model_output,
    prismatic_directions_from_plucker,
    run_joint_refit_from_face_seg,
    write_json,
)
from instruct_particulate.utils.inference_visualization_utils import (
    print_inference_summary,
    save_joint_overparam_visualization_from_model_output,
    save_segmented_visualizations,
    select_visualized_link_point_prompts,
)
from instruct_particulate.utils.postprocessing_utils import (
    find_unrefined_part_ids_for_faces,
    refine_face_part_ids_for_inference,
)


@dataclass(frozen=True)
class PreparedMeshGeometry:
    original_mesh: Any
    normalized_mesh: Any
    center: np.ndarray
    scale: float
    up_dir_rotation: np.ndarray
    render_mesh: Any
    render_to_model_rotation: np.ndarray


_INFER_OVERPARAM_AXIS_AABB_HALF_EXTENT_MIN = 1e-4


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
    *,
    model: Particulate2ArticulationModel,
    batch: dict[str, Any],
    normalized_mesh: Any,
    face_part_ids: np.ndarray,
    joint_refit_num_query_points: int,
    num_links: int,
    query_batch_size: int,
    no_point_prompt: bool,
    joint_decoding_confidence_temperature: float,
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
        query_batch_size=int(query_batch_size),
        no_point_prompt_for_unique_text=bool(no_point_prompt),
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
            confidence_temperature=joint_decoding_confidence_temperature,
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


def resolve_visualized_batch_link_point_prompts(
    *,
    batch: dict[str, Any],
    links: list[dict[str, Any]],
    no_point_prompt: bool,
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
        hide_unique_text_prompts=bool(no_point_prompt),
        link_point_prompt_dropout_eligible=dropout_eligible_np,
    )


def build_base_metadata(
    *,
    mode: str,
    input_path: Path,
    run_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    num_shape_points: int,
    segmentation_num_query_points: int,
    joint_refit_num_query_points: int,
    num_query_points_per_face_for_seg: int | None,
    query_batch_size: int,
    no_point_prompt: bool,
    enforce_connectivity_per_part: bool,
    joint_decoding_confidence_temperature: float,
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
            if num_query_points_per_face_for_seg is None
            else int(num_query_points_per_face_for_seg)
        ),
        "query_batch_size": int(query_batch_size),
        "no_point_prompt": bool(no_point_prompt),
        "enforce_connectivity_per_part": bool(enforce_connectivity_per_part),
        "joint_decoding_confidence_temperature": float(
            joint_decoding_confidence_temperature
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
    animation_frames: int,
    export_urdf_enabled: bool,
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
        animation_frames=int(animation_frames),
    )

    if export_urdf_enabled:
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
