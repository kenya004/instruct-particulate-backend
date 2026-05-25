import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from instruct_particulate.utils.articulation_utils import (
    articulate_points_with_joints,
    axis_point_to_plucker,
    joint_has_prismatic_motion,
    joint_has_prismatic_range,
    joint_has_revolute_motion,
    joint_has_revolute_range,
    joint_is_movable,
    plucker_to_axis_point,
)
from instruct_particulate.utils.data_utils import normalize_points_to_unit_extent

_STACKED_BATCH_KEYS = (
    "shape_points",
    "shape_point_normals",
    "query_points",
    "query_point_normals",
    "link_ids",
)

_OPTIONAL_STACKED_BATCH_KEYS = (
    "revolute_closest_axis_points",
    "revolute_low_points",
    "revolute_high_points",
    "revolute_axis_directions",
    "prismatic_closest_axis_points",
    "prismatic_low_points",
    "prismatic_high_points",
    "prismatic_axis_directions",
)

_JOINT_PAD_SPECS = (
    ("joint_connections", -1, torch.long),
    ("is_revolute", False, torch.bool),
    ("is_prismatic", False, torch.bool),
    ("revolute_axis_supervision_valid", False, torch.bool),
    ("prismatic_axis_supervision_valid", False, torch.bool),
    ("revolute_range_supervision_valid", False, torch.bool),
    ("prismatic_range_supervision_valid", False, torch.bool),
    ("revolute_axis", 0.0, torch.float32),
    ("prismatic_axis", 0.0, torch.float32),
    ("revolute_range", 0.0, torch.float32),
    ("prismatic_range", 0.0, torch.float32),
)

_JOINT_NUMERIC_FIELDS = (
    "revolute_axis",
    "prismatic_axis",
    "revolute_range",
    "prismatic_range",
)

_JOINT_SUPERVISION_VALID_FIELDS = (
    "revolute_axis_supervision_valid",
    "prismatic_axis_supervision_valid",
    "revolute_range_supervision_valid",
    "prismatic_range_supervision_valid",
)


def _disable_range_less_motion_for_articulation(
    joints: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Copies joints while disabling motion components that lack finite ranges.

    This is only for articulation-based visualization. The original joint
    records should still be used anywhere axis-only supervision needs to remain
    visible.
    """
    articulation_joints: List[Dict[str, Any]] = []
    for joint in joints:
        articulation_joint = dict(joint)
        if articulation_joint.get("revolute_range") is None:
            articulation_joint["is_revolute"] = False
            articulation_joint["revolute_axis"] = None
        if articulation_joint.get("prismatic_range") is None:
            articulation_joint["is_prismatic"] = False
            articulation_joint["prismatic_axis"] = None
        articulation_joints.append(articulation_joint)
    return articulation_joints


def _decode_json_field(value: Any, field_name: str) -> Any:
    """Decodes a JSON payload stored as a scalar NumPy field.

    Args:
        value: Stored field value loaded from an `.npz` file.
        field_name: Name of the field, used in error messages.

    Returns:
        The decoded Python object.

    Raises:
        ValueError: If the field is a non-scalar NumPy array.
        TypeError: If the field is not stored as a JSON string.
    """
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


def _sample_indices_with_replacement_when_needed(
    candidates: np.ndarray,
    size: int,
) -> np.ndarray:
    """Samples indices from a candidate pool.

    Args:
        candidates: Candidate indices to sample from.
        size: Number of indices to sample.

    Returns:
        Sampled indices as an `int64` array.
    """
    return np.random.choice(
        candidates,
        size=size,
        replace=size > len(candidates),
    ).astype(np.int64, copy=False)


def sample_points(
    sharp_point_flag: np.ndarray,
    num_points: int,
    sharp_point_ratio: float,
    candidate_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Samples point indices using stratified sharp/non-sharp sampling.

    Args:
        sharp_point_flag: Boolean array indicating whether each point is sharp.
        num_points: Number of points to sample.
        sharp_point_ratio: Target fraction of sharp points in the sample.
        candidate_mask: Optional boolean mask restricting which points may be
            sampled.

    Returns:
        Array of sampled point indices.

    Raises:
        ValueError: If `num_points` is negative or there are no valid
            candidates to sample from.
    """
    if num_points < 0:
        raise ValueError(f"num_points must be non-negative, got {num_points}")
    if num_points == 0:
        return np.zeros((0,), dtype=np.int64)

    if candidate_mask is None:
        candidate_indices = np.arange(len(sharp_point_flag))
    else:
        candidate_indices = np.flatnonzero(candidate_mask)
    if len(candidate_indices) == 0:
        raise ValueError("No candidate points are available for sampling")

    sharp_candidates = candidate_indices[sharp_point_flag[candidate_indices]]
    non_sharp_candidates = candidate_indices[~sharp_point_flag[candidate_indices]]

    num_sharp_desired = int(num_points * sharp_point_ratio)
    num_non_sharp_desired = num_points - num_sharp_desired

    if len(sharp_candidates) == 0:
        sampled_indices = _sample_indices_with_replacement_when_needed(
            non_sharp_candidates,
            size=num_points,
        )
    elif len(non_sharp_candidates) == 0:
        sampled_indices = _sample_indices_with_replacement_when_needed(
            sharp_candidates,
            size=num_points,
        )
    else:
        sampled_sharp = _sample_indices_with_replacement_when_needed(
            sharp_candidates,
            size=num_sharp_desired,
        )
        sampled_non_sharp = _sample_indices_with_replacement_when_needed(
            non_sharp_candidates,
            size=num_non_sharp_desired,
        )
        sampled_indices = np.concatenate([sampled_sharp, sampled_non_sharp], axis=0)
        np.random.shuffle(sampled_indices)

    return sampled_indices


def sample_query_points(
    link_ids: np.ndarray,
    num_points: int,
    query_balance_ratio: float,
    query_boundary_ratio: float = 0.0,
    query_boundary_temperature: float = 0.1,
    point_raw_link_ids: np.ndarray | None = None,
    query_boundary_distance_link_ids: np.ndarray | None = None,
    query_boundary_distance_to_raw_link: np.ndarray | None = None,
    link_scales: np.ndarray | None = None,
) -> np.ndarray:
    """Samples labeled query-point indices with optional link balancing.

    The sampling distribution is a mixture of:
    1. uniform-over-points sampling over all labeled points, and
    2. uniform-over-present-links sampling, where each point on a link shares
       that link's mass equally.
    3. optional within-link redistribution toward cached boundary-near points.

    Args:
        link_ids: Per-point link IDs where `-1` marks unlabeled points.
        num_points: Number of query points to sample.
        query_balance_ratio: Mixture weight in `[0, 1]` controlling how much of
            the sampling distribution is reweighted toward equal expected mass
            per present link.
        query_boundary_ratio: Mixture weight in `[0, 1]` controlling how much
            of the final sampling distribution is redistributed toward cached
            boundary-near points within each present link.
        query_boundary_temperature: Positive temperature applied to the
            normalized cached boundary distances before exponentiation.
        point_raw_link_ids: Optional per-point raw link IDs before any
            dataset-time link dropout or dense remapping.
        query_boundary_distance_link_ids: Optional raw link ID columns aligned
            with `query_boundary_distance_to_raw_link`.
        query_boundary_distance_to_raw_link: Optional cached matrix with shape
            `(N, K)` storing per-point distances to each represented raw link.
        link_scales: Optional current per-link scales used to normalize cached
            boundary distances for boundary-aware sampling.

    Returns:
        Array of sampled query-point indices.

    Raises:
        ValueError: If `num_points` is negative, either ratio is out of range,
            the boundary temperature is non-positive, required cached boundary
            fields are missing, or there are no labeled candidate points to
            sample from.
    """
    if num_points < 0:
        raise ValueError(f"num_points must be non-negative, got {num_points}")
    if not 0.0 <= query_balance_ratio <= 1.0:
        raise ValueError(
            "query_balance_ratio must be in [0, 1], "
            f"got {query_balance_ratio}"
        )
    if not 0.0 <= query_boundary_ratio <= 1.0:
        raise ValueError(
            "query_boundary_ratio must be in [0, 1], "
            f"got {query_boundary_ratio}"
        )
    if query_boundary_temperature <= 0.0:
        raise ValueError(
            "query_boundary_temperature must be positive, "
            f"got {query_boundary_temperature}"
        )
    if num_points == 0:
        return np.zeros((0,), dtype=np.int64)

    link_ids = np.asarray(link_ids, dtype=np.int64)
    candidate_indices = np.flatnonzero(link_ids >= 0)
    if len(candidate_indices) == 0:
        raise ValueError("No labeled query points are available for sampling")

    candidate_link_ids = link_ids[candidate_indices]
    sampling_probabilities = _compute_query_sampling_probabilities(
        candidate_link_ids=candidate_link_ids,
        query_balance_ratio=query_balance_ratio,
    )
    if query_boundary_ratio > 0.0:
        if (
            point_raw_link_ids is None
            or query_boundary_distance_link_ids is None
            or query_boundary_distance_to_raw_link is None
            or link_scales is None
        ):
            raise ValueError(
                "query_boundary_ratio > 0 requires point_raw_link_ids, "
                "query_boundary_distance_link_ids, query_boundary_distance_to_raw_link, "
                "and link_scales"
            )
        boundary_probabilities = _compute_boundary_aware_query_sampling_probabilities(
            candidate_indices=candidate_indices,
            candidate_link_ids=candidate_link_ids,
            base_probabilities=sampling_probabilities,
            current_link_ids=link_ids,
            point_raw_link_ids=np.asarray(point_raw_link_ids, dtype=np.int64),
            query_boundary_distance_link_ids=np.asarray(
                query_boundary_distance_link_ids,
                dtype=np.int64,
            ),
            query_boundary_distance_to_raw_link=np.asarray(
                query_boundary_distance_to_raw_link,
            ),
            link_scales=np.asarray(link_scales, dtype=np.float32),
            query_boundary_temperature=float(query_boundary_temperature),
        )
        sampling_probabilities = (
            (1.0 - query_boundary_ratio) * sampling_probabilities
            + query_boundary_ratio * boundary_probabilities
        )
        sampling_probabilities /= sampling_probabilities.sum()

    sampled_indices = np.random.choice(
        candidate_indices,
        size=num_points,
        replace=num_points > len(candidate_indices),
        p=sampling_probabilities,
    )
    return np.asarray(sampled_indices, dtype=np.int64)


def _compute_query_sampling_probabilities(
    *,
    candidate_link_ids: np.ndarray,
    query_balance_ratio: float,
) -> np.ndarray:
    """Builds the per-point query sampling probabilities.

    Args:
        candidate_link_ids: Link IDs for the labeled query candidates.
        query_balance_ratio: Mixture weight in `[0, 1]` for the link-balanced
            component of the sampling distribution.

    Returns:
        Per-candidate sampling probabilities that sum to 1.
    """
    _, inverse_link_indices = np.unique(candidate_link_ids, return_inverse=True)
    link_counts = np.bincount(inverse_link_indices).astype(np.float64, copy=False)

    num_candidates = float(len(candidate_link_ids))
    num_present_links = float(len(link_counts))
    per_point_uniform_mass = (1.0 - query_balance_ratio) / num_candidates
    per_point_balanced_mass = (
        query_balance_ratio
        / (num_present_links * link_counts[inverse_link_indices])
    )
    sampling_probabilities = per_point_uniform_mass + per_point_balanced_mass
    sampling_probabilities /= sampling_probabilities.sum()
    return sampling_probabilities


def _build_raw_to_current_link_id_map(
    *,
    point_raw_link_ids: np.ndarray,
    current_link_ids: np.ndarray,
) -> dict[int, int]:
    """Returns the current remapped link ID for every represented raw link."""
    if point_raw_link_ids.shape != current_link_ids.shape:
        raise ValueError(
            "point_raw_link_ids must match current_link_ids shape, "
            f"got {point_raw_link_ids.shape} and {current_link_ids.shape}"
        )
    raw_to_current_link_id: dict[int, int] = {}
    for raw_link_id, current_link_id in zip(
        point_raw_link_ids.tolist(),
        current_link_ids.tolist(),
        strict=True,
    ):
        raw_link_id = int(raw_link_id)
        if raw_link_id < 0:
            continue
        current_link_id = int(current_link_id)
        previous_current_link_id = raw_to_current_link_id.get(raw_link_id)
        if previous_current_link_id is None:
            raw_to_current_link_id[raw_link_id] = current_link_id
        elif previous_current_link_id != current_link_id:
            raise ValueError(
                "A raw link must map to exactly one current link after dataset preprocessing, "
                f"but raw_link_id={raw_link_id} maps to both {previous_current_link_id} "
                f"and {current_link_id}"
            )
    return raw_to_current_link_id


def _compute_boundary_aware_query_sampling_probabilities(
    *,
    candidate_indices: np.ndarray,
    candidate_link_ids: np.ndarray,
    base_probabilities: np.ndarray,
    current_link_ids: np.ndarray,
    point_raw_link_ids: np.ndarray,
    query_boundary_distance_link_ids: np.ndarray,
    query_boundary_distance_to_raw_link: np.ndarray,
    link_scales: np.ndarray,
    query_boundary_temperature: float,
) -> np.ndarray:
    """Redistributes base query mass toward cached boundary-near points.

    The returned probabilities preserve the total base mass assigned to each
    current link and only redistribute that mass within the link.
    """
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_link_ids = np.asarray(candidate_link_ids, dtype=np.int64)
    base_probabilities = np.asarray(base_probabilities, dtype=np.float64)
    current_link_ids = np.asarray(current_link_ids, dtype=np.int64)
    point_raw_link_ids = np.asarray(point_raw_link_ids, dtype=np.int64)
    query_boundary_distance_link_ids = np.asarray(
        query_boundary_distance_link_ids,
        dtype=np.int64,
    )
    query_boundary_distance_to_raw_link = np.asarray(
        query_boundary_distance_to_raw_link,
        dtype=np.float32,
    )
    link_scales = np.asarray(link_scales, dtype=np.float32)

    if candidate_indices.shape != candidate_link_ids.shape:
        raise ValueError(
            "candidate_indices must match candidate_link_ids shape, "
            f"got {candidate_indices.shape} and {candidate_link_ids.shape}"
        )
    if base_probabilities.shape != candidate_indices.shape:
        raise ValueError(
            "base_probabilities must match candidate_indices shape, "
            f"got {base_probabilities.shape} and {candidate_indices.shape}"
        )
    if point_raw_link_ids.shape != current_link_ids.shape:
        raise ValueError(
            "point_raw_link_ids must match current_link_ids shape, "
            f"got {point_raw_link_ids.shape} and {current_link_ids.shape}"
        )
    if query_boundary_distance_to_raw_link.shape[0] != current_link_ids.shape[0]:
        raise ValueError(
            "query_boundary_distance_to_raw_link must align with the per-point arrays, "
            f"got first dimension {query_boundary_distance_to_raw_link.shape[0]} and "
            f"{current_link_ids.shape[0]}"
        )
    if query_boundary_distance_to_raw_link.shape[1] != query_boundary_distance_link_ids.shape[0]:
        raise ValueError(
            "query_boundary_distance_to_raw_link must have one column per cached raw link ID, "
            f"got shape {query_boundary_distance_to_raw_link.shape} and "
            f"{query_boundary_distance_link_ids.shape}"
        )
    if np.any(candidate_link_ids >= len(link_scales)):
        raise ValueError(
            "candidate_link_ids reference missing link scales, "
            f"got max candidate link id {int(np.max(candidate_link_ids)) if candidate_link_ids.size > 0 else -1} "
            f"for link_scales shape {link_scales.shape}"
        )

    present_current_link_ids = np.unique(candidate_link_ids)
    if present_current_link_ids.size <= 1:
        return base_probabilities.copy()

    raw_to_current_link_id = _build_raw_to_current_link_id_map(
        point_raw_link_ids=point_raw_link_ids,
        current_link_ids=current_link_ids,
    )
    cached_column_current_link_ids = np.asarray(
        [
            raw_to_current_link_id.get(int(raw_link_id), -1)
            for raw_link_id in query_boundary_distance_link_ids.tolist()
        ],
        dtype=np.int64,
    )
    candidate_boundary_distance_matrix = query_boundary_distance_to_raw_link[candidate_indices]
    boundary_probabilities = np.zeros_like(base_probabilities, dtype=np.float64)
    for current_link_id in present_current_link_ids.tolist():
        link_mask = candidate_link_ids == current_link_id
        if not bool(np.any(link_mask)):
            continue
        link_mass = float(base_probabilities[link_mask].sum())
        foreign_column_mask = (
            (cached_column_current_link_ids >= 0)
            & (cached_column_current_link_ids != current_link_id)
        )
        if not bool(np.any(foreign_column_mask)):
            boundary_probabilities[link_mask] = link_mass / float(np.sum(link_mask))
            continue

        min_foreign_distance = candidate_boundary_distance_matrix[link_mask][:, foreign_column_mask].min(
            axis=1
        )
        current_link_scale = max(float(link_scales[int(current_link_id)]), 1e-6)
        normalized_distance = min_foreign_distance.astype(np.float64, copy=False) / current_link_scale
        boundary_affinity = np.exp(-normalized_distance / query_boundary_temperature)
        affinity_sum = float(boundary_affinity.sum())
        if not np.isfinite(affinity_sum) or affinity_sum <= 0.0:
            boundary_probabilities[link_mask] = link_mass / float(np.sum(link_mask))
            continue
        boundary_probabilities[link_mask] = link_mass * (boundary_affinity / affinity_sum)

    total_probability = float(boundary_probabilities.sum())
    if not np.isfinite(total_probability) or total_probability <= 0.0:
        return base_probabilities.copy()
    boundary_probabilities /= total_probability
    return boundary_probabilities


def _pad_tensor_list(
    values: Sequence[torch.Tensor],
    *,
    padding_value: float | int = 0,
) -> torch.Tensor:
    """Pads a variable-length tensor list along the leading dimension."""
    return pad_sequence(values, batch_first=True, padding_value=padding_value)


def _build_valid_flag(lengths: torch.Tensor) -> torch.Tensor:
    """Builds a boolean validity mask from per-sample sequence lengths."""
    max_length = int(lengths.max().item()) if lengths.numel() > 0 else 0
    return torch.arange(max_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def _stack_batch_tensors(
    batch: Sequence[Dict[str, Any]],
    keys: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Stacks fixed-shape tensor fields across the batch."""
    return {
        key: torch.stack([sample[key] for sample in batch], dim=0)
        for key in keys
    }


def _initialize_link_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Allocates padded link-level tensors for a batch."""
    max_links = max(sample["link_point_prompts"].shape[0] for sample in batch)
    batch_size = len(batch)
    sample_link_text_embeddings = [sample["link_text_embeddings"] for sample in batch]
    if not all(isinstance(embedding, torch.Tensor) for embedding in sample_link_text_embeddings):
        raise ValueError("Every sample in the batch must provide cached link_text_embeddings")
    language_dims = {
        int(embedding.shape[-1])
        for embedding in sample_link_text_embeddings
    }
    if len(language_dims) != 1:
        raise ValueError(
            "All samples in the batch must use the same cached link_text_embeddings dimension, "
            f"got {sorted(language_dims)}"
        )
    language_dim = next(iter(language_dims))
    batched: Dict[str, Any] = {
        "link_valid_flag": torch.zeros((batch_size, max_links), dtype=torch.bool),
        "link_point_prompt_dropout_eligible": torch.zeros((batch_size, max_links), dtype=torch.bool),
        "link_point_prompts": torch.zeros((batch_size, max_links, 3), dtype=torch.float32),
        "link_point_prompt_normals": torch.zeros((batch_size, max_links, 3), dtype=torch.float32),
        "link_scales": torch.zeros((batch_size, max_links), dtype=torch.float32),
        "link_text_embeddings": torch.zeros(
            (batch_size, max_links, language_dim),
            dtype=torch.float32,
        ),
    }
    return batched


def _copy_link_fields_into_batch(
    batched: Dict[str, Any],
    batch: Sequence[Dict[str, Any]],
) -> None:
    """Copies variable-length link fields into preallocated padded tensors."""
    for batch_index, sample in enumerate(batch):
        num_links = sample["link_point_prompts"].shape[0]
        batched["link_valid_flag"][batch_index, :num_links] = True
        batched["link_point_prompts"][batch_index, :num_links] = sample["link_point_prompts"]
        batched["link_point_prompt_normals"][batch_index, :num_links] = sample["link_point_prompt_normals"]
        link_point_prompt_dropout_eligible = sample.get("link_point_prompt_dropout_eligible")
        if link_point_prompt_dropout_eligible is None:
            batched["link_point_prompt_dropout_eligible"][batch_index, :num_links] = True
        else:
            batched["link_point_prompt_dropout_eligible"][batch_index, :num_links] = (
                link_point_prompt_dropout_eligible
            )
        link_scales = sample.get("link_scales")
        if link_scales is None:
            batched["link_scales"][batch_index, :num_links] = 1.0
        else:
            batched["link_scales"][batch_index, :num_links] = link_scales
        batched["link_text_embeddings"][batch_index, :num_links] = sample["link_text_embeddings"]


def _pad_joint_batch_fields(batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Pads joint-level tensors and builds their validity mask."""
    joint_lengths = torch.tensor(
        [sample["joint_connections"].shape[0] for sample in batch],
        dtype=torch.long,
    )
    batched: Dict[str, torch.Tensor] = {
        "joint_valid_flag": _build_valid_flag(joint_lengths),
    }
    for key, padding_value, dtype in _JOINT_PAD_SPECS:
        batched[key] = _pad_tensor_list(
            [sample[key] for sample in batch],
            padding_value=padding_value,
        ).to(dtype=dtype)
    return batched


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a list of dataset samples into a batch dictionary.

    Tensor fields with fixed shapes are stacked. Link-level tensors are padded
    to the maximum number of links in the batch and accompanied by a validity
    mask. Joint-level tensors are padded to the maximum number of joints and
    accompanied by `joint_valid_flag`. Text prompts remain a per-sample list.

    Args:
        batch: List of dataset samples.

    Returns:
        Batched sample dictionary.
    """
    batched: Dict[str, Any] = _stack_batch_tensors(batch, _STACKED_BATCH_KEYS)
    for key in _OPTIONAL_STACKED_BATCH_KEYS:
        if all(key in sample for sample in batch):
            batched[key] = torch.stack([sample[key] for sample in batch], dim=0)

    batched.update(_initialize_link_batch(batch))
    _copy_link_fields_into_batch(batched, batch)
    batched["link_text_prompts"] = [sample["link_text_prompts"] for sample in batch]
    batched.update(_pad_joint_batch_fields(batch))
    return batched


class InstructParticulateDataset(Dataset):
    """Loads cached articulated point-cloud samples for training and debugging."""

    def __init__(
        self,
        root: Union[str, Path],
        num_shape_points: int,
        num_query_points: int,
        sharp_point_ratio: float = 0.5,
        query_balance_ratio: float = 0.3,
        query_boundary_ratio: float = 0.0,
        query_boundary_temperature: float = 0.1,
        random_pose_prob: float = 0.0,
        dropout_individual_link_prob: float = 0.0,
        max_num_links: int = 8,
        use_augmentation: bool = False,
        random_rotation_deg_std: float = 0.0,
        random_translation_range: Sequence[float] = (-0.05, 0.05),
        random_scale_range: Sequence[float] = (0.9, 1.1),
        data_paths: Optional[Sequence[Union[str, Path]]] = None,
    ):
        """Initializes the dataset.

        Args:
            root: Root directory containing cached `.npz` files.
            num_shape_points: Number of points to sample for `shape_points`.
            num_query_points: Number of labeled points to sample for
                `query_points`.
            sharp_point_ratio: Target fraction of sharp points in sampled point
                sets.
            query_balance_ratio: Mixture weight used for query-point sampling.
                `0.0` samples uniformly over labeled points, while `1.0`
                reweights sampling toward equal expected mass per present link.
            query_boundary_ratio: Mixture weight used to redistribute a portion
                of the query-point sampling mass toward cached boundary-near
                points within each present link.
            query_boundary_temperature: Positive temperature used when turning
                normalized cached boundary distances into boundary affinities.
            random_pose_prob: Probability used for the sample-level and
                per-joint random posing logic before augmentation.
            dropout_individual_link_prob: Probability of dropping each
                non-root link from the kinematic tree before dense link-ID
                remapping.
            max_num_links: Maximum number of retained links after optional
                individual link dropout.
            use_augmentation: Whether to apply geometric augmentation after any
                random posing.
            random_rotation_deg_std: Standard deviation in degrees for an
                additional Gaussian z-rotation sampled after the base
                quarter-turn augmentation rotation.
            random_translation_range: Inclusive range used to sample each
                translation component during augmentation.
            random_scale_range: Inclusive range used to sample the uniform
                scale during augmentation.
            data_paths: Optional explicit `.npz` file list to use instead of
                recursively discovering all `.npz` files under `root`.
        """
        self.root = Path(root)
        self.num_shape_points = num_shape_points
        self.num_query_points = num_query_points
        if not 0.0 <= sharp_point_ratio <= 1.0:
            raise ValueError(
                f"sharp_point_ratio must be in [0, 1], got {sharp_point_ratio}"
            )
        self.sharp_point_ratio = sharp_point_ratio
        if not 0.0 <= query_balance_ratio <= 1.0:
            raise ValueError(
                f"query_balance_ratio must be in [0, 1], got {query_balance_ratio}"
            )
        self.query_balance_ratio = query_balance_ratio
        if not 0.0 <= query_boundary_ratio <= 1.0:
            raise ValueError(
                f"query_boundary_ratio must be in [0, 1], got {query_boundary_ratio}"
            )
        self.query_boundary_ratio = query_boundary_ratio
        if query_boundary_temperature <= 0.0:
            raise ValueError(
                "query_boundary_temperature must be positive, "
                f"got {query_boundary_temperature}"
            )
        self.query_boundary_temperature = float(query_boundary_temperature)
        if not 0.0 <= random_pose_prob <= 1.0:
            raise ValueError(f"random_pose_prob must be in [0, 1], got {random_pose_prob}")
        self.random_pose_prob = random_pose_prob
        if not 0.0 <= dropout_individual_link_prob <= 1.0:
            raise ValueError(
                "dropout_individual_link_prob must be in [0, 1], "
                f"got {dropout_individual_link_prob}"
            )
        self.dropout_individual_link_prob = dropout_individual_link_prob
        if max_num_links < 1:
            raise ValueError(f"max_num_links must be at least 1, got {max_num_links}")
        self.max_num_links = int(max_num_links)
        self.use_augmentation = use_augmentation
        if not np.isfinite(random_rotation_deg_std) or random_rotation_deg_std < 0.0:
            raise ValueError(
                "random_rotation_deg_std must be finite and non-negative, "
                f"got {random_rotation_deg_std}"
            )
        self.random_rotation_deg_std = float(random_rotation_deg_std)
        self.random_translation_range = self._validate_range(
            random_translation_range,
            "random_translation_range",
        )
        self.random_scale_range = self._validate_range(
            random_scale_range,
            "random_scale_range",
        )
        if data_paths is None:
            self.data = sorted(self.root.rglob("*.npz"))
        else:
            self.data = sorted(
                [
                    path if isinstance(path, Path) else Path(path)
                    for path in data_paths
                ]
            )

        if not self.data:
            raise ValueError(f"No .npz files found under {self.root}")

    def __len__(self) -> int:
        """Returns the number of cached samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Loads a sample and retries a few times if loading fails.

        Args:
            idx: Dataset index.

        Returns:
            Sample dictionary with point, link, and joint data.
        """
        max_retry = 5
        last_error: Optional[Exception] = None
        sample_idx = idx
        for _ in range(max_retry):
            try:
                return self._getitem(sample_idx)
            except Exception as error:
                last_error = error
                print(f"Error loading {self.data[sample_idx]} at index {sample_idx}: {error}")
                sample_idx = np.random.randint(0, len(self.data))
        assert last_error is not None
        raise last_error

    def _validate_range(
        self,
        value_range: Sequence[float],
        name: str,
    ) -> Tuple[float, float]:
        """Validates a numeric range argument.

        Args:
            value_range: Two-element range `[low, high]`.
            name: Argument name used in error messages.

        Returns:
            Tuple `(low, high)`.

        Raises:
            ValueError: If the range is not length 2 or if `low > high`.
        """
        if len(value_range) != 2:
            raise ValueError(f"{name} must have length 2, got {len(value_range)}")
        low, high = float(value_range[0]), float(value_range[1])
        if low > high:
            raise ValueError(f"{name} must satisfy low <= high, got {value_range}")
        return low, high

    def _rotate_points(self, points_or_normals: np.ndarray, rotation_z: float) -> np.ndarray:
        """Rotates 3D vectors around the z-axis.

        Args:
            points_or_normals: Array of 3D vectors.
            rotation_z: Rotation angle in radians.

        Returns:
            Rotated array with the same shape as the input.
        """
        rotation_matrix_z = np.array(
            [
                [np.cos(rotation_z), -np.sin(rotation_z), 0.0],
                [np.sin(rotation_z), np.cos(rotation_z), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return points_or_normals @ rotation_matrix_z.T

    def _sample_augmentation_params(self) -> Tuple[float, float, np.ndarray]:
        """Samples augmentation parameters for one example.

        Returns:
            Tuple containing the z-rotation angle, uniform scale, and
            translation vector.
        """
        rotation_z = np.random.choice([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        if self.random_rotation_deg_std > 0.0:
            rotation_z += np.deg2rad(
                np.random.normal(loc=0.0, scale=self.random_rotation_deg_std)
            )
        scale = float(np.random.uniform(*self.random_scale_range))
        translation = np.random.uniform(
            self.random_translation_range[0],
            self.random_translation_range[1],
            size=3,
        ).astype(np.float32)
        return rotation_z, scale, translation

    def _default_augmentation_params(self) -> Tuple[float, float, np.ndarray]:
        """Returns the identity augmentation parameters.

        Returns:
            Tuple containing zero rotation, unit scale, and zero translation.
        """
        return 0.0, 1.0, np.zeros(3, dtype=np.float32)

    def _augment_points(
        self,
        points: np.ndarray,
        rotation_z: float,
        scale: float,
        translation: np.ndarray,
    ) -> np.ndarray:
        """Applies rotation, scaling, and translation to points.

        Args:
            points: Array of 3D points.
            rotation_z: Rotation angle in radians.
            scale: Uniform scale factor.
            translation: Translation vector of shape `(3,)`.

        Returns:
            Augmented points with the same shape as the input.
        """
        augmented_points = self._rotate_points(points, rotation_z)
        augmented_points = augmented_points * scale
        augmented_points = augmented_points + translation
        return augmented_points

    def _augment_joint_axis(
        self,
        joint_axis: np.ndarray,
        rotation_z: float,
        scale: float,
        translation: np.ndarray,
    ) -> np.ndarray:
        """Applies the point-space augmentation to a Plucker axis.

        Args:
            joint_axis: Plucker line coordinates of shape `(6,)`.
            rotation_z: Rotation angle in radians.
            scale: Uniform scale factor.
            translation: Translation vector of shape `(3,)`.

        Returns:
            Augmented Plucker line coordinates of shape `(6,)`.
        """
        axis, point = plucker_to_axis_point(joint_axis)
        augmented_axis = self._rotate_points(axis[None], rotation_z)[0]
        augmented_point = self._augment_points(
            point[None],
            rotation_z=rotation_z,
            scale=scale,
            translation=translation,
        )[0]
        return axis_point_to_plucker(augmented_axis, augmented_point).astype(
            np.float32,
            copy=False,
        )

    def _closest_points_on_joint_axis(
        self,
        points: np.ndarray,
        joint_axis: np.ndarray,
    ) -> np.ndarray:
        """Projects points onto the joint axis represented in Plucker form."""
        axis_direction, axis_point = plucker_to_axis_point(
            np.asarray(joint_axis, dtype=np.float32)
        )
        offsets = np.asarray(points, dtype=np.float32) - axis_point[None, :]
        projection = np.sum(offsets * axis_direction[None, :], axis=-1, keepdims=True)
        return axis_point[None, :] + projection * axis_direction[None, :]

    def _copy_joints_with_isolated_motion(
        self,
        joints: List[Dict[str, Any]],
        child_link_id: int,
        motion_type: str,
    ) -> List[Dict[str, Any]]:
        """Copies joints while keeping only one motion component for one child link."""
        if motion_type not in {"revolute", "prismatic"}:
            raise ValueError(f"motion_type must be 'revolute' or 'prismatic', got {motion_type!r}")

        isolated_joints = self._copy_joints(joints)
        for joint in isolated_joints:
            if int(joint["child_link_id"]) != int(child_link_id):
                continue
            if motion_type == "revolute":
                joint["is_prismatic"] = False
                joint["prismatic_axis"] = None
                joint["prismatic_range"] = None
            else:
                joint["is_revolute"] = False
                joint["revolute_axis"] = None
                joint["revolute_range"] = None
            break
        return isolated_joints

    def _load_raw_sample(self, data_path: Path) -> Dict[str, Any]:
        """Loads one raw cached sample from disk.

        Args:
            data_path: Path to the cached `.npz` file.

        Returns:
            Dictionary containing raw geometry and decoded metadata.
        """
        with np.load(data_path, allow_pickle=False) as data:
            joints = []
            if "joints" in data.files:
                joints = _decode_json_field(data["joints"], "joints")
            query_boundary_distance_link_ids = None
            if "query_boundary_distance_link_ids" in data.files:
                query_boundary_distance_link_ids = np.asarray(
                    data["query_boundary_distance_link_ids"],
                    dtype=np.int64,
                )
            query_boundary_distance_to_raw_link = None
            if "query_boundary_distance_to_raw_link" in data.files:
                query_boundary_distance_to_raw_link = np.asarray(
                    data["query_boundary_distance_to_raw_link"],
                    dtype=np.float16,
                )
            return {
                "points": np.asarray(data["points"], dtype=np.float32),
                "normals": np.asarray(data["normals"], dtype=np.float32),
                "link_ids": np.asarray(data["link_ids"], dtype=np.int64),
                "sharp_point_flag": np.asarray(data["sharp_point_flag"], dtype=np.bool_),
                "links": _decode_json_field(data["links"], "links"),
                "joints": joints,
                "query_boundary_distance_link_ids": query_boundary_distance_link_ids,
                "query_boundary_distance_to_raw_link": query_boundary_distance_to_raw_link,
            }

    def _build_parent_by_child_lookup(
        self,
        joints: List[Dict[str, Any]],
    ) -> Dict[int, int]:
        """Builds a `child_link_id -> parent_link_id` lookup for one kinematic tree."""
        parent_by_child: Dict[int, int] = {}
        for joint in joints:
            child_link_id = int(joint["child_link_id"])
            parent_link_id = int(joint["parent_link_id"])
            if child_link_id in parent_by_child:
                raise ValueError(f"Multiple joints target child link_id={child_link_id}")
            parent_by_child[child_link_id] = parent_link_id
        return parent_by_child

    def _nearest_retained_ancestor_link_id(
        self,
        link_id: int,
        retained_link_ids: Set[int],
        parent_by_child: Dict[int, int],
        memo: Dict[int, Optional[int]],
    ) -> Optional[int]:
        """Finds the nearest retained ancestor for a link ID."""
        if link_id in memo:
            return memo[link_id]
        if link_id in retained_link_ids:
            memo[link_id] = link_id
            return link_id
        parent_link_id = parent_by_child.get(link_id)
        if parent_link_id is None:
            memo[link_id] = None
            return None
        memo[link_id] = self._nearest_retained_ancestor_link_id(
            parent_link_id,
            retained_link_ids,
            parent_by_child,
            memo,
        )
        return memo[link_id]

    def _reduce_kinematic_metadata_to_retained_links(
        self,
        links: List[Dict[str, Any]],
        joints: List[Dict[str, Any]],
        retained_link_ids: Set[int],
        parent_by_child: Dict[int, int],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Removes dropped links and rewires joints across the dropped chains."""
        reduced_links = [
            dict(link)
            for link in links
            if int(link["link_id"]) in retained_link_ids
        ]

        ancestor_memo: Dict[int, Optional[int]] = {}
        reduced_joints: List[Dict[str, Any]] = []
        for joint in joints:
            child_link_id = int(joint["child_link_id"])
            if child_link_id not in retained_link_ids:
                continue

            surviving_parent_link_id = self._nearest_retained_ancestor_link_id(
                int(joint["parent_link_id"]),
                retained_link_ids,
                parent_by_child,
                ancestor_memo,
            )
            if surviving_parent_link_id is None:
                continue

            reduced_joint = dict(joint)
            reduced_joint["joint_id"] = len(reduced_joints)
            reduced_joint["parent_link_id"] = surviving_parent_link_id
            reduced_joint["child_link_id"] = child_link_id
            reduced_joints.append(reduced_joint)

        return reduced_links, reduced_joints

    def _sample_retained_link_ids_after_dropout(
        self,
        all_link_ids: Set[int],
        root_link_ids: Set[int],
    ) -> Set[int]:
        """Selects retained links after optional dropout and the max-link cap.

        When a rooted kinematic tree has non-root links, at least one non-root
        link is always retained so the returned segmentation target remains
        non-trivial.
        """
        retained_link_ids = set(int(link_id) for link_id in root_link_ids)
        non_root_link_ids = np.asarray(
            sorted(all_link_ids.difference(root_link_ids)),
            dtype=np.int64,
        )
        if non_root_link_ids.size == 0:
            return retained_link_ids

        num_root_links = len(root_link_ids)
        if num_root_links == 1 and self.max_num_links < 2:
            raise ValueError(
                "max_num_links must be at least 2 for articulated samples so "
                "individual link dropout does not collapse them to the base link only, "
                f"got max_num_links={self.max_num_links}"
            )

        if self.dropout_individual_link_prob <= 0.0:
            retained_link_ids.update(non_root_link_ids.tolist())
        else:
            keep_mask = np.random.uniform(0.0, 1.0, size=non_root_link_ids.size) >= (
                self.dropout_individual_link_prob
            )
            retained_non_root_link_ids = non_root_link_ids[keep_mask]
            if retained_non_root_link_ids.size == 0:
                retained_non_root_link_ids = np.random.choice(
                    non_root_link_ids,
                    size=1,
                    replace=False,
                ).astype(np.int64, copy=False)
            retained_link_ids.update(retained_non_root_link_ids.tolist())
        if len(retained_link_ids) <= self.max_num_links:
            return retained_link_ids

        if num_root_links > self.max_num_links:
            raise ValueError(
                "max_num_links cannot be smaller than the number of root links in a sample, "
                f"got max_num_links={self.max_num_links} with {num_root_links} root links"
            )

        retained_non_root_link_ids = np.asarray(
            sorted(retained_link_ids.difference(root_link_ids)),
            dtype=np.int64,
        )
        num_non_root_links_to_keep = self.max_num_links - num_root_links
        if num_non_root_links_to_keep <= 0:
            return retained_link_ids.intersection(root_link_ids)
        if retained_non_root_link_ids.size == num_non_root_links_to_keep:
            return retained_link_ids

        sampled_non_root_link_ids = np.random.choice(
            retained_non_root_link_ids,
            size=num_non_root_links_to_keep,
            replace=False,
        ).astype(np.int64, copy=False)
        retained_link_ids = set(int(link_id) for link_id in root_link_ids)
        retained_link_ids.update(sampled_non_root_link_ids.tolist())
        return retained_link_ids

    def _sample_retained_link_ids_without_kinematic_tree(
        self,
        all_link_ids: Set[int],
    ) -> Set[int]:
        """Caps links uniformly when no kinematic tree is available."""
        sorted_link_ids = np.asarray(sorted(all_link_ids), dtype=np.int64)
        if sorted_link_ids.size <= self.max_num_links:
            return set(int(link_id) for link_id in sorted_link_ids.tolist())

        sampled_link_ids = np.random.choice(
            sorted_link_ids,
            size=self.max_num_links,
            replace=False,
        ).astype(np.int64, copy=False)
        return set(int(link_id) for link_id in sampled_link_ids.tolist())

    def _apply_link_cap_without_kinematic_tree(
        self,
        link_ids: np.ndarray,
        links: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Caps links without merging when kinematic metadata is unavailable.

        Dropped links cannot merge upward without a tree, so their point labels
        are reassigned to `-1`. Those points remain available for shape-point
        sampling but are excluded from query-point sampling.
        """
        link_ids = np.asarray(link_ids, dtype=np.int64)
        if len(links) <= self.max_num_links:
            return link_ids, links, []

        all_link_ids = {int(link["link_id"]) for link in links}
        if not all_link_ids:
            raise ValueError("Expected at least one link record in cached sample metadata")

        retained_link_ids = self._sample_retained_link_ids_without_kinematic_tree(
            all_link_ids
        )
        if len(retained_link_ids) == len(all_link_ids):
            return link_ids, links, []

        capped_link_ids = link_ids.copy()
        valid_mask = capped_link_ids >= 0
        if np.any(valid_mask):
            kept_mask = np.isin(
                capped_link_ids[valid_mask],
                np.asarray(sorted(retained_link_ids), dtype=np.int64),
                assume_unique=False,
            )
            capped_valid_link_ids = capped_link_ids[valid_mask]
            capped_valid_link_ids[~kept_mask] = -1
            capped_link_ids[valid_mask] = capped_valid_link_ids

        reduced_links = [
            dict(link)
            for link in links
            if int(link["link_id"]) in retained_link_ids
        ]
        return capped_link_ids, reduced_links, []

    def _apply_individual_link_dropout(
        self,
        link_ids: np.ndarray,
        links: List[Dict[str, Any]],
        joints: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Drops or caps links and updates labels/metadata consistently.

        Geometry stays unchanged here. Only per-point link labels and the
        returned link/joint metadata are updated. The retained links are capped
        at `max_num_links` regardless of whether stochastic dropout is enabled.
        When no kinematic tree is available, stochastic dropout is skipped and
        capped-out link labels are reassigned to `-1`.
        """
        link_ids = np.asarray(link_ids, dtype=np.int64)
        if len(links) <= 1:
            return link_ids, links, joints
        if len(joints) == 0:
            return self._apply_link_cap_without_kinematic_tree(link_ids, links)

        all_link_ids = {int(link["link_id"]) for link in links}
        if not all_link_ids:
            raise ValueError("Expected at least one link record in cached sample metadata")

        parent_by_child = self._build_parent_by_child_lookup(joints)
        # Root links are always retained so every dropped link can merge upward.
        root_link_ids = all_link_ids.difference(parent_by_child.keys())
        if not root_link_ids:
            raise ValueError("Could not determine a root link for link dropout")

        retained_link_ids = self._sample_retained_link_ids_after_dropout(
            all_link_ids=all_link_ids,
            root_link_ids=root_link_ids,
        )
        if len(retained_link_ids) == len(all_link_ids):
            return link_ids, links, joints

        ancestor_memo: Dict[int, Optional[int]] = {}
        reassigned_link_ids = link_ids.copy()
        valid_mask = reassigned_link_ids >= 0
        if np.any(valid_mask):
            unique_valid_link_ids, inverse = np.unique(
                reassigned_link_ids[valid_mask],
                return_inverse=True,
            )
            remapped_unique_link_ids = unique_valid_link_ids.copy()
            for unique_idx, original_link_id in enumerate(unique_valid_link_ids.tolist()):
                surviving_link_id = self._nearest_retained_ancestor_link_id(
                    int(original_link_id),
                    retained_link_ids,
                    parent_by_child,
                    ancestor_memo,
                )
                if surviving_link_id is None:
                    raise ValueError(
                        "Dropped link has no retained ancestor during dataset dropout: "
                        f"link_id={original_link_id}"
                    )
                remapped_unique_link_ids[unique_idx] = surviving_link_id
            reassigned_link_ids[valid_mask] = remapped_unique_link_ids[inverse]

        reduced_links, reduced_joints = self._reduce_kinematic_metadata_to_retained_links(
            links=links,
            joints=joints,
            retained_link_ids=retained_link_ids,
            parent_by_child=parent_by_child,
        )
        return reassigned_link_ids, reduced_links, reduced_joints

    def _remap_sample_link_ids_to_dense_range(
        self,
        link_ids: np.ndarray,
        links: List[Dict[str, Any]],
        joints: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Remaps per-sample link IDs to a dense `0..N-1` range.

        Cached samples may store only a subset of the original links while
        keeping their original link IDs. The model and loss code assume that
        per-sample link IDs are dense, so remap the point labels plus the link
        and joint metadata into a contiguous range on load.
        """
        sorted_link_ids = sorted({int(link["link_id"]) for link in links})
        link_id_remap = {
            original_link_id: remapped_link_id
            for remapped_link_id, original_link_id in enumerate(sorted_link_ids)
        }
        if not link_id_remap:
            raise ValueError("Expected at least one link record in cached sample metadata")

        link_ids = np.asarray(link_ids, dtype=np.int64)
        point_link_ids = set(np.unique(link_ids[link_ids >= 0]).tolist())
        missing_link_records = sorted(point_link_ids - set(link_id_remap))
        if missing_link_records:
            raise ValueError(
                "Point labels reference missing link records: "
                f"{missing_link_records}"
            )

        remapped_link_ids = link_ids.copy()
        valid_link_mask = remapped_link_ids >= 0
        if np.any(valid_link_mask):
            remapped_link_ids[valid_link_mask] = np.asarray(
                [
                    link_id_remap[int(original_link_id)]
                    for original_link_id in remapped_link_ids[valid_link_mask]
                ],
                dtype=np.int64,
            )

        remapped_links: List[Dict[str, Any]] = []
        for link in links:
            remapped_link = dict(link)
            remapped_link["link_id"] = link_id_remap[int(link["link_id"])]
            remapped_links.append(remapped_link)

        remapped_joints: List[Dict[str, Any]] = []
        for joint in joints:
            parent_link_id = int(joint["parent_link_id"])
            child_link_id = int(joint["child_link_id"])
            if parent_link_id not in link_id_remap or child_link_id not in link_id_remap:
                raise ValueError(
                    "Joint references missing link IDs after remapping: "
                    f"parent={parent_link_id}, child={child_link_id}"
                )
            remapped_joint = dict(joint)
            remapped_joint["parent_link_id"] = link_id_remap[parent_link_id]
            remapped_joint["child_link_id"] = link_id_remap[child_link_id]
            remapped_joints.append(remapped_joint)

        return remapped_link_ids, remapped_links, remapped_joints

    def _copy_joints(self, joints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Copies joint records and converts numeric fields to arrays."""
        copied_joints: List[Dict[str, Any]] = []
        for joint in sorted(joints, key=lambda item: int(item["joint_id"])):
            joint_copy = dict(joint)
            for field in _JOINT_NUMERIC_FIELDS:
                if joint_copy.get(field) is not None:
                    joint_copy[field] = np.asarray(joint_copy[field], dtype=np.float32).copy()
            copied_joints.append(joint_copy)
        return copied_joints

    def _get_link_name_variants(self, link: Dict[str, Any]) -> Tuple[str, ...]:
        """Returns the allowed text-prompt variants for one link.

        A link name may be stored as either a single string or a list of
        strings. Missing names fall back to a single empty-string prompt.
        """
        raw_name = link.get("name", "")
        link_id = int(link["link_id"])
        if isinstance(raw_name, str):
            return (raw_name,)
        if isinstance(raw_name, list):
            if not raw_name:
                raise ValueError(f"Expected at least one name variant for link_id={link_id}")
            if not all(isinstance(name, str) for name in raw_name):
                raise TypeError(
                    f"Expected every name variant for link_id={link_id} to be a string"
                )
            return tuple(raw_name)
        raise TypeError(
            "Expected link 'name' to be a string or a list of strings, "
            f"got {type(raw_name).__name__} for link_id={link_id}"
        )

    def _get_link_embedding_variants(
        self,
        link: Dict[str, Any],
        *,
        name_variants: Tuple[str, ...],
    ) -> np.ndarray:
        """Returns cached link text embeddings as a `(num_variants, dim)` array."""
        embedding = link.get("embedding")
        if embedding is None:
            raise ValueError(
                "Cached text embeddings are required for every link, "
                f"but link_id={int(link['link_id'])} is missing 'embedding'"
            )

        embedding_array = np.asarray(embedding, dtype=np.float32)
        link_id = int(link["link_id"])
        if len(name_variants) == 1:
            if embedding_array.ndim != 1:
                raise ValueError(
                    "Expected cached embedding for single-name link to be 1D, "
                    f"got shape {embedding_array.shape} for link_id={link_id}"
                )
            return embedding_array[None, :]

        if embedding_array.ndim != 2 or embedding_array.shape[0] != len(name_variants):
            raise ValueError(
                "Expected cached embedding for multi-name link to have shape "
                f"({len(name_variants)}, D), got {embedding_array.shape} for link_id={link_id}"
            )
        return embedding_array

    def _build_link_point_prompt_dropout_eligibility_by_link_id(
        self,
        links: List[Dict[str, Any]],
    ) -> Dict[int, bool]:
        """Marks links whose raw text prompt variants are unique in the sample."""
        link_name_counts: Dict[Tuple[str, ...], int] = {}
        for link in links:
            link_name_signature = self._get_link_name_variants(link)
            link_name_counts[link_name_signature] = (
                link_name_counts.get(link_name_signature, 0) + 1
            )
        return {
            int(link["link_id"]): link_name_counts[self._get_link_name_variants(link)] == 1
            for link in links
        }

    def _annotate_links_with_prompt_metadata(
        self,
        links: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Copies links and attaches dataset-side prompt metadata."""
        eligibility_by_link_id = self._build_link_point_prompt_dropout_eligibility_by_link_id(
            links
        )
        return [
            {
                **link,
                "point_prompt_source_link_id": int(link["link_id"]),
                "link_point_prompt_dropout_eligible": (
                    eligibility_by_link_id[int(link["link_id"])]
                ),
            }
            for link in links
        ]

    def _sample_link_text_condition_fields(
        self,
        sorted_links: List[Dict[str, Any]],
    ) -> Tuple[List[str], torch.Tensor]:
        """Samples text conditions while reusing draws for identical name signatures."""
        sampled_prompts: List[str] = []
        sampled_embeddings: List[np.ndarray] = []
        sampled_variant_index_by_signature: Dict[Tuple[str, ...], int] = {}
        for link in sorted_links:
            name_variants = self._get_link_name_variants(link)
            embedding_variants = self._get_link_embedding_variants(
                link,
                name_variants=name_variants,
            )
            sampled_variant_index = sampled_variant_index_by_signature.get(name_variants)
            if sampled_variant_index is None:
                sampled_variant_index = int(np.random.randint(len(name_variants)))
                sampled_variant_index_by_signature[name_variants] = sampled_variant_index
            sampled_prompts.append(name_variants[sampled_variant_index])
            sampled_embeddings.append(embedding_variants[sampled_variant_index])
        return (
            sampled_prompts,
            torch.from_numpy(np.stack(sampled_embeddings, axis=0)).float(),
        )

    def _parse_links(self, links: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Converts raw link metadata into dataset output fields.

        Args:
            links: Decoded list of link records.

        Returns:
            Dictionary containing sorted link records, sampled prompt strings,
            and cached embeddings aligned with those sampled prompts.
        """
        sorted_links = sorted(links, key=lambda link: int(link["link_id"]))
        link_text_prompts, link_text_embeddings = self._sample_link_text_condition_fields(
            sorted_links
        )
        link_point_prompt_dropout_eligible = [
            bool(link.get("link_point_prompt_dropout_eligible", True))
            for link in sorted_links
        ]

        return {
            "sorted_links": sorted_links,
            "link_text_prompts": link_text_prompts,
            "link_text_embeddings": link_text_embeddings,
            "link_point_prompt_dropout_eligible": torch.tensor(
                link_point_prompt_dropout_eligible,
                dtype=torch.bool,
            ),
        }

    def _sample_link_point_prompts_and_normals(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        point_prompt_source_link_ids: np.ndarray,
        sorted_links: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples one point prompt and normal for each link.

        Args:
            points: Full point cloud for the sample.
            normals: Per-point normals for the full point cloud.
            point_prompt_source_link_ids: Per-point link IDs from before any
                link-dropout merging, used to keep prompt sampling on each
                retained link's original geometry.
            sorted_links: Link metadata sorted by `link_id`.

        Returns:
            Two tensors of shape `(num_links, 3)` containing one sampled point
            and its aligned normal for each link.
        """
        link_point_prompts = np.zeros((len(sorted_links), 3), dtype=np.float32)
        link_point_prompt_normals = np.zeros((len(sorted_links), 3), dtype=np.float32)
        for link_idx, link in enumerate(sorted_links):
            prompt_source_link_id = int(
                link.get("point_prompt_source_link_id", link["link_id"])
            )
            candidate_indices = np.flatnonzero(
                point_prompt_source_link_ids == prompt_source_link_id
            )
            if len(candidate_indices) == 0:
                raise ValueError(
                    "No point-prompt candidates found for retained link "
                    f"link_id={int(link['link_id'])} with "
                    f"point_prompt_source_link_id={prompt_source_link_id}"
                )
            sampled_index = _sample_indices_with_replacement_when_needed(
                candidate_indices,
                size=1,
            )[0]
            link_point_prompts[link_idx] = points[sampled_index]
            link_point_prompt_normals[link_idx] = normals[sampled_index]
        return (
            torch.from_numpy(link_point_prompts).float(),
            torch.from_numpy(link_point_prompt_normals).float(),
        )

    def _augment_joints(
        self,
        joints: List[Dict[str, Any]],
        rotation_z: float,
        scale: float,
        translation: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Applies point-space augmentation to joint records."""
        augmented_joints = self._copy_joints(joints)
        for joint in augmented_joints:
            if joint.get("revolute_axis") is not None:
                joint["revolute_axis"] = self._augment_joint_axis(
                    np.asarray(joint["revolute_axis"], dtype=np.float32),
                    rotation_z=rotation_z,
                    scale=scale,
                    translation=translation,
                )
            if joint.get("prismatic_axis") is not None:
                joint["prismatic_axis"] = self._augment_joint_axis(
                    np.asarray(joint["prismatic_axis"], dtype=np.float32),
                    rotation_z=rotation_z,
                    scale=scale,
                    translation=translation,
                )
            if joint.get("prismatic_range") is not None:
                joint["prismatic_range"] = (
                    np.asarray(joint["prismatic_range"], dtype=np.float32) * scale
                )
        return augmented_joints

    def _normalize_points_and_joints_to_unit_extent(
        self,
        points: np.ndarray,
        joints: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Centers points at their AABB midpoint and scales max extent to one."""
        normalized_points, center, normalization_scale = normalize_points_to_unit_extent(
            points
        )
        normalized_joints = self._augment_joints(
            joints,
            rotation_z=0.0,
            scale=normalization_scale,
            translation=(-center * np.float32(normalization_scale)).astype(np.float32),
        )
        return normalized_points, normalized_joints

    def _compute_link_point_means(
        self,
        points: np.ndarray,
        link_ids: np.ndarray,
        *,
        num_links: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes per-link point means for valid link IDs."""
        if num_links < 0:
            raise ValueError(f"num_links must be non-negative, got {num_links}")

        link_point_means = np.zeros((num_links, 3), dtype=np.float32)
        link_has_points = np.zeros((num_links,), dtype=np.bool_)
        if num_links == 0:
            return link_point_means, link_has_points

        link_ids = np.asarray(link_ids, dtype=np.int64)
        valid_mask = link_ids >= 0
        if not np.any(valid_mask):
            return link_point_means, link_has_points

        valid_link_ids = link_ids[valid_mask]
        valid_points = np.asarray(points[valid_mask], dtype=np.float32)
        counts = np.bincount(valid_link_ids, minlength=num_links)
        link_has_points = counts > 0
        if not np.any(link_has_points):
            return link_point_means, link_has_points

        point_sums = np.stack(
            [
                np.bincount(valid_link_ids, weights=valid_points[:, dim], minlength=num_links)
                for dim in range(valid_points.shape[1])
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        link_point_means[link_has_points] = (
            point_sums[link_has_points]
            / counts[link_has_points, None].astype(np.float32, copy=False)
        )
        return link_point_means, link_has_points

    def _compute_link_point_scales(
        self,
        points: np.ndarray,
        link_ids: np.ndarray,
        *,
        num_links: int,
        link_point_means: np.ndarray | None = None,
        link_has_points: np.ndarray | None = None,
    ) -> np.ndarray:
        """Computes per-link RMS point radii around each link mean."""
        if num_links < 0:
            raise ValueError(f"num_links must be non-negative, got {num_links}")

        link_point_scales = np.zeros((num_links,), dtype=np.float32)
        if num_links == 0:
            return link_point_scales

        link_ids = np.asarray(link_ids, dtype=np.int64)
        valid_mask = link_ids >= 0
        if not np.any(valid_mask):
            return link_point_scales

        if link_point_means is None or link_has_points is None:
            link_point_means, link_has_points = self._compute_link_point_means(
                points=points,
                link_ids=link_ids,
                num_links=num_links,
            )

        valid_link_ids = link_ids[valid_mask]
        valid_points = np.asarray(points[valid_mask], dtype=np.float32)
        counts = np.bincount(valid_link_ids, minlength=num_links)
        point_offsets = valid_points - link_point_means[valid_link_ids]
        squared_radii = np.square(point_offsets).sum(axis=1, dtype=np.float32)
        squared_radius_sums = np.bincount(
            valid_link_ids,
            weights=squared_radii,
            minlength=num_links,
        ).astype(np.float32, copy=False)
        populated_link_mask = np.asarray(link_has_points, dtype=np.bool_) & (counts > 0)
        link_point_scales[populated_link_mask] = np.sqrt(
            squared_radius_sums[populated_link_mask]
            / counts[populated_link_mask].astype(np.float32, copy=False)
        )
        return link_point_scales

    def _recenter_joint_axes_to_child_link_means(
        self,
        points: np.ndarray,
        link_ids: np.ndarray,
        joints: List[Dict[str, Any]],
        *,
        num_links: int,
    ) -> List[Dict[str, Any]]:
        """Moves surviving prismatic axes to pass through their child-link means."""
        recentered_joints = self._copy_joints(joints)
        if len(recentered_joints) == 0 or num_links == 0:
            return recentered_joints

        link_point_means, link_has_points = self._compute_link_point_means(
            points=points,
            link_ids=link_ids,
            num_links=num_links,
        )
        for joint in recentered_joints:
            child_link_id = int(joint["child_link_id"])
            if (
                child_link_id < 0
                or child_link_id >= num_links
                or not bool(link_has_points[child_link_id])
            ):
                # This should be rare for retained joints, but skipping keeps
                # the loader robust if metadata and final labels diverge.
                continue

            child_link_mean = link_point_means[child_link_id]
            if joint.get("prismatic_axis") is not None:
                joint["prismatic_axis"] = axis_point_to_plucker(
                    np.asarray(joint["prismatic_axis"], dtype=np.float32)[:3],
                    child_link_mean,
                ).astype(np.float32, copy=False)
        return recentered_joints

    def _sample_random_pose_state(
        self,
        joints: List[Dict[str, Any]],
    ) -> Tuple[Dict[int, float], bool]:
        """Samples per-joint articulation states for on-the-fly posing."""
        articulation_state: Dict[int, float] = {}
        movable_child_link_ids = [
            int(joint["child_link_id"])
            for joint in joints
            if joint_is_movable(joint)
        ]

        if self.random_pose_prob <= 0.0 or not movable_child_link_ids:
            return articulation_state, False
        if np.random.uniform(0.0, 1.0) >= self.random_pose_prob:
            return articulation_state, False

        for child_link_id in movable_child_link_ids:
            if np.random.uniform(0.0, 1.0) < self.random_pose_prob:
                articulation_state[child_link_id] = float(np.random.uniform(0.0, 1.0))
        return articulation_state, bool(articulation_state)

    def _apply_random_pose(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        link_ids: np.ndarray,
        joints: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Applies random on-the-fly articulation in the cached coordinate frame."""
        articulation_state, should_pose = self._sample_random_pose_state(joints)
        if not should_pose:
            return (
                points,
                normals,
                self._copy_joints(joints),
            )

        posed_points, posed_joints, posed_normals = articulate_points_with_joints(
            xyz=points,
            part_ids=link_ids,
            joints=self._copy_joints(joints),
            articulation_state=articulation_state,
            normals=normals,
        )
        return (
            posed_points.astype(np.float32, copy=False),
            posed_normals.astype(np.float32, copy=False),
            posed_joints,
        )

    def _parse_joints(
        self,
        joints: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Converts joint records into tensor outputs."""
        sorted_joints = self._copy_joints(joints)
        is_revolute = np.zeros((len(sorted_joints),), dtype=np.bool_)
        is_prismatic = np.zeros((len(sorted_joints),), dtype=np.bool_)
        revolute_axis_supervision_valid = np.zeros((len(sorted_joints),), dtype=np.bool_)
        prismatic_axis_supervision_valid = np.zeros((len(sorted_joints),), dtype=np.bool_)
        revolute_range_supervision_valid = np.zeros((len(sorted_joints),), dtype=np.bool_)
        prismatic_range_supervision_valid = np.zeros((len(sorted_joints),), dtype=np.bool_)
        revolute_axis = np.zeros((len(sorted_joints), 6), dtype=np.float32)
        prismatic_axis = np.zeros((len(sorted_joints), 6), dtype=np.float32)
        revolute_range = np.zeros((len(sorted_joints), 2), dtype=np.float32)
        prismatic_range = np.zeros((len(sorted_joints), 2), dtype=np.float32)
        joint_connections = np.zeros((len(sorted_joints), 2), dtype=np.int64)

        for joint_idx, joint in enumerate(sorted_joints):
            parent_link_id = int(joint["parent_link_id"])
            child_link_id = int(joint["child_link_id"])
            is_revolute[joint_idx] = bool(joint.get("is_revolute", False))
            is_prismatic[joint_idx] = bool(joint.get("is_prismatic", False))
            if joint.get("revolute_axis") is not None:
                revolute_axis_supervision_valid[joint_idx] = True
                revolute_axis[joint_idx] = np.asarray(joint["revolute_axis"], dtype=np.float32)
            if joint.get("prismatic_axis") is not None:
                prismatic_axis_supervision_valid[joint_idx] = True
                prismatic_axis[joint_idx] = np.asarray(joint["prismatic_axis"], dtype=np.float32)
            if joint.get("revolute_range") is not None:
                revolute_range_supervision_valid[joint_idx] = True
                revolute_range[joint_idx] = np.asarray(joint["revolute_range"], dtype=np.float32)
            if joint.get("prismatic_range") is not None:
                prismatic_range_supervision_valid[joint_idx] = True
                prismatic_range[joint_idx] = np.asarray(joint["prismatic_range"], dtype=np.float32)
            joint_connections[joint_idx] = np.array(
                [parent_link_id, child_link_id],
                dtype=np.int64,
            )

        return {
            "is_revolute": torch.from_numpy(is_revolute),
            "is_prismatic": torch.from_numpy(is_prismatic),
            "revolute_axis_supervision_valid": torch.from_numpy(
                revolute_axis_supervision_valid
            ),
            "prismatic_axis_supervision_valid": torch.from_numpy(
                prismatic_axis_supervision_valid
            ),
            "revolute_range_supervision_valid": torch.from_numpy(
                revolute_range_supervision_valid
            ),
            "prismatic_range_supervision_valid": torch.from_numpy(
                prismatic_range_supervision_valid
            ),
            "revolute_axis": torch.from_numpy(revolute_axis).float(),
            "prismatic_axis": torch.from_numpy(prismatic_axis).float(),
            "revolute_range": torch.from_numpy(revolute_range).float(),
            "prismatic_range": torch.from_numpy(prismatic_range).float(),
            "joint_connections": torch.from_numpy(joint_connections).long(),
        }

    def _build_motion_overparam_targets(
        self,
        query_points: np.ndarray,
        query_link_ids: np.ndarray,
        joints: List[Dict[str, Any]],
        motion_type: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Builds per-query over-parameterized supervision for one motion family."""
        if motion_type not in {"revolute", "prismatic"}:
            raise ValueError(f"motion_type must be 'revolute' or 'prismatic', got {motion_type!r}")

        query_points = np.asarray(query_points, dtype=np.float32)
        query_link_ids = np.asarray(query_link_ids, dtype=np.int64)
        closest_axis_points = np.zeros((len(query_points), 3), dtype=np.float32)
        low_point_targets = np.zeros((len(query_points), 3), dtype=np.float32)
        high_point_targets = np.zeros((len(query_points), 3), dtype=np.float32)
        axis_direction_targets = np.zeros((len(query_points), 3), dtype=np.float32)
        if len(query_points) == 0 or len(joints) == 0:
            return (
                torch.from_numpy(closest_axis_points).float(),
                torch.from_numpy(low_point_targets).float(),
                torch.from_numpy(high_point_targets).float(),
                torch.from_numpy(axis_direction_targets).float(),
            )

        joint_by_child = {
            int(joint["child_link_id"]): joint
            for joint in joints
        }

        for child_link_id in np.unique(query_link_ids):
            joint = joint_by_child.get(int(child_link_id))
            if joint is None:
                continue

            link_mask = query_link_ids == child_link_id
            if not np.any(link_mask):
                continue

            if motion_type == "revolute":
                if not joint_has_revolute_motion(joint):
                    continue
                joint_axis = np.asarray(joint["revolute_axis"], dtype=np.float32)
                has_range_supervision = joint_has_revolute_range(joint)
            else:
                if not joint_has_prismatic_motion(joint):
                    continue
                joint_axis = np.asarray(joint["prismatic_axis"], dtype=np.float32)
                has_range_supervision = joint_has_prismatic_range(joint)

            link_query_points = query_points[link_mask]
            axis_direction, _ = plucker_to_axis_point(joint_axis)
            closest_points = self._closest_points_on_joint_axis(
                link_query_points,
                joint_axis,
            )
            closest_axis_points[link_mask] = closest_points
            axis_direction_targets[link_mask] = axis_direction.astype(np.float32, copy=False)
            if not has_range_supervision:
                continue

            link_part_ids = np.full(link_query_points.shape[0], child_link_id, dtype=np.int64)
            isolated_joints = self._copy_joints_with_isolated_motion(
                joints=joints,
                child_link_id=int(child_link_id),
                motion_type=motion_type,
            )
            articulated_low_points, _ = articulate_points_with_joints(
                xyz=link_query_points,
                part_ids=link_part_ids,
                joints=isolated_joints,
                articulation_state={int(child_link_id): 0.0},
            )
            articulated_high_points, _ = articulate_points_with_joints(
                xyz=link_query_points,
                part_ids=link_part_ids,
                joints=isolated_joints,
                articulation_state={int(child_link_id): 1.0},
            )
            low_point_targets[link_mask] = articulated_low_points
            high_point_targets[link_mask] = articulated_high_points

        return (
            torch.from_numpy(closest_axis_points).float(),
            torch.from_numpy(low_point_targets).float(),
            torch.from_numpy(high_point_targets).float(),
            torch.from_numpy(axis_direction_targets).float(),
        )

    def _joint_records_from_sample(
        self,
        sample: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Builds lightweight joint records from a returned sample.

        This is used by the module-level visualization/debug entrypoint, where
        the sample tensors are already aligned with any on-the-fly dropout and
        dense remapping.
        """
        is_revolute = sample["is_revolute"].detach().cpu().numpy()
        is_prismatic = sample["is_prismatic"].detach().cpu().numpy()
        supervision_valid_by_field = {
            field_name: (
                sample[field_name].detach().cpu().numpy()
                if field_name in sample
                else None
            )
            for field_name in _JOINT_SUPERVISION_VALID_FIELDS
        }
        revolute_axis = sample["revolute_axis"].detach().cpu().numpy()
        prismatic_axis = sample["prismatic_axis"].detach().cpu().numpy()
        revolute_range = sample["revolute_range"].detach().cpu().numpy()
        prismatic_range = sample["prismatic_range"].detach().cpu().numpy()
        joint_connections = sample["joint_connections"].detach().cpu().numpy()

        joint_records: List[Dict[str, Any]] = []
        for joint_idx in range(joint_connections.shape[0]):
            parent_link_id, child_link_id = joint_connections[joint_idx].astype(np.int64)
            joint_record = {
                "joint_id": joint_idx,
                "parent_link_id": int(parent_link_id),
                "child_link_id": int(child_link_id),
                "joint_description": "",
                "is_revolute": bool(is_revolute[joint_idx]),
                "is_prismatic": bool(is_prismatic[joint_idx]),
                "revolute_axis": (
                    revolute_axis[joint_idx].copy()
                    if bool(
                        supervision_valid_by_field["revolute_axis_supervision_valid"][joint_idx]
                        if supervision_valid_by_field["revolute_axis_supervision_valid"] is not None
                        else is_revolute[joint_idx]
                    )
                    else None
                ),
                "prismatic_axis": (
                    prismatic_axis[joint_idx].copy()
                    if bool(
                        supervision_valid_by_field["prismatic_axis_supervision_valid"][joint_idx]
                        if supervision_valid_by_field["prismatic_axis_supervision_valid"] is not None
                        else is_prismatic[joint_idx]
                    )
                    else None
                ),
                "revolute_range": (
                    revolute_range[joint_idx].copy()
                    if bool(
                        supervision_valid_by_field["revolute_range_supervision_valid"][joint_idx]
                        if supervision_valid_by_field["revolute_range_supervision_valid"] is not None
                        else is_revolute[joint_idx]
                    )
                    else None
                ),
                "prismatic_range": (
                    prismatic_range[joint_idx].copy()
                    if bool(
                        supervision_valid_by_field["prismatic_range_supervision_valid"][joint_idx]
                        if supervision_valid_by_field["prismatic_range_supervision_valid"] is not None
                        else is_prismatic[joint_idx]
                    )
                    else None
                ),
            }
            joint_records.append(joint_record)
        return joint_records

    def _prepare_processed_sample(
        self,
        raw_sample: Dict[str, Any],
        *,
        rotation_z: float,
        scale: float,
        translation: np.ndarray,
    ) -> Dict[str, Any]:
        """Applies geometry, kinematic, and supervision preprocessing."""
        all_points = raw_sample["points"]
        all_normals = raw_sample["normals"]
        all_link_ids = raw_sample["link_ids"]
        sharp_point_flag = raw_sample["sharp_point_flag"]
        links = self._annotate_links_with_prompt_metadata(
            raw_sample["links"]
        )
        joints = raw_sample["joints"]

        all_points, all_normals, joints = self._apply_random_pose(
            points=all_points,
            normals=all_normals,
            link_ids=all_link_ids,
            joints=joints,
        )
        all_points, joints = self._normalize_points_and_joints_to_unit_extent(
            all_points,
            joints,
        )

        if self.use_augmentation:
            all_points = self._augment_points(
                all_points,
                rotation_z=rotation_z,
                scale=scale,
                translation=translation,
            )
            all_normals = self._rotate_points(all_normals, rotation_z)
            joints = self._augment_joints(
                joints,
                rotation_z=rotation_z,
                scale=scale,
                translation=translation,
            )

        # Preserve the original per-point link ownership so retained links keep
        # sampling point prompts from their own pre-merge geometry.
        point_prompt_source_link_ids = np.asarray(all_link_ids, dtype=np.int64).copy()

        # Keep geometry augmentation on the full tree. Link dropout only
        # rewrites labels/metadata for the returned supervision targets.
        all_link_ids, links, joints = self._apply_individual_link_dropout(
            all_link_ids,
            links,
            joints,
        )
        all_link_ids, links, joints = self._remap_sample_link_ids_to_dense_range(
            all_link_ids,
            links,
            joints,
        )
        joints = self._recenter_joint_axes_to_child_link_means(
            points=all_points,
            link_ids=all_link_ids,
            joints=joints,
            num_links=len(links),
        )
        return {
            "points": all_points,
            "normals": all_normals,
            "link_ids": all_link_ids,
            "point_prompt_source_link_ids": point_prompt_source_link_ids,
            "sharp_point_flag": sharp_point_flag,
            "links": links,
            "joints": joints,
            "query_boundary_distance_link_ids": raw_sample.get(
                "query_boundary_distance_link_ids"
            ),
            "query_boundary_distance_to_raw_link": raw_sample.get(
                "query_boundary_distance_to_raw_link"
            ),
        }

    def _sample_shape_and_query_indices(
        self,
        *,
        processed_sample: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Samples shape-point and query-point indices for one processed sample."""
        shape_indices = sample_points(
            sharp_point_flag=processed_sample["sharp_point_flag"],
            num_points=self.num_shape_points,
            sharp_point_ratio=self.sharp_point_ratio,
        )
        link_scales = None
        if self.query_boundary_ratio > 0.0:
            query_boundary_distance_link_ids = processed_sample.get(
                "query_boundary_distance_link_ids"
            )
            query_boundary_distance_to_raw_link = processed_sample.get(
                "query_boundary_distance_to_raw_link"
            )
            if (
                query_boundary_distance_link_ids is None
                or query_boundary_distance_to_raw_link is None
            ):
                raise ValueError(
                    "query_boundary_ratio > 0 requires cached query boundary fields in the "
                    "dataset .npz files; run the boundary backfill script first"
                )
            link_point_means, link_has_points = self._compute_link_point_means(
                points=processed_sample["points"],
                link_ids=processed_sample["link_ids"],
                num_links=len(processed_sample["links"]),
            )
            link_scales = self._compute_link_point_scales(
                points=processed_sample["points"],
                link_ids=processed_sample["link_ids"],
                num_links=len(processed_sample["links"]),
                link_point_means=link_point_means,
                link_has_points=link_has_points,
            )
        query_indices = sample_query_points(
            link_ids=processed_sample["link_ids"],
            num_points=self.num_query_points,
            query_balance_ratio=self.query_balance_ratio,
            query_boundary_ratio=self.query_boundary_ratio,
            query_boundary_temperature=self.query_boundary_temperature,
            point_raw_link_ids=processed_sample["point_prompt_source_link_ids"],
            query_boundary_distance_link_ids=processed_sample.get(
                "query_boundary_distance_link_ids"
            ),
            query_boundary_distance_to_raw_link=processed_sample.get(
                "query_boundary_distance_to_raw_link"
            ),
            link_scales=link_scales,
        )
        return shape_indices, query_indices

    def _build_sample_from_processed_state(
        self,
        processed_sample: Dict[str, Any],
        *,
        shape_indices: np.ndarray,
        query_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """Converts one processed sample state into returned tensors."""
        all_points = processed_sample["points"]
        all_normals = processed_sample["normals"]
        all_link_ids = processed_sample["link_ids"]
        point_prompt_source_link_ids = processed_sample["point_prompt_source_link_ids"]
        links = processed_sample["links"]
        joints = processed_sample["joints"]

        parsed_links = self._parse_links(links)
        parsed_joints = self._parse_joints(joints)
        link_point_means, link_has_points = self._compute_link_point_means(
            points=all_points,
            link_ids=all_link_ids,
            num_links=len(parsed_links["sorted_links"]),
        )
        link_scales = self._compute_link_point_scales(
            points=all_points,
            link_ids=all_link_ids,
            num_links=len(parsed_links["sorted_links"]),
            link_point_means=link_point_means,
            link_has_points=link_has_points,
        )
        link_point_prompts, link_point_prompt_normals = (
            self._sample_link_point_prompts_and_normals(
                points=all_points,
                normals=all_normals,
                point_prompt_source_link_ids=point_prompt_source_link_ids,
                sorted_links=parsed_links["sorted_links"],
            )
        )
        query_link_ids = all_link_ids[query_indices]
        sample = {
            "shape_points": torch.from_numpy(all_points[shape_indices]).float(),
            "shape_point_normals": torch.from_numpy(all_normals[shape_indices]).float(),
            "query_points": torch.from_numpy(all_points[query_indices]).float(),
            "query_point_normals": torch.from_numpy(all_normals[query_indices]).float(),
            "link_ids": torch.from_numpy(query_link_ids).long(),
            "link_scales": torch.from_numpy(link_scales).float(),
            "link_point_prompts": link_point_prompts,
            "link_point_prompt_normals": link_point_prompt_normals,
            "link_point_prompt_dropout_eligible": parsed_links["link_point_prompt_dropout_eligible"],
            "link_text_prompts": parsed_links["link_text_prompts"],
            "link_text_embeddings": parsed_links["link_text_embeddings"],
            "is_revolute": parsed_joints["is_revolute"],
            "is_prismatic": parsed_joints["is_prismatic"],
            "revolute_axis_supervision_valid": parsed_joints["revolute_axis_supervision_valid"],
            "prismatic_axis_supervision_valid": parsed_joints["prismatic_axis_supervision_valid"],
            "revolute_range_supervision_valid": parsed_joints["revolute_range_supervision_valid"],
            "prismatic_range_supervision_valid": parsed_joints["prismatic_range_supervision_valid"],
            "revolute_axis": parsed_joints["revolute_axis"],
            "prismatic_axis": parsed_joints["prismatic_axis"],
            "revolute_range": parsed_joints["revolute_range"],
            "prismatic_range": parsed_joints["prismatic_range"],
            "joint_connections": parsed_joints["joint_connections"],
        }
        (
            sample["revolute_closest_axis_points"],
            sample["revolute_low_points"],
            sample["revolute_high_points"],
            sample["revolute_axis_directions"],
        ) = self._build_motion_overparam_targets(
            query_points=all_points[query_indices],
            query_link_ids=query_link_ids,
            joints=joints,
            motion_type="revolute",
        )
        (
            sample["prismatic_closest_axis_points"],
            sample["prismatic_low_points"],
            sample["prismatic_high_points"],
            sample["prismatic_axis_directions"],
        ) = self._build_motion_overparam_targets(
            query_points=all_points[query_indices],
            query_link_ids=query_link_ids,
            joints=joints,
            motion_type="prismatic",
        )
        return sample

    def _getitem(self, idx: int) -> Dict[str, Any]:
        """Loads a single sample without retry logic.

        Args:
            idx: Dataset index.

        Returns:
            Sample dictionary containing sampled points, link annotations, and
            joint metadata.
        """
        data_path = self.data[idx]

        if self.use_augmentation:
            rotation_z, scale, translation = self._sample_augmentation_params()
        else:
            rotation_z, scale, translation = self._default_augmentation_params()

        raw_sample = self._load_raw_sample(data_path)
        processed_sample = self._prepare_processed_sample(
            raw_sample,
            rotation_z=rotation_z,
            scale=scale,
            translation=translation,
        )
        shape_indices, query_indices = self._sample_shape_and_query_indices(
            processed_sample=processed_sample,
        )
        return self._build_sample_from_processed_state(
            processed_sample,
            shape_indices=shape_indices,
            query_indices=query_indices,
        )


if __name__ == "__main__":
    from instruct_particulate.datasets.vis_helper import (
        build_axis_segments,
        plot_joint_overparam_visualizations,
        plot_query_visualization,
        sample_joint_overparam_visualization_subset,
        sample_query_visualization_subset,
    )

    dataset = InstructParticulateDataset(
        root=Path(__file__).parent.parent.parent / "urdf-test",
        num_shape_points=8192,
        num_query_points=8192,
        sharp_point_ratio=0.5,
        use_augmentation=True,
        random_translation_range=(-0.05, 0.05),
        random_scale_range=(0.9, 1.1),
        random_pose_prob=1.0,
        dropout_individual_link_prob=0.5,
        max_num_links=1024,
    )

    print("len(dataset):", len(dataset))
    data_path = dataset.data[0]
    first_item = dataset._getitem(0)
    print("first_item stats:")
    print("  shape_points:", first_item["shape_points"].shape, first_item["shape_points"].dtype)
    print(
        "  shape_point_normals:",
        first_item["shape_point_normals"].shape,
        first_item["shape_point_normals"].dtype,
    )
    print("  query_points:", first_item["query_points"].shape, first_item["query_points"].dtype)
    print(
        "  query_point_normals:",
        first_item["query_point_normals"].shape,
        first_item["query_point_normals"].dtype,
    )
    print("  link_ids:", first_item["link_ids"].shape, first_item["link_ids"].dtype)
    print(
        "  link_point_prompts:",
        first_item["link_point_prompts"].shape,
        first_item["link_point_prompts"].dtype,
    )
    print(
        "  link_point_prompt_normals:",
        first_item["link_point_prompt_normals"].shape,
        first_item["link_point_prompt_normals"].dtype,
    )
    print("  num_link_text_prompts:", len(first_item["link_text_prompts"]))
    print(
        "  link_text_embeddings:",
        (
            first_item["link_text_embeddings"].shape,
            first_item["link_text_embeddings"].dtype,
        ),
    )
    print("  is_revolute:", first_item["is_revolute"])
    print("  is_prismatic:", first_item["is_prismatic"])
    print("  revolute_axis:", first_item["revolute_axis"].shape, first_item["revolute_axis"].dtype)
    print("  prismatic_axis:", first_item["prismatic_axis"].shape, first_item["prismatic_axis"].dtype)
    print(
        "  revolute_range:",
        first_item["revolute_range"].shape,
        first_item["revolute_range"].dtype,
    )
    print(
        "  prismatic_range:",
        first_item["prismatic_range"].shape,
        first_item["prismatic_range"].dtype,
    )
    print(
        "  joint_connections:",
        first_item["joint_connections"].shape,
        first_item["joint_connections"].dtype,
    )
    for field_name in (
        "revolute_closest_axis_points",
        "revolute_low_points",
        "revolute_high_points",
        "prismatic_closest_axis_points",
        "prismatic_low_points",
        "prismatic_high_points",
    ):
        print(
            f"  {field_name}:",
            first_item[field_name].shape,
            first_item[field_name].dtype,
        )

    sorted_links = [
        {"link_id": link_id, "name": link_name}
        for link_id, link_name in enumerate(first_item["link_text_prompts"])
    ]
    link_point_prompt_ids = np.arange(len(sorted_links), dtype=np.int64)
    visualization_query_points, visualization_query_normals, visualization_query_link_ids = (
        sample_query_visualization_subset(
            query_points=first_item["query_points"].numpy(),
            query_normals=first_item["query_point_normals"].numpy(),
            query_link_ids=first_item["link_ids"].numpy(),
            num_points=8192,
        )
    )
    current_joints = dataset._joint_records_from_sample(first_item)
    articulation_joints = _disable_range_less_motion_for_articulation(current_joints)

    low_query_points = None
    high_query_points = None
    low_link_point_prompts = None
    high_link_point_prompts = None
    axis_segments = None
    if any(joint_is_movable(joint) for joint in current_joints):
        axis_segments = build_axis_segments(
            first_item["query_points"].numpy(),
            first_item["link_ids"].numpy(),
            current_joints,
        )
    if any(joint_is_movable(joint) for joint in articulation_joints):
        low_query_points = articulate_points_with_joints(
            xyz=visualization_query_points,
            part_ids=visualization_query_link_ids,
            joints=articulation_joints,
            articulation_state=0.0,
        )[0]
        high_query_points = articulate_points_with_joints(
            xyz=visualization_query_points,
            part_ids=visualization_query_link_ids,
            joints=articulation_joints,
            articulation_state=1.0,
        )[0]
        low_link_point_prompts = articulate_points_with_joints(
            xyz=first_item["link_point_prompts"].numpy(),
            part_ids=link_point_prompt_ids,
            joints=articulation_joints,
            articulation_state=0.0,
        )[0]
        high_link_point_prompts = articulate_points_with_joints(
            xyz=first_item["link_point_prompts"].numpy(),
            part_ids=link_point_prompt_ids,
            joints=articulation_joints,
            articulation_state=1.0,
        )[0]

    out_path = data_path.with_suffix(".query_visualization.png")
    plot_query_visualization(
        query_points=visualization_query_points,
        query_normals=visualization_query_normals,
        query_link_ids=visualization_query_link_ids,
        link_point_prompts=first_item["link_point_prompts"].numpy(),
        link_point_prompt_ids=link_point_prompt_ids,
        links=sorted_links,
        out_path=out_path,
        low_query_points=low_query_points,
        high_query_points=high_query_points,
        low_link_point_prompts=low_link_point_prompts,
        high_link_point_prompts=high_link_point_prompts,
        axis_segments=axis_segments,
        normal_limit=len(visualization_query_points),
    )
    print("  visualization:", out_path)

    query_link_ids = first_item["link_ids"].numpy()
    revolute_child_link_ids = np.array(
        [
            int(joint["child_link_id"])
            for joint in current_joints
            if joint_has_revolute_motion(joint)
        ],
        dtype=np.int64,
    )
    prismatic_child_link_ids = np.array(
        [
            int(joint["child_link_id"])
            for joint in current_joints
            if joint_has_prismatic_motion(joint)
        ],
        dtype=np.int64,
    )
    revolute_range_child_link_ids = np.array(
        [
            int(joint["child_link_id"])
            for joint in current_joints
            if joint_has_revolute_range(joint)
        ],
        dtype=np.int64,
    )
    prismatic_range_child_link_ids = np.array(
        [
            int(joint["child_link_id"])
            for joint in current_joints
            if joint_has_prismatic_range(joint)
        ],
        dtype=np.int64,
    )

    overparam_visualizations: List[Dict[str, Any]] = []

    def collect_overparam_visualization(
        motion_prefix: str,
        valid_link_ids: np.ndarray,
        range_valid_link_ids: np.ndarray,
        closest_axis_points: torch.Tensor,
        low_points: torch.Tensor,
        high_points: torch.Tensor,
    ) -> None:
        if len(valid_link_ids) == 0:
            print(f"  {motion_prefix}_overparam_visualization: skipped (no {motion_prefix} joints)")
            return

        motion_valid_mask = np.isin(query_link_ids, valid_link_ids)
        (
            sampled_query_points,
            sampled_query_link_ids,
            sampled_closest_axis_points,
            sampled_low_points,
            sampled_high_points,
        ) = sample_joint_overparam_visualization_subset(
            query_points=first_item["query_points"].numpy(),
            query_link_ids=query_link_ids,
            closest_axis_points=closest_axis_points.numpy(),
            low_points=low_points.numpy(),
            high_points=high_points.numpy(),
            valid_mask=motion_valid_mask,
            num_points=256,
        )
        if len(sampled_query_points) == 0:
            print(f"  {motion_prefix}_overparam_visualization: skipped (no valid sampled points)")
            return

        range_valid_mask = np.isin(sampled_query_link_ids, range_valid_link_ids)
        overparam_visualizations.append(
            {
                "motion_label": motion_prefix.title(),
                "query_points": sampled_query_points,
                "query_link_ids": sampled_query_link_ids,
                "closest_axis_points": sampled_closest_axis_points,
                "closest_axis_points_valid": np.ones(
                    (len(sampled_query_points),),
                    dtype=np.bool_,
                ),
                "low_points": sampled_low_points,
                "high_points": sampled_high_points,
                "low_points_valid": range_valid_mask,
                "high_points_valid": range_valid_mask,
                "context_query_points": first_item["query_points"].numpy()[~motion_valid_mask],
            }
        )

    collect_overparam_visualization(
        "revolute",
        revolute_child_link_ids,
        revolute_range_child_link_ids,
        first_item["revolute_closest_axis_points"],
        first_item["revolute_low_points"],
        first_item["revolute_high_points"],
    )
    collect_overparam_visualization(
        "prismatic",
        prismatic_child_link_ids,
        prismatic_range_child_link_ids,
        first_item["prismatic_closest_axis_points"],
        first_item["prismatic_low_points"],
        first_item["prismatic_high_points"],
    )
    if len(overparam_visualizations) == 0:
        print("  joint_overparam_visualization: skipped (no valid motion-specific sampled points)")
    else:
        motion_out_path = data_path.with_suffix(".joint_overparam_visualization.png")
        plot_joint_overparam_visualizations(
            visualizations=overparam_visualizations,
            links=sorted_links,
            out_path=motion_out_path,
        )
        print("  joint_overparam_visualization:", motion_out_path)
