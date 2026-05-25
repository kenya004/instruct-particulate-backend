"""Additional evaluation metrics beyond the base particulate protocol."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np

from instruct_particulate.utils.articulation_utils import plucker_to_axis_point
from instruct_particulate.utils.evaluation_utils import build_part_correspondence, compute_giou

JsonDict = dict[str, Any]


def extract_category(sample_name: str) -> str:
    """Extracts the Lightwheel-style category from one sample name.

    Args:
        sample_name: Object identifier such as ``CoffeeMachine039``.

    Returns:
        Category string with trailing digits removed.
    """
    return re.sub(r"\d+$", "", str(sample_name))


def _normalize_direction(direction: np.ndarray | Sequence[float] | None) -> np.ndarray | None:
    """Normalizes one direction vector."""
    if direction is None:
        return None
    direction = np.asarray(direction, dtype=np.float32).reshape(-1)
    if direction.shape[0] != 3:
        return None
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        return None
    return (direction / norm).astype(np.float32, copy=False)


def _direction_and_point_from_line_axis(
    axis_value: np.ndarray | Sequence[float] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Builds an axis direction and optional point from a stored joint axis.

    Args:
        axis_value: Either a 3D direction or a 6D Plucker line.

    Returns:
        ``(direction, point)`` where ``point`` is ``None`` when the stored axis
        does not include a line anchor.
    """
    if axis_value is None:
        return None, None
    axis_value = np.asarray(axis_value, dtype=np.float32).reshape(-1)
    if axis_value.shape[0] == 6:
        direction, point = plucker_to_axis_point(axis_value)
        return (
            _normalize_direction(direction),
            np.asarray(point, dtype=np.float32).reshape(3),
        )
    return _normalize_direction(axis_value), None


def _predicted_axis_direction_and_point(
    prediction: Mapping[str, Any],
    part_id: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Returns the predicted joint axis for one compact predicted part ID."""
    is_part_revolute = np.asarray(prediction["is_part_revolute"], dtype=np.bool_)
    is_part_prismatic = np.asarray(prediction["is_part_prismatic"], dtype=np.bool_)

    if bool(is_part_revolute[part_id]):
        return _direction_and_point_from_line_axis(prediction["revolute_plucker"][part_id])

    if bool(is_part_prismatic[part_id]):
        prismatic_plucker = prediction.get("prismatic_plucker")
        if prismatic_plucker is not None:
            direction, point = _direction_and_point_from_line_axis(prismatic_plucker[part_id])
            if direction is not None:
                return direction, point
        return _direction_and_point_from_line_axis(prediction.get("prismatic_axis", [])[part_id])

    return None, None


def _gt_axis_direction_and_point(
    joint: Mapping[str, Any] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Returns the ground-truth joint axis encoded in one joint record."""
    if joint is None:
        return None, None
    if joint.get("revolute_axis") is not None:
        return _direction_and_point_from_line_axis(joint.get("revolute_axis"))
    if joint.get("prismatic_axis") is not None:
        return _direction_and_point_from_line_axis(joint.get("prismatic_axis"))
    return None, None


def _joint_has_revolute_motion(joint: Mapping[str, Any] | None) -> bool:
    """Returns whether one joint record carries revolute motion."""
    if joint is None:
        return False
    return bool(joint.get("is_revolute", False) or joint.get("revolute_axis") is not None)


def _axis_angle_error_deg(
    predicted_direction: np.ndarray,
    gt_direction: np.ndarray,
) -> float:
    """Computes the unsigned angle between two axes in degrees."""
    dot_product = float(np.clip(np.abs(np.dot(predicted_direction, gt_direction)), 0.0, 1.0))
    return float(np.degrees(np.arccos(dot_product)))


def _line_position_error(
    predicted_point: np.ndarray,
    predicted_direction: np.ndarray,
    gt_point: np.ndarray,
    gt_direction: np.ndarray,
) -> float:
    """Computes the shortest distance between two 3D lines."""
    predicted_direction = _normalize_direction(predicted_direction)
    gt_direction = _normalize_direction(gt_direction)
    assert predicted_direction is not None
    assert gt_direction is not None

    connecting_vector = np.asarray(gt_point - predicted_point, dtype=np.float32)
    cross_direction = np.cross(predicted_direction, gt_direction)
    cross_norm = float(np.linalg.norm(cross_direction))
    if cross_norm <= 1e-8:
        return float(np.linalg.norm(np.cross(connecting_vector, predicted_direction)))
    return float(np.abs(np.dot(connecting_vector, cross_direction)) / cross_norm)


def compute_part_and_joint_metrics(
    *,
    prediction: Mapping[str, Any],
    gt_cache: Mapping[str, Any],
    input_num_parts: int | None,
    gt_joint_records: Sequence[Mapping[str, Any]] | None,
    matching_mode: str,
    hungarian_matching_cost_type: str,
) -> JsonDict:
    """Computes per-sample part coverage and matched-joint axis errors.

    Args:
        prediction: One prediction payload variant.
        gt_cache: Cached ground-truth arrays used by the particulate evaluator.
        input_num_parts: Number of parts in the original input kinematics.
        gt_joint_records: Ground-truth joint records aligned with `gt_cache`.
        matching_mode: Part correspondence strategy used by the main evaluator.
        hungarian_matching_cost_type: Part matching cost used by the evaluator.

    Returns:
        Dictionary containing per-sample counts, sums, and mean metrics.
    """
    predicted_segmented_part_ids = np.unique(
        np.asarray(prediction.get("face_part_ids", prediction["part_ids"]), dtype=np.int32)
    )
    predicted_part_ids = np.unique(np.asarray(prediction["part_ids"], dtype=np.int32))
    predicted_part_count = int(len(predicted_segmented_part_ids))
    input_part_count = None if input_num_parts is None else int(input_num_parts)
    predicted_part_coverage_pct = (
        None
        if input_part_count is None or input_part_count <= 0
        else float(100.0 * predicted_part_count / input_part_count)
    )

    if not gt_joint_records:
        return {
            "predicted_part_count": predicted_part_count,
            "input_part_count": input_part_count,
            "predicted_part_coverage_pct": predicted_part_coverage_pct,
            "matched_joint_count": 0,
            "joint_axis_angle_error_deg": None,
            "joint_axis_angle_error_sum_deg": 0.0,
            "joint_axis_angle_error_count": 0,
            "joint_axis_position_error": None,
            "joint_axis_position_error_sum": 0.0,
            "joint_axis_position_error_count": 0,
        }

    mapping_pred2gt = build_part_correspondence(
        points_src=np.asarray(prediction["points"], dtype=np.float32),
        part_ids_src=np.asarray(prediction["part_ids"], dtype=np.int32),
        points_tgt=np.asarray(gt_cache["points"], dtype=np.float32),
        part_ids_tgt=np.asarray(gt_cache["part_ids"], dtype=np.int32),
        matching_mode=matching_mode,
        hungarian_matching_cost_type=hungarian_matching_cost_type,
    )
    gt_part_ids = np.unique(np.asarray(gt_cache["part_ids"], dtype=np.int32))
    gt_joint_by_child_id = {
        int(joint["child_link_id"]): joint
        for joint in gt_joint_records
    }

    matched_joint_count = 0
    angle_error_sum_deg = 0.0
    angle_error_count = 0
    position_error_sum = 0.0
    position_error_count = 0

    for prediction_part_index, prediction_part_id in enumerate(predicted_part_ids.tolist()):
        gt_part_index = int(mapping_pred2gt[prediction_part_index, 0])
        if gt_part_index < 0:
            continue
        gt_part_id = int(gt_part_ids[gt_part_index])
        gt_joint = gt_joint_by_child_id.get(gt_part_id)
        if gt_joint is None:
            continue

        matched_joint_count += 1
        predicted_direction, predicted_point = _predicted_axis_direction_and_point(
            prediction,
            int(prediction_part_id),
        )
        gt_direction, gt_point = _gt_axis_direction_and_point(gt_joint)
        if predicted_direction is None or gt_direction is None:
            continue

        angle_error_sum_deg += _axis_angle_error_deg(predicted_direction, gt_direction)
        angle_error_count += 1

        # Joint-position error is only meaningful for revolute axes, including
        # continuous revolute joints that omit finite angle limits.
        if (
            _joint_has_revolute_motion(gt_joint)
            and predicted_point is not None
            and gt_point is not None
        ):
            position_error_sum += _line_position_error(
                predicted_point,
                predicted_direction,
                gt_point,
                gt_direction,
            )
            position_error_count += 1

    return {
        "predicted_part_count": predicted_part_count,
        "input_part_count": input_part_count,
        "predicted_part_coverage_pct": predicted_part_coverage_pct,
        "matched_joint_count": matched_joint_count,
        "joint_axis_angle_error_deg": (
            None if angle_error_count == 0 else float(angle_error_sum_deg / angle_error_count)
        ),
        "joint_axis_angle_error_sum_deg": float(angle_error_sum_deg),
        "joint_axis_angle_error_count": int(angle_error_count),
        "joint_axis_position_error": (
            None
            if position_error_count == 0
            else float(position_error_sum / position_error_count)
        ),
        "joint_axis_position_error_sum": float(position_error_sum),
        "joint_axis_position_error_count": int(position_error_count),
    }


def compute_part_matching_quality_metrics(
    *,
    prediction: Mapping[str, Any],
    gt_payload: Mapping[str, Any],
    matching_mode: str = "hungarian",
    giou_threshold: float = 0.0,
    hungarian_matching_cost_type: str = "cdist",
) -> JsonDict:
    """Computes part matching precision/recall using rest-pose part GIoU.

    A predicted part is a hit when the matched GT part has GIoU greater than or
    equal to ``giou_threshold``. Unmatched predicted and GT parts contribute to
    the precision and recall denominators through their total counts.
    """
    pred_points = np.asarray(prediction["points"], dtype=np.float32)
    pred_part_ids = np.asarray(prediction["part_ids"], dtype=np.int32)
    gt_points = np.asarray(gt_payload["points"], dtype=np.float32)
    gt_part_ids = np.asarray(gt_payload["part_ids"], dtype=np.int32)

    unique_pred_ids = np.unique(pred_part_ids)
    unique_gt_ids = np.unique(gt_part_ids)
    num_pred_parts = int(len(unique_pred_ids))
    num_gt_parts = int(len(unique_gt_ids))
    if num_pred_parts == 0 or num_gt_parts == 0:
        return {
            "part_match_precision": 0.0,
            "part_match_recall": 0.0,
            "part_match_f1": 0.0,
            "num_correctly_matched_parts": 0,
            "num_pred_parts_total": num_pred_parts,
            "num_gt_parts_total": num_gt_parts,
        }

    mapping_pred2gt = build_part_correspondence(
        points_src=pred_points,
        part_ids_src=pred_part_ids,
        points_tgt=gt_points,
        part_ids_tgt=gt_part_ids,
        matching_mode=matching_mode,
        hungarian_matching_cost_type=hungarian_matching_cost_type,
    )

    num_correctly_matched = 0
    for pred_index, pred_part_id in enumerate(unique_pred_ids.tolist()):
        gt_index = int(mapping_pred2gt[pred_index, 0])
        if gt_index < 0:
            continue
        gt_part_id = int(unique_gt_ids[gt_index])
        pred_pts = pred_points[pred_part_ids == pred_part_id]
        gt_pts = gt_points[gt_part_ids == gt_part_id]
        if len(pred_pts) == 0 or len(gt_pts) == 0:
            continue
        giou = float(compute_giou(pred_pts, gt_pts))
        if giou >= giou_threshold:
            num_correctly_matched += 1

    precision = num_correctly_matched / num_pred_parts if num_pred_parts > 0 else 0.0
    recall = num_correctly_matched / num_gt_parts if num_gt_parts > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    return {
        "part_match_precision": float(precision),
        "part_match_recall": float(recall),
        "part_match_f1": float(f1),
        "num_correctly_matched_parts": int(num_correctly_matched),
        "num_pred_parts_total": int(num_pred_parts),
        "num_gt_parts_total": int(num_gt_parts),
    }


def aggregate_part_and_joint_metrics(eval_results: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregates the additional part/matching/joint metrics across many samples.

    Args:
        eval_results: Per-sample evaluation results.

    Returns:
        Weighted aggregate metrics using total parts / total joints as the
        denominator instead of averaging per-object means.
    """
    total_predicted_part_count = 0
    total_input_part_count = 0
    total_matched_joint_count = 0
    total_angle_error_sum_deg = 0.0
    total_angle_error_count = 0
    total_position_error_sum = 0.0
    total_position_error_count = 0
    total_correctly_matched_parts = 0
    total_pred_parts_for_matching = 0
    total_gt_parts_for_matching = 0

    for result in eval_results:
        predicted_part_count = result.get("predicted_part_count")
        input_part_count = result.get("input_part_count")
        if predicted_part_count is not None:
            total_predicted_part_count += int(predicted_part_count)
        if input_part_count is not None:
            total_input_part_count += int(input_part_count)
        total_matched_joint_count += int(result.get("matched_joint_count", 0))
        total_angle_error_sum_deg += float(result.get("joint_axis_angle_error_sum_deg", 0.0))
        total_angle_error_count += int(result.get("joint_axis_angle_error_count", 0))
        total_position_error_sum += float(result.get("joint_axis_position_error_sum", 0.0))
        total_position_error_count += int(result.get("joint_axis_position_error_count", 0))
        total_correctly_matched_parts += int(result.get("num_correctly_matched_parts", 0))
        total_pred_parts_for_matching += int(result.get("num_pred_parts_total", 0))
        total_gt_parts_for_matching += int(result.get("num_gt_parts_total", 0))

    predicted_part_coverage_pct = None
    if total_input_part_count > 0:
        predicted_part_coverage_pct = float(100.0 * total_predicted_part_count / total_input_part_count)
    part_match_precision = (
        float(total_correctly_matched_parts / total_pred_parts_for_matching)
        if total_pred_parts_for_matching > 0
        else 0.0
    )
    part_match_recall = (
        float(total_correctly_matched_parts / total_gt_parts_for_matching)
        if total_gt_parts_for_matching > 0
        else 0.0
    )
    part_match_f1 = (
        float(
            2.0
            * part_match_precision
            * part_match_recall
            / (part_match_precision + part_match_recall)
        )
        if (part_match_precision + part_match_recall) > 0.0
        else 0.0
    )

    return {
        "predicted_part_coverage_pct": predicted_part_coverage_pct,
        "predicted_part_count_total": int(total_predicted_part_count),
        "input_part_count_total": int(total_input_part_count),
        "part_match_precision": part_match_precision,
        "part_match_recall": part_match_recall,
        "part_match_f1": part_match_f1,
        "num_correctly_matched_parts_total": int(total_correctly_matched_parts),
        "num_pred_parts_total": int(total_pred_parts_for_matching),
        "num_gt_parts_total": int(total_gt_parts_for_matching),
        "matched_joint_count_total": int(total_matched_joint_count),
        "joint_axis_angle_error_deg": (
            None
            if total_angle_error_count == 0
            else float(total_angle_error_sum_deg / total_angle_error_count)
        ),
        "joint_axis_angle_error_count_total": int(total_angle_error_count),
        "joint_axis_position_error": (
            None
            if total_position_error_count == 0
            else float(total_position_error_sum / total_position_error_count)
        ),
        "joint_axis_position_error_count_total": int(total_position_error_count),
    }
