"""Shared evaluation helpers for `infer.py --eval`.

This module keeps the evaluation logic out of the CLI entrypoints. It evaluates
one predicted mesh against one preprocessed meta-root input, writes one
per-object JSON under `output_dir/<sample>/eval/`, and rebuilds root-level
summary JSON files from all available per-object results under the inference
output root.
"""

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from instruct_particulate.utils.articulation_utils import articulate_points_with_joints
from instruct_particulate.utils.evaluation_extra_metrics import (
    aggregate_part_and_joint_metrics,
    compute_part_and_joint_metrics,
    compute_part_matching_quality_metrics,
    extract_category,
)
from instruct_particulate.utils.evaluation_io import (
    CONTINUOUS_REVOLUTE_MASK_KEY,
    MATCHING_MODE_HUNGARIAN,
    MATCHING_MODE_SLOT_ALIGNED,
    PRISMATIC_RANGE_SUPERVISION_VALID_KEY,
    REFINED_VARIANT_KEY,
    REVOLUTE_RANGE_SUPERVISION_VALID_KEY,
    UNREFINED_VARIANT_KEY,
    apply_joint_range_evaluation_policy,
    build_gt_payload_from_meta_root,
    build_prediction_payload_variants_from_mesh,
    joint_records_from_motion_arrays,
    part_id_scheme_from_payload,
    save_evaluation_prediction,
)
from instruct_particulate.utils.evaluation_utils import (
    build_part_correspondence,
    evaluate_articulate_result,
    symmetric_chamfer_distance,
)

JsonDict = dict[str, Any]

DEFAULT_EVAL_NUM_POINTS = 100_000
NUM_ARTICULATION_STATES = 5
PER_OBJECT_RESULTS_FILENAME = "EVAL_RESULTS.json"
OVERALL_RESULTS_FILENAME = "OVERALL_EVAL_RESULTS.json"
PER_CATEGORY_RESULTS_FILENAME = "PER_CATEGORY_EVAL_RESULTS.json"
GT_SOURCE_INPUT_META_ROOT = "input_meta_root"
HUNGARIAN_MATCHING_COST_TYPE = "cdist"
MATCHING_MODES = (
    MATCHING_MODE_SLOT_ALIGNED,
    MATCHING_MODE_HUNGARIAN,
)
MODE_RESULT_REQUIRED_KEYS = (
    "rest_per_part_avg_chamfer",
    "rest_per_part_avg_giou",
    "rest_per_part_avg_mIoU",
    "fully_per_part_articulated_avg_chamfer",
    "fully_per_part_articulated_avg_giou",
    "fully_articulated_overall_chamfer_distances",
    "per_state_chamfer_distances",
    "per_state_giou",
    "per_state_overall_chamfer_distances",
    "predicted_part_count",
    "input_part_count",
    "predicted_part_coverage_pct",
    "part_match_precision",
    "part_match_recall",
    "part_match_f1",
    "num_correctly_matched_parts",
    "num_pred_parts_total",
    "num_gt_parts_total",
    "matched_joint_count",
    "joint_axis_angle_error_deg",
    "joint_axis_angle_error_sum_deg",
    "joint_axis_angle_error_count",
    "joint_axis_position_error",
    "joint_axis_position_error_sum",
    "joint_axis_position_error_count",
    "matching_mode",
    "gt_source",
)
NOPUNISH_MODE_RESULT_KEYS = (
    "rest_per_part_avg_chamfer_nopunish",
    "fully_articulated_avg_chamfer_nopunish",
    "rest_per_part_avg_giou_nopunish",
    "fully_per_part_articulated_avg_giou_nopunish",
    "rest_per_part_avg_mIoU_nopunish",
    "fully_per_part_articulated_avg_mIoU_nopunish",
    "per_state_chamfer_distances_nopunish",
    "per_state_giou_nopunish",
    "per_state_mIoU_nopunish",
)
LEGACY_REMOVED_SUMMARY_FILENAMES = (
    "OVERALL_EVAL_RESULTS_NOPUNISH.json",
    "PER_CATEGORY_EVAL_RESULTS_NOPUNISH.json",
)
REGULAR_SUMMARY_METRIC_SPECS = (
    ("rest_per_part_avg_chamfer", "rest_per_part_avg_chamfer", False),
    ("rest_per_part_avg_giou", "rest_per_part_avg_giou", False),
    ("rest_per_part_avg_mIoU", "rest_per_part_avg_mIoU", False),
    ("fully_per_part_articulated_avg_chamfer", "per_state_chamfer_distances", True),
    ("fully_per_part_articulated_avg_giou", "per_state_giou", True),
    (
        "fully_articulated_overall_chamfer_distances",
        "per_state_overall_chamfer_distances",
        True,
    ),
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Writes one JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=4)


def _read_json(path: Path) -> JsonDict:
    """Reads one JSON payload from disk."""
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {path} to contain a JSON object")
    return payload


def per_object_result_path(eval_dir: Path) -> Path:
    """Returns the per-object evaluation JSON path under one `eval/` directory."""
    return eval_dir / PER_OBJECT_RESULTS_FILENAME


def existing_result_path_for_input(output_root: Path, input_path: Path) -> Path:
    """Returns the deterministic per-object evaluation path for one input."""
    return output_root / input_path.stem / "eval" / PER_OBJECT_RESULTS_FILENAME


def _compute_single_result(
    prediction: Mapping[str, Any],
    gt: Mapping[str, Any],
    *,
    matching_mode: str,
) -> JsonDict:
    """Evaluates one prediction against one GT payload."""
    return evaluate_articulate_result(
        xyz=np.asarray(prediction["points"], dtype=np.float32),
        xyz_gt=np.asarray(gt["points"], dtype=np.float32),
        part_ids_pred=np.asarray(prediction["part_ids"], dtype=np.int32),
        part_ids_gt=np.asarray(gt["part_ids"], dtype=np.int32),
        motion_hierarchy_pred=np.asarray(prediction["motion_hierarchy"], dtype=np.int32),
        motion_hierarchy_gt=np.asarray(gt["motion_hierarchy"], dtype=np.int32),
        is_part_revolute_pred=np.asarray(prediction["is_part_revolute"], dtype=np.bool_),
        is_part_revolute_gt=np.asarray(gt["is_part_revolute"], dtype=np.bool_),
        is_part_prismatic_pred=np.asarray(prediction["is_part_prismatic"], dtype=np.bool_),
        is_part_prismatic_gt=np.asarray(gt["is_part_prismatic"], dtype=np.bool_),
        revolute_plucker_pred=np.asarray(prediction["revolute_plucker"], dtype=np.float32),
        revolute_plucker_gt=np.asarray(gt["revolute_plucker"], dtype=np.float32),
        revolute_range_pred=np.asarray(prediction["revolute_range"], dtype=np.float32),
        revolute_range_gt=np.asarray(gt["revolute_range"], dtype=np.float32),
        prismatic_axis_pred=np.asarray(prediction["prismatic_axis"], dtype=np.float32),
        prismatic_axis_gt=np.asarray(gt["prismatic_axis"], dtype=np.float32),
        prismatic_range_pred=np.asarray(prediction["prismatic_range"], dtype=np.float32),
        prismatic_range_gt=np.asarray(gt["prismatic_range"], dtype=np.float32),
        num_articulation_states=NUM_ARTICULATION_STATES,
        matching_mode=matching_mode,
        hungarian_matching_cost_type=HUNGARIAN_MATCHING_COST_TYPE,
    )


def _require_cuda_for_evaluation() -> None:
    """Raises when evaluation is attempted without CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Evaluation requires CUDA. Run infer.py --eval on a CUDA-capable host."
        )


@torch.no_grad()
def _evaluate_prediction(
    prediction: Mapping[str, Any],
    gt: Mapping[str, Any],
    *,
    matching_mode: str,
) -> JsonDict:
    """Evaluates one prediction/GT pair with CUDA autocast enabled."""
    _require_cuda_for_evaluation()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return _compute_single_result(prediction, gt, matching_mode=matching_mode)


def _strip_nopunish_metrics(mode_result: Mapping[str, Any]) -> JsonDict:
    """Drops legacy no-punish metrics from one mode result payload."""
    return {
        key: value
        for key, value in mode_result.items()
        if key not in NOPUNISH_MODE_RESULT_KEYS
    }


def _sanitize_mode_result(mode_result: Mapping[str, Any]) -> JsonDict:
    """Normalizes one mode result to the current persisted schema."""
    sanitized_result = _strip_nopunish_metrics(mode_result)
    unrefined_result = sanitized_result.get(UNREFINED_VARIANT_KEY)
    if isinstance(unrefined_result, Mapping):
        sanitized_result[UNREFINED_VARIANT_KEY] = _strip_nopunish_metrics(unrefined_result)
    return sanitized_result


def _sanitize_sample_result(sample_result: Mapping[str, Any]) -> JsonDict:
    """Normalizes one per-object result payload to the current persisted schema."""
    sanitized_result = dict(sample_result)
    for mode in MATCHING_MODES:
        mode_result = sanitized_result.get(mode)
        if isinstance(mode_result, Mapping):
            sanitized_result[mode] = _sanitize_mode_result(mode_result)
    return sanitized_result


def _compute_mode_result(
    *,
    prediction_variants: Mapping[str, Mapping[str, Any]],
    gt_payload: Mapping[str, Any],
    matching_mode: str,
    gt_source: str,
) -> JsonDict:
    """Evaluates all available segmentation variants for one matching mode."""
    adjusted_prediction_variants = {
        variant_name: _make_prediction_order_invariant(
            apply_joint_range_evaluation_policy(variant_payload, gt_payload),
            gt_payload,
            matching_mode=matching_mode,
        )
        for variant_name, variant_payload in prediction_variants.items()
    }
    input_num_parts = int(len(np.asarray(gt_payload["link_names"])))
    gt_joint_records = joint_records_from_motion_arrays(gt_payload)

    refined_result = _evaluate_prediction(
        adjusted_prediction_variants[REFINED_VARIANT_KEY],
        gt_payload,
        matching_mode=matching_mode,
    )
    refined_result.update(
        compute_part_and_joint_metrics(
            prediction=adjusted_prediction_variants[REFINED_VARIANT_KEY],
            gt_cache=gt_payload,
            input_num_parts=input_num_parts,
            gt_joint_records=gt_joint_records,
            matching_mode=matching_mode,
            hungarian_matching_cost_type=HUNGARIAN_MATCHING_COST_TYPE,
        )
    )
    refined_result.update(
        compute_part_matching_quality_metrics(
            prediction=adjusted_prediction_variants[REFINED_VARIANT_KEY],
            gt_payload=gt_payload,
            matching_mode=matching_mode,
            hungarian_matching_cost_type=HUNGARIAN_MATCHING_COST_TYPE,
        )
    )
    refined_result = _strip_nopunish_metrics(refined_result)
    refined_result["matching_mode"] = matching_mode
    refined_result["gt_source"] = gt_source
    refined_result["gt_part_id_scheme"] = part_id_scheme_from_payload(gt_payload)

    if UNREFINED_VARIANT_KEY in adjusted_prediction_variants:
        unrefined_result = _evaluate_prediction(
            adjusted_prediction_variants[UNREFINED_VARIANT_KEY],
            gt_payload,
            matching_mode=matching_mode,
        )
        unrefined_result.update(
            compute_part_and_joint_metrics(
                prediction=adjusted_prediction_variants[UNREFINED_VARIANT_KEY],
                gt_cache=gt_payload,
                input_num_parts=input_num_parts,
                gt_joint_records=gt_joint_records,
                matching_mode=matching_mode,
                hungarian_matching_cost_type=HUNGARIAN_MATCHING_COST_TYPE,
            )
        )
        unrefined_result.update(
            compute_part_matching_quality_metrics(
                prediction=adjusted_prediction_variants[UNREFINED_VARIANT_KEY],
                gt_payload=gt_payload,
                matching_mode=matching_mode,
                hungarian_matching_cost_type=HUNGARIAN_MATCHING_COST_TYPE,
            )
        )
        unrefined_result = _strip_nopunish_metrics(unrefined_result)
        unrefined_result["matching_mode"] = matching_mode
        unrefined_result["gt_source"] = gt_source
        unrefined_result["gt_part_id_scheme"] = part_id_scheme_from_payload(gt_payload)
        refined_result[UNREFINED_VARIANT_KEY] = unrefined_result
    return refined_result


def _build_matched_gt_part_ids(
    *,
    prediction: Mapping[str, Any],
    gt_payload: Mapping[str, Any],
    matching_mode: str,
) -> dict[int, int]:
    """Maps each predicted part ID to the GT part ID used for evaluation."""
    prediction_part_ids = np.unique(np.asarray(prediction["part_ids"], dtype=np.int32))
    gt_part_ids = np.unique(np.asarray(gt_payload["part_ids"], dtype=np.int32))
    mapping_pred2gt = build_part_correspondence(
        points_src=np.asarray(prediction["points"], dtype=np.float32),
        part_ids_src=np.asarray(prediction["part_ids"], dtype=np.int32),
        points_tgt=np.asarray(gt_payload["points"], dtype=np.float32),
        part_ids_tgt=np.asarray(gt_payload["part_ids"], dtype=np.int32),
        matching_mode=matching_mode,
        hungarian_matching_cost_type=HUNGARIAN_MATCHING_COST_TYPE,
    )
    matched_gt_part_ids: dict[int, int] = {}
    for prediction_part_index, prediction_part_id in enumerate(prediction_part_ids.tolist()):
        gt_part_index = int(mapping_pred2gt[prediction_part_index, 0])
        if gt_part_index < 0:
            continue
        matched_gt_part_ids[int(prediction_part_id)] = int(gt_part_ids[gt_part_index])
    return matched_gt_part_ids


def _single_joint_endpoint_cost(
    *,
    pred_points: np.ndarray,
    pred_part_ids: np.ndarray,
    pred_joints: Sequence[Mapping[str, Any]],
    pred_child_link_id: int,
    pred_state: float,
    gt_points: np.ndarray,
    gt_part_ids: np.ndarray,
    gt_joints: Sequence[Mapping[str, Any]],
    gt_child_link_id: int,
    gt_state: float,
) -> float:
    """Returns overall point-cloud Chamfer for one individually articulated joint."""
    articulated_pred_points = articulate_points_with_joints(
        pred_points,
        pred_part_ids,
        list(pred_joints),
        {int(pred_child_link_id): float(pred_state)},
    )[0]
    articulated_gt_points = articulate_points_with_joints(
        gt_points,
        gt_part_ids,
        list(gt_joints),
        {int(gt_child_link_id): float(gt_state)},
    )[0]
    return float(
        symmetric_chamfer_distance(
            np.asarray(articulated_pred_points, dtype=np.float32),
            np.asarray(articulated_gt_points, dtype=np.float32),
        )
    )


def _should_flip_endpoint_order(
    *,
    prediction: Mapping[str, Any],
    gt_payload: Mapping[str, Any],
    pred_joint_records: Sequence[Mapping[str, Any]],
    gt_joint_records: Sequence[Mapping[str, Any]],
    pred_part_id: int,
    gt_part_id: int,
) -> bool:
    """Chooses whether one predicted joint's low/high order should be flipped.

    The decision follows the requested protocol: articulate the predicted and GT
    joint individually at endpoint states `0` and `1`, then compare the summed
    overall point-cloud Chamfer for identity versus swapped endpoint
    correspondences.
    """
    identity_cost = _single_joint_endpoint_cost(
        pred_points=np.asarray(prediction["points"], dtype=np.float32),
        pred_part_ids=np.asarray(prediction["part_ids"], dtype=np.int32),
        pred_joints=pred_joint_records,
        pred_child_link_id=pred_part_id,
        pred_state=0.0,
        gt_points=np.asarray(gt_payload["points"], dtype=np.float32),
        gt_part_ids=np.asarray(gt_payload["part_ids"], dtype=np.int32),
        gt_joints=gt_joint_records,
        gt_child_link_id=gt_part_id,
        gt_state=0.0,
    ) + _single_joint_endpoint_cost(
        pred_points=np.asarray(prediction["points"], dtype=np.float32),
        pred_part_ids=np.asarray(prediction["part_ids"], dtype=np.int32),
        pred_joints=pred_joint_records,
        pred_child_link_id=pred_part_id,
        pred_state=1.0,
        gt_points=np.asarray(gt_payload["points"], dtype=np.float32),
        gt_part_ids=np.asarray(gt_payload["part_ids"], dtype=np.int32),
        gt_joints=gt_joint_records,
        gt_child_link_id=gt_part_id,
        gt_state=1.0,
    )
    swapped_cost = _single_joint_endpoint_cost(
        pred_points=np.asarray(prediction["points"], dtype=np.float32),
        pred_part_ids=np.asarray(prediction["part_ids"], dtype=np.int32),
        pred_joints=pred_joint_records,
        pred_child_link_id=pred_part_id,
        pred_state=0.0,
        gt_points=np.asarray(gt_payload["points"], dtype=np.float32),
        gt_part_ids=np.asarray(gt_payload["part_ids"], dtype=np.int32),
        gt_joints=gt_joint_records,
        gt_child_link_id=gt_part_id,
        gt_state=1.0,
    ) + _single_joint_endpoint_cost(
        pred_points=np.asarray(prediction["points"], dtype=np.float32),
        pred_part_ids=np.asarray(prediction["part_ids"], dtype=np.int32),
        pred_joints=pred_joint_records,
        pred_child_link_id=pred_part_id,
        pred_state=1.0,
        gt_points=np.asarray(gt_payload["points"], dtype=np.float32),
        gt_part_ids=np.asarray(gt_payload["part_ids"], dtype=np.int32),
        gt_joints=gt_joint_records,
        gt_child_link_id=gt_part_id,
        gt_state=0.0,
    )
    return bool(swapped_cost < identity_cost)


def _make_prediction_order_invariant(
    prediction: Mapping[str, Any],
    gt_payload: Mapping[str, Any],
    *,
    matching_mode: str,
) -> JsonDict:
    """Aligns supervised predicted endpoint ordering to GT before evaluation.

    For each supervised finite-range joint, this first articulates that joint
    individually and compares identity versus swapped endpoint correspondences
    using overall point-cloud Chamfer. It then flips the predicted range order
    for that joint when the swapped endpoint assignment matches GT better.
    """
    aligned_prediction: JsonDict = {}
    for key, value in prediction.items():
        if isinstance(value, np.ndarray):
            aligned_prediction[key] = np.asarray(value).copy()
        else:
            aligned_prediction[key] = value

    matched_gt_part_ids = _build_matched_gt_part_ids(
        prediction=aligned_prediction,
        gt_payload=gt_payload,
        matching_mode=matching_mode,
    )
    pred_joint_records = joint_records_from_motion_arrays(aligned_prediction)
    gt_joint_records = joint_records_from_motion_arrays(gt_payload)
    pred_joint_by_child = {
        int(joint["child_link_id"]): joint for joint in pred_joint_records
    }
    gt_joint_by_child = {
        int(joint["child_link_id"]): joint for joint in gt_joint_records
    }

    revolute_range = np.asarray(aligned_prediction["revolute_range"], dtype=np.float32)
    prismatic_range = np.asarray(aligned_prediction["prismatic_range"], dtype=np.float32)
    gt_revolute_supervised = np.asarray(
        gt_payload[REVOLUTE_RANGE_SUPERVISION_VALID_KEY],
        dtype=np.bool_,
    )
    gt_prismatic_supervised = np.asarray(
        gt_payload[PRISMATIC_RANGE_SUPERVISION_VALID_KEY],
        dtype=np.bool_,
    )
    gt_continuous_revolute = np.asarray(
        gt_payload[CONTINUOUS_REVOLUTE_MASK_KEY],
        dtype=np.bool_,
    )
    predicted_is_revolute = np.asarray(
        aligned_prediction["is_part_revolute"],
        dtype=np.bool_,
    )
    predicted_is_prismatic = np.asarray(
        aligned_prediction["is_part_prismatic"],
        dtype=np.bool_,
    )

    for pred_part_id, gt_part_id in matched_gt_part_ids.items():
        if gt_continuous_revolute[gt_part_id]:
            continue
        pred_joint = pred_joint_by_child.get(int(pred_part_id))
        gt_joint = gt_joint_by_child.get(int(gt_part_id))
        if pred_joint is None or gt_joint is None:
            continue

        if gt_revolute_supervised[gt_part_id]:
            if not predicted_is_revolute[pred_part_id] or not bool(gt_joint.get("is_revolute")):
                continue
            if _should_flip_endpoint_order(
                prediction=aligned_prediction,
                gt_payload=gt_payload,
                pred_joint_records=pred_joint_records,
                gt_joint_records=gt_joint_records,
                pred_part_id=int(pred_part_id),
                gt_part_id=int(gt_part_id),
            ):
                revolute_range[pred_part_id] = revolute_range[pred_part_id][::-1]
                pred_joint_records = joint_records_from_motion_arrays(
                    {**aligned_prediction, "revolute_range": revolute_range}
                )
                pred_joint_by_child = {
                    int(joint["child_link_id"]): joint for joint in pred_joint_records
                }
            continue

        if gt_prismatic_supervised[gt_part_id]:
            if not predicted_is_prismatic[pred_part_id] or not bool(gt_joint.get("is_prismatic")):
                continue
            if _should_flip_endpoint_order(
                prediction=aligned_prediction,
                gt_payload=gt_payload,
                pred_joint_records=pred_joint_records,
                gt_joint_records=gt_joint_records,
                pred_part_id=int(pred_part_id),
                gt_part_id=int(gt_part_id),
            ):
                prismatic_range[pred_part_id] = prismatic_range[pred_part_id][::-1]
                pred_joint_records = joint_records_from_motion_arrays(
                    {**aligned_prediction, "prismatic_range": prismatic_range}
                )
                pred_joint_by_child = {
                    int(joint["child_link_id"]): joint for joint in pred_joint_records
                }

    aligned_prediction["revolute_range"] = revolute_range
    aligned_prediction["prismatic_range"] = prismatic_range
    return aligned_prediction


def evaluate_prediction_variants_for_meta_root(
    *,
    meta_root: Path,
    prediction_variants: Mapping[str, Mapping[str, Any]],
    num_points: int = DEFAULT_EVAL_NUM_POINTS,
) -> JsonDict:
    """Evaluates one meta-root prediction payload against on-the-fly GT."""
    gt_payload = build_gt_payload_from_meta_root(meta_root=meta_root, num_points=num_points)
    sample_name = meta_root.stem
    sample_result: JsonDict = {
        "sample_name": sample_name,
        "category": extract_category(sample_name),
    }
    for matching_mode in MATCHING_MODES:
        sample_result[matching_mode] = _compute_mode_result(
            prediction_variants=prediction_variants,
            gt_payload=gt_payload,
            matching_mode=matching_mode,
            gt_source=GT_SOURCE_INPUT_META_ROOT,
        )
    return sample_result


def export_and_evaluate_meta_root_prediction(
    *,
    output_dir: Path,
    meta_root: Path,
    mesh: Any,
    face_part_ids: np.ndarray,
    motion_hierarchy: Sequence[tuple[int, int]],
    is_part_revolute: np.ndarray,
    is_part_prismatic: np.ndarray,
    revolute_plucker: np.ndarray,
    revolute_range: np.ndarray,
    prismatic_axis: np.ndarray,
    prismatic_range: np.ndarray,
    face_part_ids_unrefined: np.ndarray | None = None,
    prismatic_plucker: np.ndarray | None = None,
    num_points: int = DEFAULT_EVAL_NUM_POINTS,
) -> tuple[Path, JsonDict]:
    """Exports particulate eval artifacts and writes one per-object eval JSON.

    Args:
        output_dir: Per-object inference output directory.
        meta_root: Preprocessed input directory used as GT for evaluation.
        mesh: Normalized predicted mesh.
        face_part_ids: Refined predicted face segmentation in prompt-slot space.
        motion_hierarchy: Predicted motion hierarchy edges.
        is_part_revolute: Predicted revolute mask.
        is_part_prismatic: Predicted prismatic mask.
        revolute_plucker: Predicted revolute axes.
        revolute_range: Predicted revolute ranges.
        prismatic_axis: Predicted prismatic directions.
        prismatic_range: Predicted prismatic ranges.
        face_part_ids_unrefined: Optional unrefined face segmentation.
        prismatic_plucker: Optional predicted prismatic Plucker lines.
        num_points: Number of points to sample from prediction and GT meshes.

    Returns:
        Tuple of `(eval_dir, sample_result)`.
    """
    eval_dir = save_evaluation_prediction(
        output_dir=output_dir,
        mesh=mesh,
        face_part_ids=face_part_ids,
        face_part_ids_unrefined=face_part_ids_unrefined,
        motion_hierarchy=motion_hierarchy,
        is_part_revolute=is_part_revolute,
        is_part_prismatic=is_part_prismatic,
        revolute_plucker=revolute_plucker,
        revolute_range=revolute_range,
        prismatic_axis=prismatic_axis,
        prismatic_range=prismatic_range,
        prismatic_plucker=prismatic_plucker,
    )
    prediction_variants = build_prediction_payload_variants_from_mesh(
        mesh=mesh,
        face_part_ids=face_part_ids,
        face_part_ids_unrefined=face_part_ids_unrefined,
        motion_hierarchy=motion_hierarchy,
        is_part_revolute=is_part_revolute,
        is_part_prismatic=is_part_prismatic,
        revolute_plucker=revolute_plucker,
        revolute_range=revolute_range,
        prismatic_axis=prismatic_axis,
        prismatic_range=prismatic_range,
        prismatic_plucker=prismatic_plucker,
        num_points=num_points,
    )
    sample_result = evaluate_prediction_variants_for_meta_root(
        meta_root=meta_root,
        prediction_variants=prediction_variants,
        num_points=num_points,
    )
    _write_json(per_object_result_path(eval_dir), sample_result)
    return eval_dir, sample_result


def _rounded_mean(values: Sequence[float]) -> float:
    """Rounds one mean to the particulate output precision."""
    return float(np.round(np.mean(values), 4))


def _round_optional(value: float | None) -> float | None:
    """Rounds an optional scalar to the particulate output precision."""
    if value is None:
        return None
    return float(np.round(value, 4))


def _summary_metric_value(
    result: Mapping[str, Any],
    *,
    source_key: str,
    use_last_state: bool,
) -> float:
    """Extracts one scalar metric used in an aggregate summary."""
    value: Any = result[source_key]
    if use_last_state:
        value = value[-1]
    return float(value)


def _with_additional_metric_aggregates(
    summary: JsonDict,
    eval_results: Sequence[Mapping[str, Any]],
) -> None:
    """Adds weighted part/joint aggregates to one summary payload."""
    weighted_metrics = aggregate_part_and_joint_metrics(eval_results)
    summary.update(
        {
            "predicted_part_coverage_pct": _round_optional(
                weighted_metrics["predicted_part_coverage_pct"]
            ),
            "predicted_part_count_total": int(weighted_metrics["predicted_part_count_total"]),
            "input_part_count_total": int(weighted_metrics["input_part_count_total"]),
            "part_match_precision": _round_optional(
                weighted_metrics["part_match_precision"]
            ),
            "part_match_recall": _round_optional(weighted_metrics["part_match_recall"]),
            "part_match_f1": _round_optional(weighted_metrics["part_match_f1"]),
            "num_correctly_matched_parts_total": int(
                weighted_metrics["num_correctly_matched_parts_total"]
            ),
            "num_pred_parts_total": int(weighted_metrics["num_pred_parts_total"]),
            "num_gt_parts_total": int(weighted_metrics["num_gt_parts_total"]),
            "matched_joint_count_total": int(weighted_metrics["matched_joint_count_total"]),
            "joint_axis_angle_error_deg": _round_optional(
                weighted_metrics["joint_axis_angle_error_deg"]
            ),
            "joint_axis_angle_error_count_total": int(
                weighted_metrics["joint_axis_angle_error_count_total"]
            ),
            "joint_axis_position_error": _round_optional(
                weighted_metrics["joint_axis_position_error"]
            ),
            "joint_axis_position_error_count_total": int(
                weighted_metrics["joint_axis_position_error_count_total"]
            ),
        }
    )


def _collect_unrefined_results(
    eval_results: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Collects optional unrefined per-sample results for one matching mode."""
    return [
        result[UNREFINED_VARIANT_KEY]
        for result in eval_results
        if UNREFINED_VARIANT_KEY in result
    ]


def _build_summary_block(
    eval_results: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Builds one aggregate summary block for one matching mode."""
    summary: JsonDict = {
        "num_samples": len(eval_results),
        "matching_mode": str(eval_results[0]["matching_mode"]),
    }
    for summary_key, source_key, use_last_state in REGULAR_SUMMARY_METRIC_SPECS:
        summary[summary_key] = _rounded_mean(
            [
                _summary_metric_value(
                    result,
                    source_key=source_key,
                    use_last_state=use_last_state,
                )
                for result in eval_results
            ]
        )
    _with_additional_metric_aggregates(summary, eval_results)
    return summary


def _build_mode_summary(eval_results: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Builds the regular aggregate summary for one matching mode."""
    summary = _build_summary_block(eval_results)
    unrefined_results = _collect_unrefined_results(eval_results)
    if unrefined_results:
        summary[UNREFINED_VARIANT_KEY] = _build_summary_block(unrefined_results)
    return summary


def _mode_results(
    sample_results: Sequence[Mapping[str, Any]],
    mode: str,
) -> list[Mapping[str, Any]]:
    """Collects the per-sample results for one matching mode."""
    return [sample_result[mode] for sample_result in sample_results]


def _group_results_by_category(
    sample_results: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Groups sample results by category."""
    grouped_results: dict[str, list[Mapping[str, Any]]] = {}
    for sample_result in sample_results:
        category = str(sample_result.get("category") or extract_category(str(sample_result["sample_name"])))
        grouped_results.setdefault(category, []).append(sample_result)
    return grouped_results


def _build_overall_payload(sample_results: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Builds the aggregate summary JSON across all matching modes."""
    return {
        "num_samples": len(sample_results),
        **{
            mode: _build_mode_summary(_mode_results(sample_results, mode))
            for mode in MATCHING_MODES
        },
    }


def _build_per_category_payload(sample_results: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Builds the regular per-category summary JSON."""
    return {
        category: _build_overall_payload(category_results)
        for category, category_results in sorted(_group_results_by_category(sample_results).items())
    }


def _is_mode_result_valid(mode_result: Mapping[str, Any]) -> bool:
    """Returns whether one mode block matches the current schema."""
    if any(key not in mode_result for key in MODE_RESULT_REQUIRED_KEYS):
        return False
    unrefined_result = mode_result.get(UNREFINED_VARIANT_KEY)
    if unrefined_result is not None:
        if not isinstance(unrefined_result, Mapping):
            return False
        if any(key not in unrefined_result for key in MODE_RESULT_REQUIRED_KEYS):
            return False
    return True


def is_sample_result_valid(sample_result: Mapping[str, Any]) -> bool:
    """Returns whether one per-object evaluation JSON matches the current schema."""
    if "sample_name" not in sample_result or "category" not in sample_result:
        return False
    return all(
        isinstance(sample_result.get(mode), Mapping)
        and _is_mode_result_valid(sample_result[mode])
        for mode in MATCHING_MODES
    )


def load_cached_sample_result(output_root: Path, input_path: Path) -> JsonDict | None:
    """Loads one existing per-object evaluation JSON when it is still valid."""
    result_path = existing_result_path_for_input(output_root, input_path)
    if not result_path.exists():
        return None
    try:
        cached_result = _read_json(result_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not is_sample_result_valid(cached_result):
        return None
    cached_result = _sanitize_sample_result(cached_result)
    if str(cached_result.get("sample_name")) != input_path.stem:
        return None
    return cached_result


def _find_per_object_result_paths(output_root: Path) -> list[Path]:
    """Finds all per-object evaluation JSONs under one inference output root."""
    return sorted(
        result_path
        for result_path in output_root.rglob(PER_OBJECT_RESULTS_FILENAME)
        if result_path.parent.name == "eval"
    )


def refresh_output_root_summaries(output_root: Path) -> list[JsonDict]:
    """Rebuilds root-level evaluation summaries from per-object JSON files.

    Args:
        output_root: Root inference output directory whose immediate children are
            per-object output directories.

    Returns:
        Valid sample results discovered under ``output_root``.
    """
    sample_results: list[JsonDict] = []
    for result_path in _find_per_object_result_paths(output_root):
        try:
            sample_result = _read_json(result_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            print(f"Skipping invalid evaluation JSON: {result_path}")
            continue
        if not is_sample_result_valid(sample_result):
            print(f"Skipping outdated evaluation JSON: {result_path}")
            continue
        sanitized_result = _sanitize_sample_result(sample_result)
        if sanitized_result != sample_result:
            _write_json(result_path, sanitized_result)
        sample_result = sanitized_result
        sample_results.append(sample_result)

    sample_results.sort(key=lambda payload: str(payload["sample_name"]))
    for filename in LEGACY_REMOVED_SUMMARY_FILENAMES:
        legacy_summary_path = output_root / filename
        if legacy_summary_path.exists():
            legacy_summary_path.unlink()
    if not sample_results:
        return []

    _write_json(output_root / OVERALL_RESULTS_FILENAME, _build_overall_payload(sample_results))
    _write_json(
        output_root / PER_CATEGORY_RESULTS_FILENAME,
        _build_per_category_payload(sample_results),
    )
    return sample_results
