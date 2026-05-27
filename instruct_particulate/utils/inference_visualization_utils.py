from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import trimesh

from instruct_particulate.datasets.vis_helper import (
    plot_joint_overparam_visualizations,
    plot_query_visualization,
)
from instruct_particulate.utils.export_utils import export_animated_glb_file
from instruct_particulate.utils.visualization_utils import (
    COLORS,
    create_motion_axis_meshes,
    create_textured_mesh_parts,
)


def _denormalize_visualization_points(
    points: np.ndarray | None,
    *,
    center: np.ndarray,
    scale: float,
) -> np.ndarray | None:
    if points is None:
        return None
    return (
        np.asarray(points, dtype=np.float32) / np.float32(scale)
        + np.asarray(center, dtype=np.float32)
    ).astype(np.float32, copy=False)


def _collect_motion_overparam_visualization(
    *,
    motion_label: str,
    query_points: np.ndarray,
    query_link_ids: np.ndarray,
    query_confidences: np.ndarray | None,
    valid_link_ids: np.ndarray,
    closest_axis_points: np.ndarray | None,
    low_points: np.ndarray | None,
    high_points: np.ndarray | None,
    num_points: int,
) -> dict[str, Any] | None:
    if (
        closest_axis_points is None
        or low_points is None
        or high_points is None
        or len(valid_link_ids) == 0
    ):
        return None

    motion_valid_mask = np.isin(query_link_ids, valid_link_ids)
    valid_indices = np.flatnonzero(motion_valid_mask)
    if len(valid_indices) == 0:
        return None
    if len(valid_indices) > num_points:
        sampled_indices = np.random.choice(
            valid_indices,
            size=num_points,
            replace=False,
        ).astype(np.int64, copy=False)
        sampled_indices.sort()
    else:
        sampled_indices = valid_indices.astype(np.int64, copy=False)

    sampled_query_points = np.asarray(query_points, dtype=np.float32)[sampled_indices]
    sampled_query_link_ids = np.asarray(query_link_ids, dtype=np.int64)[sampled_indices]
    sampled_closest_axis_points = np.asarray(
        closest_axis_points,
        dtype=np.float32,
    )[sampled_indices]
    sampled_low_points = np.asarray(low_points, dtype=np.float32)[sampled_indices]
    sampled_high_points = np.asarray(high_points, dtype=np.float32)[sampled_indices]
    if len(sampled_query_points) == 0:
        return None

    if query_confidences is None:
        sampled_query_confidences = np.ones((len(sampled_indices),), dtype=np.float32)
    else:
        sampled_query_confidences = np.asarray(
            query_confidences,
            dtype=np.float32,
        )[sampled_indices]
    target_valid_mask = np.ones((len(sampled_indices),), dtype=np.bool_)

    palette = np.asarray(COLORS, dtype=np.float32) / 255.0
    base_point_colors = np.full((len(sampled_indices), 3), 0.7, dtype=np.float32)
    valid_link_mask = sampled_query_link_ids >= 0
    if np.any(valid_link_mask):
        base_point_colors[valid_link_mask] = palette[
            sampled_query_link_ids[valid_link_mask] % len(palette)
        ]
    point_confidences = np.clip(sampled_query_confidences, 0.0, 1.0).reshape(-1, 1)
    point_colors = 1.0 - point_confidences * (1.0 - base_point_colors)

    return {
        "motion_label": motion_label,
        "query_points": sampled_query_points,
        "query_link_ids": sampled_query_link_ids,
        "point_colors": point_colors.astype(np.float32, copy=False),
        "closest_axis_points": sampled_closest_axis_points,
        "closest_axis_points_valid": target_valid_mask.copy(),
        "low_points": sampled_low_points,
        "low_points_valid": target_valid_mask.copy(),
        "high_points": sampled_high_points,
        "high_points_valid": target_valid_mask.copy(),
        "context_query_points": np.asarray(query_points, dtype=np.float32)[~motion_valid_mask],
    }


def _output_array(
    output: Mapping[str, Any],
    key: str,
    *,
    dtype: np.dtype[Any] | type = np.float32,
) -> np.ndarray | None:
    value = output.get(key)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value_array = np.asarray(value)
    if value_array.ndim > 0 and value_array.shape[0] == 1:
        value_array = value_array[0]
    return np.asarray(value_array, dtype=dtype)


def print_inference_summary(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    unique_part_ids: np.ndarray,
    visualization_path: Path | None = None,
    overparam_visualization_path: Path | None = None,
) -> None:
    print(f"Saved inference outputs to {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    if visualization_path is not None:
        print(f"Query visualization: {visualization_path}")
    if overparam_visualization_path is not None:
        print(f"Overparam visualization: {overparam_visualization_path}")
    print(f"Segmented parts present: {unique_part_ids.tolist()}")


def save_segmented_visualizations(
    output_dir: Path,
    mesh,
    face_part_ids: np.ndarray,
    *,
    motion_hierarchy: Sequence[tuple[int, int]],
    is_part_revolute: np.ndarray,
    is_part_prismatic: np.ndarray,
    revolute_plucker: np.ndarray,
    revolute_range: np.ndarray,
    prismatic_axis: np.ndarray,
    prismatic_range: np.ndarray,
    animation_frames: int,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    unique_part_ids = np.unique(face_part_ids).astype(np.int32, copy=False)
    mesh_parts_original = [
        mesh.submesh([face_part_ids == part_id], append=True)
        for part_id in unique_part_ids
    ]
    mesh_parts_segmented = create_textured_mesh_parts(
        [mesh_part.copy() for mesh_part in mesh_parts_original]
    )
    axes = create_motion_axis_meshes(
        mesh_parts_original,
        unique_part_ids,
        is_part_revolute,
        is_part_prismatic,
        revolute_plucker,
        prismatic_axis,
    )

    trimesh.Scene(mesh_parts_segmented + axes).export(output_dir / "mesh_parts_with_axes.glb")
    export_animated_glb_file(
        mesh_parts_original,
        unique_part_ids,
        list(motion_hierarchy),
        is_part_revolute,
        is_part_prismatic,
        revolute_plucker,
        revolute_range,
        prismatic_axis,
        prismatic_range,
        animation_frames,
        str(output_dir / "animated_textured.glb"),
        include_axes=False,
        axes_meshes=None,
    )
    return mesh_parts_original, unique_part_ids


def _resolve_link_point_prompt_inputs(
    link_point_prompts: np.ndarray | None,
    link_point_prompt_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if link_point_prompts is None:
        prompt_array = np.zeros((0, 3), dtype=np.float32)
    else:
        prompt_array = np.asarray(link_point_prompts, dtype=np.float32)
    if prompt_array.ndim != 2 or prompt_array.shape[-1] != 3:
        raise ValueError(
            f"Expected link_point_prompts to have shape (N, 3), got {prompt_array.shape}"
        )

    if link_point_prompt_ids is None:
        prompt_ids = np.arange(len(prompt_array), dtype=np.int64)
    else:
        prompt_ids = np.asarray(link_point_prompt_ids, dtype=np.int64)
        if prompt_ids.shape != (len(prompt_array),):
            raise ValueError(
                "link_point_prompt_ids must align with link_point_prompts, "
                f"got {prompt_ids.shape} and {prompt_array.shape}"
            )
    return prompt_array, prompt_ids


def _sample_query_visualization_inputs(
    *,
    query_points: np.ndarray,
    query_normals: np.ndarray,
    query_link_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(query_points) == 0:
        raise ValueError("Point visualization requires at least one query point")
    query_points = np.asarray(query_points, dtype=np.float32)
    query_normals = np.asarray(query_normals, dtype=np.float32)
    query_link_ids = np.asarray(query_link_ids, dtype=np.int64)
    sample_size = min(8192, len(query_points))
    sample_indices = np.random.choice(len(query_points), size=sample_size, replace=False)
    sample_indices = sample_indices.astype(np.int64, copy=False)
    return (
        query_points[sample_indices],
        query_normals[sample_indices],
        query_link_ids[sample_indices],
        sample_indices,
    )


def save_predicted_point_query_rest_visualization(
    output_dir: Path,
    *,
    query_points: np.ndarray,
    query_normals: np.ndarray,
    predicted_part_ids: np.ndarray,
    link_point_prompts: np.ndarray | None,
    link_point_prompt_ids: np.ndarray | None = None,
    links: Sequence[dict[str, Any]],
    predicted_link_confidences: Mapping[int, float] | None = None,
) -> Path:
    link_point_prompts, link_point_prompt_ids = _resolve_link_point_prompt_inputs(
        link_point_prompts,
        link_point_prompt_ids,
    )
    (
        visualization_query_points,
        visualization_query_normals,
        visualization_predicted_link_ids,
        _,
    ) = _sample_query_visualization_inputs(
        query_points=query_points,
        query_normals=query_normals,
        query_link_ids=predicted_part_ids,
    )

    out_path = output_dir / "query_visualization.png"
    plot_query_visualization(
        query_points=visualization_query_points,
        query_normals=visualization_query_normals,
        query_link_ids=visualization_predicted_link_ids,
        link_point_prompts=link_point_prompts,
        link_point_prompt_ids=link_point_prompt_ids,
        links=list(links),
        predicted_link_confidences=predicted_link_confidences,
        out_path=out_path,
        normal_limit=0,
    )
    return out_path


def select_visualized_link_point_prompts(
    *,
    link_point_prompts: np.ndarray,
    links: Sequence[dict[str, Any]],
    hide_unique_text_prompts: bool = False,
    link_point_prompt_dropout_eligible: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    link_point_prompts = np.asarray(link_point_prompts, dtype=np.float32)
    if link_point_prompts.ndim != 2 or link_point_prompts.shape[-1] != 3:
        raise ValueError(
            f"Expected link_point_prompts to have shape (N, 3), got {link_point_prompts.shape}"
        )
    if len(link_point_prompts) != len(links):
        raise ValueError(
            "link_point_prompts must align with links, "
            f"got {len(link_point_prompts)} prompts for {len(links)} links"
        )
    link_point_prompt_ids = np.arange(len(links), dtype=np.int64)
    if not hide_unique_text_prompts:
        return link_point_prompts, link_point_prompt_ids
    if link_point_prompt_dropout_eligible is None:
        raise ValueError(
            "hide_unique_text_prompts=True requires link_point_prompt_dropout_eligible"
        )
    link_point_prompt_dropout_eligible = np.asarray(
        link_point_prompt_dropout_eligible,
        dtype=np.bool_,
    )
    if link_point_prompt_dropout_eligible.shape != (len(links),):
        raise ValueError(
            "link_point_prompt_dropout_eligible must align with links, "
            f"got {link_point_prompt_dropout_eligible.shape} for {len(links)} links"
        )
    visible_prompt_mask = ~link_point_prompt_dropout_eligible
    return (
        link_point_prompts[visible_prompt_mask],
        link_point_prompt_ids[visible_prompt_mask],
    )


def save_joint_overparam_visualization(
    output_dir: Path,
    *,
    query_points: np.ndarray,
    query_link_ids: np.ndarray,
    query_confidences: np.ndarray | None,
    links: Sequence[dict[str, Any]],
    joints: Sequence[dict[str, Any]],
    revolute_closest_axis_points: np.ndarray | None,
    revolute_low_points: np.ndarray | None,
    revolute_high_points: np.ndarray | None,
    prismatic_closest_axis_points: np.ndarray | None,
    prismatic_low_points: np.ndarray | None,
    prismatic_high_points: np.ndarray | None,
    num_points: int = 256,
) -> Path | None:
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")

    query_points = np.asarray(query_points, dtype=np.float32)
    query_link_ids = np.asarray(query_link_ids, dtype=np.int64)
    if query_points.shape != (len(query_link_ids), 3):
        raise ValueError(
            "query_points and query_link_ids must align, "
            f"got {query_points.shape} and {query_link_ids.shape}"
        )

    revolute_child_link_ids = np.asarray(
        [
            int(joint["child_link_id"])
            for joint in joints
            if bool(joint.get("is_revolute", False))
        ],
        dtype=np.int64,
    )
    prismatic_child_link_ids = np.asarray(
        [
            int(joint["child_link_id"])
            for joint in joints
            if bool(joint.get("is_prismatic", False))
        ],
        dtype=np.int64,
    )

    visualizations: list[dict[str, Any]] = []
    revolute_visualization = _collect_motion_overparam_visualization(
        motion_label="Revolute",
        query_points=query_points,
        query_link_ids=query_link_ids,
        query_confidences=query_confidences,
        valid_link_ids=revolute_child_link_ids,
        closest_axis_points=revolute_closest_axis_points,
        low_points=revolute_low_points,
        high_points=revolute_high_points,
        num_points=num_points,
    )
    if revolute_visualization is not None:
        visualizations.append(revolute_visualization)

    prismatic_visualization = _collect_motion_overparam_visualization(
        motion_label="Prismatic",
        query_points=query_points,
        query_link_ids=query_link_ids,
        query_confidences=query_confidences,
        valid_link_ids=prismatic_child_link_ids,
        closest_axis_points=prismatic_closest_axis_points,
        low_points=prismatic_low_points,
        high_points=prismatic_high_points,
        num_points=num_points,
    )
    if prismatic_visualization is not None:
        visualizations.append(prismatic_visualization)

    if len(visualizations) == 0:
        return None

    out_path = output_dir / "overparam.png"
    plot_joint_overparam_visualizations(
        visualizations=visualizations,
        links=list(links),
        out_path=out_path,
    )
    return out_path


def save_joint_overparam_visualization_from_model_output(
    output_dir: Path,
    *,
    output: Mapping[str, Any],
    query_points: np.ndarray,
    query_link_ids: np.ndarray,
    links: Sequence[dict[str, Any]],
    joints: Sequence[dict[str, Any]],
    center: np.ndarray | None = None,
    scale: float | None = None,
    num_points: int = 256,
) -> Path | None:
    if (center is None) != (scale is None):
        raise ValueError("center and scale must be provided together for denormalization")

    revolute_closest_axis_points = _output_array(output, "revolute_closest_axis_points")
    revolute_low_points = _output_array(output, "revolute_low_points")
    revolute_high_points = _output_array(output, "revolute_high_points")
    prismatic_closest_axis_points = _output_array(output, "prismatic_closest_axis_points")
    prismatic_low_points = _output_array(output, "prismatic_low_points")
    prismatic_high_points = _output_array(output, "prismatic_high_points")
    if center is not None and scale is not None:
        revolute_closest_axis_points = _denormalize_visualization_points(
            revolute_closest_axis_points,
            center=center,
            scale=scale,
        )
        revolute_low_points = _denormalize_visualization_points(
            revolute_low_points,
            center=center,
            scale=scale,
        )
        revolute_high_points = _denormalize_visualization_points(
            revolute_high_points,
            center=center,
            scale=scale,
        )
        prismatic_closest_axis_points = _denormalize_visualization_points(
            prismatic_closest_axis_points,
            center=center,
            scale=scale,
        )
        prismatic_low_points = _denormalize_visualization_points(
            prismatic_low_points,
            center=center,
            scale=scale,
        )
        prismatic_high_points = _denormalize_visualization_points(
            prismatic_high_points,
            center=center,
            scale=scale,
        )

    return save_joint_overparam_visualization(
        output_dir,
        query_points=query_points,
        query_link_ids=query_link_ids,
        query_confidences=_output_array(
            output,
            "joint_decoding_confidences",
            dtype=np.float32,
        ),
        links=links,
        joints=joints,
        revolute_closest_axis_points=revolute_closest_axis_points,
        revolute_low_points=revolute_low_points,
        revolute_high_points=revolute_high_points,
        prismatic_closest_axis_points=prismatic_closest_axis_points,
        prismatic_low_points=prismatic_low_points,
        prismatic_high_points=prismatic_high_points,
        num_points=num_points,
    )
