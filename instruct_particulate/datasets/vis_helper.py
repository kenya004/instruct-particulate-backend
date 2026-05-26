from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from instruct_particulate.utils.articulation_utils import (
    joint_has_prismatic_motion,
    joint_has_revolute_motion,
    plucker_to_axis_point,
)
from instruct_particulate.utils.visualization_utils import (
    ARROW_COLOR_PRISMATIC,
    ARROW_COLOR_REVOLUTE,
    COLORS,
)

LINK_POINT_PROMPT_COLOR = np.array([1.0, 0.0, 0.0], dtype=np.float32)
OVERPARAM_CONTEXT_QUERY_COLOR = np.full((3,), 0.85, dtype=np.float32)
VISUALIZATION_VIEW_ELEV_DEG = 20.0
VISUALIZATION_VIEW_AZIM_DEG = 225.0


def _rgb01(color: Sequence[float]) -> np.ndarray:
    """Converts an RGB color to the `[0, 1]` range."""
    rgb = np.asarray(color[:3], dtype=np.float32)
    if np.any(rgb > 1.0):
        rgb = rgb / 255.0
    return rgb


def _point_colors(link_ids: np.ndarray) -> np.ndarray:
    """Builds per-point colors from link IDs."""
    palette = np.asarray(COLORS, dtype=np.float32) / 255.0
    colors = np.full((len(link_ids), 3), 0.7, dtype=np.float32)
    valid_mask = link_ids >= 0
    if np.any(valid_mask):
        colors[valid_mask] = palette[link_ids[valid_mask] % len(palette)]
    return colors


def _set_equal_axes(ax: Any, cloud: np.ndarray) -> None:
    """Sets equal bounds for a 3D axis."""
    min_corner = cloud.min(axis=0)
    max_corner = cloud.max(axis=0)
    center = (min_corner + max_corner) * 0.5
    radius = max(np.ptp(cloud, axis=0).max() * 0.5, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def build_axis_segments(
    points: np.ndarray,
    point_link_ids: np.ndarray,
    joints: List[Dict[str, Any]],
    extension_ratio: float = 0.1,
    fallback_len_ratio: float = 0.25,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Builds line segments for visualizing joint axes.

    Args:
        points: Point cloud used to determine visualization scale.
        point_link_ids: Per-point link IDs aligned with `points`.
        joints: Joint records in the current pose.
        extension_ratio: Fraction of the child-link projected extent added past
            each end of the segment.
        fallback_len_ratio: Fallback segment length as a fraction of cloud span
            when the child link has no points or negligible extent.

    Returns:
        List of `(start, end, color)` axis segments.
    """
    span = np.ptp(points, axis=0).max() if len(points) else 0.0
    fallback_half_length = max(span * fallback_len_ratio, 1e-3)
    min_extension = max(span * 0.02, 1e-3)
    segments: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add_segment(link_id: int, axis: np.ndarray, axis_color: np.ndarray) -> None:
        axis = np.asarray(axis, dtype=np.float32)
        child_points = points[point_link_ids == link_id]
        if axis.shape[-1] == 6:
            axis_dir, axis_point = plucker_to_axis_point(axis)
        elif axis.shape[-1] == 3:
            axis_dir = axis
            axis_point = (
                child_points.mean(axis=0)
                if len(child_points)
                else np.zeros((3,), dtype=np.float32)
            )
        else:
            raise ValueError(f"Expected axis to have 3 or 6 values, got shape {axis.shape}")
        axis_norm = np.linalg.norm(axis_dir)
        if axis_norm <= 1e-8:
            return
        axis_dir = axis_dir / axis_norm
        if len(child_points) == 0:
            segment_start = axis_point - fallback_half_length * axis_dir
            segment_end = axis_point + fallback_half_length * axis_dir
        else:
            projections = np.dot(child_points - axis_point, axis_dir)
            proj_min = float(projections.min())
            proj_max = float(projections.max())
            proj_extent = proj_max - proj_min
            if proj_extent <= 1e-8:
                center_projection = float(projections.mean())
                extension = min_extension
                segment_start = axis_point + (center_projection - extension) * axis_dir
                segment_end = axis_point + (center_projection + extension) * axis_dir
            else:
                extension = max(proj_extent * extension_ratio, min_extension)
                segment_start = axis_point + (proj_min - extension) * axis_dir
                segment_end = axis_point + (proj_max + extension) * axis_dir
        segments.append((segment_start, segment_end, axis_color))

    for joint in sorted(joints, key=lambda item: int(item["joint_id"])):
        if not joint_has_revolute_motion(joint) and not joint_has_prismatic_motion(joint):
            continue

        link_id = int(joint["child_link_id"])
        if joint_has_revolute_motion(joint):
            add_segment(
                link_id,
                np.asarray(joint["revolute_axis"], dtype=np.float32),
                _rgb01(ARROW_COLOR_REVOLUTE),
            )
        if joint_has_prismatic_motion(joint):
            add_segment(
                link_id,
                np.asarray(joint["prismatic_axis"], dtype=np.float32),
                _rgb01(ARROW_COLOR_PRISMATIC),
            )

    return segments


def plot_query_visualization(
    *,
    query_points: np.ndarray,
    query_normals: np.ndarray,
    query_link_ids: np.ndarray,
    link_point_prompts: np.ndarray,
    link_point_prompt_ids: np.ndarray,
    links: List[Dict[str, Any]],
    out_path: Path,
    predicted_link_confidences: Optional[Mapping[int, float]] = None,
    low_query_points: Optional[np.ndarray] = None,
    high_query_points: Optional[np.ndarray] = None,
    low_link_point_prompts: Optional[np.ndarray] = None,
    high_link_point_prompts: Optional[np.ndarray] = None,
    axis_segments: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    normal_limit: int = 500,
) -> None:
    """Plots query points and optional articulated low/high positions.

    Args:
        query_points: Query points in the current frame.
        query_normals: Query normals in the current frame.
        query_link_ids: Per-query link IDs.
        link_point_prompts: Per-link prompt points.
        link_point_prompt_ids: Link IDs corresponding to `link_point_prompts`.
        links: Link metadata used for the legend.
        predicted_link_confidences: Optional per-link prediction confidences for the legend.
        out_path: Output image path.
        low_query_points: Optional low-limit articulated query points.
        high_query_points: Optional high-limit articulated query points.
        low_link_point_prompts: Optional low-limit articulated prompt points.
        high_link_point_prompts: Optional high-limit articulated prompt points.
        axis_segments: Optional axis line segments for the original panel.
        normal_limit: Maximum number of query normals to render as arrows.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    point_colors = _point_colors(query_link_ids)
    palette = np.asarray(COLORS, dtype=np.float32) / 255.0
    all_clouds = [query_points, link_point_prompts]
    titles = ["Query points"]

    if low_query_points is not None and high_query_points is not None:
        all_clouds.extend([low_query_points, high_query_points])
        if low_link_point_prompts is not None:
            all_clouds.append(low_link_point_prompts)
        if high_link_point_prompts is not None:
            all_clouds.append(high_link_point_prompts)
        titles.extend(["Low-limit pose", "High-limit pose"])

    bounds_cloud = np.concatenate(all_clouds, axis=0)
    num_panels = len(titles)
    fig = plt.figure(figsize=(6 * num_panels, 6))

    plot_specs = [
        ("Query points", query_points, link_point_prompts, True, True),
    ]
    if low_query_points is not None and high_query_points is not None:
        plot_specs.extend(
            [
                ("Low-limit pose", low_query_points, low_link_point_prompts, False, False),
                ("High-limit pose", high_query_points, high_link_point_prompts, False, False),
            ]
        )

    def render_panel(
        ax: Any,
        title: str,
        cloud: np.ndarray,
        prompt_cloud: Optional[np.ndarray],
        draw_axes: bool,
        draw_normals: bool,
    ) -> None:
        ax.scatter(
            cloud[:, 0],
            cloud[:, 1],
            cloud[:, 2],
            c=point_colors,
            s=14,
            linewidths=0,
            depthshade=False,
            zorder=1,
        )
        if draw_normals and len(cloud) and normal_limit > 0:
            normal_count = min(normal_limit, len(cloud))
            normal_indices = np.linspace(0, len(cloud) - 1, num=normal_count, dtype=int)
            normal_scale = max(np.ptp(bounds_cloud, axis=0).max() * 0.04, 1e-3)
            ax.quiver(
                cloud[normal_indices, 0],
                cloud[normal_indices, 1],
                cloud[normal_indices, 2],
                query_normals[normal_indices, 0],
                query_normals[normal_indices, 1],
                query_normals[normal_indices, 2],
                length=normal_scale,
                normalize=True,
                color="black",
                linewidth=0.5,
                alpha=0.6,
                zorder=2,
            )
        if prompt_cloud is not None and len(prompt_cloud):
            ax.scatter(
                prompt_cloud[:, 0],
                prompt_cloud[:, 1],
                prompt_cloud[:, 2],
                color=LINK_POINT_PROMPT_COLOR,
                s=150,
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                depthshade=False,
                zorder=3,
            )
        if draw_axes and axis_segments is not None:
            for start_point, end_point, axis_color in axis_segments:
                ax.plot(
                    [start_point[0], end_point[0]],
                    [start_point[1], end_point[1]],
                    [start_point[2], end_point[2]],
                    color=axis_color,
                    lw=1.5,
                    alpha=0.9,
                    zorder=2,
                )
        _set_equal_axes(ax, bounds_cloud)
        ax.view_init(
            elev=VISUALIZATION_VIEW_ELEV_DEG,
            azim=VISUALIZATION_VIEW_AZIM_DEG,
        )
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    for panel_idx, plot_spec in enumerate(plot_specs, start=1):
        render_panel(
            fig.add_subplot(1, num_panels, panel_idx, projection="3d", computed_zorder=False),
            *plot_spec,
        )

    has_any_prompt_points = any(
        prompt_cloud is not None and len(prompt_cloud) > 0
        for prompt_cloud in (
            link_point_prompts,
            low_link_point_prompts,
            high_link_point_prompts,
        )
    )
    legend_handles = []
    if has_any_prompt_points:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="None",
                markersize=10,
                markerfacecolor=LINK_POINT_PROMPT_COLOR,
                markeredgecolor="black",
                label="Link point prompt",
            )
        )
    for link in sorted(links, key=lambda item: int(item["link_id"])):
        link_id = int(link["link_id"])
        color = palette[link_id % len(palette)]
        link_name = str(link.get("name", f"link_{link_id}"))
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor=color,
                markeredgecolor=color,
                label=f"{link_id}: {link_name}",
            )
        )

    fig.legend(handles=legend_handles, loc="lower center", ncol=min(4, len(legend_handles)), frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_joint_overparam_visualizations(
    visualizations: List[Dict[str, Any]],
    links: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    """Plots one or more motion-specific over-parameterized visualizations in one figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if len(visualizations) == 0:
        return

    palette = np.asarray(COLORS, dtype=np.float32) / 255.0
    plot_specs = [
        ("Closest-axis targets", "closest_axis_points", "closest_axis_points_valid", "x"),
        ("Low-pose targets", "low_points", "low_points_valid", "^"),
        ("High-pose targets", "high_points", "high_points_valid", "s"),
    ]
    normalized_visualizations: List[Dict[str, Any]] = []
    for visualization in visualizations:
        query_points = np.asarray(visualization["query_points"], dtype=np.float32)
        if len(query_points) == 0:
            continue
        query_link_ids = np.asarray(visualization["query_link_ids"], dtype=np.int64)
        expected_shape = (len(query_points), 3)
        raw_point_colors = visualization.get("point_colors")
        if raw_point_colors is None:
            point_colors = _point_colors(query_link_ids)
        else:
            point_colors = np.asarray(raw_point_colors, dtype=np.float32)
            if point_colors.shape != expected_shape:
                raise ValueError(
                    f"point_colors must have shape {expected_shape}, got {tuple(point_colors.shape)}"
                )
        normalized_visualization = {
            "motion_label": str(visualization.get("motion_label", "")).strip(),
            "query_points": query_points,
            "query_link_ids": query_link_ids,
            "point_colors": point_colors,
        }
        bounds_clouds = [query_points]
        for _, key, valid_key, _ in plot_specs:
            value = visualization.get(key)
            if value is None:
                value_array = np.zeros(expected_shape, dtype=np.float32)
                valid_mask = np.zeros((len(query_points),), dtype=np.bool_)
            else:
                value_array = np.asarray(value, dtype=np.float32)
                if value_array.shape != expected_shape:
                    raise ValueError(
                        f"{key} must have shape {expected_shape}, got {tuple(value_array.shape)}"
                    )
                raw_valid_mask = visualization.get(valid_key)
                if raw_valid_mask is None:
                    valid_mask = np.ones((len(query_points),), dtype=np.bool_)
                else:
                    valid_mask = np.asarray(raw_valid_mask, dtype=np.bool_)
                    if valid_mask.shape != (len(query_points),):
                        raise ValueError(
                            f"{valid_key} must have shape {(len(query_points),)}, "
                            f"got {tuple(valid_mask.shape)}"
                        )
            normalized_visualization[key] = value_array
            normalized_visualization[valid_key] = valid_mask
            if np.any(valid_mask):
                bounds_clouds.append(value_array[valid_mask])
        context_query_points = visualization.get("context_query_points")
        if context_query_points is not None and len(context_query_points):
            context_query_points = np.asarray(context_query_points, dtype=np.float32)
            normalized_visualization["context_query_points"] = context_query_points
            bounds_clouds.append(context_query_points)
        else:
            normalized_visualization["context_query_points"] = None
        normalized_visualization["bounds_cloud"] = np.concatenate(bounds_clouds, axis=0)
        normalized_visualizations.append(normalized_visualization)

    if len(normalized_visualizations) == 0:
        return

    num_rows = len(normalized_visualizations)
    num_cols = len(plot_specs)
    fig = plt.figure(figsize=(6 * num_cols, 6 * num_rows))

    def render_panel(
        ax: Any,
        *,
        motion_label: str,
        query_points: np.ndarray,
        point_colors: np.ndarray,
        target_points: np.ndarray,
        target_valid_mask: np.ndarray,
        target_title: str,
        target_marker: str,
        bounds_cloud: np.ndarray,
        context_query_points: Optional[np.ndarray],
    ) -> None:
        title = f"{motion_label}: {target_title}" if motion_label else target_title
        query_alpha = 0.45
        visible_query_mask = target_valid_mask
        if context_query_points is not None and len(context_query_points):
            ax.scatter(
                context_query_points[:, 0],
                context_query_points[:, 1],
                context_query_points[:, 2],
                color=OVERPARAM_CONTEXT_QUERY_COLOR,
                s=10,
                linewidths=0,
                alpha=0.65,
                depthshade=False,
                zorder=0,
            )
        if np.any(visible_query_mask):
            ax.scatter(
                query_points[visible_query_mask, 0],
                query_points[visible_query_mask, 1],
                query_points[visible_query_mask, 2],
                c=point_colors[visible_query_mask],
                s=28,
                linewidths=0,
                alpha=query_alpha,
                depthshade=False,
                zorder=1,
            )
        if np.any(target_valid_mask):
            valid_query_points = query_points[target_valid_mask]
            valid_target_points = target_points[target_valid_mask]
            valid_point_colors = point_colors[target_valid_mask]
            for point, target_point, color in zip(
                valid_query_points,
                valid_target_points,
                valid_point_colors,
                strict=True,
            ):
                ax.plot(
                    [point[0], target_point[0]],
                    [point[1], target_point[1]],
                    [point[2], target_point[2]],
                    color=color,
                    alpha=0.35,
                    lw=0.8,
                    zorder=1,
                )
            ax.scatter(
                valid_target_points[:, 0],
                valid_target_points[:, 1],
                valid_target_points[:, 2],
                c=valid_point_colors,
                s=36,
                marker=target_marker,
                linewidths=1.0,
                depthshade=False,
                zorder=2,
            )
            if not np.all(target_valid_mask):
                ax.text2D(
                    0.03,
                    0.97,
                    "masked targets hidden",
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top",
                )
        else:
            ax.text2D(
                0.03,
                0.97,
                "no visible target",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
            )
        _set_equal_axes(ax, bounds_cloud)
        ax.view_init(
            elev=VISUALIZATION_VIEW_ELEV_DEG,
            azim=VISUALIZATION_VIEW_AZIM_DEG,
        )
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    for row_idx, visualization in enumerate(normalized_visualizations):
        for col_idx, (target_title, target_key, valid_key, target_marker) in enumerate(plot_specs):
            render_panel(
                fig.add_subplot(
                    num_rows,
                    num_cols,
                    row_idx * num_cols + col_idx + 1,
                    projection="3d",
                    computed_zorder=False,
                ),
                motion_label=visualization["motion_label"],
                query_points=visualization["query_points"],
                point_colors=visualization["point_colors"],
                target_points=visualization[target_key],
                target_valid_mask=visualization[valid_key],
                target_title=target_title,
                target_marker=target_marker,
                bounds_cloud=visualization["bounds_cloud"],
                context_query_points=visualization["context_query_points"],
            )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=OVERPARAM_CONTEXT_QUERY_COLOR,
            markeredgecolor=OVERPARAM_CONTEXT_QUERY_COLOR,
            label="Other query point",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="black",
            label="Current query point",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            markersize=7,
            color="black",
            label="Closest-axis target",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            markersize=7,
            color="black",
            label="Low-pose target",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=7,
            color="black",
            label="High-pose target",
        ),
    ]
    for link in sorted(links, key=lambda item: int(item["link_id"])):
        link_id = int(link["link_id"])
        color = palette[link_id % len(palette)]
        link_name = str(link.get("name", f"link_{link_id}"))
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor=color,
                markeredgecolor=color,
                label=f"{link_id}: {link_name}",
            )
        )

    fig.legend(handles=legend_handles, loc="lower center", ncol=min(4, len(legend_handles)), frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

