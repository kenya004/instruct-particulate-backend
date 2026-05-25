from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree


def compute_query_boundary_distance_cache(
    points: np.ndarray,
    link_ids: np.ndarray,
    *,
    output_dtype: np.dtype[np.floating] | type[np.floating] = np.float16,
    query_workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns per-point distances to every represented raw link.

    Args:
        points: Point array with shape `(N, 3)`.
        link_ids: Per-point raw link IDs with shape `(N,)`; unlabeled points may
            use `-1` and are still assigned distances to every represented raw
            link.
        output_dtype: Floating dtype used for the returned distance matrix.
        query_workers: Worker count forwarded to `cKDTree.query`.

    Returns:
        A tuple `(represented_link_ids, distance_to_raw_link)` where
        `represented_link_ids` has shape `(K,)` and
        `distance_to_raw_link` has shape `(N, K)`.
    """
    points = np.asarray(points, dtype=np.float32)
    link_ids = np.asarray(link_ids, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if link_ids.shape != (points.shape[0],):
        raise ValueError(
            f"link_ids must have shape ({points.shape[0]},), got {link_ids.shape}"
        )
    if query_workers == 0:
        raise ValueError("query_workers must be non-zero")

    represented_link_ids = np.unique(link_ids[link_ids >= 0]).astype(np.int64, copy=False)
    if represented_link_ids.size == 0:
        raise ValueError("At least one non-negative link_id is required to build boundary cache")

    distance_to_raw_link = np.empty(
        (points.shape[0], represented_link_ids.shape[0]),
        dtype=np.float32,
    )
    for column_index, raw_link_id in enumerate(represented_link_ids.tolist()):
        link_points = points[link_ids == raw_link_id]
        if link_points.shape[0] == 0:
            raise ValueError(
                "represented_link_ids includes a raw link without any points, "
                f"got raw_link_id={raw_link_id}"
            )
        tree = cKDTree(link_points)
        distance_to_raw_link[:, column_index] = tree.query(
            points,
            k=1,
            workers=query_workers,
        )[0].astype(np.float32, copy=False)
    return represented_link_ids, distance_to_raw_link.astype(output_dtype, copy=False)
