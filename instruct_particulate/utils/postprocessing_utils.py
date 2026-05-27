import os
import time
import heapq
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import trimesh
from numba import njit, prange
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


DEFAULT_FACE_GROUP_DISTANCE_THRESHOLD = 1e-4


def _get_face_centroids(mesh):
    """Return face centroids in a compact dtype for NN queries."""
    return np.asarray(mesh.triangles_center, dtype=np.float32)


def _query_nearest(tree, query_points):
    """
    Query nearest neighbors with all available CPU workers.

    SciPy exposes ``workers`` in cKDTree.query; using -1 lets it parallelize.
    """
    return tree.query(query_points, k=1, workers=-1)


def _mesh_face_connected_components(mesh):
    """Return mesh face connected components as dense int64 arrays."""
    components = trimesh.graph.connected_components(
        edges=mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces)),
        min_len=1,
    )
    return [np.array(list(component), dtype=np.int64) for component in components]


def assign_undefined_faces_to_nearest_defined(mesh, face_part_ids):
    """
    Fill undefined (-1) face labels by nearest defined face label.

    For scalability this uses a KD-tree over defined face centroids.
    """
    face_part_ids_filled = face_part_ids.copy()
    undefined_faces = np.flatnonzero(face_part_ids_filled == -1)
    if len(undefined_faces) == 0:
        return face_part_ids_filled

    defined_faces = np.flatnonzero(face_part_ids_filled != -1)
    if len(defined_faces) == 0:
        return face_part_ids_filled

    centroids = _get_face_centroids(mesh)
    tree = cKDTree(centroids[defined_faces])
    _, nearest_local = _query_nearest(tree, centroids[undefined_faces])
    nearest_local = np.atleast_1d(nearest_local)
    nearest_defined_faces = defined_faces[nearest_local]

    face_part_ids_filled[undefined_faces] = face_part_ids_filled[nearest_defined_faces]
    return face_part_ids_filled


def refine_part_ids_strict(mesh, face_part_ids):
    """
    Refine face part IDs by treating each connected component (CC) independently.

    For each CC:
    - If it has any defined labels, all faces are overwritten with the dominant
      part ID by surface area.
    - If all faces are undefined (-1), assign all faces from the nearest defined CC.
    
    Args:
        mesh: trimesh object
        face_part_ids: part ID for each face [num_faces]
    
    Returns:
        refined_face_part_ids: refined part ID for each face [num_faces]
    """
    face_part_ids = np.asarray(face_part_ids, dtype=np.int32).copy()
    mesh_components = _mesh_face_connected_components(mesh)

    component_dominant_part_id = {}
    undefined_components = []

    # For each connected component, find the dominant part ID by surface area.
    for comp_idx, component in enumerate(mesh_components):
        if len(component) == 0:
            continue

        component_part_ids = face_part_ids[component]
        valid_mask = component_part_ids != -1
        if not np.any(valid_mask):
            undefined_components.append(comp_idx)
            continue

        valid_part_ids = component_part_ids[valid_mask].astype(np.int64)
        valid_face_areas = mesh.area_faces[component[valid_mask]]
        unique_part_ids, inverse = np.unique(valid_part_ids, return_inverse=True)
        part_area_sums = np.bincount(inverse, weights=valid_face_areas)
        dominant_part_id = int(unique_part_ids[np.argmax(part_area_sums)])

        component_dominant_part_id[comp_idx] = dominant_part_id
        face_part_ids[component] = dominant_part_id

    # Components that are entirely undefined are assigned from the nearest
    # component that has a defined dominant part label.
    if undefined_components and component_dominant_part_id:
        centroids = _get_face_centroids(mesh)
        face_to_component = np.full(len(mesh.faces), -1, dtype=np.int32)
        defined_face_chunks = []
        undefined_face_chunks = []
        for comp_idx in component_dominant_part_id.keys():
            comp_faces = mesh_components[comp_idx]
            face_to_component[comp_faces] = comp_idx
            defined_face_chunks.append(comp_faces)
        for comp_idx in undefined_components:
            comp_faces = mesh_components[comp_idx]
            face_to_component[comp_faces] = comp_idx
            undefined_face_chunks.append(comp_faces)

        defined_faces = np.concatenate(defined_face_chunks, axis=0)
        undefined_faces = np.concatenate(undefined_face_chunks, axis=0)
        tree = cKDTree(centroids[defined_faces])
        nearest_dist, nearest_local = _query_nearest(tree, centroids[undefined_faces])
        nearest_local = np.atleast_1d(nearest_local)
        nearest_dist = np.atleast_1d(nearest_dist)

        undefined_face_components = face_to_component[undefined_faces]
        nearest_defined_faces = defined_faces[nearest_local]
        nearest_defined_components = face_to_component[nearest_defined_faces]

        order = np.argsort(undefined_face_components, kind="mergesort")
        sorted_undefined_comps = undefined_face_components[order]
        sorted_dists = nearest_dist[order]
        sorted_nearest_defined_comps = nearest_defined_components[order]

        unique_undefined_comps, group_start = np.unique(sorted_undefined_comps, return_index=True)
        group_end = np.concatenate([group_start[1:], np.array([len(sorted_undefined_comps)])])

        for comp_idx, start, end in zip(unique_undefined_comps, group_start, group_end):
            best_local = start + int(np.argmin(sorted_dists[start:end]))
            nearest_defined_comp = int(sorted_nearest_defined_comps[best_local])
            face_part_ids[mesh_components[int(comp_idx)]] = component_dominant_part_id[nearest_defined_comp]
    
    return face_part_ids


def _majority_vote_face_part_ids(mesh, part_ids, face_indices):
    """Assigns each sampled face the majority label of its query points.

    Faces that never received a sampled query point remain `-1`.

    Args:
        mesh: trimesh object
        part_ids: part IDs for each sampled point [num_points]
        face_indices: which face each point lies on (-1 means on edge) [num_points]

    Returns:
        Face labels with unresolved faces marked as `-1`.
    """
    num_faces = len(mesh.faces)
    face_part_ids = np.full(num_faces, -1, dtype=np.int32)

    face_to_points = {}
    for point_idx, face_idx in enumerate(face_indices):
        if face_idx == -1:
            continue
        if face_idx not in face_to_points:
            face_to_points[face_idx] = []
        face_to_points[face_idx].append(part_ids[point_idx])

    for face_idx, point_part_ids in face_to_points.items():
        counts = np.bincount(point_part_ids)
        majority_part_id = np.argmax(counts)
        face_part_ids[face_idx] = majority_part_id
    return face_part_ids


def find_unrefined_part_ids_for_faces(mesh, part_ids, face_indices):
    """Builds the face labels used before connected-component refinement.

    This matches the user's requested "unrefined" representation:
    1. majority-vote query labels per sampled face
    2. nearest-neighbor fill for unsampled faces

    Args:
        mesh: trimesh object
        part_ids: part IDs for each sampled point [num_points]
        face_indices: which face each point lies on (-1 means on edge) [num_points]

    Returns:
        Face labels after majority vote plus nearest-face fill.
    """
    initial_face_part_ids = _majority_vote_face_part_ids(mesh, part_ids, face_indices)
    return assign_undefined_faces_to_nearest_defined(mesh, initial_face_part_ids)


def refine_face_part_ids(mesh, face_part_ids, strict=False):
    """Apply the base face-label post-processing stage.

    Args:
        mesh: trimesh object
        face_part_ids: Face labels after the unrefined majority-vote stage.
        strict: Whether to use strict refinement. When False, the unrefined
            labels are returned unchanged.

    Returns:
        Base per-face part IDs used for the final segmentation export.
    """
    if strict:
        return refine_part_ids_strict(mesh, face_part_ids)
    return np.asarray(face_part_ids, dtype=np.int32).copy()


def point_to_triangle_distance_batch(points_batch, tri_verts_batch):
    """Compute squared distances from batched points to batched triangles."""
    v0 = tri_verts_batch[:, 0, :]
    v1 = tri_verts_batch[:, 1, :]
    v2 = tri_verts_batch[:, 2, :]

    edge0 = v1 - v0
    edge1 = v2 - v0
    normals = np.cross(edge0, edge1)
    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)

    valid_mask = normal_norms[:, 0] >= 1e-10
    normals = normals / np.maximum(normal_norms, 1e-10)

    to_points = points_batch - v0[:, np.newaxis, :]
    dist_to_plane = np.einsum("mnk,mk->mn", to_points, normals)
    points_on_plane = (
        points_batch - dist_to_plane[:, :, np.newaxis] * normals[:, np.newaxis, :]
    )

    v = points_on_plane - v0[:, np.newaxis, :]
    d00 = np.einsum("mk,mk->m", edge0, edge0)
    d01 = np.einsum("mk,mk->m", edge0, edge1)
    d11 = np.einsum("mk,mk->m", edge1, edge1)
    d20 = np.einsum("mnk,mk->mn", v, edge0)
    d21 = np.einsum("mnk,mk->mn", v, edge1)
    denom = d00 * d11 - d01 * d01

    valid_denom = np.abs(denom) >= 1e-10
    denom = np.where(valid_denom, denom, 1.0)[:, np.newaxis]

    bary_v = (d11[:, np.newaxis] * d20 - d01[:, np.newaxis] * d21) / denom
    bary_w = (d00[:, np.newaxis] * d21 - d01[:, np.newaxis] * d20) / denom
    bary_u = 1.0 - bary_v - bary_w

    inside_mask = (bary_u >= -1e-10) & (bary_v >= -1e-10) & (bary_w >= -1e-10)
    inside_mask = inside_mask & valid_mask[:, np.newaxis] & valid_denom[:, np.newaxis]

    distances_sq = dist_to_plane * dist_to_plane
    outside_mask = ~inside_mask

    if np.any(outside_mask):
        edge = edge0
        edge_len_sq = d00
        ap = points_batch - v0[:, np.newaxis, :]
        t = np.clip(
            np.einsum("mnk,mk->mn", ap, edge)
            / np.maximum(edge_len_sq[:, np.newaxis], 1e-10),
            0,
            1,
        )
        proj = v0[:, np.newaxis, :] + t[:, :, np.newaxis] * edge[:, np.newaxis, :]
        diff = points_batch - proj
        dist_edge0_sq = np.einsum("mnk,mnk->mn", diff, diff)

        edge = v2 - v1
        edge_len_sq = np.einsum("mk,mk->m", edge, edge)
        ap = points_batch - v1[:, np.newaxis, :]
        t = np.clip(
            np.einsum("mnk,mk->mn", ap, edge)
            / np.maximum(edge_len_sq[:, np.newaxis], 1e-10),
            0,
            1,
        )
        proj = v1[:, np.newaxis, :] + t[:, :, np.newaxis] * edge[:, np.newaxis, :]
        diff = points_batch - proj
        dist_edge1_sq = np.einsum("mnk,mnk->mn", diff, diff)

        edge = v0 - v2
        edge_len_sq = np.einsum("mk,mk->m", edge, edge)
        ap = points_batch - v2[:, np.newaxis, :]
        t = np.clip(
            np.einsum("mnk,mk->mn", ap, edge)
            / np.maximum(edge_len_sq[:, np.newaxis], 1e-10),
            0,
            1,
        )
        proj = v2[:, np.newaxis, :] + t[:, :, np.newaxis] * edge[:, np.newaxis, :]
        diff = points_batch - proj
        dist_edge2_sq = np.einsum("mnk,mnk->mn", diff, diff)

        min_edge_dist_sq = np.minimum(
            dist_edge0_sq,
            np.minimum(dist_edge1_sq, dist_edge2_sq),
        )
        distances_sq = np.where(outside_mask, min_edge_dist_sq, distances_sq)

    return distances_sq


def resolve_point_prompt_face_ids(
    mesh,
    point_prompts,
    *,
    exact_batch_size=8192,
):
    """Resolve each point prompt to the nearest mesh face in the same coordinate frame.

    This is used for backward compatibility when saved prompt-face IDs are unavailable.
    The implementation is exact but keeps the work bounded by:
    - initializing the best candidate from a face-centroid KD-tree
    - pruning exact triangle checks with face AABB lower bounds
    """
    point_prompts = np.asarray(point_prompts, dtype=np.float32)
    if point_prompts.ndim != 2 or point_prompts.shape[1] != 3:
        raise ValueError(
            "point_prompts must have shape (num_prompts, 3), "
            f"got {point_prompts.shape}"
        )
    if point_prompts.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    face_verts = np.asarray(mesh.triangles, dtype=np.float64)
    if face_verts.shape[0] == 0:
        raise ValueError("cannot resolve point-prompt faces on a mesh with zero faces")
    bbox_mins = np.min(face_verts, axis=1)
    bbox_maxs = np.max(face_verts, axis=1)
    face_centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    centroid_tree = cKDTree(face_centroids)

    face_ids = np.zeros((point_prompts.shape[0],), dtype=np.int64)
    all_face_ids = np.arange(len(face_verts), dtype=np.int64)

    for prompt_idx, point_prompt in enumerate(point_prompts.astype(np.float64, copy=False)):
        _, seed_face_local = _query_nearest(centroid_tree, point_prompt[None, :])
        seed_face_id = int(np.atleast_1d(seed_face_local)[0])
        best_face_id = seed_face_id
        best_sq = float(
            point_to_triangle_distance_batch(
                np.broadcast_to(point_prompt, (1, 1, 3)),
                face_verts[[seed_face_id]],
            )[0, 0]
        )

        axis_gap = np.maximum(
            np.maximum(bbox_mins - point_prompt[None, :], point_prompt[None, :] - bbox_maxs),
            0.0,
        )
        lower_bound_sq = np.einsum("ij,ij->i", axis_gap, axis_gap)
        candidate_face_ids = all_face_ids[lower_bound_sq < best_sq]
        if candidate_face_ids.size == 0:
            face_ids[prompt_idx] = best_face_id
            continue

        for start in range(0, len(candidate_face_ids), int(exact_batch_size)):
            batch_face_ids = candidate_face_ids[start:start + int(exact_batch_size)]
            batch_dist_sq = point_to_triangle_distance_batch(
                np.broadcast_to(point_prompt, (len(batch_face_ids), 1, 3)),
                face_verts[batch_face_ids],
            )[:, 0]
            batch_best_local = int(np.argmin(batch_dist_sq))
            batch_best_sq = float(batch_dist_sq[batch_best_local])
            if batch_best_sq < best_sq:
                best_sq = batch_best_sq
                best_face_id = int(batch_face_ids[batch_best_local])

        face_ids[prompt_idx] = best_face_id

    return face_ids


def segment_segment_distance_sq_batch(p1, q1, p2, q2):
    """Compute squared distances between batched 3D line segments."""
    eps = 1e-12

    u = q1 - p1
    v = q2 - p2
    w = p1 - p2

    a = np.einsum("ij,ij->i", u, u)
    b = np.einsum("ij,ij->i", u, v)
    c = np.einsum("ij,ij->i", v, v)
    d = np.einsum("ij,ij->i", u, w)
    e = np.einsum("ij,ij->i", v, w)
    det = a * c - b * b

    s_n = np.empty_like(det)
    t_n = np.empty_like(det)
    s_d = np.empty_like(det)
    t_d = np.empty_like(det)

    parallel_mask = det < eps
    non_parallel_mask = ~parallel_mask

    s_n[parallel_mask] = 0.0
    s_d[parallel_mask] = 1.0
    t_n[parallel_mask] = e[parallel_mask]
    t_d[parallel_mask] = c[parallel_mask]

    s_n[non_parallel_mask] = (
        b[non_parallel_mask] * e[non_parallel_mask]
        - c[non_parallel_mask] * d[non_parallel_mask]
    )
    t_n[non_parallel_mask] = (
        a[non_parallel_mask] * e[non_parallel_mask]
        - b[non_parallel_mask] * d[non_parallel_mask]
    )
    s_d[non_parallel_mask] = det[non_parallel_mask]
    t_d[non_parallel_mask] = det[non_parallel_mask]

    mask = non_parallel_mask & (s_n < 0.0)
    s_n[mask] = 0.0
    t_n[mask] = e[mask]
    t_d[mask] = c[mask]

    mask = non_parallel_mask & (s_n > s_d)
    s_n[mask] = s_d[mask]
    t_n[mask] = e[mask] + b[mask]
    t_d[mask] = c[mask]

    mask = t_n < 0.0
    t_n[mask] = 0.0
    s_n[mask] = -d[mask]
    s_d[mask] = a[mask]

    mask2 = mask & (s_n < 0.0)
    s_n[mask2] = 0.0
    mask2 = mask & (s_n > s_d)
    s_n[mask2] = s_d[mask2]

    mask = t_n > t_d
    t_n[mask] = t_d[mask]
    s_n[mask] = -d[mask] + b[mask]
    s_d[mask] = a[mask]

    mask2 = mask & (s_n < 0.0)
    s_n[mask2] = 0.0
    mask2 = mask & (s_n > s_d)
    s_n[mask2] = s_d[mask2]

    sc = np.zeros_like(s_n)
    tc = np.zeros_like(t_n)
    valid_s = np.abs(s_d) > eps
    valid_t = np.abs(t_d) > eps
    sc[valid_s] = s_n[valid_s] / s_d[valid_s]
    tc[valid_t] = t_n[valid_t] / t_d[valid_t]

    delta = w + sc[:, np.newaxis] * u - tc[:, np.newaxis] * v
    return np.einsum("ij,ij->i", delta, delta)


def segment_intersects_triangle_batch(seg_start, seg_end, tri_verts_batch, eps=1e-10):
    """Test batched segment-triangle intersections."""
    direction = seg_end - seg_start

    v0 = tri_verts_batch[:, 0, :]
    v1 = tri_verts_batch[:, 1, :]
    v2 = tri_verts_batch[:, 2, :]
    edge1 = v1 - v0
    edge2 = v2 - v0

    pvec = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    non_parallel = np.abs(det) > eps

    inv_det = np.zeros_like(det)
    inv_det[non_parallel] = 1.0 / det[non_parallel]

    tvec = seg_start - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det

    qvec = np.cross(tvec, edge1)
    v = np.einsum("ij,ij->i", direction, qvec) * inv_det
    t = np.einsum("ij,ij->i", edge2, qvec) * inv_det

    return (
        non_parallel
        & (u >= -eps)
        & (v >= -eps)
        & (u + v <= 1.0 + eps)
        & (t >= -eps)
        & (t <= 1.0 + eps)
    )


def triangle_pairs_within_threshold_batch(tri_a_batch, tri_b_batch, threshold_sq):
    """Return mask of triangle pairs with exact distance < threshold."""
    num_pairs = len(tri_a_batch)
    if num_pairs == 0:
        return np.zeros(0, dtype=bool)

    adjacent = np.zeros(num_pairs, dtype=bool)
    edge_indices = ((0, 1), (1, 2), (2, 0))

    min_vv_sq = np.full(num_pairs, np.inf, dtype=tri_a_batch.dtype)
    for ia in range(3):
        pa = tri_a_batch[:, ia, :]
        for ib in range(3):
            pb = tri_b_batch[:, ib, :]
            diff = pa - pb
            vv_sq = np.einsum("ij,ij->i", diff, diff)
            min_vv_sq = np.minimum(min_vv_sq, vv_sq)
    adjacent |= min_vv_sq < threshold_sq

    remaining_mask = ~adjacent
    if not np.any(remaining_mask):
        return adjacent

    remaining_idx = np.flatnonzero(remaining_mask)
    tri_a_rem = tri_a_batch[remaining_idx]
    tri_b_rem = tri_b_batch[remaining_idx]

    d_a_to_b_sq = point_to_triangle_distance_batch(tri_a_rem, tri_b_rem)
    d_b_to_a_sq = point_to_triangle_distance_batch(tri_b_rem, tri_a_rem)
    min_pt_sq = np.minimum(
        np.min(d_a_to_b_sq, axis=1),
        np.min(d_b_to_a_sq, axis=1),
    )
    pt_adjacent = min_pt_sq < threshold_sq
    if np.any(pt_adjacent):
        adjacent[remaining_idx[pt_adjacent]] = True

    remaining_mask = ~adjacent
    if not np.any(remaining_mask):
        return adjacent

    remaining_idx = np.flatnonzero(remaining_mask)
    tri_a_rem = tri_a_batch[remaining_idx]
    tri_b_rem = tri_b_batch[remaining_idx]
    min_edge_sq = np.full(len(remaining_idx), np.inf, dtype=tri_a_rem.dtype)

    for a0, a1 in edge_indices:
        p1 = tri_a_rem[:, a0, :]
        q1 = tri_a_rem[:, a1, :]
        for b0, b1 in edge_indices:
            p2 = tri_b_rem[:, b0, :]
            q2 = tri_b_rem[:, b1, :]
            edge_dist_sq = segment_segment_distance_sq_batch(p1, q1, p2, q2)
            min_edge_sq = np.minimum(min_edge_sq, edge_dist_sq)

    edge_adjacent = min_edge_sq < threshold_sq
    if np.any(edge_adjacent):
        adjacent[remaining_idx[edge_adjacent]] = True

    remaining_mask = ~adjacent
    if not np.any(remaining_mask):
        return adjacent

    remaining_idx = np.flatnonzero(remaining_mask)
    tri_a_rem = tri_a_batch[remaining_idx]
    tri_b_rem = tri_b_batch[remaining_idx]
    intersects = np.zeros(len(remaining_idx), dtype=bool)

    for a0, a1 in edge_indices:
        intersects |= segment_intersects_triangle_batch(
            tri_a_rem[:, a0, :],
            tri_a_rem[:, a1, :],
            tri_b_rem,
        )
        intersects |= segment_intersects_triangle_batch(
            tri_b_rem[:, a0, :],
            tri_b_rem[:, a1, :],
            tri_a_rem,
        )

    if np.any(intersects):
        adjacent[remaining_idx[intersects]] = True

    return adjacent


def _triangle_pair_distance_sq_batch(tri_a_batch, tri_b_batch):
    """Compute exact triangle-triangle squared distances for a batch."""
    num_pairs = len(tri_a_batch)
    if num_pairs == 0:
        return np.zeros(0, dtype=np.float64)

    edge_indices = ((0, 1), (1, 2), (2, 0))
    min_sq = np.full(num_pairs, np.inf, dtype=np.float64)

    d_a_to_b_sq = point_to_triangle_distance_batch(tri_a_batch, tri_b_batch)
    d_b_to_a_sq = point_to_triangle_distance_batch(tri_b_batch, tri_a_batch)
    min_sq = np.minimum(min_sq, np.min(d_a_to_b_sq, axis=1))
    min_sq = np.minimum(min_sq, np.min(d_b_to_a_sq, axis=1))

    for a0, a1 in edge_indices:
        p1 = tri_a_batch[:, a0, :]
        q1 = tri_a_batch[:, a1, :]
        for b0, b1 in edge_indices:
            p2 = tri_b_batch[:, b0, :]
            q2 = tri_b_batch[:, b1, :]
            min_sq = np.minimum(
                min_sq,
                segment_segment_distance_sq_batch(p1, q1, p2, q2),
            )

    intersects = np.zeros(num_pairs, dtype=bool)
    for a0, a1 in edge_indices:
        intersects |= segment_intersects_triangle_batch(
            tri_a_batch[:, a0, :],
            tri_a_batch[:, a1, :],
            tri_b_batch,
        )
        intersects |= segment_intersects_triangle_batch(
            tri_b_batch[:, a0, :],
            tri_b_batch[:, a1, :],
            tri_a_batch,
        )
    min_sq[intersects] = 0.0
    return min_sq


@njit(cache=True, parallel=True)
def generate_candidate_pairs_sweep_numba(
    order,
    mins_a,
    maxs_a,
    mins_b,
    maxs_b,
    upper_bounds,
    distance_threshold,
):
    """Generate exact bbox candidate pairs using the reviewed parallel sweep-line logic."""
    n_faces = len(order)

    counts = np.zeros(n_faces, dtype=np.int64)
    for i in prange(n_faces):
        upper_bound = upper_bounds[i]
        if upper_bound <= i + 1:
            continue

        min_ai = mins_a[i]
        max_ai = maxs_a[i]
        min_bi = mins_b[i]
        max_bi = maxs_b[i]

        local_count = 0
        for j in range(i + 1, upper_bound):
            if min_ai - maxs_a[j] >= distance_threshold:
                continue
            if mins_a[j] - max_ai >= distance_threshold:
                continue
            if min_bi - maxs_b[j] >= distance_threshold:
                continue
            if mins_b[j] - max_bi >= distance_threshold:
                continue
            local_count += 1

        counts[i] = local_count

    offsets = np.empty(n_faces, dtype=np.int64)
    total_count = 0
    for i in range(n_faces):
        offsets[i] = total_count
        total_count += counts[i]

    candidate_pairs = np.empty((total_count, 2), dtype=np.int64)

    for i in prange(n_faces):
        upper_bound = upper_bounds[i]
        if upper_bound <= i + 1:
            continue

        min_ai = mins_a[i]
        max_ai = maxs_a[i]
        min_bi = mins_b[i]
        max_bi = maxs_b[i]
        face_i = order[i]
        out_idx = offsets[i]

        for j in range(i + 1, upper_bound):
            if min_ai - maxs_a[j] >= distance_threshold:
                continue
            if mins_a[j] - max_ai >= distance_threshold:
                continue
            if min_bi - maxs_b[j] >= distance_threshold:
                continue
            if mins_b[j] - max_bi >= distance_threshold:
                continue

            face_j = order[j]
            if face_i < face_j:
                candidate_pairs[out_idx, 0] = face_i
                candidate_pairs[out_idx, 1] = face_j
            else:
                candidate_pairs[out_idx, 0] = face_j
                candidate_pairs[out_idx, 1] = face_i
            out_idx += 1

    return candidate_pairs


def filter_adjacent_pairs_batch(batch_pairs, verts, faces, threshold_sq):
    """Filter candidate pairs to those with exact triangle distance < threshold."""
    if len(batch_pairs) == 0:
        return batch_pairs

    face_i_indices = batch_pairs[:, 0]
    face_j_indices = batch_pairs[:, 1]
    face_i_vids = faces[face_i_indices]
    face_j_vids = faces[face_j_indices]

    adjacent_mask = np.any(
        face_i_vids[:, :, np.newaxis] == face_j_vids[:, np.newaxis, :],
        axis=(1, 2),
    )

    remaining_mask = ~adjacent_mask
    if np.any(remaining_mask):
        remaining_idx = np.flatnonzero(remaining_mask)
        face_i_vids_rem = face_i_vids[remaining_idx]
        face_j_vids_rem = face_j_vids[remaining_idx]

        face_i_verts_rem = verts[face_i_vids_rem]
        face_j_verts_rem = verts[face_j_vids_rem]

        mins_i = np.min(face_i_verts_rem, axis=1)
        maxs_i = np.max(face_i_verts_rem, axis=1)
        mins_j = np.min(face_j_verts_rem, axis=1)
        maxs_j = np.max(face_j_verts_rem, axis=1)

        axis_gap = np.maximum(mins_i - maxs_j, mins_j - maxs_i)
        axis_gap = np.maximum(axis_gap, 0.0)
        lower_bound_sq = np.einsum("ij,ij->i", axis_gap, axis_gap)
        maybe_adjacent = lower_bound_sq < threshold_sq

        if np.any(maybe_adjacent):
            geom_idx = remaining_idx[maybe_adjacent]
            tri_adjacent_mask = triangle_pairs_within_threshold_batch(
                face_i_verts_rem[maybe_adjacent],
                face_j_verts_rem[maybe_adjacent],
                threshold_sq,
            )
            if np.any(tri_adjacent_mask):
                adjacent_mask[geom_idx[tri_adjacent_mask]] = True

    return batch_pairs[adjacent_mask]


def build_face_edge_adjacency(mesh):
    """Build adjacency from exact mesh face-edge connectivity."""
    face_adjacency = defaultdict(set)
    for face_i, face_j in np.asarray(mesh.face_adjacency, dtype=np.int64):
        face_adjacency[int(face_i)].add(int(face_j))
        face_adjacency[int(face_j)].add(int(face_i))
    return face_adjacency


def build_face_distance_adjacency(
    verts,
    faces,
    distance_threshold=DEFAULT_FACE_GROUP_DISTANCE_THRESHOLD,
    max_distance_workers=None,
    component_labels=None,
    cross_component_only=False,
    log_prefix="",
):
    """Build face adjacency based on exact triangle-to-triangle distances."""
    n_faces = len(faces)
    face_verts_all = verts[faces]
    bbox_mins = np.min(face_verts_all, axis=1)
    bbox_maxs = np.max(face_verts_all, axis=1)

    if cross_component_only:
        if component_labels is None:
            raise ValueError("component_labels is required when cross_component_only=True")
        component_labels = np.asarray(component_labels, dtype=np.int32)
        if component_labels.shape != (n_faces,):
            raise ValueError(
                "component_labels must have one entry per face, "
                f"got shape {component_labels.shape} for {n_faces} faces"
            )

    step1_start = time.time()
    n_pairs_total = n_faces * (n_faces - 1) // 2
    if cross_component_only:
        component_sizes = np.bincount(component_labels.astype(np.int64, copy=False))
        intra_component_pairs = int(
            np.sum(component_sizes * np.maximum(component_sizes - 1, 0) // 2)
        )
        n_pairs_total -= intra_component_pairs

    idx = np.arange(n_faces, dtype=np.int64)
    best_axis = 0
    best_order = np.argsort(bbox_mins[:, 0], kind="mergesort")
    mins_axis = bbox_mins[best_order, 0]
    maxs_axis = bbox_maxs[best_order, 0]
    best_upper = np.searchsorted(
        mins_axis,
        maxs_axis + distance_threshold,
        side="left",
    ).astype(np.int64, copy=False)
    window = best_upper.copy()
    window -= idx
    window -= 1
    window[window < 0] = 0
    best_estimated_checks = int(np.sum(window))

    for axis in (1, 2):
        order_axis = np.argsort(bbox_mins[:, axis], kind="mergesort")
        mins_axis = bbox_mins[order_axis, axis]
        maxs_axis = bbox_maxs[order_axis, axis]
        upper_axis = np.searchsorted(
            mins_axis,
            maxs_axis + distance_threshold,
            side="left",
        ).astype(np.int64, copy=False)
        window_axis = upper_axis.copy()
        window_axis -= idx
        window_axis -= 1
        window_axis[window_axis < 0] = 0
        estimated_checks = int(np.sum(window_axis))
        if estimated_checks < best_estimated_checks:
            best_estimated_checks = estimated_checks
            best_axis = axis
            best_order = order_axis
            best_upper = upper_axis

    other_axes = [axis for axis in (0, 1, 2) if axis != best_axis]
    axis_a, axis_b = other_axes
    sorted_mins = bbox_mins[best_order]
    sorted_maxs = bbox_maxs[best_order]
    candidate_pairs = generate_candidate_pairs_sweep_numba(
        best_order,
        sorted_mins[:, axis_a],
        sorted_maxs[:, axis_a],
        sorted_mins[:, axis_b],
        sorted_maxs[:, axis_b],
        best_upper,
        distance_threshold,
    )
    if cross_component_only and len(candidate_pairs) > 0:
        cross_component_mask = (
            component_labels[candidate_pairs[:, 0]] != component_labels[candidate_pairs[:, 1]]
        )
        candidate_pairs = candidate_pairs[cross_component_mask]

    step1_time = time.time() - step1_start
    axis_name = "xyz"[best_axis]
    sparsity = len(candidate_pairs) / n_pairs_total if n_pairs_total > 0 else 0.0
    prefix = f"{log_prefix} " if log_prefix else ""
    print(
        f"{prefix}Step 1 (Exact sparse candidate generation): {step1_time:.4f}s - "
        f"{best_estimated_checks} axis-{axis_name} sweep checks -> "
        f"{len(candidate_pairs):,} candidates ({sparsity:.8f} of all pairs)"
    )

    step2_start = time.time()
    face_adjacency = defaultdict(set)
    batch_size = 200_000
    threshold_sq = distance_threshold * distance_threshold

    if max_distance_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = min(cpu_count, 64)
    else:
        max_workers = max(1, int(max_distance_workers))

    accepted_rows = []
    accepted_cols = []
    total_distance_batches = 0

    if max_workers == 1:
        for start in range(0, len(candidate_pairs), batch_size):
            end = min(start + batch_size, len(candidate_pairs))
            adjacent_pairs = filter_adjacent_pairs_batch(
                candidate_pairs[start:end],
                verts,
                faces,
                threshold_sq,
            )
            total_distance_batches += 1
            if len(adjacent_pairs) > 0:
                accepted_rows.append(adjacent_pairs[:, 0].astype(np.int32, copy=False))
                accepted_cols.append(adjacent_pairs[:, 1].astype(np.int32, copy=False))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for start in range(0, len(candidate_pairs), batch_size):
                end = min(start + batch_size, len(candidate_pairs))
                futures.append(
                    executor.submit(
                        filter_adjacent_pairs_batch,
                        candidate_pairs[start:end],
                        verts,
                        faces,
                        threshold_sq,
                    )
                )
                total_distance_batches += 1

            for future in as_completed(futures):
                adjacent_pairs = future.result()
                if len(adjacent_pairs) > 0:
                    accepted_rows.append(adjacent_pairs[:, 0].astype(np.int32, copy=False))
                    accepted_cols.append(adjacent_pairs[:, 1].astype(np.int32, copy=False))

    if accepted_rows:
        rows = np.concatenate(accepted_rows, axis=0)
        cols = np.concatenate(accepted_cols, axis=0)
        sym_rows = np.concatenate((rows, cols), axis=0)
        sym_cols = np.concatenate((cols, rows), axis=0)
        data = np.ones(len(sym_rows), dtype=np.uint8)
        adjacency_csr = coo_matrix(
            (data, (sym_rows, sym_cols)),
            shape=(n_faces, n_faces),
            dtype=np.uint8,
        ).tocsr()
        indptr = adjacency_csr.indptr
        indices = adjacency_csr.indices
        non_empty_rows = np.flatnonzero(np.diff(indptr))
        for face_i in non_empty_rows:
            start = indptr[face_i]
            end = indptr[face_i + 1]
            face_adjacency[int(face_i)] = set(indices[start:end].tolist())

    step2_time = time.time() - step2_start
    print(
        f"{prefix}Step 2 (Precise distance computation): {step2_time:.4f}s - "
        f"{len(candidate_pairs):,} candidates across {total_distance_batches:,} "
        f"distance batches; {len(face_adjacency)} faces have adjacencies"
    )
    return face_adjacency


def _component_face_ids_from_labels(component_labels, n_components):
    """Group face IDs by connected-component label without scanning once per component."""
    component_labels = np.asarray(component_labels, dtype=np.int64)
    order = np.argsort(component_labels, kind="mergesort")
    counts = np.bincount(component_labels, minlength=int(n_components))
    offsets = np.concatenate(
        (
            np.zeros((1,), dtype=np.int64),
            np.cumsum(counts, dtype=np.int64),
        )
    )
    return [
        order[offsets[component_id]:offsets[component_id + 1]]
        for component_id in range(int(n_components))
    ]


def _component_bounds_from_face_bounds(
    bbox_mins,
    bbox_maxs,
    component_labels,
    n_components,
):
    component_bbox_mins = np.full((int(n_components), 3), np.inf, dtype=np.float64)
    component_bbox_maxs = np.full((int(n_components), 3), -np.inf, dtype=np.float64)
    np.minimum.at(component_bbox_mins, component_labels, bbox_mins)
    np.maximum.at(component_bbox_maxs, component_labels, bbox_maxs)
    return component_bbox_mins, component_bbox_maxs


def find_face_groups(faces, face_labels, adjacency=None, verts=None):
    """Find connected components of faces with the same link ID."""
    num_faces = len(faces)

    if adjacency is None:
        if verts is None:
            raise ValueError("verts must be provided if adjacency is None")
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        adjacency = defaultdict(set)
        for face_i, face_j in mesh.face_adjacency:
            adjacency[int(face_i)].add(int(face_j))
            adjacency[int(face_j)].add(int(face_i))

    visited = np.zeros(num_faces, dtype=bool)
    groups = []

    for start_face_id in range(num_faces):
        if visited[start_face_id]:
            continue

        link_id = int(face_labels[start_face_id])
        component = []
        queue = deque([start_face_id])
        visited[start_face_id] = True

        while queue:
            face_id = queue.popleft()
            component.append(face_id)
            for neighbor_id in adjacency.get(face_id, []):
                if not visited[neighbor_id] and int(face_labels[neighbor_id]) == link_id:
                    visited[neighbor_id] = True
                    queue.append(neighbor_id)

        groups.append((link_id, component))

    return groups


def find_connected_components_fast(n_nodes, adjacency):
    """Find connected components in an undirected sparse graph."""
    row_indices = []
    col_indices = []
    for node_idx in range(n_nodes):
        for neighbor_idx in adjacency.get(node_idx, set()):
            row_indices.append(node_idx)
            col_indices.append(neighbor_idx)

    if not row_indices:
        return n_nodes, np.arange(n_nodes, dtype=np.int32)

    adjacency_matrix = csr_matrix(
        (np.ones(len(row_indices), dtype=bool), (row_indices, col_indices)),
        shape=(n_nodes, n_nodes),
    )
    n_components, labels = connected_components(adjacency_matrix, directed=False)
    return int(n_components), labels.astype(np.int32, copy=False)


def _copy_face_adjacency(adjacency):
    return {int(face_id): set(int(neighbor) for neighbor in neighbors) for face_id, neighbors in adjacency.items()}


def _merge_face_adjacency(base_adjacency, extra_adjacency):
    """Merge undirected face adjacency maps."""
    merged_adjacency = _copy_face_adjacency(base_adjacency)
    for face_id, neighbors in extra_adjacency.items():
        merged_adjacency.setdefault(int(face_id), set()).update(
            int(neighbor) for neighbor in neighbors
        )
    return merged_adjacency


def _closest_face_pair_between_components(
    face_component_a,
    face_component_b,
    *,
    face_verts,
    bbox_mins,
    bbox_maxs,
    face_centroids,
    upper_bound_sq=np.inf,
):
    """Find the exact closest face pair across two disconnected components."""
    face_component_a = np.asarray(face_component_a, dtype=np.int64)
    face_component_b = np.asarray(face_component_b, dtype=np.int64)
    if face_component_a.size == 0 or face_component_b.size == 0:
        raise ValueError("component face lists must be non-empty")

    if face_component_a.size > face_component_b.size:
        best_pair, best_sq = _closest_face_pair_between_components(
            face_component_b,
            face_component_a,
            face_verts=face_verts,
            bbox_mins=bbox_mins,
            bbox_maxs=bbox_maxs,
            face_centroids=face_centroids,
            upper_bound_sq=upper_bound_sq,
        )
        return best_pair[::-1], best_sq

    centroid_tree = cKDTree(face_centroids[face_component_a])
    centroid_distances, nearest_local = _query_nearest(
        centroid_tree,
        face_centroids[face_component_b],
    )
    centroid_distances = np.atleast_1d(centroid_distances)
    nearest_local = np.atleast_1d(nearest_local)
    best_b_local = int(np.argmin(centroid_distances))
    best_face_a = int(face_component_a[nearest_local[best_b_local]])
    best_face_b = int(face_component_b[best_b_local])
    best_sq = min(
        float(upper_bound_sq),
        float(
            _triangle_pair_distance_sq_batch(
                face_verts[[best_face_a]],
                face_verts[[best_face_b]],
            )[0]
        ),
    )
    best_pair = (best_face_a, best_face_b)
    if best_sq <= 0.0:
        return best_pair, best_sq

    block_size = 128
    exact_batch_size = 8_192

    for start_a in range(0, len(face_component_a), block_size):
        face_ids_a = face_component_a[start_a:start_a + block_size]
        mins_a = bbox_mins[face_ids_a]
        maxs_a = bbox_maxs[face_ids_a]

        for start_b in range(0, len(face_component_b), block_size):
            face_ids_b = face_component_b[start_b:start_b + block_size]
            mins_b = bbox_mins[face_ids_b]
            maxs_b = bbox_maxs[face_ids_b]

            axis_gap = np.maximum(
                mins_a[:, np.newaxis, :] - maxs_b[np.newaxis, :, :],
                mins_b[np.newaxis, :, :] - maxs_a[:, np.newaxis, :],
            )
            axis_gap = np.maximum(axis_gap, 0.0)
            lower_bound_sq = np.einsum("ijk,ijk->ij", axis_gap, axis_gap)
            candidate_mask = lower_bound_sq < best_sq
            if not np.any(candidate_mask):
                continue

            candidate_a_local, candidate_b_local = np.nonzero(candidate_mask)
            candidate_lower_bounds = lower_bound_sq[candidate_a_local, candidate_b_local]
            candidate_order = np.argsort(candidate_lower_bounds, kind="mergesort")
            candidate_a = face_ids_a[candidate_a_local[candidate_order]]
            candidate_b = face_ids_b[candidate_b_local[candidate_order]]
            candidate_lower_bounds = candidate_lower_bounds[candidate_order]

            for batch_start in range(0, len(candidate_a), exact_batch_size):
                batch_end = min(batch_start + exact_batch_size, len(candidate_a))
                if candidate_lower_bounds[batch_start] >= best_sq:
                    break
                batch_face_ids_a = candidate_a[batch_start:batch_end]
                batch_face_ids_b = candidate_b[batch_start:batch_end]
                batch_dist_sq = _triangle_pair_distance_sq_batch(
                    face_verts[batch_face_ids_a],
                    face_verts[batch_face_ids_b],
                )
                batch_best_local = int(np.argmin(batch_dist_sq))
                batch_best_sq = float(batch_dist_sq[batch_best_local])
                if batch_best_sq < best_sq:
                    best_sq = batch_best_sq
                    best_pair = (
                        int(batch_face_ids_a[batch_best_local]),
                        int(batch_face_ids_b[batch_best_local]),
                    )
                    if best_sq <= 0.0:
                        return best_pair, best_sq

    return best_pair, best_sq


class _UnionFind:
    def __init__(self, n_items):
        self.parent = np.arange(int(n_items), dtype=np.int32)
        self.rank = np.zeros((int(n_items),), dtype=np.int8)
        self.n_sets = int(n_items)

    def find(self, item):
        item = int(item)
        parent = self.parent
        while int(parent[item]) != item:
            parent[item] = parent[int(parent[item])]
            item = int(parent[item])
        return item

    def union(self, item_a, item_b):
        root_a = self.find(item_a)
        root_b = self.find(item_b)
        if root_a == root_b:
            return False
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1
        self.n_sets -= 1
        return True


def _component_bbox_pair_lower_bound_sq(
    component_bbox_mins,
    component_bbox_maxs,
    component_indices_a,
    component_indices_b,
):
    """Return exact AABB lower-bound distances for component bbox pairs."""
    mins_a = component_bbox_mins[component_indices_a]
    maxs_a = component_bbox_maxs[component_indices_a]
    mins_b = component_bbox_mins[component_indices_b]
    maxs_b = component_bbox_maxs[component_indices_b]
    axis_gap = np.maximum(mins_a - maxs_b, mins_b - maxs_a)
    axis_gap = np.maximum(axis_gap, 0.0)
    return np.einsum("ij,ij->i", axis_gap, axis_gap)


def _sorted_component_bbox_lower_bound_pairs(
    component_bbox_mins,
    component_bbox_maxs,
):
    """Return all component pairs sorted by their exact AABB distance lower bound."""
    n_components = len(component_bbox_mins)
    n_pairs = n_components * (n_components - 1) // 2
    pair_rows = np.empty((n_pairs,), dtype=np.int32)
    pair_cols = np.empty((n_pairs,), dtype=np.int32)
    pair_lower_bound_sq = np.empty((n_pairs,), dtype=np.float64)

    write_offset = 0
    for component_idx in range(n_components - 1):
        component_indices_b = np.arange(
            component_idx + 1,
            n_components,
            dtype=np.int32,
        )
        n_row_pairs = len(component_indices_b)
        pair_rows[write_offset:write_offset + n_row_pairs] = component_idx
        pair_cols[write_offset:write_offset + n_row_pairs] = component_indices_b
        pair_lower_bound_sq[write_offset:write_offset + n_row_pairs] = (
            _component_bbox_pair_lower_bound_sq(
                component_bbox_mins,
                component_bbox_maxs,
                np.full((n_row_pairs,), component_idx, dtype=np.int32),
                component_indices_b,
            )
        )
        write_offset += n_row_pairs

    sort_order = np.lexsort((pair_cols, pair_rows, pair_lower_bound_sq))
    return (
        pair_rows[sort_order],
        pair_cols[sort_order],
        pair_lower_bound_sq[sort_order],
    )


def _exact_component_bridge_edges(
    component_face_ids,
    *,
    face_verts,
    bbox_mins,
    bbox_maxs,
    face_centroids,
    component_bbox_mins,
    component_bbox_maxs,
):
    """Return exact MST face-pair bridges over disconnected face components.

    Components are connected by the exact closest triangle-to-triangle face pair.
    AABB distances are used only as lower bounds for lazy Kruskal ordering.
    """
    n_components = len(component_face_ids)
    if n_components <= 1:
        return []

    lower_bound_start = time.time()
    pair_rows, pair_cols, pair_lower_bound_sq = _sorted_component_bbox_lower_bound_pairs(
        component_bbox_mins,
        component_bbox_maxs,
    )
    lower_bound_time = time.time() - lower_bound_start

    union_find = _UnionFind(n_components)
    bridge_edges = []
    exact_edge_heap = []
    pair_cursor = 0
    exact_evaluations = 0
    skipped_same_set_lower_bound_pairs = 0
    discarded_same_set_exact_edges = 0
    exact_eval_time = 0.0
    n_pairs = len(pair_rows)

    while union_find.n_sets > 1:
        while (
            pair_cursor < n_pairs
            and (
                not exact_edge_heap
                or float(pair_lower_bound_sq[pair_cursor]) < float(exact_edge_heap[0][0])
            )
        ):
            component_idx_a = int(pair_rows[pair_cursor])
            component_idx_b = int(pair_cols[pair_cursor])
            pair_cursor += 1

            if union_find.find(component_idx_a) == union_find.find(component_idx_b):
                skipped_same_set_lower_bound_pairs += 1
                continue

            exact_start = time.time()
            face_pair, distance_sq = _closest_face_pair_between_components(
                component_face_ids[component_idx_a],
                component_face_ids[component_idx_b],
                face_verts=face_verts,
                bbox_mins=bbox_mins,
                bbox_maxs=bbox_maxs,
                face_centroids=face_centroids,
            )
            exact_eval_time += time.time() - exact_start
            exact_evaluations += 1
            heapq.heappush(
                exact_edge_heap,
                (
                    float(distance_sq),
                    component_idx_a,
                    component_idx_b,
                    int(face_pair[0]),
                    int(face_pair[1]),
                ),
            )

        if not exact_edge_heap:
            raise RuntimeError("failed to connect face-distance adjacency components")

        _, component_idx_a, component_idx_b, face_i, face_j = heapq.heappop(
            exact_edge_heap
        )
        if union_find.union(component_idx_a, component_idx_b):
            bridge_edges.append((face_i, face_j))
        else:
            discarded_same_set_exact_edges += 1

    print(
        "Exact component bridge search: "
        f"{lower_bound_time:.4f}s bbox lower bounds for {n_pairs:,} component pairs; "
        f"{exact_evaluations:,} exact component-pair evaluations in {exact_eval_time:.4f}s; "
        f"accepted {len(bridge_edges):,} bridges; "
        f"skipped {skipped_same_set_lower_bound_pairs:,} same-set lower-bound pairs; "
        f"discarded {discarded_same_set_exact_edges:,} same-set exact edges"
    )
    return bridge_edges


def ensure_face_adjacency_is_connected(mesh, face_adjacency):
    """Bridge disconnected face-distance components by component-level nearest links."""
    connected_adjacency = _copy_face_adjacency(face_adjacency)
    num_faces = len(mesh.faces)
    n_components, component_labels = find_connected_components_fast(
        num_faces,
        connected_adjacency,
    )
    if n_components <= 1:
        return connected_adjacency

    bridge_start = time.time()
    print(
        "Face connectivity graph has "
        f"{n_components} connected components after threshold links; "
        "adding nearest component bridges"
    )
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_verts = verts[faces]
    bbox_mins = np.min(face_verts, axis=1)
    bbox_maxs = np.max(face_verts, axis=1)
    face_centroids = np.asarray(mesh.triangles_center, dtype=np.float64)

    component_face_ids = _component_face_ids_from_labels(component_labels, n_components)
    component_bbox_mins, component_bbox_maxs = _component_bounds_from_face_bounds(
        bbox_mins,
        bbox_maxs,
        component_labels,
        n_components,
    )
    bridge_edges = _exact_component_bridge_edges(
        component_face_ids,
        face_verts=face_verts,
        bbox_mins=bbox_mins,
        bbox_maxs=bbox_maxs,
        face_centroids=face_centroids,
        component_bbox_mins=component_bbox_mins,
        component_bbox_maxs=component_bbox_maxs,
    )
    for best_pair in bridge_edges:
        face_i, face_j = best_pair
        connected_adjacency.setdefault(face_i, set()).add(face_j)
        connected_adjacency.setdefault(face_j, set()).add(face_i)

    print(
        "Component connectivity bridge step: "
        f"{time.time() - bridge_start:.4f}s - added {len(bridge_edges):,} "
        "nearest component bridges"
    )
    return connected_adjacency


def build_face_connectivity_adjacency_for_inference(
    mesh,
    *,
    distance_threshold=DEFAULT_FACE_GROUP_DISTANCE_THRESHOLD,
    max_distance_workers=None,
):
    """Build the inference connectivity graph from edge adjacency plus cross-CC distance links."""
    base_face_adjacency = build_face_edge_adjacency(mesh)
    num_faces = len(mesh.faces)
    n_components, component_labels = find_connected_components_fast(
        num_faces,
        base_face_adjacency,
    )
    if n_components <= 1:
        return base_face_adjacency

    print(
        "Base face-edge adjacency has "
        f"{n_components} connected components across {num_faces} faces; "
        "adding face-level cross-component distance links"
    )
    cross_component_adjacency = build_face_distance_adjacency(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
        distance_threshold=float(distance_threshold),
        component_labels=component_labels,
        cross_component_only=True,
        max_distance_workers=max_distance_workers,
        log_prefix="Cross-component",
    )
    return _merge_face_adjacency(base_face_adjacency, cross_component_adjacency)


def _compute_face_probability_statistics(num_faces, point_part_probabilities, face_indices):
    """Aggregate point softmax probabilities onto faces."""
    point_part_probabilities = np.asarray(point_part_probabilities, dtype=np.float32)
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if point_part_probabilities.ndim != 2:
        raise ValueError(
            "point_part_probabilities must have shape (num_points, num_parts), "
            f"got {point_part_probabilities.shape}"
        )
    if face_indices.shape != (point_part_probabilities.shape[0],):
        raise ValueError(
            "face_indices must have one entry per point, "
            f"got {face_indices.shape} for {point_part_probabilities.shape[0]} points"
        )

    num_parts = point_part_probabilities.shape[1]
    face_probability_sums = np.zeros((num_faces, num_parts), dtype=np.float64)
    face_probability_counts = np.zeros(num_faces, dtype=np.int64)

    valid_mask = face_indices >= 0
    if np.any(valid_mask):
        valid_face_indices = face_indices[valid_mask]
        np.add.at(
            face_probability_sums,
            valid_face_indices,
            point_part_probabilities[valid_mask],
        )
        np.add.at(
            face_probability_counts,
            valid_face_indices,
            np.ones(valid_face_indices.shape[0], dtype=np.int64),
        )

    return face_probability_sums, face_probability_counts


def _build_filled_face_probability_means(mesh, face_probability_sums, face_probability_counts):
    """Fill unsampled face probabilities from the nearest sampled face."""
    num_faces, num_parts = face_probability_sums.shape
    filled_face_probability_means = np.zeros((num_faces, num_parts), dtype=np.float32)
    defined_mask = face_probability_counts > 0

    if np.any(defined_mask):
        defined_faces = np.flatnonzero(defined_mask)
        filled_face_probability_means[defined_faces] = (
            face_probability_sums[defined_faces]
            / face_probability_counts[defined_faces, np.newaxis]
        ).astype(np.float32, copy=False)
    else:
        filled_face_probability_means.fill(1.0 / max(num_parts, 1))
        return filled_face_probability_means

    undefined_faces = np.flatnonzero(~defined_mask)
    if undefined_faces.size == 0:
        return filled_face_probability_means

    centroids = _get_face_centroids(mesh)
    tree = cKDTree(centroids[defined_faces])
    _, nearest_local = _query_nearest(tree, centroids[undefined_faces])
    nearest_local = np.atleast_1d(nearest_local)
    nearest_defined_faces = defined_faces[nearest_local]
    filled_face_probability_means[undefined_faces] = filled_face_probability_means[
        nearest_defined_faces
    ]
    return filled_face_probability_means


def _group_confidence_vector(
    group_face_ids,
    face_probability_sums,
    face_probability_counts,
    filled_face_probability_means,
):
    """Aggregate point softmax probabilities for one face group."""
    group_face_ids = np.asarray(group_face_ids, dtype=np.int64)
    group_point_count = int(face_probability_counts[group_face_ids].sum())
    if group_point_count > 0:
        return (
            face_probability_sums[group_face_ids].sum(axis=0)
            / float(group_point_count)
        ).astype(np.float32, copy=False)
    return filled_face_probability_means[group_face_ids].mean(axis=0).astype(
        np.float32,
        copy=False,
    )


def _adjacent_group_indices_by_part_id_for_group(groups, face_adjacency, face_to_group, group_idx):
    adjacent_groups_by_part_id = defaultdict(set)
    for face_id in groups[group_idx][1]:
        for adjacent_face_id in face_adjacency.get(face_id, set()):
            adjacent_group_idx = int(face_to_group[int(adjacent_face_id)])
            if adjacent_group_idx < 0 or adjacent_group_idx == group_idx:
                continue
            adjacent_part_id = int(groups[adjacent_group_idx][0])
            if adjacent_part_id >= 0:
                adjacent_groups_by_part_id[adjacent_part_id].add(adjacent_group_idx)
    return adjacent_groups_by_part_id


def _iterative_single_group_reassignment(
    face_part_ids,
    *,
    face_adjacency,
    input_part_ids,
    face_probability_sums,
    face_probability_counts,
    filled_face_probability_means,
):
    """Iteratively enforce one face group per part ID."""
    input_part_ids = np.asarray(input_part_ids, dtype=np.int64)
    face_part_ids = np.asarray(face_part_ids, dtype=np.int32).copy()
    max_iterations = max(8, 4 * max(1, int(input_part_ids.size)))
    seen_states = set()

    for _ in range(max_iterations):
        state_key = face_part_ids.tobytes()
        if state_key in seen_states:
            return face_part_ids
        seen_states.add(state_key)

        groups = find_face_groups(
            np.empty((face_part_ids.shape[0], 3), dtype=np.int64),
            face_part_ids,
            adjacency=face_adjacency,
        )
        groups_by_part_id = defaultdict(list)
        group_confidences = []
        group_sizes = []
        for group_idx, (part_id, group_face_ids) in enumerate(groups):
            groups_by_part_id[int(part_id)].append(group_idx)
            group_face_ids = np.asarray(group_face_ids, dtype=np.int64)
            group_confidences.append(
                _group_confidence_vector(
                    group_face_ids,
                    face_probability_sums,
                    face_probability_counts,
                    filled_face_probability_means,
                )
            )
            group_sizes.append(len(group_face_ids))

        duplicate_part_ids = [
            int(part_id)
            for part_id, group_indices in groups_by_part_id.items()
            if len(group_indices) > 1 and part_id >= 0
        ]
        if not duplicate_part_ids:
            return face_part_ids

        existing_part_ids = {
            int(part_id)
            for part_id in np.unique(face_part_ids)
            if int(part_id) >= 0
        }
        missing_part_ids = sorted(
            int(part_id) for part_id in input_part_ids if int(part_id) not in existing_part_ids
        )

        updates = {}
        groups_to_keep = {}
        for part_id in duplicate_part_ids:
            groups_to_keep[part_id] = max(
                groups_by_part_id[part_id],
                key=lambda group_idx: (
                    group_sizes[group_idx],
                    float(group_confidences[group_idx][part_id]),
                    -group_idx,
                ),
            )

        face_to_group = np.full(face_part_ids.shape[0], -1, dtype=np.int32)
        for group_idx, (_, group_face_ids) in enumerate(groups):
            face_to_group[np.asarray(group_face_ids, dtype=np.int64)] = int(group_idx)

        available_missing_part_ids = set(missing_part_ids)
        for part_id in duplicate_part_ids:
            group_to_keep = groups_to_keep[part_id]
            for group_to_update in groups_by_part_id[part_id]:
                if group_to_update == group_to_keep:
                    continue

                adjacent_groups_by_part_id = _adjacent_group_indices_by_part_id_for_group(
                    groups,
                    face_adjacency,
                    face_to_group,
                    group_to_update,
                )
                safe_adjacent_part_ids = set()
                for adjacent_part_id, adjacent_group_indices in adjacent_groups_by_part_id.items():
                    if adjacent_part_id == part_id:
                        continue
                    target_group_indices = groups_by_part_id.get(adjacent_part_id, [])
                    if len(target_group_indices) == 1:
                        safe_adjacent_part_ids.add(adjacent_part_id)
                        continue
                    target_group_to_keep = groups_to_keep.get(adjacent_part_id)
                    if (
                        target_group_to_keep is not None
                        and target_group_to_keep in adjacent_group_indices
                    ):
                        safe_adjacent_part_ids.add(adjacent_part_id)

                replacement_candidates = sorted(
                    safe_adjacent_part_ids | available_missing_part_ids
                )
                if not replacement_candidates:
                    continue

                confidence_vector = group_confidences[group_to_update]
                best_replacement_part_id = max(
                    replacement_candidates,
                    key=lambda candidate_part_id: (
                        float(confidence_vector[candidate_part_id]),
                        -int(candidate_part_id),
                    ),
                )
                updates[group_to_update] = int(best_replacement_part_id)
                available_missing_part_ids.discard(int(best_replacement_part_id))

        if not updates:
            return face_part_ids

        updated_face_part_ids = face_part_ids.copy()
        for group_idx, replacement_part_id in updates.items():
            updated_face_part_ids[np.asarray(groups[group_idx][1], dtype=np.int64)] = replacement_part_id
        if np.array_equal(updated_face_part_ids, face_part_ids):
            return face_part_ids
        face_part_ids = updated_face_part_ids

    raise RuntimeError("single-group face post-processing did not converge")


def refine_face_part_ids_for_inference(
    mesh,
    face_part_ids,
    *,
    point_part_probabilities=None,
    face_indices=None,
    input_part_ids=None,
    strict=False,
    enforce_connectivity_per_part=False,
    distance_threshold=DEFAULT_FACE_GROUP_DISTANCE_THRESHOLD,
):
    """Inference-time face post-processing layered on top of the base face pass."""
    face_part_ids = np.asarray(face_part_ids, dtype=np.int32)
    base_face_part_ids = refine_face_part_ids(
        mesh,
        face_part_ids,
        strict=bool(strict),
    )
    if not enforce_connectivity_per_part:
        return base_face_part_ids

    if point_part_probabilities is None:
        raise ValueError(
            "point_part_probabilities is required when enforce_connectivity_per_part is enabled"
        )
    if face_indices is None:
        raise ValueError("face_indices is required when enforce_connectivity_per_part is enabled")
    if input_part_ids is None:
        raise ValueError("input_part_ids is required when enforce_connectivity_per_part is enabled")

    point_part_probabilities = np.asarray(point_part_probabilities, dtype=np.float32)
    face_indices = np.asarray(face_indices, dtype=np.int64)
    input_part_ids = np.asarray(input_part_ids, dtype=np.int64)

    if point_part_probabilities.ndim != 2:
        raise ValueError("point_part_probabilities must have shape [num_points, num_parts]")
    if point_part_probabilities.shape[1] <= 0:
        raise ValueError("point_part_probabilities must contain at least one part column")
    if np.any(input_part_ids < 0):
        raise ValueError("input_part_ids must be non-negative")
    if np.any(input_part_ids >= point_part_probabilities.shape[1]):
        raise ValueError(
            "input_part_ids must be within the probability columns, "
            f"got max {int(input_part_ids.max())} for {point_part_probabilities.shape[1]} columns"
        )

    face_connectivity_adjacency = build_face_connectivity_adjacency_for_inference(
        mesh,
        distance_threshold=float(distance_threshold),
    )
    face_connectivity_adjacency = ensure_face_adjacency_is_connected(
        mesh,
        face_connectivity_adjacency,
    )
    faces = np.asarray(mesh.faces, dtype=np.int64)

    face_probability_sums, face_probability_counts = _compute_face_probability_statistics(
        len(faces),
        point_part_probabilities,
        face_indices,
    )
    filled_face_probability_means = _build_filled_face_probability_means(
        mesh,
        face_probability_sums,
        face_probability_counts,
    )

    return _iterative_single_group_reassignment(
        base_face_part_ids,
        face_adjacency=face_connectivity_adjacency,
        input_part_ids=input_part_ids,
        face_probability_sums=face_probability_sums,
        face_probability_counts=face_probability_counts,
        filled_face_probability_means=filled_face_probability_means,
    )
