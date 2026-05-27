import numpy as np
from typing import List, Tuple


def axis_point_to_plucker(axis: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Convert axis-point coordinates to plucker coordinates.
    """
    assert axis.shape[-1] == 3
    assert point.shape[-1] == 3
    l = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-8)
    m = np.cross(l, point, axis=-1)
    return np.concatenate([l, m], axis=-1)


def plucker_to_axis_point(plucker: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert plucker coordinates to axis-point coordinates.
    """
    assert plucker.shape[-1] == 6
    l, m = plucker[..., :3], plucker[..., 3:]
    axis = l / (np.linalg.norm(l, axis=-1, keepdims=True) + 1e-8)
    point = np.cross(m, axis, axis=-1)
    return axis, point


def plucker_to_4x4_transform_matrix(plucker: np.ndarray, angle: float) -> np.ndarray:
    """
    Convert plucker coordinates to a 4x4 transformation matrix.
    """
    assert plucker.shape == (6,)
    axis, point = plucker_to_axis_point(plucker)

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = point - R @ point
    
    return T


def transform_points(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Transform points by a 4x4 transformation matrix.

    points: (..., 3)
    transform_matrix: (4, 4)
    """
    return points @ transform_matrix[:3, :3].T + transform_matrix[:3, 3]


def transform_direction(direction: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Transform a direction vector by a 4x4 transformation matrix.

    direction: (..., 3)
    transform_matrix: (4, 4)
    """
    return direction @ transform_matrix[:3, :3].T


def transform_plucker(plucker: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """Transforms a Plucker line by a 4x4 transform matrix."""
    axis, point = plucker_to_axis_point(np.asarray(plucker, dtype=np.float32))
    transformed_axis = transform_direction(axis, transform_matrix)
    transformed_point = transform_points(point, transform_matrix)
    return axis_point_to_plucker(transformed_axis, transformed_point).astype(
        np.float32,
        copy=False,
    )


def get_subtree_part_ids(motion_hierarchy: List[Tuple[int, int]], part_id: int) -> List[int]:
    """
    Get the subtree part ids for a given part id.
    """
    subtree_part_ids = [part_id]
    for parent_id, child_id in motion_hierarchy:
        if parent_id == part_id:
            subtree_part_ids.extend(get_subtree_part_ids(motion_hierarchy, child_id))
    return subtree_part_ids


def get_part_order_from_root(motion_hierarchy: List[Tuple[int, int]]) -> List[int]:
    """
    Depth-first search to get the part order from the root.
    """
    part_order = []
    visited = set()
    def dfs(part_id):
        if part_id in visited:
            return
        part_order.append(part_id)
        visited.add(part_id)
        for parent_id, child_id in motion_hierarchy:
            if parent_id == part_id:
                dfs(child_id)

    # Find the base/root part id
    all_part_ids = set([parent_id for parent_id, _ in motion_hierarchy])
    all_part_ids.update([child_id for _, child_id in motion_hierarchy])

    # Find the root part id
    for _, child_id in motion_hierarchy:
        all_part_ids.remove(child_id)

    # assert len(all_part_ids) == 1
    root_part_id = all_part_ids.pop()

    dfs(root_part_id) # Populate part_order
    return part_order


def compute_part_transforms(
    unique_part_ids,
    motion_hierarchy,
    is_part_revolute,
    is_part_prismatic,
    revolute_plucker,
    revolute_range,
    prismatic_axis,
    prismatic_range,
    articulation_state
):
    """
    Compute the 4x4 transformation matrix for each part at a given articulation state.
    Returns a dictionary mapping part_id to its cumulative transformation matrix.
    
    The transformation represents how to transform each part from its rest pose to the articulated pose.
    """
    if len(motion_hierarchy) == 0:
        return {pid: np.eye(4) for pid in unique_part_ids}
    
    # Collect all relevant part IDs from motion hierarchy and unique_part_ids
    all_part_ids = set(unique_part_ids)
    for parent, child in motion_hierarchy:
        all_part_ids.add(parent)
        all_part_ids.add(child)
        
    transforms = {pid: np.eye(4) for pid in all_part_ids}
    
    # Process parts in hierarchical order (BFS/DFS from root)
    part_order = get_part_order_from_root(motion_hierarchy)
    
    for pid in part_order:
        affected_part_ids = get_subtree_part_ids(motion_hierarchy, pid)
        part_articulation_state = (
            articulation_state
            if np.isscalar(articulation_state) or np.asarray(articulation_state).ndim == 0
            else articulation_state[pid]
        )
        
        # Compute transformation for this part's joint
        joint_transform = np.eye(4)
        
        if is_part_revolute[pid]:
            low_limit, high_limit = revolute_range[pid]
            angle = low_limit + part_articulation_state * (high_limit - low_limit)
            joint_transform = plucker_to_4x4_transform_matrix(revolute_plucker[pid], angle)
        
        elif is_part_prismatic[pid]:
            low_limit, high_limit = prismatic_range[pid]
            displacement = low_limit + part_articulation_state * (high_limit - low_limit)
            paxis = prismatic_axis[pid]
            joint_transform[:3, 3] = displacement * paxis
        
        # Apply joint transformation to all affected (descendant) parts
        for affected_pid in affected_part_ids:
            if affected_pid in transforms:
                transforms[affected_pid] = joint_transform @ transforms[affected_pid]
    
    return transforms
