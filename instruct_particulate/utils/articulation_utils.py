import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

_DEFAULT_CONTINUOUS_REVOLUTE_RANGE = np.asarray(
    [0.0, 2.0 * np.pi],
    dtype=np.float32,
)


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


def shift_axes_plucker(
    axes_plucker: np.ndarray,
    x_center: float, y_center: float, z_center: float, scale: float
) -> np.ndarray:
    """Shift the axes plucker coordinates."""
    axis, point = plucker_to_axis_point(axes_plucker)
    point_new = point - np.array([x_center, y_center, z_center])
    point_new *= scale
    return axis_point_to_plucker(axis, point_new)


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


def joint_has_revolute_motion(joint: Dict[str, Any]) -> bool:
    """Returns whether a joint has usable revolute axis metadata."""
    return (
        bool(joint.get("is_revolute", False))
        and joint.get("revolute_axis") is not None
    )


def joint_has_prismatic_motion(joint: Dict[str, Any]) -> bool:
    """Returns whether a joint has usable prismatic axis metadata."""
    return (
        bool(joint.get("is_prismatic", False))
        and joint.get("prismatic_axis") is not None
    )


def joint_has_revolute_range(joint: Dict[str, Any]) -> bool:
    """Returns whether a joint carries finite revolute range supervision."""
    return joint_has_revolute_motion(joint) and joint.get("revolute_range") is not None


def joint_has_prismatic_range(joint: Dict[str, Any]) -> bool:
    """Returns whether a joint carries finite prismatic range supervision."""
    return joint_has_prismatic_motion(joint) and joint.get("prismatic_range") is not None


def get_revolute_articulation_range(joint: Dict[str, Any]) -> Optional[np.ndarray]:
    """Returns the revolute range used for articulation, including continuous joints."""
    if not joint_has_revolute_motion(joint):
        return None
    if joint.get("revolute_range") is None:
        return _DEFAULT_CONTINUOUS_REVOLUTE_RANGE.copy()
    return np.asarray(joint["revolute_range"], dtype=np.float32)


def get_prismatic_articulation_range(joint: Dict[str, Any]) -> Optional[np.ndarray]:
    """Returns the prismatic range used for articulation, if available."""
    if not joint_has_prismatic_range(joint):
        return None
    return np.asarray(joint["prismatic_range"], dtype=np.float32)


def joint_is_movable(joint: Dict[str, Any]) -> bool:
    """Returns whether a joint can currently be articulated."""
    return (
        get_revolute_articulation_range(joint) is not None
        or get_prismatic_articulation_range(joint) is not None
    )


def articulate_points_with_joints(
    xyz: np.ndarray,
    part_ids: np.ndarray,
    joints: List[Dict[str, Any]],
    articulation_state: Union[float, np.ndarray, Dict[int, float]],
    normals: Optional[np.ndarray] = None,
) -> Union[
    Tuple[np.ndarray, List[Dict[str, Any]]],
    Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray],
]:
    """Articulates points and joint metadata using joint records directly.

    Args:
        xyz: Point coordinates.
        part_ids: Per-point link IDs.
        joints: Joint records containing `parent_link_id`, `child_link_id`,
            `is_revolute`, `is_prismatic`, and optionally the corresponding
            axes and ranges.
        articulation_state: Either a scalar state applied to every movable
            joint, a dense array indexed by child link ID, or a dictionary
            keyed by child link ID. When a dictionary is provided, only the
            listed joints are articulated.
        normals: Optional normals aligned with `xyz`.

    Returns:
        `(articulated_xyz, articulated_joints)` when `normals` is `None`,
        otherwise `(articulated_xyz, articulated_joints, articulated_normals)`.
    """
    articulated_xyz = np.asarray(xyz, dtype=np.float32).copy()
    articulated_normals = None if normals is None else np.asarray(normals, dtype=np.float32).copy()

    articulated_joints: List[Dict[str, Any]] = []
    for joint in sorted(joints, key=lambda item: int(item["joint_id"])):
        joint_copy = dict(joint)
        if joint_copy.get("revolute_axis") is not None:
            joint_copy["revolute_axis"] = np.asarray(
                joint_copy["revolute_axis"],
                dtype=np.float32,
            ).copy()
        if joint_copy.get("prismatic_axis") is not None:
            joint_copy["prismatic_axis"] = np.asarray(
                joint_copy["prismatic_axis"],
                dtype=np.float32,
            ).copy()
        if joint_copy.get("revolute_range") is not None:
            joint_copy["revolute_range"] = np.asarray(
                joint_copy["revolute_range"],
                dtype=np.float32,
            ).copy()
        if joint_copy.get("prismatic_range") is not None:
            joint_copy["prismatic_range"] = np.asarray(
                joint_copy["prismatic_range"],
                dtype=np.float32,
            ).copy()
        articulated_joints.append(joint_copy)

    motion_hierarchy = [
        (int(joint["parent_link_id"]), int(joint["child_link_id"]))
        for joint in articulated_joints
    ]
    if len(motion_hierarchy) == 0:
        if articulated_normals is None:
            return articulated_xyz, articulated_joints
        return articulated_xyz, articulated_joints, articulated_normals

    joint_by_child = {
        int(joint["child_link_id"]): joint
        for joint in articulated_joints
    }

    def get_joint_state(child_link_id: int) -> float:
        if isinstance(articulation_state, dict):
            return float(articulation_state.get(child_link_id, 0.0))
        if np.isscalar(articulation_state) or np.asarray(articulation_state).ndim == 0:
            return float(articulation_state)
        return float(np.asarray(articulation_state, dtype=np.float32)[child_link_id])

    def should_articulate_joint(child_link_id: int) -> bool:
        if isinstance(articulation_state, dict):
            return child_link_id in articulation_state
        return True

    def apply_transform_to_subtree(
        child_link_id: int,
        transform_matrix: np.ndarray,
        transform_normals_flag: bool,
    ) -> None:
        affected_part_ids = get_subtree_part_ids(motion_hierarchy, child_link_id)
        for affected_part_id in affected_part_ids:
            part_mask = part_ids == affected_part_id
            articulated_xyz[part_mask] = transform_points(
                articulated_xyz[part_mask],
                transform_matrix,
            )
            if articulated_normals is not None and transform_normals_flag:
                articulated_normals[part_mask] = transform_direction(
                    articulated_normals[part_mask],
                    transform_matrix,
                )

            affected_joint = joint_by_child.get(affected_part_id)
            if affected_joint is not None:
                if joint_has_revolute_motion(affected_joint):
                    affected_joint["revolute_axis"] = transform_plucker(
                        np.asarray(affected_joint["revolute_axis"], dtype=np.float32),
                        transform_matrix,
                    )
                if joint_has_prismatic_motion(affected_joint):
                    prismatic_axis = np.asarray(
                        affected_joint["prismatic_axis"],
                        dtype=np.float32,
                    )
                    if prismatic_axis.shape[-1] == 6:
                        affected_joint["prismatic_axis"] = transform_plucker(
                            prismatic_axis,
                            transform_matrix,
                        )
                    else:
                        affected_joint["prismatic_axis"] = transform_direction(
                            prismatic_axis,
                            transform_matrix,
                        )

    part_order = get_part_order_from_root(motion_hierarchy)
    for child_link_id in part_order:
        joint = joint_by_child.get(child_link_id)
        if joint is None or not should_articulate_joint(child_link_id):
            continue

        part_state = get_joint_state(child_link_id)
        revolute_range = get_revolute_articulation_range(joint)
        if revolute_range is not None:
            joint_axis = np.asarray(joint["revolute_axis"], dtype=np.float32)
            low_limit, high_limit = revolute_range
            angle = low_limit + part_state * (high_limit - low_limit)
            if joint.get("revolute_range") is not None:
                joint["revolute_range"] = np.array(
                    [low_limit - angle, high_limit - angle],
                    dtype=np.float32,
                )
            transform_matrix = plucker_to_4x4_transform_matrix(
                joint_axis,
                float(angle),
            ).astype(np.float32, copy=False)
            apply_transform_to_subtree(
                child_link_id,
                transform_matrix,
                transform_normals_flag=True,
            )

        prismatic_range = get_prismatic_articulation_range(joint)
        if prismatic_range is not None:
            joint_axis = np.asarray(joint["prismatic_axis"], dtype=np.float32)
            low_limit, high_limit = prismatic_range
            displacement = low_limit + part_state * (high_limit - low_limit)
            if joint_axis.shape[-1] == 6:
                axis_dir, _ = plucker_to_axis_point(joint_axis)
            else:
                axis_dir = joint_axis / max(np.linalg.norm(joint_axis), 1e-8)
            transform_matrix = np.eye(4, dtype=np.float32)
            transform_matrix[:3, 3] = displacement * axis_dir
            joint["prismatic_range"] = np.array(
                [low_limit - displacement, high_limit - displacement],
                dtype=np.float32,
            )
            apply_transform_to_subtree(
                child_link_id,
                transform_matrix,
                transform_normals_flag=False,
            )

    if articulated_normals is None:
        return articulated_xyz, articulated_joints
    return articulated_xyz, articulated_joints, articulated_normals


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


def articulate_points(
    xyz: np.ndarray,
    part_ids: np.ndarray,
    motion_hierarchy: List[Tuple[int, int]],
    is_part_revolute: np.ndarray,
    is_part_prismatic: np.ndarray,
    revolute_plucker: np.ndarray,
    revolute_range: np.ndarray,
    prismatic_axis: np.ndarray,
    prismatic_range: np.ndarray,
    articulation_state: Union[float, np.ndarray],  # Value between 0 and 1
    normals: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Articulate points based on given articulation state.
    
    Args:
        articulation_state: Value between 0 (low limit) and 1 (high limit)
    
    Returns:
        Tuple of (articulated_xyz, articulated_revolute_plucker, articulated_prismatic_axis)
    """
    articulated_xyz = xyz.copy()
    articulated_revolute_plucker = revolute_plucker.copy()
    articulated_prismatic_axis = prismatic_axis.copy()
    articulated_revolute_range = revolute_range.copy()
    articulated_prismatic_range = prismatic_range.copy()
    if normals is not None:
        articulated_normals = normals.copy()

    if len(motion_hierarchy) == 0:
        return articulated_xyz, articulated_revolute_plucker, articulated_prismatic_axis
    
    part_order = get_part_order_from_root(motion_hierarchy)
    
    for pid in part_order:
        affected_part_ids = get_subtree_part_ids(motion_hierarchy, pid)
        
        if is_part_revolute[pid]:
            low_limit, high_limit = revolute_range[pid]
            part_articulation_state = articulation_state if isinstance(articulation_state, float) else articulation_state[pid]
            # Interpolate between low and high limits
            angle = low_limit + part_articulation_state * (high_limit - low_limit)
            articulated_revolute_range[pid] = np.array([low_limit - angle, high_limit - angle])
            transform_matrix = plucker_to_4x4_transform_matrix(articulated_revolute_plucker[pid], angle)
            
            for affected_pid in affected_part_ids:
                # Transform points
                articulated_xyz[part_ids == affected_pid] = transform_points(
                    articulated_xyz[part_ids == affected_pid], transform_matrix
                )
                # Transform normals
                if normals is not None:
                    articulated_normals[part_ids == affected_pid] = transform_direction(
                        articulated_normals[part_ids == affected_pid], transform_matrix
                    )
                
                # Transform revolute axes for affected parts
                if is_part_revolute[affected_pid]:
                    current_axis, current_point = plucker_to_axis_point(articulated_revolute_plucker[affected_pid])
                    new_axis = transform_direction(current_axis, transform_matrix)
                    new_point = transform_points(current_point, transform_matrix)
                    articulated_revolute_plucker[affected_pid] = axis_point_to_plucker(new_axis, new_point)
                
                # Transform prismatic axes for affected parts
                if is_part_prismatic[affected_pid]:
                    articulated_prismatic_axis[affected_pid] = transform_direction(
                        articulated_prismatic_axis[affected_pid], transform_matrix
                    )
        
        if is_part_prismatic[pid]:
            low_limit, high_limit = prismatic_range[pid]
            part_articulation_state = articulation_state if isinstance(articulation_state, float) else articulation_state[pid]
            # Interpolate between low and high limits
            displacement = low_limit + part_articulation_state * (high_limit - low_limit)
            articulated_prismatic_range[pid] = np.array([low_limit - displacement, high_limit - displacement])
            paxis = articulated_prismatic_axis[pid]
            
            for affected_pid in affected_part_ids:
                # Translate points
                articulated_xyz[part_ids == affected_pid] = (
                    articulated_xyz[part_ids == affected_pid] + displacement * paxis
                )
                # Translation does not affect normals
                
                # Translate revolute axes for affected parts (only the point, not the direction)
                if is_part_revolute[affected_pid]:
                    current_axis, current_point = plucker_to_axis_point(articulated_revolute_plucker[affected_pid])
                    new_point = current_point + displacement * paxis
                    articulated_revolute_plucker[affected_pid] = axis_point_to_plucker(current_axis, new_point)
    
    output = (articulated_xyz, articulated_revolute_plucker, articulated_prismatic_axis, articulated_revolute_range, articulated_prismatic_range)
    if normals is not None:
        output = output + (articulated_normals,)
    return output


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


def articulate_bbox(
    bbox_vertices: np.ndarray,
    part_ids: np.ndarray,
    motion_hierarchy: List[Tuple[int, int]],
    is_part_revolute: np.ndarray,
    is_part_prismatic: np.ndarray,
    revolute_plucker: np.ndarray,
    revolute_range: np.ndarray,
    prismatic_axis: np.ndarray,
    prismatic_range: np.ndarray,
    articulation_state: Union[float, np.ndarray],  # Value between 0 and 1
    rotation_origin: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Articulate bbox vertices based on given articulation state.
    
    Args:
        articulation_state: Value between 0 (low limit) and 1 (high limit)
    
    Returns:
        Tuple of (articulated_xyz, articulated_revolute_plucker, articulated_prismatic_axis)
    """
    articulated_revolute_plucker = revolute_plucker.copy()
    articulated_prismatic_axis = prismatic_axis.copy()
    articulated_revolute_range = revolute_range.copy()
    articulated_prismatic_range = prismatic_range.copy()

    part_transformations = [[] for _ in range(len(np.unique(part_ids)))]
    if len(motion_hierarchy) == 0:
        return part_transformations
    
    part_order = get_part_order_from_root(motion_hierarchy)
    # reverse part order
    part_order = part_order[::-1]

    for pid in part_order:
        if pid not in part_ids:
            continue
        affected_part_ids = get_subtree_part_ids(motion_hierarchy, pid)
        applied_tramsformation_matrix = np.eye(4, dtype=np.float32)
        applied_rotation_axis_origin = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        applied_transformation_type = "none"
        children_idxs = affected_part_ids

        if is_part_revolute[pid]:
            low_limit, high_limit = articulated_revolute_range[pid]
            part_articulation_state = articulation_state if isinstance(articulation_state, float) else articulation_state[pid]
            # Interpolate between low and high limits
            angle = part_articulation_state * (high_limit - low_limit) # check if this is radian 
            axis, origin = plucker_to_axis_point(articulated_revolute_plucker[pid])
            transform_matrix = plucker_to_4x4_transform_matrix(articulated_revolute_plucker[pid], angle)
            rotation_axis_origin = origin if rotation_origin is None else rotation_origin[pid]
            applied_tramsformation_matrix[:3, :3] = transform_matrix[:3, :3]
            applied_rotation_axis_origin = rotation_axis_origin
            applied_transformation_type = "rotation"
            
        
        elif is_part_prismatic[pid]:
            low_limit, high_limit = prismatic_range[pid]
            part_articulation_state = articulation_state if isinstance(articulation_state, float) else articulation_state[pid]
            # Interpolate between low and high limits
            displacement = part_articulation_state * (high_limit - low_limit)
            articulated_prismatic_range[pid] = np.array([low_limit - displacement, high_limit - displacement])
            paxis = articulated_prismatic_axis[pid]

            translation = paxis * displacement
            applied_tramsformation_matrix[:3, 3] = translation
            applied_transformation_type = "translation"
        
        if not applied_transformation_type == "none":
            record = {
                "type": applied_transformation_type,
                "matrix": applied_tramsformation_matrix,
                "rotation_axis_origin": applied_rotation_axis_origin
            }
            for child_idx in list(set([pid] + children_idxs)):
                # part_transformations[child_idx].append(record)
                # turn part_id to index
                if child_idx not in part_ids:
                    continue
                try:
                    child_idx_index = np.where(np.unique(part_ids) == child_idx)[0][0]
                    part_transformations[child_idx_index].append(record)
                except:
                    breakpoint()


    return part_transformations
