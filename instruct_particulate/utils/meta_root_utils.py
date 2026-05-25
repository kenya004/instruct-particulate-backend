from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from instruct_particulate.utils.data_utils import get_face_to_link_mapping, load_obj_raw_preserve


def is_supported_meta_root(meta_root: Path) -> bool:
    """Returns whether `meta_root` matches a supported preprocessed asset layout."""
    if not meta_root.is_dir():
        return False
    original_obj_path = meta_root / "original.obj"
    if not original_obj_path.exists():
        return False
    if (meta_root / "seg.npy").exists() and (meta_root / "kinematic.json").exists():
        return True
    if (
        (meta_root / "meta.npz").exists()
        and (meta_root / "link_axes_plucker.npz").exists()
        and (meta_root / "link_range.npz").exists()
    ):
        return True
    return False


def load_kinematic_metadata_json(
    json_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Loads `links` and `joints` from a kinematic JSON file."""
    with json_path.open("r", encoding="utf-8") as fh:
        annotation = json.load(fh)
    if not isinstance(annotation, Mapping):
        raise ValueError(
            f"Expected {json_path} to contain a JSON object, got {type(annotation).__name__}"
        )
    raw_links = annotation.get("links", [])
    raw_joints = annotation.get("joints", [])
    if not isinstance(raw_links, Sequence) or isinstance(raw_links, (str, bytes)):
        raise TypeError(
            f"Expected 'links' in {json_path} to be a sequence, got {type(raw_links).__name__}"
        )
    if not isinstance(raw_joints, Sequence) or isinstance(raw_joints, (str, bytes)):
        raise TypeError(
            f"Expected 'joints' in {json_path} to be a sequence, got {type(raw_joints).__name__}"
        )
    return [dict(link) for link in raw_links], [dict(joint) for joint in raw_joints]


def _legacy_motion_arrays_from_particulate_meta(
    *,
    meta_root: Path,
    num_links: int,
) -> dict[str, np.ndarray]:
    link_axes_plucker_path = meta_root / "link_axes_plucker.npz"
    link_range_path = meta_root / "link_range.npz"
    if not link_axes_plucker_path.exists() or not link_range_path.exists():
        raise FileNotFoundError(
            f"Legacy particulate-format meta-root {meta_root} is missing "
            "`link_axes_plucker.npz` or `link_range.npz`"
        )

    link_axes_plucker_raw = np.load(link_axes_plucker_path)
    link_range_raw = np.load(link_range_path)
    link_axes_plucker = np.zeros((num_links, 12), dtype=np.float32)
    link_range = np.zeros((num_links, 4), dtype=np.float32)
    for key, value in link_axes_plucker_raw.items():
        link_axes_plucker[int(key)] = np.asarray(value, dtype=np.float32)
    for key, value in link_range_raw.items():
        link_range[int(key)] = np.asarray(value, dtype=np.float32)
    return {
        "link_axes_plucker": link_axes_plucker,
        "link_range": link_range,
    }


def load_meta_root_annotation(
    meta_root: Path,
) -> tuple[Path, np.ndarray, list[dict[str, Any]], list[dict[str, Any]], str]:
    """Loads either the new or legacy particulate meta-root format.

    Returns `(original_obj_path, face_link_ids, links, joints, format_name)`.
    """
    original_obj_path = meta_root / "original.obj"
    if not original_obj_path.exists():
        raise FileNotFoundError(f"Meta-root input {meta_root} is missing required file original.obj")

    face_seg_path = meta_root / "seg.npy"
    kinematic_json_path = meta_root / "kinematic.json"
    if face_seg_path.exists() and kinematic_json_path.exists():
        vertices, faces = load_obj_raw_preserve(original_obj_path)
        face_gt_link_ids = np.load(face_seg_path).astype(np.int64, copy=False)
        if face_gt_link_ids.shape != (len(faces),):
            raise ValueError(
                f"Expected {face_seg_path} to contain one link id per face, got shape "
                f"{face_gt_link_ids.shape} for {len(faces)} faces"
            )
        links, joints = load_kinematic_metadata_json(kinematic_json_path)
        return original_obj_path, face_gt_link_ids, links, joints, "kinematic_json"

    legacy_meta_path = meta_root / "meta.npz"
    if legacy_meta_path.exists():
        vertices, faces = load_obj_raw_preserve(original_obj_path)
        meta = np.load(legacy_meta_path, allow_pickle=True)
        if "vert_to_bone" not in meta or "bone_structure" not in meta:
            raise ValueError(
                f"Legacy particulate-format meta-root {meta_root} is missing "
                "`vert_to_bone` or `bone_structure` inside meta.npz"
            )

        vert_to_bone = np.asarray(meta["vert_to_bone"], dtype=np.int64)
        if vert_to_bone.shape != (len(vertices),):
            raise ValueError(
                f"Expected vert_to_bone in {legacy_meta_path} to have shape ({len(vertices)},), "
                f"got {vert_to_bone.shape}"
            )

        face_gt_link_ids = get_face_to_link_mapping(vert_to_bone, faces)
        if face_gt_link_ids is None:
            raise ValueError(
                f"Cannot derive face segmentation for legacy particulate-format meta-root {meta_root}: "
                "some faces span multiple link IDs"
            )
        face_gt_link_ids = np.asarray(face_gt_link_ids, dtype=np.int64, copy=False)

        bone_structure = np.asarray(meta["bone_structure"], dtype=np.int64)
        if bone_structure.ndim != 2 or bone_structure.shape[1] != 2:
            raise ValueError(
                f"Expected bone_structure in {legacy_meta_path} to have shape (J, 2), "
                f"got {bone_structure.shape}"
            )

        num_links = int(max(vert_to_bone.max(initial=-1), bone_structure.max(initial=-1)) + 1)
        motion_arrays = _legacy_motion_arrays_from_particulate_meta(
            meta_root=meta_root,
            num_links=num_links,
        )
        links = [
            {
                "link_id": link_id,
                "name": f"link_{link_id}",
            }
            for link_id in range(num_links)
        ]
        joints: list[dict[str, Any]] = []
        for joint_id, (parent_link_id, child_link_id) in enumerate(bone_structure.tolist()):
            parent_link_id = int(parent_link_id)
            child_link_id = int(child_link_id)
            child_axes = motion_arrays["link_axes_plucker"][child_link_id]
            child_range = motion_arrays["link_range"][child_link_id]
            is_revolute = bool(np.linalg.norm(child_axes[:3]) > 0.0)
            is_prismatic = bool(np.linalg.norm(child_axes[6:9]) > 0.0)
            joints.append(
                {
                    "joint_id": int(joint_id),
                    "parent_link_id": parent_link_id,
                    "child_link_id": child_link_id,
                    "joint_description": f"link_{parent_link_id} -> link_{child_link_id}",
                    "is_revolute": is_revolute,
                    "is_prismatic": is_prismatic,
                    "revolute_axis": (
                        np.asarray(child_axes[:6], dtype=np.float32).tolist() if is_revolute else None
                    ),
                    "prismatic_axis": (
                        np.asarray(child_axes[6:], dtype=np.float32).tolist() if is_prismatic else None
                    ),
                    "revolute_range": (
                        np.asarray(child_range[:2], dtype=np.float32).tolist() if is_revolute else None
                    ),
                    "prismatic_range": (
                        np.asarray(child_range[2:], dtype=np.float32).tolist() if is_prismatic else None
                    ),
                }
            )

        return original_obj_path, face_gt_link_ids, links, joints, "legacy_particulate"

    raise FileNotFoundError(
        f"Meta-root input {meta_root} is missing supported metadata files. Expected either "
        "seg.npy + kinematic.json or meta.npz + link_axes_plucker.npz + link_range.npz."
    )
