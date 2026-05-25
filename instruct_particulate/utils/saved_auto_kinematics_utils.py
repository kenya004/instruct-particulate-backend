from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


SAVED_AUTO_KINEMATICS_VISUALIZATION_UP_DIR = "+Y"


@dataclass(frozen=True)
class SavedAutoKinematicsRoot:
    root_path: Path
    mesh_path: Path
    geometry_metadata_path: Path
    parsed_response_path: Path
    lifted_prompts_path: Path
    mesh_up_dir: str
    link_names: list[str]
    joint_specs: list[tuple[int, int, str]]
    link_point_prompts: np.ndarray
    link_point_prompt_normals: np.ndarray
    link_point_prompt_face_ids: np.ndarray | None


def is_saved_auto_kinematics_root(path: Path) -> bool:
    root = path.expanduser()
    if not root.is_dir():
        return False
    required_paths = (
        root / "_auto_kinematics_visualization_mesh.glb",
        root / "auto_kinematics_parsed.json",
        root / "auto_kinematics_lifted_point_prompts.json",
    )
    return all(required_path.exists() for required_path in required_paths)


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {path} to contain a JSON object, got {type(payload).__name__}")
    return payload


def load_saved_auto_kinematics_root(root: Path) -> SavedAutoKinematicsRoot:
    auto_root = root.expanduser().resolve()
    if not is_saved_auto_kinematics_root(auto_root):
        raise FileNotFoundError(
            "Saved auto-kinematics root is missing required files: "
            f"{auto_root}"
        )

    parsed_response_path = auto_root / "auto_kinematics_parsed.json"
    lifted_prompts_path = auto_root / "auto_kinematics_lifted_point_prompts.json"
    parsed_payload = _load_json_object(parsed_response_path)
    lifted_payload = _load_json_object(lifted_prompts_path)

    raw_links = parsed_payload.get("links")
    raw_joints = parsed_payload.get("joints")
    if not isinstance(raw_links, list):
        raise ValueError(f"{parsed_response_path} is missing a 'links' list.")
    if not isinstance(raw_joints, list):
        raise ValueError(f"{parsed_response_path} is missing a 'joints' list.")

    sorted_links = sorted(raw_links, key=lambda link: int(link["link_id"]))
    link_names: list[str] = []
    expected_link_ids = list(range(len(sorted_links)))
    observed_link_ids: list[int] = []
    for raw_link in sorted_links:
        if not isinstance(raw_link, dict):
            raise ValueError(
                f"Each entry in {parsed_response_path}['links'] must be an object."
            )
        link_id = int(raw_link.get("link_id", -1))
        link_name = str(raw_link.get("name", "")).strip()
        if not link_name:
            raise ValueError(
                f"Link {link_id} in {parsed_response_path} is missing a non-empty name."
            )
        observed_link_ids.append(link_id)
        link_names.append(link_name)
    if observed_link_ids != expected_link_ids:
        raise ValueError(
            f"Link IDs in {parsed_response_path} must be continuous from 0, got {observed_link_ids}"
        )

    sorted_joints = sorted(raw_joints, key=lambda joint: int(joint["joint_id"]))
    joint_specs: list[tuple[int, int, str]] = []
    expected_joint_ids = list(range(len(sorted_joints)))
    observed_joint_ids: list[int] = []
    for raw_joint in sorted_joints:
        if not isinstance(raw_joint, dict):
            raise ValueError(
                f"Each entry in {parsed_response_path}['joints'] must be an object."
            )
        joint_id = int(raw_joint.get("joint_id", -1))
        parent_link_id = int(raw_joint.get("parent_link_id", -1))
        child_link_id = int(raw_joint.get("child_link_id", -1))
        joint_type = str(raw_joint.get("joint_type", "")).strip().lower()
        if joint_type not in {"revolute", "prismatic"}:
            raise ValueError(
                f"Joint {joint_id} in {parsed_response_path} has invalid joint_type={joint_type!r}."
            )
        observed_joint_ids.append(joint_id)
        joint_specs.append((parent_link_id, child_link_id, joint_type))
    if observed_joint_ids != expected_joint_ids:
        raise ValueError(
            f"Joint IDs in {parsed_response_path} must be continuous from 0, got {observed_joint_ids}"
        )

    raw_lifted_links = lifted_payload.get("links")
    if not isinstance(raw_lifted_links, list):
        raise ValueError(f"{lifted_prompts_path} is missing a 'links' list.")

    raw_mesh_path = lifted_payload.get("point_prompt_face_mesh_path")
    if raw_mesh_path is None:
        raw_mesh_path = lifted_payload.get("visualization_mesh_path")
    if raw_mesh_path is None:
        observed_mesh_paths = {
            str(record.get("point_prompt_face_mesh_path")).strip()
            for record in raw_lifted_links
            if record.get("point_prompt_face_mesh_path") is not None
        }
        if len(observed_mesh_paths) == 1:
            raw_mesh_path = observed_mesh_paths.pop()
        elif len(observed_mesh_paths) > 1:
            raise ValueError(
                f"{lifted_prompts_path} contains inconsistent point_prompt_face_mesh_path values."
            )
    if raw_mesh_path is None:
        mesh_path = auto_root / "_auto_kinematics_visualization_mesh.glb"
    elif not isinstance(raw_mesh_path, str) or not raw_mesh_path.strip():
        raise ValueError(
            f"{lifted_prompts_path} has invalid point_prompt_face_mesh_path={raw_mesh_path!r}."
        )
    else:
        candidate_mesh_path = Path(raw_mesh_path)
        mesh_path = (
            candidate_mesh_path.expanduser().resolve()
            if candidate_mesh_path.is_absolute()
            else (auto_root / candidate_mesh_path).resolve()
        )
    if not mesh_path.exists():
        raise FileNotFoundError(f"Saved visualization mesh does not exist: {mesh_path}")

    records_by_link_id: dict[int, dict[str, Any]] = {}
    for raw_record in raw_lifted_links:
        if not isinstance(raw_record, dict):
            raise ValueError(
                f"Each entry in {lifted_prompts_path}['links'] must be an object."
            )
        link_id = int(raw_record.get("link_id", -1))
        if link_id in records_by_link_id:
            raise ValueError(
                f"Duplicate link_id={link_id} in {lifted_prompts_path}."
            )
        records_by_link_id[link_id] = raw_record

    prompt_points: list[np.ndarray] = []
    prompt_normals: list[np.ndarray] = []
    prompt_face_ids: list[int] = []
    have_prompt_face_ids = True
    for link_id in expected_link_ids:
        if link_id not in records_by_link_id:
            raise ValueError(
                f"{lifted_prompts_path} is missing lifted prompt data for link_id={link_id}."
            )
        record = records_by_link_id[link_id]
        point_prompt_normalized = np.asarray(
            record.get("point_prompt_normalized"),
            dtype=np.float32,
        )
        if point_prompt_normalized.shape != (3,):
            raise ValueError(
                f"Lifted point_prompt_normalized for link_id={link_id} must have shape (3,), "
                f"got {point_prompt_normalized.shape}"
            )
        normal = np.asarray(record.get("normal"), dtype=np.float32)
        if normal.shape != (3,):
            raise ValueError(
                f"Lifted normal for link_id={link_id} must have shape (3,), got {normal.shape}"
            )
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-8:
            raise ValueError(
                f"Lifted normal for link_id={link_id} is degenerate: {normal.tolist()}"
            )
        face_id = record.get("point_prompt_face_id")
        if face_id is None:
            have_prompt_face_ids = False
        elif have_prompt_face_ids:
            prompt_face_ids.append(int(face_id))
        prompt_points.append(point_prompt_normalized.astype(np.float32, copy=False))
        prompt_normals.append((normal / normal_norm).astype(np.float32, copy=False))

    return SavedAutoKinematicsRoot(
        root_path=auto_root,
        mesh_path=mesh_path,
        geometry_metadata_path=auto_root / "_geometry_metadata.npz",
        parsed_response_path=parsed_response_path,
        lifted_prompts_path=lifted_prompts_path,
        mesh_up_dir=SAVED_AUTO_KINEMATICS_VISUALIZATION_UP_DIR,
        link_names=list(link_names),
        joint_specs=list(joint_specs),
        link_point_prompts=np.stack(prompt_points, axis=0).astype(np.float32, copy=False),
        link_point_prompt_normals=np.stack(prompt_normals, axis=0).astype(np.float32, copy=False),
        link_point_prompt_face_ids=(
            np.asarray(prompt_face_ids, dtype=np.int64)
            if have_prompt_face_ids
            else None
        ),
    )
