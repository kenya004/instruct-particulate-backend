from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw

from instruct_particulate.utils.data_utils import (
    load_trimesh,
    normalize_mesh,
    reorient_mesh_to_z_up,
    up_dir_rotation_matrix,
)
from instruct_particulate.utils.inference_utils import (
    prepare_inference_batch_from_mesh_with_prompts,
)
from instruct_particulate.utils.postprocessing_utils import resolve_point_prompt_face_ids


DEFAULT_AUTO_KINEMATICS_AZIMUTHS = (-45.0, 45.0, 135.0, 215.0)
DEFAULT_RENDER_PITCH_DEG = 110.0
DEFAULT_RENDER_ELEVATION_DEG = 90.0 - DEFAULT_RENDER_PITCH_DEG
DEFAULT_RENDER_DISTANCE = 2.0
DEFAULT_RENDER_RESOLUTION = 512
DEFAULT_POINT_PROMPT_SNAP_RADIUS_PX = 24
DEFAULT_BLENDER_RENDER_ENGINE = "CYCLES"
DEFAULT_BLENDER_RENDER_SAMPLES = 256


@dataclass(frozen=True)
class RenderedView:
    image_id: int
    image_path: Path
    camera_path: Path
    azimuth_deg: float
    elevation_deg: float
    width: int
    height: int
    intrinsic: np.ndarray
    camera_to_world: np.ndarray
    world_to_camera: np.ndarray
    face_ids: np.ndarray
    hit_points: np.ndarray
    normals: np.ndarray
    depth: np.ndarray


def detect_provider(model_id: str) -> str:
    lowered = str(model_id).strip().lower()
    if "gemini" in lowered:
        return "gemini"
    return "openai"


def image_detail_for_model(model_id: str) -> str:
    normalized = str(model_id).strip().lower()
    match = re.match(r"^gpt-(\d+)\.(\d+)", normalized)
    if match is not None:
        major = int(match.group(1))
        minor = int(match.group(2))
        if major > 5 or (major == 5 and minor >= 4):
            return "original"
    return "high"


def _normalize_reasoning_effort(reasoning_effort: str | None) -> str:
    normalized = str(reasoning_effort or "").strip().lower()
    if normalized == "med":
        normalized = "medium"
    if normalized not in {"low", "medium", "high"}:
        return "medium"
    return normalized


def _build_gemini_thinking_config(types: Any, reasoning_effort: str) -> Any | None:
    try:
        return types.ThinkingConfig(thinking_level=_normalize_reasoning_effort(reasoning_effort))
    except Exception:
        return None


def _extract_openai_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(response, "output", None)
    text_fragments: list[str] = []
    if isinstance(output, list):
        for item in output:
            if item is None:
                continue
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if item_type != "message":
                continue
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                if part is None:
                    continue
                part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                if part_type not in {"output_text", "text", "input_text"}:
                    continue
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if isinstance(text, str) and text.strip():
                    text_fragments.append(text.strip())
    if text_fragments:
        return "\n".join(text_fragments).strip()

    if isinstance(response, dict):
        fallback = response.get("output_text")
        return fallback.strip() if isinstance(fallback, str) else ""

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            fallback = dumped.get("output_text")
            if isinstance(fallback, str) and fallback.strip():
                return fallback.strip()
    return ""


def extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            raise ValueError("Model response was valid JSON but not a JSON object.")
        return parsed

    raise ValueError("Model response was not parseable as a JSON object.")


def _load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    return path.read_text(encoding="utf-8").strip()


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"


def _encode_image_data(path: Path) -> tuple[bytes, str]:
    return path.read_bytes(), _guess_mime_type(path)


def resolve_blender_binary(explicit_path: str | os.PathLike[str] | None = None) -> Path:
    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Blender binary does not exist: {candidate}")
        return candidate

    home = Path.home()
    candidates = [
        home / "blender" / "blender",
        home / "blender/blender",
    ]
    for directory in sorted(home.glob("blender*-linux-x64"), reverse=True):
        candidates.append(directory / "blender")

    which_blender = shutil.which("blender")
    if which_blender:
        candidates.append(Path(which_blender))

    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find Blender. Pass --blender-bin or install Blender under your home directory."
    )


def render_mesh_auto_kinematics_views(
    mesh: trimesh.Trimesh,
    *,
    output_dir: Path,
    azimuths_deg: Sequence[float],
    mesh_path: str | os.PathLike[str] | None = None,
    up_dir: str | None = None,
    pitch_deg: float = DEFAULT_RENDER_PITCH_DEG,
    camera_distance: float = DEFAULT_RENDER_DISTANCE,
    resolution: int = DEFAULT_RENDER_RESOLUTION,
    blender_bin: str | os.PathLike[str] | None = None,
) -> list[RenderedView]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_views: list[RenderedView] = []
    azimuths = [float(azimuth_deg) for azimuth_deg in azimuths_deg]
    blender_path = resolve_blender_binary(blender_bin)
    repo_root = Path(__file__).resolve().parents[2]
    blender_script_path = repo_root / "scripts" / "render_auto_kinematics_blender.py"
    if not blender_script_path.exists():
        raise FileNotFoundError(f"Auto-kinematics Blender script does not exist: {blender_script_path}")

    if mesh_path is None:
        render_mesh_path = output_dir / "_auto_kinematics_render_mesh.glb"
        mesh.export(render_mesh_path)
    else:
        render_mesh_path = Path(mesh_path).expanduser().resolve()
        if not render_mesh_path.exists():
            raise FileNotFoundError(f"Auto-kinematics render mesh path does not exist: {render_mesh_path}")
    command = [
        str(blender_path),
        "-b",
        "-P",
        str(blender_script_path),
        "--",
        "--mesh-path",
        str(render_mesh_path),
        "--output-dir",
        str(output_dir),
        "--resolution",
        str(int(resolution)),
        "--camera-distance",
        repr(float(camera_distance)),
        "--pitch-deg",
        repr(float(pitch_deg)),
        "--engine",
        DEFAULT_BLENDER_RENDER_ENGINE,
        "--samples",
        str(int(DEFAULT_BLENDER_RENDER_SAMPLES)),
        "--azimuths",
        *[repr(float(azimuth_deg)) for azimuth_deg in azimuths],
    ]
    if up_dir is not None and str(up_dir).strip():
        command.extend(["--up-dir", str(up_dir)])
    print(
        f"Rendering {len(azimuths)} auto-kinematics views with Blender at {blender_path}"
    )
    subprocess.run(command, check=True)

    total_views = len(azimuths)
    for image_id, azimuth_deg in enumerate(azimuths):
        image_path = output_dir / f"view_{image_id:03d}.png"
        camera_path = output_dir / f"view_{image_id:03d}_camera.npz"
        if not image_path.exists():
            raise FileNotFoundError(f"Blender render output is missing image: {image_path}")
        if not camera_path.exists():
            raise FileNotFoundError(f"Blender render output is missing camera data: {camera_path}")

        print(
            f"Loading auto-kinematics view {image_id + 1}/{total_views} "
            f"(azimuth={float(azimuth_deg):.1f} deg)"
        )
        camera_payload = np.load(camera_path)
        intrinsic = np.asarray(camera_payload["intrinsic"], dtype=np.float64)
        camera_to_world = np.asarray(camera_payload["camera_to_world"], dtype=np.float64)
        world_to_camera = np.asarray(camera_payload["world_to_camera"], dtype=np.float64)
        face_ids = np.asarray(camera_payload["face_ids"], dtype=np.int32)
        hit_points = np.asarray(camera_payload["hit_points"], dtype=np.float32)
        normals = np.asarray(camera_payload["normals"], dtype=np.float32)
        depth = np.asarray(camera_payload["depth"], dtype=np.float32)
        elevation_deg = (
            float(camera_payload["elevation_deg"])
            if "elevation_deg" in camera_payload.files
            else float(DEFAULT_RENDER_ELEVATION_DEG)
        )
        rendered_views.append(
            RenderedView(
                image_id=int(image_id),
                image_path=image_path,
                camera_path=camera_path,
                azimuth_deg=float(azimuth_deg),
                elevation_deg=elevation_deg,
                width=int(resolution),
                height=int(resolution),
                intrinsic=intrinsic.astype(np.float32),
                camera_to_world=camera_to_world.astype(np.float32),
                world_to_camera=world_to_camera.astype(np.float32),
                face_ids=face_ids.astype(np.int32, copy=False),
                hit_points=hit_points.astype(np.float32, copy=False),
                normals=normals.astype(np.float32, copy=False),
                depth=depth.astype(np.float32, copy=False),
            )
        )
    return rendered_views


def build_auto_kinematics_user_prompt(rendered_views: Sequence[RenderedView]) -> str:
    lines = [
        "Analyze the rendered object views and infer the rigid-part kinematic structure.",
        "Return only a valid JSON object matching the schema in the system instruction.",
        "Use only rigid parts. Do not include non-rigid or decorative subparts.",
        "Do not omit small independently moving rigid parts; controls or attachments such as knobs, buttons, switches, sliders, levers, handles, trays, lids, doors, or latches must each be their own link when they move independently.",
        "For every link, choose one representative visible surface point.",
        "For every point_prompt, return integer x and y bins in the inclusive range [0, 1000].",
        "These bins represent normalized image coordinates with top-left origin:",
        "x_normalized = x / 1000 and y_normalized = y / 1000.",
        "So (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner.",
        "Each image is identified by the explicit image ID text that appears immediately before it.",
        "Point prompts for different links do not need to come from the same image.",
        "For each link independently, choose the image where that link is most visible and easiest to localize.",
        f"Number of images: {len(rendered_views)}.",
    ]
    for view in rendered_views:
        lines.append(
            f"Image ID {view.image_id}: square render with size {view.width}x{view.height} pixels."
        )
    return "\n".join(lines)


def _get_openai_api_key(explicit_api_key: str | None) -> str | None:
    return explicit_api_key or os.getenv("OPENAI_API_KEY")


def _get_gemini_api_key(explicit_api_key: str | None) -> str | None:
    return explicit_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def call_auto_kinematics_model(
    *,
    model_id: str,
    system_prompt_path: Path,
    rendered_views: Sequence[RenderedView],
    provider: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str = "medium",
) -> dict[str, Any]:
    resolved_provider = detect_provider(model_id) if provider is None else provider
    system_prompt = _load_text(system_prompt_path)
    user_prompt = build_auto_kinematics_user_prompt(rendered_views)
    if resolved_provider == "openai":
        return _call_openai(
            model_id=model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            rendered_views=rendered_views,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
    if resolved_provider == "gemini":
        return _call_gemini(
            model_id=model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            rendered_views=rendered_views,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
    raise ValueError(f"Unsupported provider: {resolved_provider}")


def _call_openai(
    *,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    rendered_views: Sequence[RenderedView],
    api_key: str | None,
    reasoning_effort: str,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("OpenAI support requires the `openai` package to be installed.") from exc

    resolved_api_key = _get_openai_api_key(api_key)
    if not resolved_api_key:
        raise EnvironmentError("Missing OpenAI API key. Set OPENAI_API_KEY.")

    content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
    image_detail = image_detail_for_model(model_id)
    for view in rendered_views:
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"Image ID {view.image_id}: the next image corresponds to this ID. "
                    "Use this integer in point_prompt.image_id."
                ),
            }
        )
        image_bytes, mime_type = _encode_image_data(view.image_path)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{encoded}",
                "detail": image_detail,
            }
        )

    client = OpenAI(api_key=resolved_api_key)
    request_payload: dict[str, Any] = {
        "model": model_id,
        "reasoning": {"effort": _normalize_reasoning_effort(reasoning_effort)},
        "text": {"format": {"type": "json_object"}},
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        "store": False,
    }
    response = client.responses.create(**request_payload)
    content_text = _extract_openai_response_text(response)
    if not content_text:
        raise ValueError("Unexpected OpenAI response format: response output text was empty.")
    return extract_json_object(content_text)


def _call_gemini(
    *,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    rendered_views: Sequence[RenderedView],
    api_key: str | None,
    reasoning_effort: str,
) -> dict[str, Any]:
    resolved_api_key = _get_gemini_api_key(api_key)
    if not resolved_api_key:
        raise EnvironmentError("Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError("Gemini support requires the `google-genai` package.") from exc

    client = genai.Client(api_key=resolved_api_key)
    parts: list[Any] = [types.Part.from_text(text=user_prompt)]
    for view in rendered_views:
        parts.append(
            types.Part.from_text(
                text=(
                    f"Image ID {view.image_id}: the next image corresponds to this ID. "
                    "Use this integer in point_prompt.image_id."
                )
            )
        )
        image_bytes, mime_type = _encode_image_data(view.image_path)
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

    contents = [types.Content(role="user", parts=parts)]
    config_kwargs: dict[str, Any] = {
        "system_instruction": system_prompt,
        "response_mime_type": "application/json",
        "temperature": 0.1,
    }
    thinking_config = _build_gemini_thinking_config(types, reasoning_effort)
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    response_text = getattr(response, "text", None)
    if not response_text:
        raise ValueError("Gemini returned an empty response.")
    return extract_json_object(response_text)


def parse_auto_kinematics_response(response: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise TypeError(f"Expected a JSON object, got {type(response).__name__}")
    total_links = int(response.get("total_links", 0))
    raw_links = response.get("links")
    raw_joints = response.get("joints")
    if not isinstance(raw_links, list):
        raise ValueError("Response must contain a 'links' list.")
    if not isinstance(raw_joints, list):
        raise ValueError("Response must contain a 'joints' list.")
    if total_links != len(raw_links):
        raise ValueError(
            f"total_links={total_links} does not match len(links)={len(raw_links)}"
        )

    links: list[dict[str, Any]] = []
    seen_link_ids: set[int] = set()
    for raw_link in raw_links:
        if not isinstance(raw_link, dict):
            raise ValueError("Each entry in 'links' must be an object.")
        link_id = int(raw_link.get("link_id", -1))
        if link_id in seen_link_ids:
            raise ValueError(f"Duplicate link_id {link_id} in response.")
        seen_link_ids.add(link_id)
        link_name = str(raw_link.get("name", "")).strip()
        if not link_name:
            raise ValueError(f"Link {link_id} is missing a non-empty name.")
        point_prompt = raw_link.get("point_prompt")
        if not isinstance(point_prompt, dict):
            raise ValueError(f"Link {link_id} is missing a point_prompt object.")
        image_id = int(point_prompt.get("image_id", -1))
        x_bin = point_prompt.get("x")
        y_bin = point_prompt.get("y")
        if x_bin is None or y_bin is None:
            coordinate = point_prompt.get("coordinate")
            if (
                not isinstance(coordinate, (list, tuple))
                or len(coordinate) != 2
            ):
                raise ValueError(
                    "Link "
                    f"{link_id} point_prompt must contain integer x/y bins or a length-2 coordinate array, "
                    f"got {point_prompt!r}"
                )
            x_bin = coordinate[0]
            y_bin = coordinate[1]
        x_bin = int(round(float(x_bin)))
        y_bin = int(round(float(y_bin)))
        if not (0 <= x_bin <= 1000 and 0 <= y_bin <= 1000):
            raise ValueError(
                f"Link {link_id} point_prompt bins must lie in [0, 1000], got x={x_bin}, y={y_bin}"
            )
        links.append(
            {
                "link_id": link_id,
                "name": link_name,
                "point_prompt": {
                    "image_id": image_id,
                    "x": x_bin,
                    "y": y_bin,
                    "normalized_coordinate": [
                        float(x_bin) / 1000.0,
                        float(y_bin) / 1000.0,
                    ],
                },
            }
        )
    links.sort(key=lambda link: int(link["link_id"]))
    expected_link_ids = list(range(len(links)))
    observed_link_ids = [int(link["link_id"]) for link in links]
    if observed_link_ids != expected_link_ids:
        raise ValueError(
            f"Link IDs must be continuous from 0, got {observed_link_ids}"
        )

    joints: list[dict[str, Any]] = []
    seen_joint_ids: set[int] = set()
    for raw_joint in raw_joints:
        if not isinstance(raw_joint, dict):
            raise ValueError("Each entry in 'joints' must be an object.")
        joint_id = int(raw_joint.get("joint_id", -1))
        if joint_id in seen_joint_ids:
            raise ValueError(f"Duplicate joint_id {joint_id} in response.")
        seen_joint_ids.add(joint_id)
        parent_link_id = int(raw_joint.get("parent_link_id", -1))
        child_link_id = int(raw_joint.get("child_link_id", -1))
        joint_type = str(raw_joint.get("joint_type", "")).strip().lower()
        if joint_type not in {"revolute", "prismatic"}:
            raise ValueError(
                f"Joint {joint_id} must have joint_type 'revolute' or 'prismatic', got {joint_type!r}"
            )
        joints.append(
            {
                "joint_id": joint_id,
                "parent_link_id": parent_link_id,
                "child_link_id": child_link_id,
                "joint_type": joint_type,
            }
        )
    joints.sort(key=lambda joint: int(joint["joint_id"]))
    expected_joint_ids = list(range(len(joints)))
    observed_joint_ids = [int(joint["joint_id"]) for joint in joints]
    if observed_joint_ids != expected_joint_ids:
        raise ValueError(
            f"Joint IDs must be continuous from 0, got {observed_joint_ids}"
        )

    link_names = [str(link["name"]) for link in links]
    joint_specs = [
        (
            int(joint["parent_link_id"]),
            int(joint["child_link_id"]),
            str(joint["joint_type"]),
        )
        for joint in joints
    ]
    return {
        "object_name": str(response.get("object_name", "")).strip(),
        "total_links": total_links,
        "links": links,
        "joints": joints,
        "link_names": link_names,
        "joint_specs": joint_specs,
    }


def _resolve_prompt_pixel(
    view: RenderedView,
    *,
    point_prompt: dict[str, Any],
    max_snap_radius_px: int,
) -> tuple[int, int, float, float, float]:
    raw_x = float(np.clip(float(point_prompt["x"]) / 1000.0, 0.0, 1.0)) * float(view.width - 1)
    raw_y = float(np.clip(float(point_prompt["y"]) / 1000.0, 0.0, 1.0)) * float(view.height - 1)
    requested_x = int(np.floor(raw_x + 0.5))
    requested_y = int(np.floor(raw_y + 0.5))
    raw_x = float(np.clip(raw_x, 0.0, float(view.width - 1)))
    raw_y = float(np.clip(raw_y, 0.0, float(view.height - 1)))
    clipped_x = int(np.clip(requested_x, 0, view.width - 1))
    clipped_y = int(np.clip(requested_y, 0, view.height - 1))
    if view.face_ids[clipped_y, clipped_x] >= 0:
        return clipped_x, clipped_y, float(np.hypot(clipped_x - raw_x, clipped_y - raw_y)), raw_x, raw_y

    valid_pixels = np.argwhere(view.face_ids >= 0)
    if valid_pixels.size == 0:
        raise ValueError(f"Rendered view {view.image_id} does not contain any visible mesh pixels.")
    deltas = valid_pixels - np.asarray([clipped_y, clipped_x], dtype=np.int32)
    squared_distances = (deltas.astype(np.float64) ** 2).sum(axis=1)
    best_index = int(np.argmin(squared_distances))
    best_distance = float(np.sqrt(squared_distances[best_index]))
    if best_distance > float(max_snap_radius_px):
        raise ValueError(
            "Point prompt lies too far from the rendered object surface: "
            f"view={view.image_id}, requested=({raw_x:.2f}, {raw_y:.2f}), nearest_distance={best_distance:.2f}px"
        )
    best_y, best_x = valid_pixels[best_index]
    return int(best_x), int(best_y), best_distance, raw_x, raw_y


def lift_point_prompts_from_rendered_views(
    *,
    links: Sequence[dict[str, Any]],
    rendered_views: Sequence[RenderedView],
    max_snap_radius_px: int = DEFAULT_POINT_PROMPT_SNAP_RADIUS_PX,
) -> dict[str, Any]:
    views_by_id = {int(view.image_id): view for view in rendered_views}
    lifted_points: list[np.ndarray] = []
    lifted_normals: list[np.ndarray] = []
    debug_records: list[dict[str, Any]] = []
    for link in links:
        point_prompt = link["point_prompt"]
        image_id = int(point_prompt["image_id"])
        if image_id not in views_by_id:
            raise ValueError(
                f"Link {link['link_id']} references image_id={image_id}, which is not among the rendered views."
            )
        view = views_by_id[image_id]
        resolved_x, resolved_y, snap_distance_px, requested_x_px, requested_y_px = _resolve_prompt_pixel(
            view,
            point_prompt=point_prompt,
            max_snap_radius_px=max_snap_radius_px,
        )
        point = np.asarray(view.hit_points[resolved_y, resolved_x], dtype=np.float32)
        normal = np.asarray(view.normals[resolved_y, resolved_x], dtype=np.float32)
        if not np.isfinite(point).all():
            raise ValueError(
                f"Resolved point prompt for link {link['link_id']} did not map to a finite 3D point."
            )
        normal_norm = float(np.linalg.norm(normal))
        if not np.isfinite(normal).all() or normal_norm <= 1e-8:
            raise ValueError(
                f"Resolved point prompt for link {link['link_id']} did not map to a valid normal."
            )
        normal = normal / normal_norm
        lifted_points.append(point)
        lifted_normals.append(normal.astype(np.float32, copy=False))
        debug_records.append(
            {
                "link_id": int(link["link_id"]),
                "name": str(link["name"]),
                "image_id": image_id,
                "requested_coordinate_bins": [
                    int(point_prompt["x"]),
                    int(point_prompt["y"]),
                ],
                "requested_coordinate_normalized": [
                    float(point_prompt["x"]) / 1000.0,
                    float(point_prompt["y"]) / 1000.0,
                ],
                "requested_coordinate_pixels": [
                    float(requested_x_px),
                    float(requested_y_px),
                ],
                "resolved_coordinate": [int(resolved_x), int(resolved_y)],
                "snap_distance_px": float(snap_distance_px),
                "point_prompt_render_normalized": point.astype(np.float32).tolist(),
                "point_prompt_normalized": point.astype(np.float32).tolist(),
                "normal": normal.astype(np.float32).tolist(),
            }
        )
    return {
        "points": np.stack(lifted_points, axis=0).astype(np.float32),
        "normals": np.stack(lifted_normals, axis=0).astype(np.float32),
        "records": debug_records,
    }


def save_point_prompt_visualization(
    *,
    mesh_path: Path,
    point_prompts_world: np.ndarray,
    output_path: Path,
    blender_bin: str | os.PathLike[str] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_path = output_path.with_name(f"{output_path.stem}_points.json")
    prompts_path.write_text(
        json.dumps(np.asarray(point_prompts_world, dtype=np.float32).tolist()),
        encoding="utf-8",
    )
    blender_path = resolve_blender_binary(blender_bin)
    repo_root = Path(__file__).resolve().parents[2]
    blender_script_path = repo_root / "scripts" / "export_point_prompt_visualization_blender.py"
    if not blender_script_path.exists():
        raise FileNotFoundError(
            f"Point-prompt visualization Blender script does not exist: {blender_script_path}"
        )
    subprocess.run(
        [
            str(blender_path),
            "-b",
            "-P",
            str(blender_script_path),
            "--",
            "--mesh-path",
            str(mesh_path),
            "--points-json",
            str(prompts_path),
            "--output-path",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def save_2d_point_prompt_visualizations(
    *,
    rendered_views: Sequence[RenderedView],
    lifted_prompt_records: Sequence[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_image_id: dict[int, list[dict[str, Any]]] = {}
    for record in lifted_prompt_records:
        records_by_image_id.setdefault(int(record["image_id"]), []).append(record)

    colors = [
        (255, 64, 64),
        (64, 200, 255),
        (255, 196, 64),
        (120, 255, 120),
        (255, 120, 220),
        (180, 140, 255),
    ]
    saved_paths: list[Path] = []
    for view in rendered_views:
        image_records = records_by_image_id.get(int(view.image_id), [])
        if not image_records:
            continue
        with Image.open(view.image_path) as image:
            annotated = image.convert("RGBA")
        draw = ImageDraw.Draw(annotated)
        for record_idx, record in enumerate(image_records):
            x, y = [int(v) for v in record["resolved_coordinate"]]
            color = colors[record_idx % len(colors)]
            radius = 8
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
            draw.line((x - 12, y, x + 12, y), fill=color, width=2)
            draw.line((x, y - 12, x, y + 12), fill=color, width=2)
            label = f"{int(record['link_id'])}: {record['name']}"
            draw.text((x + 12, y - 18), label, fill=color)
        output_path = output_dir / f"view_{int(view.image_id):03d}.png"
        annotated.save(output_path)
        saved_paths.append(output_path)
    return saved_paths


def _denormalize_points(
    points: np.ndarray,
    *,
    center: np.ndarray,
    scale: float,
) -> np.ndarray:
    return (np.asarray(points, dtype=np.float32) / np.float32(scale) + center).astype(
        np.float32,
        copy=False,
    )


def load_normalized_visualization_mesh(
    mesh_path: Path,
    *,
    mesh_up_dir: str = "+Y",
) -> trimesh.Trimesh:
    """Loads a saved visualization mesh and converts it back to normalized model space."""
    raw_mesh = load_trimesh(mesh_path)
    original_mesh, _ = reorient_mesh_to_z_up(raw_mesh, mesh_up_dir)
    normalized_mesh, _, _ = normalize_mesh(original_mesh)
    return normalized_mesh


def prepare_lifted_prompt_records_for_saving(
    *,
    normalized_mesh: trimesh.Trimesh,
    lifted: dict[str, Any],
    center: np.ndarray,
    scale: float,
    render_to_model_rotation: np.ndarray | None = None,
) -> dict[str, Any]:
    """Transforms lifted prompt records into model space and annotates face IDs."""
    prompt_points = np.asarray(lifted["points"], dtype=np.float32)
    prompt_normals = np.asarray(lifted["normals"], dtype=np.float32)
    if render_to_model_rotation is not None:
        rotation = np.asarray(render_to_model_rotation, dtype=np.float32)
        prompt_points = prompt_points @ rotation.T
        prompt_normals = prompt_normals @ rotation.T
    normal_norms = np.linalg.norm(prompt_normals, axis=1, keepdims=True)
    normal_norms = np.clip(normal_norms, a_min=1e-8, a_max=None)
    prompt_normals = prompt_normals / normal_norms

    records = list(lifted["records"])
    for record, point, normal in zip(records, prompt_points, prompt_normals, strict=True):
        record["point_prompt_normalized"] = point.astype(np.float32).tolist()
        record["normal"] = normal.astype(np.float32).tolist()

    prompt_face_ids = resolve_point_prompt_face_ids(normalized_mesh, prompt_points)
    for record, face_id in zip(records, prompt_face_ids.tolist(), strict=True):
        record["point_prompt_face_id"] = int(face_id)

    point_prompts_world = _denormalize_points(
        prompt_points,
        center=np.asarray(center, dtype=np.float32),
        scale=float(scale),
    )
    return {
        "point_prompts": prompt_points.astype(np.float32, copy=False),
        "point_prompt_normals": prompt_normals.astype(np.float32, copy=False),
        "point_prompt_face_ids": prompt_face_ids.astype(np.int64, copy=False),
        "point_prompts_world": point_prompts_world.astype(np.float32, copy=False),
        "records": records,
    }


def save_auto_kinematics_artifacts(
    *,
    output_dir: Path,
    raw_response: dict[str, Any],
    parsed_response: dict[str, Any],
    lifted_prompt_records: Sequence[dict[str, Any]],
    center: np.ndarray,
    scale: float,
    visualization_mesh_relative_path: str = "_auto_kinematics_visualization_mesh.glb",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_response_path = output_dir / "auto_kinematics_raw_response.json"
    parsed_response_path = output_dir / "auto_kinematics_parsed.json"
    lifted_prompts_path = output_dir / "auto_kinematics_lifted_point_prompts.json"
    raw_response_path.write_text(json.dumps(raw_response, indent=2) + "\n", encoding="utf-8")
    parsed_response_path.write_text(json.dumps(parsed_response, indent=2) + "\n", encoding="utf-8")

    mesh_path_pointer = str(visualization_mesh_relative_path)
    lifted_payload = {
        "normalization": {
            "center": np.asarray(center, dtype=np.float32).tolist(),
            "scale": float(scale),
        },
        "visualization_mesh_path": mesh_path_pointer,
        "point_prompt_face_mesh_path": mesh_path_pointer,
        "links": list(lifted_prompt_records),
    }
    for link_record in lifted_payload["links"]:
        if "point_prompt_face_id" in link_record and "point_prompt_face_mesh_path" not in link_record:
            link_record["point_prompt_face_mesh_path"] = mesh_path_pointer
        if "point_prompt_world" not in link_record:
            point_prompt_normalized = np.asarray(
                link_record["point_prompt_normalized"],
                dtype=np.float32,
            )
            point_prompt_world = _denormalize_points(
                point_prompt_normalized[None, :],
                center=np.asarray(center, dtype=np.float32),
                scale=float(scale),
            )[0]
            link_record["point_prompt_world"] = point_prompt_world.tolist()
    lifted_prompts_path.write_text(
        json.dumps(lifted_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "raw_response_path": raw_response_path,
        "parsed_response_path": parsed_response_path,
        "lifted_prompts_path": lifted_prompts_path,
    }


def prepare_auto_kinematics_mesh_batch(
    *,
    mesh: trimesh.Trimesh,
    original_mesh: trimesh.Trimesh,
    render_mesh: trimesh.Trimesh | None = None,
    output_dir: Path,
    auto_kinematics_model_id: str,
    azimuths_deg: Sequence[float],
    system_prompt_path: Path,
    num_shape_points: int,
    num_query_points: int,
    num_query_points_per_face_for_seg: int | None = None,
    sharp_point_ratio: float,
    device: torch.device,
    center: np.ndarray,
    scale: float,
    render_mesh_space: str | None = None,
    render_to_batch_rotation: np.ndarray | None = None,
    reasoning_effort: str = "medium",
    blender_bin: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    auto_output_dir = output_dir / "auto_kinematics"
    auto_output_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = auto_output_dir / "renders"
    render_source_mesh = mesh if render_mesh is None else render_mesh
    rendered_views = render_mesh_auto_kinematics_views(
        render_source_mesh,
        output_dir=renders_dir,
        azimuths_deg=azimuths_deg,
        blender_bin=blender_bin,
    )
    provider = detect_provider(auto_kinematics_model_id)
    print(
        f"Requesting auto kinematics from {provider} model {auto_kinematics_model_id} "
        f"using {len(rendered_views)} rendered views"
    )
    raw_response = call_auto_kinematics_model(
        model_id=auto_kinematics_model_id,
        system_prompt_path=system_prompt_path,
        rendered_views=rendered_views,
        provider=provider,
        reasoning_effort=reasoning_effort,
    )
    parsed_response = parse_auto_kinematics_response(raw_response)
    print(f"Lifted {len(parsed_response['links'])} auto-kinematics links from model response")
    lifted = lift_point_prompts_from_rendered_views(
        links=parsed_response["links"],
        rendered_views=rendered_views,
    )
    prepared_prompts = prepare_lifted_prompt_records_for_saving(
        normalized_mesh=mesh,
        lifted=lifted,
        center=np.asarray(center, dtype=np.float32),
        scale=float(scale),
        render_to_model_rotation=render_to_batch_rotation,
    )
    prompt_points = np.asarray(prepared_prompts["point_prompts"], dtype=np.float32)
    prompt_normals = np.asarray(prepared_prompts["point_prompt_normals"], dtype=np.float32)
    prompt_face_ids = np.asarray(prepared_prompts["point_prompt_face_ids"], dtype=np.int64)
    link_point_prompts = torch.from_numpy(prompt_points).float()
    link_point_prompt_normals = torch.from_numpy(prompt_normals).float()
    batch, query_face_indices = prepare_inference_batch_from_mesh_with_prompts(
        mesh,
        num_shape_points=num_shape_points,
        num_query_points=num_query_points,
        num_query_points_per_face_for_seg=num_query_points_per_face_for_seg,
        sharp_point_ratio=sharp_point_ratio,
        link_names=parsed_response["link_names"],
        joint_specs=parsed_response["joint_specs"],
        device=device,
        link_point_prompts=link_point_prompts,
        link_point_prompt_normals=link_point_prompt_normals,
        require_unique_link_names=False,
    )
    point_prompts_world = np.asarray(prepared_prompts["point_prompts_world"], dtype=np.float32)
    visualization_mesh_path = auto_output_dir / "_auto_kinematics_visualization_mesh.glb"
    visualization_mesh = original_mesh.copy()
    visualization_export_transform = np.eye(4, dtype=np.float32)
    visualization_export_transform[:3, :3] = up_dir_rotation_matrix("+Z", "+Y")
    visualization_mesh.apply_transform(visualization_export_transform)
    visualization_mesh.export(visualization_mesh_path)
    prompt_visualization_path = save_point_prompt_visualization(
        mesh_path=visualization_mesh_path,
        point_prompts_world=point_prompts_world,
        output_path=auto_output_dir / "point_prompts_visualization.glb",
        blender_bin=blender_bin,
    )
    saved_paths = save_auto_kinematics_artifacts(
        output_dir=auto_output_dir,
        raw_response=raw_response,
        parsed_response=parsed_response,
        lifted_prompt_records=prepared_prompts["records"],
        center=np.asarray(center, dtype=np.float32),
        scale=float(scale),
    )
    point_prompt_overlay_paths = save_2d_point_prompt_visualizations(
        rendered_views=rendered_views,
        lifted_prompt_records=prepared_prompts["records"],
        output_dir=auto_output_dir / "point_prompt_overlays",
    )
    metadata = {
        "auto_kinematics_model_id": auto_kinematics_model_id,
        "auto_kinematics_provider": provider,
        "reasoning_effort": _normalize_reasoning_effort(reasoning_effort),
        "system_prompt_path": str(system_prompt_path),
        "blender_bin": str(resolve_blender_binary(blender_bin)),
        "render_dir": str(renders_dir),
        "azimuths_deg": [float(azimuth) for azimuth in azimuths_deg],
        "elevation_deg": float(DEFAULT_RENDER_ELEVATION_DEG),
        "pitch_deg": float(DEFAULT_RENDER_PITCH_DEG),
        "render_distance": float(DEFAULT_RENDER_DISTANCE),
        "render_resolution": int(DEFAULT_RENDER_RESOLUTION),
        "render_mesh_space": (
            str(render_mesh_space)
            if render_mesh_space is not None
            else ("batch_mesh_space" if render_mesh is None else "render_mesh_pre_normalization")
        ),
        "render_normalized_in_blender": True,
        "visualization_mesh_path": str(visualization_mesh_path),
        "prompt_visualization_path": str(prompt_visualization_path),
        "point_prompt_overlay_paths": [str(path) for path in point_prompt_overlay_paths],
        **{key: str(path) for key, path in saved_paths.items()},
    }
    return {
        "batch": batch,
        "query_face_indices": query_face_indices,
        "link_names": parsed_response["link_names"],
        "joint_specs": parsed_response["joint_specs"],
        "metadata": metadata,
        "link_point_prompt_face_ids": prompt_face_ids.astype(np.int64, copy=False),
    }
