from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

UP_DIRECTIONS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
DEFAULT_MODEL = "gemini-2.5-flash"


class SetupError(RuntimeError):
    pass


class InvalidMeshError(ValueError):
    pass


class VLMError(RuntimeError):
    pass


def normalize_up_dir(value: str) -> str:
    token = value.strip().upper()
    if token in {"X", "Y", "Z"}:
        token = "+" + token
    if token not in UP_DIRECTIONS:
        raise ValueError("up_dir must be +X, -X, +Y, -Y, +Z, or -Z.")
    return token


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    output_root: Path
    model: str = DEFAULT_MODEL
    up_dir: str = "+Z"
    blender_bin: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        repo = Path(os.getenv("INSTRUCT_PARTICULATE_REPO", ".")).expanduser().resolve()
        output = os.getenv("IP_OBJECT_NAME_OUTPUT_ROOT")
        return cls(
            repo_root=repo,
            output_root=Path(output).expanduser().resolve() if output else repo / "data" / "object_names",
            model=os.getenv("IP_OBJECT_NAME_MODEL", DEFAULT_MODEL),
            up_dir=normalize_up_dir(os.getenv("IP_OBJECT_NAME_UP_DIR", "+Z")),
            blender_bin=os.getenv("BLENDER_BIN") or None,
        )


def _api_key(*, required: bool = True) -> str | None:
    from .credentials import CredentialSettingsError, configured_api_key

    try:
        key = configured_api_key()
    except CredentialSettingsError as error:
        raise SetupError(str(error)) from None
    if required and not key:
        raise SetupError("Set GEMINI_API_KEY in ip_object_name/local_settings.py or in the environment.")
    return key


def _upstream(repo_root: Path):
    repo = repo_root.expanduser().resolve()
    helper = repo / "instruct_particulate" / "utils" / "auto_kinematics_utils.py"
    script = repo / "scripts" / "render_auto_kinematics_blender.py"
    if not helper.is_file() or not script.is_file():
        raise SetupError("Instruct-Particulate sources were not found. Set INSTRUCT_PARTICULATE_REPO or --repo.")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        renderer = importlib.import_module("instruct_particulate.utils.auto_kinematics_utils")
        data_utils = importlib.import_module("instruct_particulate.utils.data_utils")
    except ImportError as exc:
        raise SetupError("Cannot import the upstream renderer. Use the existing Instruct-Particulate .venv and dependencies.") from exc
    return renderer, data_utils


def check_environment(settings: Settings, *, require_key: bool = True) -> dict[str, Any]:
    normalize_up_dir(settings.up_dir)
    if not settings.model.strip():
        raise SetupError("The VLM model ID is empty. Set IP_OBJECT_NAME_MODEL or --model.")
    if require_key:
        _api_key()
    renderer, _ = _upstream(settings.repo_root)
    try:
        blender = renderer.resolve_blender_binary(settings.blender_bin)
    except FileNotFoundError as exc:
        raise SetupError("Blender was not found. Set BLENDER_BIN to the WSL Blender executable.") from exc
    return {
        "repo_root": str(settings.repo_root),
        "blender_bin": str(blender),
        "vlm_model": settings.model,
        "up_dir": normalize_up_dir(settings.up_dir),
        "api_key_configured": bool(_api_key(required=False)),
        "note": "Local configuration checked; Blender execution and VLM access are not tested here.",
    }


def validate_glb(path: Path) -> None:
    if path.suffix.lower() != ".glb":
        raise InvalidMeshError("Input must be a .glb file.")
    if not path.is_file():
        raise InvalidMeshError("The input GLB file does not exist.")
    with path.open("rb") as source:
        header = source.read(12)
    if len(header) != 12:
        raise InvalidMeshError("The file is not a valid binary glTF 2.0 file.")
    magic, version, length = struct.unpack("<4sII", header)
    if magic != b"glTF" or version != 2 or length != path.stat().st_size:
        raise InvalidMeshError("Invalid GLB header, version, or file length.")


def validate_prediction(payload: Any) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("object_name"), str):
        raise VLMError("VLM response must contain a string object_name.")
    name = " ".join(payload["object_name"].split())
    if not name or len(name) > 120:
        raise VLMError("VLM returned an empty or excessively long object_name.")
    return name


def request_object_name(views: list, *, model: str) -> str:
    from google import genai
    from google.genai import types

    prompt_path = Path(__file__).with_name("object-name-system-prompt.md")
    parts = [types.Part.from_text(text="Identify the single object shown in these views of the same 3D mesh.")]
    for index, view in enumerate(views):
        parts.append(types.Part.from_text(text=f"View {index + 1} of the same object:"))
        parts.append(types.Part.from_bytes(data=Path(view.image_path).read_bytes(), mime_type="image/png"))
    try:
        with genai.Client(api_key=_api_key(), http_options=types.HttpOptions(timeout=120_000)) as client:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=prompt_path.read_text(encoding="utf-8"),
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {"object_name": {"type": "STRING"}},
                        "required": ["object_name"],
                    },
                ),
            )
    except SetupError:
        raise
    except Exception as exc:
        raise VLMError("VLM request failed. Check the API key, model access, network, and quota.") from exc
    try:
        payload = json.loads(response.text)
    except (TypeError, ValueError) as exc:
        raise VLMError("VLM did not return valid JSON.") from exc
    return validate_prediction(payload)


def identify_object(
    mesh_path: Path,
    settings: Settings,
    *,
    render_only: bool = False,
    source_filename: str | None = None,
) -> dict[str, Any]:
    mesh_path = mesh_path.expanduser().resolve()
    validate_glb(mesh_path)
    environment = check_environment(settings, require_key=not render_only)
    renderer, data_utils = _upstream(settings.repo_root)
    try:
        mesh = data_utils.load_trimesh(mesh_path)
    except Exception as exc:
        raise InvalidMeshError("Could not read nonempty triangle mesh geometry from this GLB.") from exc

    # Keep the original GLB and its render orientation for the later segmentation step.
    model_id = uuid4().hex
    output_dir = settings.output_root.expanduser().resolve() / model_id
    output_dir.mkdir(parents=True)
    input_path = output_dir / "input.glb"
    shutil.copyfile(mesh_path, input_path)
    with input_path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    views = renderer.render_mesh_auto_kinematics_views(
        mesh,
        output_dir=output_dir / "renders",
        azimuths_deg=renderer.DEFAULT_AUTO_KINEMATICS_AZIMUTHS,
        mesh_path=input_path,
        up_dir=environment["up_dir"],
        blender_bin=environment["blender_bin"],
    )
    if not views or any(not Path(view.image_path).is_file() for view in views):
        raise RuntimeError("The renderer did not produce the expected images.")
    name = None if render_only else request_object_name(views, model=settings.model)
    result = {
        "model_id": model_id,
        "object_name": name,
        "status": "rendered" if render_only else "identified",
        "source_filename": source_filename or mesh_path.name,
        "source_sha256": digest,
        "up_dir": environment["up_dir"],
        "vlm_model": None if render_only else settings.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "input_mesh": str(input_path),
        "render_images": [str(view.image_path) for view in views],
    }
    (output_dir / "object_name.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
