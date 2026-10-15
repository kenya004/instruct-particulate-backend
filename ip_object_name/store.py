from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

from . import core


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def error_details(error: Exception) -> dict[str, str]:
    cause = error.__cause__ or error
    message = str(cause)
    keys = [os.getenv("GEMINI_API_KEY"), os.getenv("GOOGLE_API_KEY")]
    try:
        keys.append(core._api_key())
    except core.SetupError:
        pass
    for key in keys:
        if key:
            message = message.replace(key, "[REDACTED]")
    return {"type": type(cause).__name__, "detail": message[:4000]}


class SourceChangedError(RuntimeError):
    """The input file changed while its snapshot was being registered."""


class ModelStore:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._sources: dict[tuple[str, int, int], str] = {}
        for path in self.root.glob("*/metadata.json"):
            if path.parent.name.startswith("."):
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            self._sources[self._source_key(data)] = path.parent.name

    @staticmethod
    def _source_key(data: dict) -> tuple[str, int, int]:
        return data["source_path"], data["source_size"], data["source_mtime_ns"]

    def _folder(self, model_id: str) -> Path:
        # IDs, unlike source filenames, are used to construct paths inside the store.
        if len(model_id) != 32 or any(c not in "0123456789abcdef" for c in model_id):
            raise ValueError("Invalid model_id.")
        return self.root / model_id

    def get(self, model_id: str) -> dict[str, Any]:
        return json.loads((self._folder(model_id) / "metadata.json").read_text(encoding="utf-8"))

    def save(self, metadata: dict[str, Any]) -> None:
        write_json(self._folder(metadata["model_id"]) / "metadata.json", metadata)

    def find_source(self, path: Path, signature: tuple[int, int]) -> dict[str, Any] | None:
        model_id = self._sources.get((str(path.resolve()), *signature))
        return self.get(model_id) if model_id else None

    def register(self, path: Path, *, up_dir: str, expected_signature: tuple[int, int] | None = None) -> dict[str, Any]:
        path = path.expanduser().resolve()
        direction = core.normalize_up_dir(up_dir)
        if path.suffix.lower() != ".glb":
            raise core.InvalidMeshError("Input must be a .glb file.")
        before = source_signature(path)
        if expected_signature is not None and before != expected_signature:
            raise SourceChangedError("The input changed before registration.")
        existing = self.find_source(path, before)
        if existing:
            return existing
        model_id = uuid4().hex
        target = self._folder(model_id)
        with tempfile.TemporaryDirectory(prefix=".register-", dir=self.root) as temporary:
            snapshot = Path(temporary) / "input.glb"
            shutil.copyfile(path, snapshot)
            if source_signature(path) != before or snapshot.stat().st_size != before[0]:
                raise SourceChangedError("The input changed during registration.")
            with snapshot.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            metadata = {
                "model_id": model_id,
                "source_filename": path.name,
                "source_path": str(path),
                "source_size": before[0],
                "source_mtime_ns": before[1],
                "source_sha256": digest,
                "input_mesh": str(target / "input.glb"),
                "up_dir": direction,
                "units": None,
                "created_at": utc_now(),
                "status": "pending",
                "object_name": None,
                "error": None,
            }
            write_json(Path(temporary) / "metadata.json", metadata)
            os.replace(temporary, target)
        self._sources[self._source_key(metadata)] = model_id
        return metadata

    def mark_interrupted(self) -> int:
        count = 0
        for model_id in set(self._sources.values()):
            metadata = self.get(model_id)
            if metadata["status"] == "processing":
                metadata.update(
                    status="failed", finished_at=utc_now(),
                    error={"type": "Interrupted", "detail": "The previous process stopped before completion. Use --retry-failed to retry."},
                )
                self.save(metadata)
                count += 1
        return count

    def recognize(self, model_id: str, settings: core.Settings, *, retry_failed: bool = False) -> dict[str, Any]:
        metadata = self.get(model_id)
        if metadata["status"] == "completed":
            return metadata
        if metadata["status"] == "failed" and not retry_failed:
            return metadata
        if metadata["status"] == "processing":
            raise RuntimeError("This model is already being processed.")
        folder = self._folder(model_id)
        run_id = uuid4().hex
        run_dir = folder / "recognition" / run_id
        run_dir.mkdir(parents=True)
        settings = replace(settings, up_dir=metadata["up_dir"])
        metadata.update(
            status="processing", started_at=utc_now(), finished_at=None,
            error=None, recognition_run_id=run_id, vlm_model=settings.model,
        )
        self.save(metadata)
        try:
            input_path = folder / "input.glb"
            core.validate_glb(input_path)
            environment = core.check_environment(settings)
            renderer, data_utils = core._upstream(settings.repo_root)
            mesh = data_utils.load_trimesh(input_path)
            prompt = Path(core.__file__).with_name("object-name-system-prompt.md").read_text(encoding="utf-8")
            (run_dir / "system_prompt.md").write_text(prompt, encoding="utf-8")
            config = {
                "model_id": model_id,
                "run_id": run_id,
                "vlm_model": settings.model,
                "google_genai_version": version("google-genai"),
                "up_dir": metadata["up_dir"],
                "system_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "generation_parameters": "API defaults; only JSON response format and schema are specified.",
                "azimuths_deg": list(renderer.DEFAULT_AUTO_KINEMATICS_AZIMUTHS),
                "resolution": getattr(renderer, "DEFAULT_RENDER_RESOLUTION", None),
                "pitch_deg": getattr(renderer, "DEFAULT_RENDER_PITCH_DEG", None),
                "render_samples": getattr(renderer, "DEFAULT_BLENDER_RENDER_SAMPLES", None),
                "blender_bin": environment["blender_bin"],
            }
            write_json(run_dir / "request.json", config)
            views = renderer.render_mesh_auto_kinematics_views(
                mesh,
                output_dir=run_dir / "renders",
                azimuths_deg=renderer.DEFAULT_AUTO_KINEMATICS_AZIMUTHS,
                mesh_path=input_path,
                up_dir=metadata["up_dir"],
                blender_bin=environment["blender_bin"],
            )
            if not views or any(not Path(view.image_path).is_file() for view in views):
                raise RuntimeError("The renderer did not produce the expected images.")
            name = core.validate_prediction({"object_name": core.request_object_name(views, model=settings.model)})
            result = {
                **config, "object_name": name, "status": "identified",
                "source_filename": metadata["source_filename"],
                "source_sha256": metadata["source_sha256"],
                "input_mesh": metadata["input_mesh"],
                "created_at": utc_now(),
                "render_images": [str(view.image_path) for view in views],
            }
            write_json(run_dir / "object_name.json", result)
            write_json(folder / "recognition" / "object_name.json", result)
            metadata.update(
                status="completed", object_name=name, finished_at=utc_now(),
                recognition_result=str(run_dir / "object_name.json"),
            )
        except Exception as error:
            metadata.update(status="failed", finished_at=utc_now(), error=error_details(error))
            write_json(run_dir / "error.json", metadata["error"])
        self.save(metadata)
        return metadata
