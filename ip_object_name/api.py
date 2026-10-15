from __future__ import annotations

import logging
import tempfile
import threading
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .core import InvalidMeshError, Settings, SetupError, VLMError, identify_object, normalize_up_dir


MAX_UPLOAD_BYTES = 100 * 1024 * 1024
logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Instruct-Particulate Object Name", version="0.1.0")
    app.state.settings = settings or Settings.from_env()
    app.state.inference_lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "alive", "service": "ip-object-name"}

    @app.post("/identify-object")
    def identify(file: UploadFile = File(...), up_dir: str | None = Form(None)):
        try:
            filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
            if Path(filename).suffix.lower() != ".glb":
                raise HTTPException(status_code=422, detail="Upload a .glb file.")
            try:
                settings = app.state.settings
                if up_dir is not None:
                    settings = replace(settings, up_dir=normalize_up_dir(up_dir))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if file.size is not None and file.size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="GLB must be at most 100 MiB.")
            if not app.state.inference_lock.acquire(blocking=False):
                raise HTTPException(status_code=409, detail="Another recognition request is running. Retry after it finishes.")
            try:
                with tempfile.TemporaryDirectory(prefix="ip_object_name_upload_") as temporary:
                    mesh_path = Path(temporary) / "input.glb"
                    total = 0
                    with mesh_path.open("wb") as destination:
                        while chunk := file.file.read(1024 * 1024):
                            total += len(chunk)
                            if total > MAX_UPLOAD_BYTES:
                                raise HTTPException(status_code=413, detail="GLB must be at most 100 MiB.")
                            destination.write(chunk)
                    return identify_object(mesh_path, settings, source_filename=filename)
            finally:
                app.state.inference_lock.release()
        except InvalidMeshError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SetupError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except VLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Object-name recognition failed")
            raise HTTPException(status_code=500, detail="Rendering or recognition failed. Check the server log.") from exc
        finally:
            file.file.close()

    return app


app = create_app()
