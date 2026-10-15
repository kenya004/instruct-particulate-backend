from __future__ import annotations

import importlib
import os


class CredentialSettingsError(RuntimeError):
    pass


def configured_api_key() -> str | None:
    module_name = f"{__package__}.local_settings"
    local_key = None
    try:
        settings = importlib.import_module(module_name)
        local_key = getattr(settings, "GEMINI_API_KEY", None)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise CredentialSettingsError("Cannot load local_settings.py. Check its imports.") from None
    except Exception:
        raise CredentialSettingsError("Cannot load local_settings.py. Check its syntax and contents.") from None
    for value in (local_key, os.getenv("GEMINI_API_KEY"), os.getenv("GOOGLE_API_KEY")):
        if value is not None and not isinstance(value, str):
            raise CredentialSettingsError("GEMINI_API_KEY in local_settings.py must be a string.")
        if value and value.strip():
            return value.strip()
    return None
