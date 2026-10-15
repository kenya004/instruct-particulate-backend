from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from .core import Settings, check_environment, identify_object, normalize_up_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Identify an object from a GLB using Instruct-Particulate rendered views.")
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--up-dir", type=normalize_up_dir)
    parser.add_argument("--blender-bin")
    parser.add_argument("--check", action="store_true", help="Check local configuration without rendering or calling the VLM.")
    parser.add_argument("--render-only", action="store_true", help="Generate images without an API key or VLM request.")
    args = parser.parse_args()
    if not args.check and args.mesh is None:
        parser.error("--mesh is required unless --check is used.")
    try:
        settings = Settings.from_env()
        overrides = {}
        for name in ("model", "up_dir", "blender_bin"):
            if getattr(args, name) is not None:
                overrides[name] = getattr(args, name)
        if args.repo is not None:
            overrides["repo_root"] = args.repo.expanduser().resolve()
            if args.output_dir is None and not os.getenv("IP_OBJECT_NAME_OUTPUT_ROOT"):
                overrides["output_root"] = overrides["repo_root"] / "data" / "object_names"
        if args.output_dir is not None:
            overrides["output_root"] = args.output_dir.expanduser().resolve()
        settings = replace(settings, **overrides)
        with redirect_stdout(sys.stderr):
            result = (
                check_environment(settings, require_key=not args.render_only)
                if args.check
                else identify_object(args.mesh, settings, render_only=args.render_only)
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Object-name recognition failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
