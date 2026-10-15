from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from filelock import FileLock, Timeout

from . import core
from .store import ModelStore, SourceChangedError, source_signature


class FolderWatcher:
    def __init__(
        self, inbox: Path, store: ModelStore, settings: core.Settings, *,
        settle_seconds: float = 2.0, retry_failed: bool = False,
    ):
        if settle_seconds < 0:
            raise ValueError("settle_seconds must be nonnegative.")
        self.inbox = inbox.expanduser().resolve()
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.settings = settings
        self.settle_seconds = settle_seconds
        self.retry_failed = retry_failed
        self.waiting = False
        self._observed: dict[Path, tuple[tuple[int, int], float]] = {}
        self._attempted: set[str] = set()

    def scan(self) -> list[dict]:
        self.waiting = False
        results = []
        paths = sorted(
            (p for p in self.inbox.iterdir() if p.is_file() and p.suffix.lower() == ".glb"),
            key=lambda p: p.name.casefold(),
        )
        present = set(paths)
        self._observed = {p: value for p, value in self._observed.items() if p in present}
        for path in paths:
            try:
                signature = source_signature(path)
                known = self.store.find_source(path, signature)
                if known:
                    if known["status"] == "completed" or known["model_id"] in self._attempted:
                        continue
                    if known["status"] == "failed" and not self.retry_failed:
                        continue
                now = time.monotonic()
                observed = self._observed.get(path)
                if observed is None or observed[0] != signature:
                    observed = (signature, now)
                    self._observed[path] = observed
                if now - observed[1] < self.settle_seconds:
                    self.waiting = True
                    continue
                metadata = self.store.register(
                    path, up_dir=self.settings.up_dir, expected_signature=signature
                )
            except (FileNotFoundError, SourceChangedError):
                self._observed.pop(path, None)
                self.waiting = True
                continue
            model_id = metadata["model_id"]
            self._attempted.add(model_id)
            print(f"[processing] {path.name} -> {model_id}", flush=True)
            with redirect_stdout(sys.stderr):
                result = self.store.recognize(model_id, self.settings, retry_failed=self.retry_failed)
            if result["status"] == "completed":
                print(f"[completed] {path.name}: {result['object_name']} ({model_id})", flush=True)
            else:
                print(f"[failed] {path.name}: {json.dumps(result['error'], ensure_ascii=False)} ({model_id})", flush=True)
            results.append(result)
        return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Register new inbox GLBs and recognize their object names.")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--inbox", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--up-dir", type=core.normalize_up_dir)
    parser.add_argument("--blender-bin")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="Process available files and exit after they settle.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry each failed input once during this invocation.")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.settle_seconds < 0:
        parser.error("--poll-seconds must be positive and --settle-seconds must be nonnegative.")
    try:
        settings = core.Settings.from_env()
        overrides = {name: getattr(args, name) for name in ("model", "up_dir", "blender_bin") if getattr(args, name) is not None}
        if args.repo is not None:
            overrides["repo_root"] = args.repo.expanduser().resolve()
        settings = replace(settings, **overrides)
        inbox = (args.inbox or settings.repo_root / "data" / "inbox").expanduser().resolve()
        root = (args.models_dir or settings.repo_root / "data" / "models").expanduser().resolve()
        if inbox == root or inbox in root.parents or root in inbox.parents:
            parser.error("--inbox and --models-dir must be separate, non-nested directories.")
        core.check_environment(settings)
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(root / ".watch.lock", timeout=0):
            store = ModelStore(root)
            interrupted = store.mark_interrupted()
            if interrupted:
                print(f"Marked {interrupted} interrupted request(s) as failed. Use --retry-failed to retry.", flush=True)
            watcher = FolderWatcher(
                inbox, store, settings,
                settle_seconds=args.settle_seconds, retry_failed=args.retry_failed,
            )
            print(f"Inbox: {inbox}\nModels: {root}\nVLM: {settings.model}\nUp direction: {settings.up_dir}", flush=True)
            failed = False
            while True:
                results = watcher.scan()
                failed = failed or any(result["status"] == "failed" for result in results)
                if args.once and not watcher.waiting:
                    return 1 if failed else 0
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\nStopped. Completed models will not be processed again.")
        return 0
    except Timeout:
        print("Another watcher already owns this model directory.", file=sys.stderr)
        return 1
    except Exception as error:
        from .store import error_details

        print(json.dumps(error_details(error), ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
