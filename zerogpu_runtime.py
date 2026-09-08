"""Resolve missing ZeroGPU cgroup mounts before importing ``spaces``."""

import logging
import os
from pathlib import Path


def configure_zerogpu_cgroup() -> None:
    """Use procfs only if it exposes the host's actual device cgroup."""
    if os.environ.get("SPACES_ZERO_GPU", "").lower() not in {"1", "t", "true"}:
        return

    configured = Path(os.environ.get("ZEROGPU_PROC_SELF_CGROUP_PATH", "/proc/self/cgroup"))
    if configured.exists():
        return

    fallback = Path("/proc/self/cgroup")
    try:
        lines = fallback.read_text().strip().splitlines()
    except OSError as exc:
        raise RuntimeError(
            f"ZeroGPU cgroup file {configured} is missing and {fallback} is unreadable. "
            "Restart the Hugging Face Space to restore its runtime mounts."
        ) from exc

    # Match spaces.zero.utils.self_cgroup_device_path: prefer the v1 devices
    # controller, otherwise use the unified v2 hierarchy.
    cgroup = None
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[1] == "devices":
            cgroup = fields[2]
            break
    if cgroup is None and len(lines) == 1 and lines[0].startswith("0::"):
        cgroup = lines[0][3:]

    # A private cgroup namespace exposes '/' rather than the host container
    # path. It cannot identify this process to the ZeroGPU scheduler.
    if not cgroup or not cgroup.startswith("/") or cgroup == "/" or ".." in cgroup.split("/"):
        raise RuntimeError(
            f"ZeroGPU cgroup file {configured} is missing and {fallback} does not "
            "expose a usable host device cgroup. Restart the Hugging Face Space "
            "to restore its runtime mounts; if this persists, contact Hugging Face support."
        )

    os.environ["ZEROGPU_PROC_SELF_CGROUP_PATH"] = str(fallback)
    logging.getLogger(__name__).warning(
        "ZeroGPU cgroup file %s is missing; using %s for GPU scheduling",
        configured,
        fallback,
    )
