from __future__ import annotations

import argparse
import importlib
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.request import urlretrieve

import torch
from torch import Tensor, nn


PARTFIELD_FEATURE_DIM = 448
_PARTFIELD_SDF_CHANNELS = 64
_PARTFIELD_CHECKPOINT_REPO_ID = "mikaelaangel/partfield-ckpt"
_PARTFIELD_CHECKPOINT_FILENAME = "model_objaverse.ckpt"
_PARTFIELD_CHECKPOINT_URL = (
    "https://huggingface.co/"
    f"{_PARTFIELD_CHECKPOINT_REPO_ID}/resolve/main/{_PARTFIELD_CHECKPOINT_FILENAME}"
)
_PARTFIELD_ROOT = Path(__file__).resolve().parents[2] / "partfield"
_PARTFIELD_CHECKPOINT_PATH = _PARTFIELD_ROOT / "model" / _PARTFIELD_CHECKPOINT_FILENAME
_PARTFIELD_CONFIG_PATH = _PARTFIELD_ROOT / "configs" / "final" / "demo.yaml"


def _resolve_partfield_root() -> Path:
    if _PARTFIELD_ROOT.exists():
        return _PARTFIELD_ROOT
    raise FileNotFoundError(
        f"Could not find vendored PartField checkout at {_PARTFIELD_ROOT}"
    )


def _ensure_partfield_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=_PARTFIELD_CHECKPOINT_REPO_ID,
            filename=_PARTFIELD_CHECKPOINT_FILENAME,
            local_dir=str(checkpoint_path.parent),
            local_dir_use_symlinks=False,
        )
    except ImportError:
        pass
    if not checkpoint_path.exists():
        urlretrieve(_PARTFIELD_CHECKPOINT_URL, checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not download PartField checkpoint to {checkpoint_path}")
    return checkpoint_path


def _resolve_partfield_config_path() -> Path:
    if not _PARTFIELD_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Could not find PartField config: {_PARTFIELD_CONFIG_PATH}")
    return _PARTFIELD_CONFIG_PATH


class PartFieldFeatureExtractor:
    """Lazy wrapper around the local PartField checkpoint used by `particulate`."""

    def __init__(
        self,
    ) -> None:
        self.partfield_root = _resolve_partfield_root()
        self.partfield_checkpoint_path = _ensure_partfield_checkpoint(_PARTFIELD_CHECKPOINT_PATH)
        self.partfield_config_path = _resolve_partfield_config_path()
        self._model: Optional[nn.Module] = None
        self._sample_triplane_feat: Any = None
        self._target_device: Optional[torch.device] = None

    def _ensure_imports_loaded(self) -> None:
        if self._sample_triplane_feat is not None:
            return
        partfield_root_str = str(self.partfield_root)
        if partfield_root_str not in sys.path:
            sys.path.append(partfield_root_str)

        encoder_pc = importlib.import_module("partfield.model.PVCNN.encoder_pc")
        trainer_module = importlib.import_module("partfield.model_trainer_pvcnn_only_demo")
        config_module = importlib.import_module("partfield.config")
        self._sample_triplane_feat = encoder_pc.sample_triplane_feat
        self._partfield_model_cls = trainer_module.Model
        self._partfield_setup = config_module.setup

    def _load_model(self, device: torch.device) -> nn.Module:
        self._ensure_imports_loaded()
        cfg = self._partfield_setup(
            argparse.Namespace(
                config_file=str(self.partfield_config_path),
                opts=[],
            ),
            freeze=False,
        )
        model = self._partfield_model_cls.load_from_checkpoint(
            str(self.partfield_checkpoint_path),
            cfg=cfg,
            map_location="cpu",
        )
        model.eval()
        model.requires_grad_(False)
        model.to(device=device)
        self._model = model
        self._target_device = device
        return model

    def _get_model(self, device: torch.device) -> nn.Module:
        if self._model is None:
            return self._load_model(device)
        if self._target_device != device:
            self._model.to(device=device)
            self._target_device = device
        return self._model

    def _autocast_context(self, device: torch.device):
        if device.type != "cuda":
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=torch.get_autocast_gpu_dtype(),
            enabled=torch.is_autocast_enabled(),
        )

    def _normalize_points(
        self,
        encode_points: Tensor,
        *decode_points: Tensor | None,
    ) -> Tuple[Tensor, Tuple[Tensor | None, ...]]:
        bbmin = encode_points.amin(dim=-2, keepdim=True)
        bbmax = encode_points.amax(dim=-2, keepdim=True)
        center = (bbmin + bbmax) * 0.5
        extent = (bbmax - bbmin).amax(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = 2.0 * 0.9 / extent

        normalized_encode_points = (encode_points - center) * scale
        normalized_decode_points = tuple(
            None if points is None else (points - center) * scale
            for points in decode_points
        )
        return normalized_encode_points, normalized_decode_points

    @torch.no_grad()
    def extract(
        self,
        *,
        encode_points: Tensor,
        decode_shape_points: Tensor | None = None,
        decode_query_points: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        model = self._get_model(encode_points.device)
        normalized_encode_points, (
            normalized_shape_points,
            normalized_query_points,
        ) = self._normalize_points(
            encode_points,
            decode_shape_points,
            decode_query_points,
        )

        with self._autocast_context(encode_points.device):
            encoded = model.pvcnn(normalized_encode_points, normalized_encode_points)
            planes = model.triplane_transformer(encoded)
            part_planes = planes[:, :, _PARTFIELD_SDF_CHANNELS:]
            shape_features = (
                None
                if normalized_shape_points is None
                else self._sample_triplane_feat(part_planes, normalized_shape_points)
            )
            query_features = (
                None
                if normalized_query_points is None
                else self._sample_triplane_feat(part_planes, normalized_query_points)
            )

        return (
            None if shape_features is None else shape_features.float(),
            None if query_features is None else query_features.float(),
        )
