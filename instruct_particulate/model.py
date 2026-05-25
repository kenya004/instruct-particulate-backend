from __future__ import annotations

import math
import os
from contextlib import nullcontext
from itertools import chain
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from instruct_particulate.utils.inference_utils import (
    axis_point_to_plucker_torch,
    estimate_prismatic_limit_torch,
    estimate_revolute_limit_torch,
    fit_axis_to_closest_points_torch,
)
from instruct_particulate.utils.partfield_feature_utils import (
    PARTFIELD_FEATURE_DIM,
    PartFieldFeatureExtractor,
)
from instruct_particulate.utils.text_embedding_utils import (
    encode_clip_text_prompts,
    load_clip_text_encoder,
)


def _make_silu_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    bias: bool = True,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim, bias=bias),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim, bias=bias),
    )


class _OverparamQueryMasks(NamedTuple):
    revolute_axis: Tensor
    revolute_point: Tensor
    prismatic_axis: Tensor
    prismatic_point: Tensor


class _ActiveLinkGroups(NamedTuple):
    active_flat_mask: Tensor
    group_inverse: Tensor
    num_groups: int


class _OverparamQueryGroups(NamedTuple):
    revolute_axis: _ActiveLinkGroups
    revolute_point: _ActiveLinkGroups
    prismatic_axis: _ActiveLinkGroups
    prismatic_point: _ActiveLinkGroups


JOINT_LOSS_LOGGING_KEYS = (
    "revolute_axis_l1_loss",
    "prismatic_axis_l1_loss",
    "revolute_point_l1_loss",
    "prismatic_point_l1_loss",
    "revolute_direction_loss",
    "prismatic_direction_loss",
    "revolute_range_l1_loss",
    "prismatic_range_l1_loss",
)

_OVERPARAM_LINK_SCALE_MIN = 0.05
_OVERPARAM_AXIS_AABB_HALF_EXTENT_MIN = 1e-4
_PLAIN_JOINT_DECODE_TYPES = frozenset({"plain", "plain+fm"})
_OVERPARAM_JOINT_DECODE_TYPES = frozenset(
    {"overparametrized", "overparam+dir", "overparam+singledir"}
)


def _normalize_joint_decode_type(joint_decode_type: str) -> str:
    normalized_joint_decode_type = str(joint_decode_type).lower()
    if normalized_joint_decode_type in {
        "plain+flowmatching",
        "plain+flow-matching",
        "plain+fm",
    }:
        return "plain+fm"
    if normalized_joint_decode_type in {
        "overparameterization",
        "overparameterized",
        "overparam",
    }:
        return "overparametrized"
    if normalized_joint_decode_type in {
        "overparameterization+dir",
        "overparameterized+dir",
        "overparametrized+dir",
        "overparam+dir",
    }:
        return "overparam+dir"
    if normalized_joint_decode_type in {
        "overparameterization+singledir",
        "overparameterized+singledir",
        "overparametrized+singledir",
        "overparam+singledir",
        "overparameterization+single-dir",
        "overparameterized+single-dir",
        "overparametrized+single-dir",
        "overparam+single-dir",
    }:
        return "overparam+singledir"
    return normalized_joint_decode_type


def _normalize_joint_fm_prediction_type(prediction_type: str) -> str:
    normalized_prediction_type = str(prediction_type).lower()
    if normalized_prediction_type in {"x", "xpred", "x-pred", "x_pred"}:
        return "x"
    if normalized_prediction_type in {"v", "vpred", "v-pred", "v_pred"}:
        return "v"
    return normalized_prediction_type


def _normalize_overparam_closest_axis_space(closest_axis_space: str) -> str:
    normalized_closest_axis_space = str(closest_axis_space).lower()
    if normalized_closest_axis_space in {
        "world",
        "sample",
        "global",
        "global-space",
        "world-space",
    }:
        return "world"
    if normalized_closest_axis_space in {
        "part_aabb",
        "part-aabb",
        "aabb",
        "local_aabb",
        "local-aabb",
    }:
        return "part_aabb"
    return normalized_closest_axis_space


def _coerce_joint_fm_state_stat(
    values: Optional[Sequence[float]],
    *,
    default_value: float,
    name: str,
) -> Tensor:
    if values is None:
        values = [default_value] * 8
    if len(values) != 8:
        raise ValueError(f"{name} must have length 8, got {len(values)}")
    tensor = torch.tensor([float(value) for value in values], dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values, got {values!r}")
    if name.endswith("_std") and torch.any(tensor <= 0.0):
        raise ValueError(f"{name} must contain only positive values, got {values!r}")
    return tensor


def _sample_logitnorm_t(
    *,
    size: Sequence[int],
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    logits = torch.randn(size, device=device, dtype=torch.float32)
    return torch.sigmoid(logits * float(std) + float(mean)).to(dtype=dtype)


def _gather_joint_link_latents(
    *,
    link_latents: Tensor,
    joint_connections: Tensor,
) -> Tuple[Tensor, Tensor]:
    parent_indices = joint_connections[..., 0].clamp_min(0)
    child_indices = joint_connections[..., 1].clamp_min(0)
    gather_index = parent_indices.unsqueeze(-1).expand(-1, -1, link_latents.shape[-1])
    parent_latents = link_latents.gather(dim=1, index=gather_index)
    gather_index = child_indices.unsqueeze(-1).expand(-1, -1, link_latents.shape[-1])
    child_latents = link_latents.gather(dim=1, index=gather_index)
    return parent_latents, child_latents


def _build_joint_motion_condition_inputs(
    *,
    parent_latents: Tensor,
    child_latents: Tensor,
    motion_type: str,
    revolute_embedding: Tensor,
    prismatic_embedding: Tensor,
) -> Tensor:
    if parent_latents.shape != child_latents.shape:
        raise ValueError(
            "parent_latents and child_latents must share the same shape, "
            f"got {tuple(parent_latents.shape)} and {tuple(child_latents.shape)}"
        )
    if motion_type == "revolute":
        type_embedding = revolute_embedding
    elif motion_type == "prismatic":
        type_embedding = prismatic_embedding
    else:
        raise ValueError(
            "motion_type must be 'revolute' or 'prismatic', "
            f"got {motion_type!r}"
        )
    type_embeddings = type_embedding.to(
        device=parent_latents.device,
        dtype=parent_latents.dtype,
    ).view(1, 1, -1).expand_as(parent_latents)
    return torch.cat((type_embeddings, parent_latents, child_latents), dim=-1)


def _build_joint_supervision_mask(
    *,
    joint_valid_flag: Tensor,
    motion_flag: Tensor,
    supervision_valid: Tensor | None,
) -> Tensor:
    supervision_mask = joint_valid_flag & motion_flag
    if supervision_valid is not None:
        supervision_mask = supervision_mask & supervision_valid.to(
            device=joint_valid_flag.device,
            dtype=torch.bool,
        )
    return supervision_mask


def _reduce_l1_error_to_mask_shape(
    l1_error: Tensor,
    mask: Tensor,
) -> Tensor:
    if l1_error.ndim < mask.ndim:
        raise ValueError(
            f"l1_error must have at least as many dims as mask, got {l1_error.ndim} and {mask.ndim}"
        )
    if l1_error.shape[: mask.ndim] != mask.shape:
        raise ValueError(
            "Leading l1_error dims must match mask shape, "
            f"got {tuple(l1_error.shape[: mask.ndim])} and {tuple(mask.shape)}"
        )
    if l1_error.ndim == mask.ndim:
        return l1_error
    trailing_dims = tuple(range(mask.ndim, l1_error.ndim))
    return l1_error.mean(dim=trailing_dims)


def _masked_mean_error(error: Tensor, mask: Tensor) -> Tensor:
    mask_weights = mask.to(dtype=error.dtype)
    masked_error = error * mask_weights
    normalizer = mask_weights.sum()
    return masked_error.sum() / normalizer.clamp_min(1.0)


def _masked_flip_invariant_joint_parameter_losses(
    *,
    predicted_axis: Tensor,
    target_axis: Tensor,
    axis_mask: Tensor,
    predicted_range: Tensor,
    target_range: Tensor,
    range_mask: Tensor,
    axis_weight: float = 1.0,
    range_weight: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    direct_axis_error = _reduce_l1_error_to_mask_shape(
        F.l1_loss(predicted_axis, target_axis, reduction="none"),
        axis_mask,
    )
    flipped_axis_error = _reduce_l1_error_to_mask_shape(
        F.l1_loss(-predicted_axis, target_axis, reduction="none"),
        axis_mask,
    )
    direct_range_error = _reduce_l1_error_to_mask_shape(
        F.l1_loss(predicted_range, target_range, reduction="none"),
        range_mask,
    )
    flipped_range_error = _reduce_l1_error_to_mask_shape(
        F.l1_loss(-predicted_range, target_range, reduction="none"),
        range_mask,
    )

    direct_joint_error = (
        float(axis_weight) * direct_axis_error * axis_mask.to(dtype=direct_axis_error.dtype)
        + float(range_weight) * direct_range_error * range_mask.to(dtype=direct_range_error.dtype)
    )
    flipped_joint_error = (
        float(axis_weight) * flipped_axis_error * axis_mask.to(dtype=flipped_axis_error.dtype)
        + float(range_weight) * flipped_range_error * range_mask.to(dtype=flipped_range_error.dtype)
    )
    use_flipped = flipped_joint_error < direct_joint_error
    selected_axis_error = torch.where(use_flipped, flipped_axis_error, direct_axis_error)
    selected_range_error = torch.where(use_flipped, flipped_range_error, direct_range_error)
    return (
        _masked_mean_error(selected_axis_error, axis_mask),
        _masked_mean_error(selected_range_error, range_mask),
    )


class LossComputationResult(NamedTuple):
    losses: Dict[str, Tensor]
    joint_loss_support_counts: Dict[str, Tensor]


class SwiGLUFeedForward(nn.Module):
    """Modern gated MLP used after attention blocks."""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        multiplier: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * multiplier)
        self.in_proj = nn.Linear(dim, 2 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        x = value * F.silu(gate)
        x = self.dropout(x)
        return self.out_proj(x)


class FrequencyMLPEmbedder(nn.Module):
    """Embeds coordinates or normals with Fourier features and a small MLP."""

    def __init__(
        self,
        output_dim: int,
        num_frequencies: int,
        input_dim: int = 3,
        hidden_dim: Optional[int] = None,
        max_frequency: float = 32.0,
        include_raw: bool = True,
    ):
        super().__init__()
        if num_frequencies <= 0:
            raise ValueError(f"num_frequencies must be positive, got {num_frequencies}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if max_frequency <= 0.0:
            raise ValueError(f"max_frequency must be positive, got {max_frequency}")

        hidden_dim = hidden_dim or output_dim
        self.input_dim = input_dim
        self.include_raw = include_raw

        if num_frequencies == 1:
            frequencies = torch.tensor([math.pi], dtype=torch.float32)
        else:
            frequencies = torch.exp(
                torch.linspace(0.0, math.log(max_frequency), steps=num_frequencies, dtype=torch.float32)
            ) * math.pi
        self.register_buffer("frequencies", frequencies, persistent=False)

        encoded_dim = input_dim * (2 * num_frequencies + int(include_raw))
        self.input_norm = nn.RMSNorm(encoded_dim)
        self.mlp = _make_silu_mlp(
            input_dim=encoded_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected inputs with last dimension {self.input_dim}, got {tuple(x.shape)}"
            )

        x_float = x.float()
        angles = x_float.unsqueeze(-1) * self.frequencies
        encoded = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=-2)
        if self.include_raw:
            encoded = torch.cat((encoded, x_float), dim=-1)
        encoded = encoded.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(self.input_norm(encoded))


class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion/flow timesteps with sinusoidal features."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        frequency_embedding_dim: int = 256,
        max_period: float = 10_000.0,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if frequency_embedding_dim <= 0:
            raise ValueError(
                f"frequency_embedding_dim must be positive, got {frequency_embedding_dim}"
            )
        if max_period <= 0.0:
            raise ValueError(f"max_period must be positive, got {max_period}")
        self.frequency_embedding_dim = int(frequency_embedding_dim)
        self.max_period = float(max_period)
        self.mlp = _make_silu_mlp(
            input_dim=self.frequency_embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )

    def _frequency_embedding(self, t: Tensor) -> Tensor:
        half_dim = self.frequency_embedding_dim // 2
        if half_dim == 0:
            return t.unsqueeze(-1)
        frequency_exponents = torch.arange(
            half_dim,
            device=t.device,
            dtype=torch.float32,
        )
        frequency_exponents = frequency_exponents / max(half_dim, 1)
        frequencies = torch.exp(-math.log(self.max_period) * frequency_exponents)
        angles = t.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if self.frequency_embedding_dim % 2 != 0:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        if t.ndim != 1:
            raise ValueError(f"expected a 1D timestep tensor, got shape {tuple(t.shape)}")
        frequency_embedding = self._frequency_embedding(t).to(
            dtype=self.mlp[0].weight.dtype
        )
        return self.mlp(frequency_embedding)


class SDPASelfAttention(nn.Module):
    """Self-attention implemented with PyTorch SDPA."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qk_norm: bool = True,
        qkv_bias: bool = True,
    ):
        super().__init__()
        if head_dim is None:
            if model_dim % num_heads != 0:
                raise ValueError(
                    f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads}) "
                    "when head_dim is not specified"
                )
            head_dim = model_dim // num_heads

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.attn_dropout = attn_dropout

        self.qkv_proj = nn.Linear(model_dim, 3 * self.inner_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(self.inner_dim, model_dim)
        self.out_dropout = nn.Dropout(proj_dropout)

        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x: Tensor, *, mask: Tensor | None = None) -> Tensor:
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.q_norm(self._reshape_heads(q))
        k = self.k_norm(self._reshape_heads(k))
        v = self._reshape_heads(v)
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, :, None] & mask[:, None, None, :]

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            x.shape[0],
            x.shape[1],
            self.inner_dim,
        )
        return self.out_dropout(self.out_proj(attn_output))


class SDPACrossAttention(nn.Module):
    """Cross-attention implemented with PyTorch SDPA."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qk_norm: bool = True,
        q_bias: bool = True,
        kv_bias: bool = True,
    ):
        super().__init__()
        if head_dim is None:
            if model_dim % num_heads != 0:
                raise ValueError(
                    f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads}) "
                    "when head_dim is not specified"
                )
            head_dim = model_dim // num_heads

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.attn_dropout = attn_dropout

        self.q_proj = nn.Linear(model_dim, self.inner_dim, bias=q_bias)
        self.kv_proj = nn.Linear(model_dim, 2 * self.inner_dim, bias=kv_bias)
        self.out_proj = nn.Linear(self.inner_dim, model_dim)
        self.out_dropout = nn.Dropout(proj_dropout)

        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        context: Tensor,
        *,
        query_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        q = self.q_norm(self._reshape_heads(self.q_proj(query)))
        k, v = self.kv_proj(context).chunk(2, dim=-1)
        k = self.k_norm(self._reshape_heads(k))
        v = self._reshape_heads(v)

        if query_mask is None and context_mask is None:
            attn_mask = None
        else:
            if query_mask is None:
                query_mask = torch.ones(query.shape[:2], device=query.device, dtype=torch.bool)
            if context_mask is None:
                context_mask = torch.ones(
                    context.shape[:2],
                    device=context.device,
                    dtype=torch.bool,
                )
            attn_mask = query_mask[:, None, :, None] & context_mask[:, None, None, :]

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            query.shape[0],
            query.shape[1],
            self.inner_dim,
        )
        return self.out_dropout(self.out_proj(attn_output))


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block with a gated MLP."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ffn_multiplier: float = 4.0,
        ffn_dropout: float = 0.0,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.query_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.context_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.attn = SDPACrossAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            qk_norm=qk_norm,
        )
        self.ffn_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.ffn = SwiGLUFeedForward(
            dim=model_dim,
            multiplier=ffn_multiplier,
            dropout=ffn_dropout,
        )

    def forward(
        self,
        query: Tensor,
        context: Tensor,
        *,
        query_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        query = query + self.attn(
            self.query_norm(query),
            self.context_norm(context),
            query_mask=query_mask,
            context_mask=context_mask,
        )
        if query_mask is not None:
            query = query.masked_fill(~query_mask.unsqueeze(-1), 0)
        query = query + self.ffn(self.ffn_norm(query))
        if query_mask is not None:
            query = query.masked_fill(~query_mask.unsqueeze(-1), 0)
        return query


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention block with a gated MLP."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ffn_multiplier: float = 4.0,
        ffn_dropout: float = 0.0,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attn_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.attn = SDPASelfAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            qk_norm=qk_norm,
        )
        self.ffn_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.ffn = SwiGLUFeedForward(
            dim=model_dim,
            multiplier=ffn_multiplier,
            dropout=ffn_dropout,
        )

    def forward(self, x: Tensor, *, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), mask=mask)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = x + self.ffn(self.ffn_norm(x))
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        return x


class ShapeLatentEncoder(nn.Module):
    """Encodes a point cloud into a fixed set of shape latents."""

    def __init__(
        self,
        model_dim: int,
        num_shape_latents: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ffn_multiplier: float = 4.0,
        ffn_dropout: float = 0.0,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.shape_latents = nn.Parameter(torch.empty(num_shape_latents, model_dim))
        self.block = CrossAttentionBlock(
            model_dim=model_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            ffn_multiplier=ffn_multiplier,
            ffn_dropout=ffn_dropout,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
        )
        self.output_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.shape_latents, std=0.02)

    def forward(self, point_tokens: Tensor) -> Tensor:
        batch_size = point_tokens.shape[0]
        latents = self.shape_latents.to(dtype=point_tokens.dtype).unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        latents = self.block(latents, point_tokens)
        return self.output_norm(latents)


class SegmentationDecoder(nn.Module):
    """Pairwise query-to-link scoring over padded link tokens."""

    def __init__(
        self,
        model_dim: int,
        decode_type: str = "dot",
        bias: bool = True,
        query_chunk_size: Optional[int] = None,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.decode_type = str(decode_type)
        if self.decode_type not in {"dot", "mlp"}:
            raise ValueError(
                f"decode_type must be 'dot' or 'mlp', got {decode_type!r}"
            )
        if query_chunk_size is not None and int(query_chunk_size) <= 0:
            raise ValueError(f"query_chunk_size must be positive when set, got {query_chunk_size}")
        self.query_chunk_size = None if query_chunk_size is None else int(query_chunk_size)

        if self.decode_type == "mlp":
            self.decoder = _make_silu_mlp(
                input_dim=model_dim * 2,
                hidden_dim=model_dim * 4,
                output_dim=1,
                bias=bias,
            )

    def _decode_mlp_logits(
        self,
        query_latents: Tensor,
        link_latents: Tensor,
    ) -> Tensor:
        expanded_link_latents = link_latents.unsqueeze(1).expand(
            -1,
            query_latents.shape[1],
            -1,
            -1,
        )
        expanded_query_latents = query_latents.unsqueeze(2).expand(
            -1,
            -1,
            link_latents.shape[1],
            -1,
        )
        pair_latents = torch.cat(
            (expanded_link_latents, expanded_query_latents),
            dim=-1,
        )
        return self.decoder(pair_latents).squeeze(-1)

    def forward(
        self,
        query_latents: Tensor,
        link_latents: Tensor,
        link_valid_flag: Tensor,
    ) -> Tensor:
        if self.decode_type == "dot":
            logits = torch.matmul(query_latents, link_latents.transpose(-1, -2))
        else:
            if (
                self.query_chunk_size is None
                or query_latents.shape[1] <= self.query_chunk_size
            ):
                logits = self._decode_mlp_logits(query_latents, link_latents)
            else:
                logits = torch.cat(
                    [
                        self._decode_mlp_logits(
                            query_latents[:, start : start + self.query_chunk_size],
                            link_latents,
                        )
                        for start in range(0, query_latents.shape[1], self.query_chunk_size)
                    ],
                    dim=1,
                )
        return logits.masked_fill(~link_valid_flag.unsqueeze(1), float("-inf"))


class JointDecoderPlain(nn.Module):
    """Decodes per-joint axes and ranges from parent/child link latents."""

    motion_state_dim = 8

    def __init__(
        self,
        model_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.revolute_embedding = nn.Parameter(torch.empty(model_dim))
        self.prismatic_embedding = nn.Parameter(torch.empty(model_dim))
        self.decoder = _make_silu_mlp(
            input_dim=model_dim * 3,
            hidden_dim=model_dim * 4,
            output_dim=self.motion_state_dim,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.revolute_embedding, std=0.02)
        nn.init.trunc_normal_(self.prismatic_embedding, std=0.02)

    def _predict_motion_parameters(
        self,
        *,
        parent_latents: Tensor,
        child_latents: Tensor,
        active_mask: Tensor,
        motion_type: str,
    ) -> Tuple[Tensor, Tensor]:
        decoded = self.decoder(
            _build_joint_motion_condition_inputs(
                parent_latents=parent_latents,
                child_latents=child_latents,
                motion_type=motion_type,
                revolute_embedding=self.revolute_embedding,
                prismatic_embedding=self.prismatic_embedding,
            )
        )
        decoded = decoded.masked_fill(~active_mask.unsqueeze(-1), 0)
        return decoded[..., :6], decoded[..., 6:8]

    def predict(
        self,
        link_latents: Tensor,
        joint_connections: Tensor,
        joint_valid_flag: Tensor,
        is_revolute: Tensor,
        is_prismatic: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decodes padded joint tensors and zeros invalid rows in the outputs."""
        parent_latents, child_latents = _gather_joint_link_latents(
            link_latents=link_latents,
            joint_connections=joint_connections,
        )
        motion_flags = dict(revolute=is_revolute, prismatic=is_prismatic)
        motion_outputs: Dict[str, Tensor] = {}
        for motion_type in ("revolute", "prismatic"):
            motion_outputs[f"{motion_type}_axis"], motion_outputs[f"{motion_type}_range"] = self._predict_motion_parameters(
                parent_latents=parent_latents,
                child_latents=child_latents,
                active_mask=joint_valid_flag & motion_flags[motion_type],
                motion_type=motion_type)
        return (
            motion_outputs["revolute_axis"],
            motion_outputs["prismatic_axis"],
            motion_outputs["revolute_range"],
            motion_outputs["prismatic_range"],
        )

    def compute_loss(
        self,
        *,
        link_latents: Tensor,
        joint_connections: Tensor,
        joint_valid_flag: Tensor,
        is_revolute: Tensor,
        is_prismatic: Tensor,
        revolute_axis: Tensor,
        prismatic_axis: Tensor,
        revolute_range: Tensor,
        prismatic_range: Tensor,
        revolute_axis_supervision_valid: Tensor | None = None,
        prismatic_axis_supervision_valid: Tensor | None = None,
        revolute_range_supervision_valid: Tensor | None = None,
        prismatic_range_supervision_valid: Tensor | None = None,
        fm_batch_size: int = 1024,
    ) -> Dict[str, Tensor]:
        del fm_batch_size
        parent_latents, child_latents = _gather_joint_link_latents(
            link_latents=link_latents,
            joint_connections=joint_connections,
        )
        axis_targets = dict(revolute=revolute_axis, prismatic=prismatic_axis)
        range_targets = dict(revolute=revolute_range, prismatic=prismatic_range)
        axis_supervision_valid = dict(revolute=revolute_axis_supervision_valid, prismatic=prismatic_axis_supervision_valid)
        range_supervision_valid = dict(revolute=revolute_range_supervision_valid, prismatic=prismatic_range_supervision_valid)
        motion_flags = dict(revolute=is_revolute, prismatic=is_prismatic)
        losses: Dict[str, Tensor] = {}
        for motion_type in ("revolute", "prismatic"):
            predicted_axis, predicted_range = self._predict_motion_parameters(
                parent_latents=parent_latents,
                child_latents=child_latents,
                active_mask=joint_valid_flag & motion_flags[motion_type],
                motion_type=motion_type,
            )
            axis_mask = _build_joint_supervision_mask(
                joint_valid_flag=joint_valid_flag,
                motion_flag=motion_flags[motion_type],
                supervision_valid=axis_supervision_valid[motion_type],
            )
            range_mask = _build_joint_supervision_mask(
                joint_valid_flag=joint_valid_flag,
                motion_flag=motion_flags[motion_type],
                supervision_valid=range_supervision_valid[motion_type],
            )
            axis_loss, range_loss = _masked_flip_invariant_joint_parameter_losses(
                predicted_axis=predicted_axis,
                target_axis=axis_targets[motion_type],
                axis_mask=axis_mask,
                predicted_range=predicted_range,
                target_range=range_targets[motion_type],
                range_mask=range_mask,
            )
            losses[f"{motion_type}_joint_axis_l1_loss"] = axis_loss
            losses[f"{motion_type}_joint_range_l1_loss"] = range_loss
        return losses

class JointDecoderPlainFlowMatching(nn.Module):
    """Predicts plain joint FM states with configurable x/v parameterization."""

    motion_state_dim = 8

    def __init__(
        self,
        model_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        prediction_type: str = "v",
        time_embedding_dim: int = 256,
        inference_steps: int = 100,
        time_scale: float = 1000.0,
        sigma_min: float = 0.0,
        rescale_t: float = 1.0,
        cfg_scale: float = 1.0,
        revolute_state_mean: Optional[Sequence[float]] = None,
        revolute_state_std: Optional[Sequence[float]] = None,
        prismatic_state_mean: Optional[Sequence[float]] = None,
        prismatic_state_std: Optional[Sequence[float]] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.hidden_dim = int(hidden_dim or model_dim)
        self.inference_steps = int(inference_steps)
        if self.inference_steps <= 0:
            raise ValueError(f"inference_steps must be positive, got {inference_steps}")
        prediction_type = _normalize_joint_fm_prediction_type(prediction_type)
        if prediction_type not in {"x", "v"}:
            raise ValueError(
                "prediction_type must be 'x' or 'v', "
                f"got {prediction_type!r}"
            )
        self.prediction_type = prediction_type
        self.time_scale = float(time_scale)
        self.sigma_min = float(sigma_min)
        self.rescale_t = float(rescale_t)
        self.cfg_scale = float(cfg_scale)
        self.register_buffer(
            "revolute_state_mean",
            _coerce_joint_fm_state_stat(
                revolute_state_mean,
                default_value=0.0,
                name="revolute_state_mean",
            ),
            persistent=True,
        )
        self.register_buffer(
            "revolute_state_std",
            _coerce_joint_fm_state_stat(
                revolute_state_std,
                default_value=1.0,
                name="revolute_state_std",
            ),
            persistent=True,
        )
        self.register_buffer(
            "prismatic_state_mean",
            _coerce_joint_fm_state_stat(
                prismatic_state_mean,
                default_value=0.0,
                name="prismatic_state_mean",
            ),
            persistent=True,
        )
        self.register_buffer(
            "prismatic_state_std",
            _coerce_joint_fm_state_stat(
                prismatic_state_std,
                default_value=1.0,
                name="prismatic_state_std",
            ),
            persistent=True,
        )
        self.revolute_embedding = nn.Parameter(torch.empty(model_dim))
        self.prismatic_embedding = nn.Parameter(torch.empty(model_dim))
        self.condition_projector = _make_silu_mlp(
            input_dim=model_dim * 3,
            hidden_dim=self.hidden_dim * 2,
            output_dim=self.hidden_dim,
            bias=bias,
        )
        self.state_projector = nn.Linear(
            self.motion_state_dim,
            self.hidden_dim,
            bias=bias,
        )
        self.time_embedder = TimestepEmbedder(
            self.hidden_dim,
            frequency_embedding_dim=time_embedding_dim,
        )
        self.input_norm = nn.RMSNorm(self.hidden_dim)
        self.residual_block_1 = _make_silu_mlp(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim * 4,
            output_dim=self.hidden_dim,
            bias=bias,
        )
        self.residual_block_2 = _make_silu_mlp(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim * 4,
            output_dim=self.hidden_dim,
            bias=bias,
        )
        self.output_norm = nn.RMSNorm(self.hidden_dim)
        self.output_projector = nn.Linear(
            self.hidden_dim,
            self.motion_state_dim,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.revolute_embedding, std=0.02)
        nn.init.trunc_normal_(self.prismatic_embedding, std=0.02)

    def _prepare_t_sequence(
        self,
        *,
        device: torch.device,
        steps: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        num_steps = self.inference_steps if steps is None else int(steps)
        if num_steps <= 0:
            raise ValueError(f"steps must be positive, got {num_steps}")
        t_sequence = torch.linspace(
            0.0,
            1.0,
            num_steps + 1,
            device=device,
            dtype=torch.float32,
        )
        if self.rescale_t:
            t_sequence = t_sequence / (
                1.0 + (self.rescale_t - 1.0) * (1.0 - t_sequence)
            )
        return t_sequence.to(dtype=dtype)

    def _motion_state_mean_and_std(
        self,
        *,
        motion_type: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        if motion_type == "revolute":
            mean = self.revolute_state_mean
            std = self.revolute_state_std
        elif motion_type == "prismatic":
            mean = self.prismatic_state_mean
            std = self.prismatic_state_std
        else:
            raise ValueError(
                "motion_type must be 'revolute' or 'prismatic', "
                f"got {motion_type!r}"
            )
        safe_std = std.to(device=device, dtype=dtype).clamp_min(1.0e-6)
        return mean.to(device=device, dtype=dtype), safe_std

    def _normalize_motion_state(
        self,
        motion_state: Tensor,
        *,
        motion_type: str,
    ) -> Tensor:
        mean, std = self._motion_state_mean_and_std(
            motion_type=motion_type,
            device=motion_state.device,
            dtype=motion_state.dtype,
        )
        return (motion_state - mean) / std

    def _unnormalize_motion_state(
        self,
        motion_state: Tensor,
        *,
        motion_type: str,
    ) -> Tensor:
        mean, std = self._motion_state_mean_and_std(
            motion_type=motion_type,
            device=motion_state.device,
            dtype=motion_state.dtype,
        )
        return motion_state * std + mean

    def _sample_motion_state(
        self,
        *,
        condition_embeddings: Tensor,
        steps: int | None = None,
        cfg_scale: float | None = None,
    ) -> Tensor:
        num_samples = condition_embeddings.shape[0]
        if num_samples == 0:
            return condition_embeddings.new_zeros((0, self.motion_state_dim))
        guidance_scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        motion_state = torch.randn(
            (num_samples, self.motion_state_dim),
            device=condition_embeddings.device,
            dtype=condition_embeddings.dtype,
        )
        t_sequence = self._prepare_t_sequence(
            device=condition_embeddings.device,
            steps=steps,
            dtype=condition_embeddings.dtype,
        )
        for t0, t1 in zip(t_sequence[:-1], t_sequence[1:]):
            time_vector = motion_state.new_full((num_samples,), t0)
            if guidance_scale == 1.0:
                velocity = self.predict_velocity(
                    x_t=motion_state,
                    t=time_vector,
                    condition_embeddings=condition_embeddings,
                )
            else:
                velocity = self._predict_guided_velocity(
                    x_t=motion_state,
                    t=time_vector,
                    condition_embeddings=condition_embeddings,
                    cfg_scale=guidance_scale,
                )
            motion_state = motion_state + (t1 - t0).to(dtype=motion_state.dtype) * velocity
        return motion_state

    def _bridge_noise_scale(self, t: Tensor) -> Tensor:
        return 1.0 - (1.0 - self.sigma_min) * t

    def _predict_model_output(
        self,
        *,
        x_t: Tensor,
        t: Tensor,
        condition_embeddings: Tensor,
    ) -> Tensor:
        if x_t.ndim != 2:
            raise ValueError(f"x_t must be rank-2, got shape {tuple(x_t.shape)}")
        if x_t.shape[-1] != self.motion_state_dim:
            raise ValueError(
                f"x_t last dim must be {self.motion_state_dim}, got {tuple(x_t.shape)}"
            )
        if condition_embeddings.ndim != 2:
            raise ValueError(
                "condition_embeddings must be rank-2, "
                f"got shape {tuple(condition_embeddings.shape)}"
            )
        if condition_embeddings.shape[0] != x_t.shape[0]:
            raise ValueError(
                "condition_embeddings batch dim must match x_t, "
                f"got {tuple(condition_embeddings.shape)} and {tuple(x_t.shape)}"
            )
        if t.ndim != 1 or t.shape[0] != x_t.shape[0]:
            raise ValueError(
                "t must be a vector matching x_t batch size, "
                f"got {tuple(t.shape)} and {tuple(x_t.shape)}"
            )
        x_hidden = self.state_projector(x_t.to(dtype=self.state_projector.weight.dtype))
        time_hidden = self.time_embedder(t * self.time_scale)
        condition_embeddings = condition_embeddings.to(dtype=x_hidden.dtype)
        hidden = self.input_norm(x_hidden + time_hidden + condition_embeddings)
        hidden = hidden + self.residual_block_1(hidden)
        hidden = hidden + self.residual_block_2(hidden)
        return self.output_projector(self.output_norm(hidden))

    def _predicted_clean_state_to_velocity(
        self,
        *,
        predicted_clean_state: Tensor,
        x_t: Tensor,
        t: Tensor,
    ) -> Tensor:
        bridge_noise_scale = self._bridge_noise_scale(t).unsqueeze(-1).to(
            dtype=predicted_clean_state.dtype,
        )
        bridge_noise_scale = bridge_noise_scale.clamp_min(
            torch.finfo(predicted_clean_state.dtype).eps
        )
        x_t = x_t.to(dtype=predicted_clean_state.dtype)
        return (
            predicted_clean_state - (1.0 - self.sigma_min) * x_t
        ) / bridge_noise_scale

    def predict_velocity(
        self,
        *,
        x_t: Tensor,
        t: Tensor,
        condition_embeddings: Tensor,
    ) -> Tensor:
        model_output = self._predict_model_output(
            x_t=x_t,
            t=t,
            condition_embeddings=condition_embeddings,
        )
        if self.prediction_type == "v":
            return model_output
        return self._predicted_clean_state_to_velocity(
            predicted_clean_state=model_output,
            x_t=x_t,
            t=t,
        )

    def _predict_guided_velocity(
        self,
        *,
        x_t: Tensor,
        t: Tensor,
        condition_embeddings: Tensor,
        cfg_scale: float | None = None,
    ) -> Tensor:
        guidance_scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        if not math.isfinite(guidance_scale) or guidance_scale < 0.0:
            raise ValueError(
                "cfg_scale must be a finite non-negative number, "
                f"got {guidance_scale!r}"
            )
        conditional_velocity = self.predict_velocity(
            x_t=x_t,
            t=t,
            condition_embeddings=condition_embeddings,
        )
        unconditional_velocity = self.predict_velocity(
            x_t=x_t,
            t=t,
            condition_embeddings=torch.zeros_like(condition_embeddings),
        )
        return unconditional_velocity + guidance_scale * (
            conditional_velocity - unconditional_velocity
        )

    def predict(
        self,
        link_latents: Tensor,
        joint_connections: Tensor,
        joint_valid_flag: Tensor,
        is_revolute: Tensor,
        is_prismatic: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        parent_latents, child_latents = _gather_joint_link_latents(
            link_latents=link_latents,
            joint_connections=joint_connections,
        )
        batch_size, max_joints = joint_valid_flag.shape
        motion_flags = dict(revolute=is_revolute, prismatic=is_prismatic)
        motion_outputs: Dict[str, Tensor] = {}
        for motion_type in ("revolute", "prismatic"):
            motion_outputs[f"{motion_type}_axis"] = link_latents.new_zeros(
                (batch_size, max_joints, 6)
            )
            motion_outputs[f"{motion_type}_range"] = link_latents.new_zeros(
                (batch_size, max_joints, 2)
            )
            active_mask = joint_valid_flag & motion_flags[motion_type]
            if not torch.any(active_mask):
                continue
            condition_embeddings = self.condition_projector(
                _build_joint_motion_condition_inputs(
                    parent_latents=parent_latents,
                    child_latents=child_latents,
                    motion_type=motion_type,
                    revolute_embedding=self.revolute_embedding,
                    prismatic_embedding=self.prismatic_embedding,
                )
            )[active_mask]
            motion_state = self._sample_motion_state(
                condition_embeddings=condition_embeddings,
            )
            motion_state = self._unnormalize_motion_state(
                motion_state,
                motion_type=motion_type,
            )
            motion_outputs[f"{motion_type}_axis"][active_mask] = motion_state[..., :6]
            motion_outputs[f"{motion_type}_range"][active_mask] = motion_state[..., 6:8]
        return (
            motion_outputs["revolute_axis"],
            motion_outputs["prismatic_axis"],
            motion_outputs["revolute_range"],
            motion_outputs["prismatic_range"],
        )

    def _zero_fm_loss(self) -> Tensor:
        connected_zero = None
        for parameter in self.parameters():
            term = parameter.reshape(-1)[0]
            connected_zero = term if connected_zero is None else connected_zero + term
        if connected_zero is None:
            raise RuntimeError("JointDecoderPlainFlowMatching unexpectedly has no parameters")
        return connected_zero * 0.0

    def _compute_motion_fm_loss(
        self,
        *,
        link_latents: Tensor,
        joint_connections: Tensor,
        joint_valid_flag: Tensor,
        motion_type: str,
        motion_flag: Tensor,
        target_axis: Tensor,
        target_range: Tensor,
        axis_supervision_valid: Tensor | None,
        range_supervision_valid: Tensor | None,
        fm_batch_size: int,
        training_time_mean: float,
        training_time_std: float,
        cfg_dropout: float,
    ) -> Tensor:
        axis_mask = _build_joint_supervision_mask(
            joint_valid_flag=joint_valid_flag,
            motion_flag=motion_flag,
            supervision_valid=axis_supervision_valid,
        )
        range_mask = _build_joint_supervision_mask(
            joint_valid_flag=joint_valid_flag,
            motion_flag=motion_flag,
            supervision_valid=range_supervision_valid,
        )
        pair_mask = axis_mask | range_mask
        if not torch.any(pair_mask):
            return self._zero_fm_loss()

        parent_latents, child_latents = _gather_joint_link_latents(
            link_latents=link_latents,
            joint_connections=joint_connections,
        )
        condition_embeddings = self.condition_projector(
            _build_joint_motion_condition_inputs(
                parent_latents=parent_latents,
                child_latents=child_latents,
                motion_type=motion_type,
                revolute_embedding=self.revolute_embedding,
                prismatic_embedding=self.prismatic_embedding,
            )
        )[pair_mask]
        target_state = torch.cat((target_axis[pair_mask], target_range[pair_mask]), dim=-1).to(
            dtype=condition_embeddings.dtype
        )
        target_state = self._normalize_motion_state(
            target_state,
            motion_type=motion_type,
        )
        state_mask = torch.cat(
            (
                axis_mask[pair_mask].unsqueeze(-1).expand(-1, 6),
                range_mask[pair_mask].unsqueeze(-1).expand(-1, 2),
            ),
            dim=-1,
        )
        flat_batch_size = target_state.shape[0] * int(fm_batch_size)
        time_samples = _sample_logitnorm_t(
            size=(flat_batch_size,),
            device=condition_embeddings.device,
            mean=training_time_mean,
            std=training_time_std,
            dtype=condition_embeddings.dtype,
        )
        noise = torch.randn(
            (flat_batch_size, self.motion_state_dim),
            device=condition_embeddings.device,
            dtype=condition_embeddings.dtype,
        )
        tiled_target_state = target_state.unsqueeze(1).expand(-1, int(fm_batch_size), -1).reshape(
            flat_batch_size,
            self.motion_state_dim,
        )
        tiled_condition_embeddings = condition_embeddings.unsqueeze(1).expand(
            -1,
            int(fm_batch_size),
            -1,
        ).reshape(flat_batch_size, condition_embeddings.shape[-1])
        if cfg_dropout > 0.0:
            dropout_mask = torch.rand(
                flat_batch_size,
                device=tiled_condition_embeddings.device,
            ) < cfg_dropout
            tiled_condition_embeddings = tiled_condition_embeddings.masked_fill(
                dropout_mask.unsqueeze(-1),
                0,
            )
        tiled_state_mask = state_mask.unsqueeze(1).expand(-1, int(fm_batch_size), -1).reshape(
            flat_batch_size,
            self.motion_state_dim,
        )
        t_broadcast = time_samples.unsqueeze(-1)
        x_t = self._bridge_noise_scale(t_broadcast) * noise + t_broadcast * tiled_target_state
        target_velocity = tiled_target_state - (1.0 - self.sigma_min) * noise
        predicted_velocity = self.predict_velocity(
            x_t=x_t,
            t=time_samples,
            condition_embeddings=tiled_condition_embeddings,
        )
        squared_error = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
        mask_weights = tiled_state_mask.to(dtype=squared_error.dtype)
        return (squared_error * mask_weights).sum() / mask_weights.sum().clamp_min(1.0)

    def compute_loss(
        self,
        *,
        link_latents: Tensor,
        joint_connections: Tensor,
        joint_valid_flag: Tensor,
        is_revolute: Tensor,
        is_prismatic: Tensor,
        revolute_axis: Tensor,
        prismatic_axis: Tensor,
        revolute_range: Tensor,
        prismatic_range: Tensor,
        revolute_axis_supervision_valid: Tensor | None = None,
        prismatic_axis_supervision_valid: Tensor | None = None,
        revolute_range_supervision_valid: Tensor | None = None,
        prismatic_range_supervision_valid: Tensor | None = None,
        fm_batch_size: int = 1024,
        training_time_mean: float = 0.0,
        training_time_std: float = 1.0,
        cfg_dropout: float = 0.0,
    ) -> Dict[str, Tensor]:
        if fm_batch_size <= 0:
            raise ValueError(f"fm_batch_size must be positive, got {fm_batch_size}")
        if not math.isfinite(cfg_dropout) or cfg_dropout < 0.0 or cfg_dropout > 1.0:
            raise ValueError(
                "cfg_dropout must be a finite number in [0, 1], "
                f"got {cfg_dropout!r}"
            )
        axis_targets = dict(revolute=revolute_axis, prismatic=prismatic_axis)
        range_targets = dict(revolute=revolute_range, prismatic=prismatic_range)
        axis_supervision_valid = dict(revolute=revolute_axis_supervision_valid, prismatic=prismatic_axis_supervision_valid)
        range_supervision_valid = dict(revolute=revolute_range_supervision_valid, prismatic=prismatic_range_supervision_valid)
        motion_flags = dict(revolute=is_revolute, prismatic=is_prismatic)
        losses: Dict[str, Tensor] = {}
        for motion_type in ("revolute", "prismatic"):
            losses[f"{motion_type}_joint_fm_loss"] = self._compute_motion_fm_loss(
                link_latents=link_latents,
                joint_connections=joint_connections,
                joint_valid_flag=joint_valid_flag,
                motion_type=motion_type,
                motion_flag=motion_flags[motion_type],
                target_axis=axis_targets[motion_type],
                target_range=range_targets[motion_type],
                axis_supervision_valid=axis_supervision_valid[motion_type],
                range_supervision_valid=range_supervision_valid[motion_type],
                fm_batch_size=fm_batch_size,
                training_time_mean=training_time_mean,
                training_time_std=training_time_std,
                cfg_dropout=cfg_dropout,
            )
        return losses

class JointDecoderOverParametrized(nn.Module):
    """Decodes per-query over-parameterized joint supervision targets."""

    def __init__(
        self,
        model_dim: int,
        output_dim: int = 9,
        bias: bool = True,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.output_dim = int(output_dim)
        if self.output_dim not in {9, 12}:
            raise ValueError(
                f"JointDecoderOverParametrized output_dim must be 9 or 12, got {output_dim}"
            )
        self.revolute_embedding = nn.Parameter(torch.empty(model_dim))
        self.prismatic_embedding = nn.Parameter(torch.empty(model_dim))
        self.joint_decoder = _make_silu_mlp(
            input_dim=model_dim * 4,
            hidden_dim=model_dim * 4,
            output_dim=self.output_dim,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.revolute_embedding, std=0.02)
        nn.init.trunc_normal_(self.prismatic_embedding, std=0.02)

    def forward(
        self,
        query_latents: Tensor,
        link_latents: Tensor,
        assigned_link_ids: Tensor,
        joint_connections: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Decodes `(closest_axis_point, low_pose_point, high_pose_point)` per query."""
        batch_size, num_links = link_latents.shape[:2]
        parent_link_ids = torch.arange(
            num_links,
            device=assigned_link_ids.device,
            dtype=assigned_link_ids.dtype,
        ).view(1, -1).expand(batch_size, -1).clone() # [batch_size, num_links]
        valid_joint_mask = joint_connections[..., 1] >= 0
        child_link_ids = joint_connections[..., 1].clamp_min(0)
        batch_indices = torch.arange(
            batch_size,
            device=joint_connections.device,
        ).unsqueeze(1).expand_as(child_link_ids)
        parent_link_ids[
            batch_indices[valid_joint_mask],
            child_link_ids[valid_joint_mask],
        ] = joint_connections[..., 0][valid_joint_mask]

        assigned_link_latents = link_latents.gather(
            dim=1,
            index=assigned_link_ids.unsqueeze(-1).expand(-1, -1, link_latents.shape[-1]),
        )
        assigned_parent_link_ids = parent_link_ids.gather(dim=1, index=assigned_link_ids)
        assigned_parent_link_latents = link_latents.gather(
            dim=1,
            index=assigned_parent_link_ids.unsqueeze(-1).expand(-1, -1, link_latents.shape[-1]),
        )
        revolute_type_embeddings = self.revolute_embedding.to(
            device=assigned_link_latents.device,
            dtype=assigned_link_latents.dtype,
        ).view(1, 1, -1).expand_as(assigned_link_latents)
        prismatic_type_embeddings = self.prismatic_embedding.to(
            device=assigned_link_latents.device,
            dtype=assigned_link_latents.dtype,
        ).view(1, 1, -1).expand_as(assigned_link_latents)
        return (
            self.joint_decoder(
                torch.cat(
                    (
                        revolute_type_embeddings,
                        query_latents,
                        assigned_parent_link_latents,
                        assigned_link_latents,
                    ),
                    dim=-1,
                )
            ),
            self.joint_decoder(
                torch.cat(
                    (
                        prismatic_type_embeddings,
                        query_latents,
                        assigned_parent_link_latents,
                        assigned_link_latents,
                    ),
                    dim=-1,
                )
            ),
        )


class JointDecoderSingleDirection(nn.Module):
    """Decodes one axis direction per joint from parent/child link latents."""

    direction_dim = 3

    def __init__(
        self,
        model_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.revolute_embedding = nn.Parameter(torch.empty(model_dim))
        self.prismatic_embedding = nn.Parameter(torch.empty(model_dim))
        self.decoder = _make_silu_mlp(
            input_dim=model_dim * 3,
            hidden_dim=model_dim * 4,
            output_dim=self.direction_dim,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.revolute_embedding, std=0.02)
        nn.init.trunc_normal_(self.prismatic_embedding, std=0.02)

    def _predict_directions(
        self,
        *,
        parent_latents: Tensor,
        child_latents: Tensor,
        active_mask: Tensor,
        motion_type: str,
    ) -> Tensor:
        decoded = self.decoder(
            _build_joint_motion_condition_inputs(
                parent_latents=parent_latents,
                child_latents=child_latents,
                motion_type=motion_type,
                revolute_embedding=self.revolute_embedding,
                prismatic_embedding=self.prismatic_embedding,
            )
        )
        return decoded.masked_fill(~active_mask.unsqueeze(-1), 0)

    def predict(
        self,
        *,
        link_latents: Tensor,
        joint_connections: Tensor,
        joint_valid_flag: Tensor,
        is_revolute: Tensor,
        is_prismatic: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        parent_latents, child_latents = _gather_joint_link_latents(
            link_latents=link_latents,
            joint_connections=joint_connections,
        )
        return (
            self._predict_directions(
                parent_latents=parent_latents,
                child_latents=child_latents,
                active_mask=joint_valid_flag & is_revolute,
                motion_type="revolute",
            ),
            self._predict_directions(
                parent_latents=parent_latents,
                child_latents=child_latents,
                active_mask=joint_valid_flag & is_prismatic,
                motion_type="prismatic",
            ),
        )


class Particulate2Encoder(nn.Module):
    """Encoder backbone for articulated point-cloud understanding."""

    def __init__(
        self,
        model_dim: int = 768,
        num_heads: int = 12,
        head_dim: Optional[int] = None,
        num_shape_latents: int = 256,
        num_attention_blocks: int = 6,
        coordinate_num_frequencies: int = 32,
        normal_num_frequencies: int = 32,
        coordinate_max_frequency: float = 64.0,
        normal_max_frequency: float = 16.0,
        link_text_feature_dim: int = 768,
        embedding_hidden_dim: Optional[int] = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ffn_multiplier: float = 4.0,
        ffn_dropout: float = 0.0,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        compute_link_text_embeddings_on_the_fly: bool = False,
        use_text_conditioning: bool = True,
        clip_text_batch_size: int = 64,
        dropout_all_normals_prob: float = 0.3,
        dropout_all_point_prompts_prob: float = 0.0,
        dropout_individual_point_prompt_prob: float = 0.0,
        dropout_individual_text_conditioning_prob: float = 0.0,
        use_pretrained_features_shape: bool = False,
        use_pretrained_features_query: bool = False,
        use_pretrained_features_point_prompt: bool = False,
        pretrained_semantic_point_feature_dim: int = PARTFIELD_FEATURE_DIM,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.link_text_feature_dim = link_text_feature_dim
        self.clip_model_name = clip_model_name
        self.compute_link_text_embeddings_on_the_fly = bool(compute_link_text_embeddings_on_the_fly)
        self.use_text_conditioning = bool(use_text_conditioning)
        if int(clip_text_batch_size) <= 0:
            raise ValueError(f"clip_text_batch_size must be positive, got {clip_text_batch_size}")
        self.clip_text_batch_size = int(clip_text_batch_size)
        self.dropout_all_normals_prob = float(dropout_all_normals_prob)
        if not 0.0 <= self.dropout_all_normals_prob <= 1.0:
            raise ValueError(
                "dropout_all_normals_prob must be in [0, 1], "
                f"got {dropout_all_normals_prob}"
            )
        self.dropout_all_point_prompts_prob = dropout_all_point_prompts_prob
        self.dropout_individual_point_prompt_prob = dropout_individual_point_prompt_prob
        self.dropout_individual_text_conditioning_prob = float(
            dropout_individual_text_conditioning_prob
        )
        if not 0.0 <= self.dropout_individual_text_conditioning_prob <= 1.0:
            raise ValueError(
                "dropout_individual_text_conditioning_prob must be in [0, 1], "
                f"got {dropout_individual_text_conditioning_prob}"
            )
        if not self.use_text_conditioning and (
            self.dropout_all_point_prompts_prob != 0.0
            or self.dropout_individual_point_prompt_prob != 0.0
            or self.dropout_individual_text_conditioning_prob != 0.0
        ):
            raise ValueError(
                "use_text_conditioning=False requires "
                "dropout_all_point_prompts_prob=0 and "
                "dropout_individual_point_prompt_prob=0 and "
                "dropout_individual_text_conditioning_prob=0 to avoid ambiguous "
                "text-free vs. point-prompt-dropped training examples"
            )
        self.use_pretrained_features_shape = bool(use_pretrained_features_shape)
        self.use_pretrained_features_query = bool(use_pretrained_features_query)
        self.use_pretrained_features_point_prompt = bool(use_pretrained_features_point_prompt)
        self.pretrained_semantic_point_feature_dim = int(pretrained_semantic_point_feature_dim)
        self._partfield_feature_extractor: Optional[PartFieldFeatureExtractor] = None
        self.text_tokenizer: Any = None
        self.text_model: Optional[nn.Module] = None
        attn_block_kwargs = {
            "model_dim": model_dim,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "attn_dropout": attn_dropout,
            "proj_dropout": proj_dropout,
            "ffn_multiplier": ffn_multiplier,
            "ffn_dropout": ffn_dropout,
            "qk_norm": qk_norm,
            "norm_eps": norm_eps,
        }

        self.coordinate_embedder = FrequencyMLPEmbedder(
            output_dim=model_dim,
            num_frequencies=coordinate_num_frequencies,
            input_dim=3,
            hidden_dim=embedding_hidden_dim,
            max_frequency=coordinate_max_frequency,
        )
        self.normal_embedder = FrequencyMLPEmbedder(
            output_dim=model_dim,
            num_frequencies=normal_num_frequencies,
            input_dim=3,
            hidden_dim=embedding_hidden_dim,
            max_frequency=normal_max_frequency,
        )
        self.shape_encoder = ShapeLatentEncoder(
            num_shape_latents=num_shape_latents,
            **attn_block_kwargs,
        )
        text_hidden_dim = embedding_hidden_dim or model_dim
        self.link_text_input_norm = nn.RMSNorm(link_text_feature_dim, eps=norm_eps)
        self.link_text_projector = _make_silu_mlp(
            input_dim=link_text_feature_dim,
            hidden_dim=text_hidden_dim,
            output_dim=model_dim,
        )
        if (
            self.use_pretrained_features_shape
            or self.use_pretrained_features_query
            or self.use_pretrained_features_point_prompt
        ):
            self.pretrained_feature_input_norm = nn.RMSNorm(
                self.pretrained_semantic_point_feature_dim,
                eps=norm_eps,
            )
            self.pretrained_feature_projector = _make_silu_mlp(
                input_dim=self.pretrained_semantic_point_feature_dim,
                hidden_dim=text_hidden_dim,
                output_dim=model_dim,
            )
        else:
            self.pretrained_feature_input_norm = None
            self.pretrained_feature_projector = None
        self.link_to_shape_cross_attn = nn.ModuleList(
            [
                CrossAttentionBlock(**attn_block_kwargs)
                for _ in range(num_attention_blocks)
            ]
        )
        self.link_self_attn = nn.ModuleList(
            [
                SelfAttentionBlock(**attn_block_kwargs)
                for _ in range(num_attention_blocks)
            ]
        )
        self.query_to_shape_cross_attn = nn.ModuleList(
            [
                CrossAttentionBlock(**attn_block_kwargs)
                for _ in range(num_attention_blocks)
            ]
        )
        self.query_to_link_cross_attn = nn.ModuleList(
            [
                CrossAttentionBlock(**attn_block_kwargs)
                for _ in range(num_attention_blocks)
            ]
        )
        self.no_point_prompt_embedding = nn.Parameter(torch.zeros(model_dim))
        self.no_text_conditioning_embedding = nn.Parameter(torch.zeros(model_dim))
        self.link_output_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.query_output_norm = nn.RMSNorm(model_dim, eps=norm_eps)

    def encode_shape(
        self,
        shape_points: Tensor,
        shape_point_normals: Tensor,
        pretrained_features: Tensor | None = None,
        drop_normal_mask: Tensor | None = None,
    ) -> Tensor:
        point_tokens = self._embed_point_tokens(
            shape_points,
            shape_point_normals,
            pretrained_features=pretrained_features,
            drop_normal_mask=drop_normal_mask,
        )
        return self.shape_encoder(point_tokens)

    def encode_links(
        self,
        link_point_prompts: Tensor | None,
        link_point_prompt_normals: Tensor | None,
        link_valid_flag: Tensor,
        link_point_prompt_dropout_eligible: Tensor | None = None,
        forced_no_point_prompt_mask: Tensor | None = None,
        drop_normal_mask: Tensor | None = None,
        link_point_prompt_pretrained_features: Tensor | None = None,
        link_text_prompts: Optional[Sequence[Sequence[str]]] = None,
        link_text_embeddings: Tensor | None = None,
    ) -> Tensor:
        """Embeds valid link prompts and leaves padded link slots at zero.

        When prompt tensors are omitted, valid links use the learned
        `no_point_prompt_embedding` instead of point-derived features. When
        `forced_no_point_prompt_mask` is provided, those valid links also use
        the no-prompt embedding even in eval mode.
        """
        batch_size, max_links = link_valid_flag.shape
        no_point_prompt_mask = self._resolve_no_point_prompt_mask(
            link_point_prompts=link_point_prompts,
            link_valid_flag=link_valid_flag,
            link_point_prompt_dropout_eligible=link_point_prompt_dropout_eligible,
            forced_no_point_prompt_mask=forced_no_point_prompt_mask,
        )
        point_features = self._resolve_valid_link_point_features(
            link_point_prompts=link_point_prompts,
            link_point_prompt_normals=link_point_prompt_normals,
            link_valid_flag=link_valid_flag,
            no_point_prompt_mask=no_point_prompt_mask,
            drop_normal_mask=drop_normal_mask,
            link_point_prompt_pretrained_features=link_point_prompt_pretrained_features,
        )
        if self.use_text_conditioning:
            text_features = self._resolve_valid_link_text_features(
                link_text_prompts=link_text_prompts,
                link_text_embeddings=link_text_embeddings,
                link_valid_flag=link_valid_flag,
            )
        else:
            text_features = torch.zeros(
                (int(link_valid_flag.sum().item()), self.link_text_feature_dim),
                device=link_valid_flag.device,
                dtype=self.link_text_projector[0].weight.dtype,
            )
        projected_text_features = self.link_text_projector(
            self.link_text_input_norm(text_features)
        )
        if self.use_text_conditioning:
            text_conditioning_dropout_mask = self._sample_text_conditioning_dropout_mask(
                link_valid_flag,
                no_point_prompt_mask=no_point_prompt_mask,
            )
            dropped_valid_text_links = text_conditioning_dropout_mask[link_valid_flag]
            projected_text_features = torch.where(
                dropped_valid_text_links.unsqueeze(-1),
                self.no_text_conditioning_embedding.to(
                    device=projected_text_features.device,
                    dtype=projected_text_features.dtype,
                ).unsqueeze(0).expand_as(projected_text_features),
                projected_text_features,
            )
        if point_features.dtype != projected_text_features.dtype:
            point_features = point_features.to(dtype=projected_text_features.dtype)
        link_latents = projected_text_features.new_zeros((batch_size, max_links, self.model_dim))
        valid_link_features = point_features + projected_text_features
        link_latents[link_valid_flag] = valid_link_features
        return link_latents

    def _run_attention_blocks(
        self,
        shape_latents: Tensor,
        link_latents: Tensor,
        query_latents: Tensor,
        link_valid_flag: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        for (
            link_to_shape_cross_attn,
            link_self_attn,
            query_to_shape_cross_attn,
            query_to_link_cross_attn,
        ) in zip(
            self.link_to_shape_cross_attn,
            self.link_self_attn,
            self.query_to_shape_cross_attn,
            self.query_to_link_cross_attn,
            strict=True,
        ):
            link_latents = link_to_shape_cross_attn(
                link_latents,
                shape_latents,
                query_mask=link_valid_flag,
            )
            link_latents = link_self_attn(link_latents, mask=link_valid_flag)
            query_latents = query_to_shape_cross_attn(query_latents, shape_latents)
            query_latents = query_to_link_cross_attn(
                query_latents,
                link_latents,
                context_mask=link_valid_flag,
            )

        # Residual blocks can write into padded rows unless they are zeroed again.
        link_latents = self.link_output_norm(link_latents)
        link_latents = link_latents.masked_fill(~link_valid_flag.unsqueeze(-1), 0)
        query_latents = self.query_output_norm(query_latents)
        return link_latents, query_latents

    def forward(
        self,
        shape_points: Tensor | None = None,
        shape_point_normals: Tensor | None = None,
        query_points: Tensor | None = None,
        query_point_normals: Tensor | None = None,
        link_point_prompts: Tensor | None = None,
        link_point_prompt_normals: Tensor | None = None,
        link_valid_flag: Tensor | None = None,
        link_point_prompt_dropout_eligible: Tensor | None = None,
        link_text_prompts: Sequence[Sequence[str]] | None = None,
        link_text_embeddings: Tensor | None = None,
    ) -> Dict[str, Any]:
        """Encodes shape, link, and query inputs into latent tokens.

        Args:
            shape_points: Shape points with shape `(B, Ns, 3)`.
            shape_point_normals: Shape normals with shape `(B, Ns, 3)`.
            query_points: Query points with shape `(B, Nq, 3)`.
            query_point_normals: Query normals with shape `(B, Nq, 3)`.
            link_point_prompts: Optional link prompt points with shape
                `(B, L, 3)`. When omitted together with
                `link_point_prompt_normals`, valid link slots use the learned
                no-prompt embedding instead.
            link_point_prompt_normals: Optional link prompt normals aligned
                with `link_point_prompts`, with shape `(B, L, 3)`.
            link_valid_flag: Boolean tensor with shape `(B, L)` indicating
                which padded link slots are valid.
            link_point_prompt_dropout_eligible: Optional boolean tensor with
                shape `(B, L)` indicating which valid links may have their
                point prompt and prompt normal dropped during training.
            link_text_prompts: Per-link text prompts as a batch list of string
                lists, where sample `i` has length `sum(link_valid_flag[i])`.
            link_text_embeddings: Optional padded language features with shape
                `(B, L, D_lang)`.
        """
        if (link_point_prompts is None) != (link_point_prompt_normals is None):
            raise ValueError(
                "link_point_prompts and link_point_prompt_normals must both be provided or both be None"
            )

        if (
            shape_points is None
            or shape_point_normals is None
            or query_points is None
            or query_point_normals is None
            or link_valid_flag is None
        ):
            raise ValueError(
                "forward requires shape/query tensors and link_valid_flag"
            )
        if (
            self.use_text_conditioning
            and link_text_prompts is None
            and link_text_embeddings is None
        ):
            raise ValueError(
                "forward requires either link_text_prompts or link_text_embeddings "
                "when use_text_conditioning=True"
            )

        (
            shape_pretrained_features,
            query_pretrained_features,
            link_point_prompt_pretrained_features,
        ) = (
            self._compute_pretrained_point_features(
                shape_points=shape_points,
                query_points=query_points,
                link_point_prompts=link_point_prompts,
            )
        )
        drop_normal_mask = self._sample_all_normals_dropout_mask(
            batch_size=int(shape_points.shape[0]),
            device=shape_points.device,
        )
        shape_latents = self.encode_shape(
            shape_points,
            shape_point_normals,
            pretrained_features=shape_pretrained_features,
            drop_normal_mask=drop_normal_mask,
        )
        link_latents = self.encode_links(
            link_point_prompts=link_point_prompts,
            link_point_prompt_normals=link_point_prompt_normals,
            link_valid_flag=link_valid_flag,
            link_point_prompt_dropout_eligible=link_point_prompt_dropout_eligible,
            drop_normal_mask=drop_normal_mask,
            link_point_prompt_pretrained_features=link_point_prompt_pretrained_features,
            link_text_prompts=link_text_prompts,
            link_text_embeddings=link_text_embeddings,
        )
        query_latents = self._embed_point_tokens(
            query_points,
            query_point_normals,
            pretrained_features=query_pretrained_features,
            drop_normal_mask=drop_normal_mask,
        )
        link_latents, query_latents = self._run_attention_blocks(
            shape_latents=shape_latents,
            link_latents=link_latents,
            query_latents=query_latents,
            link_valid_flag=link_valid_flag,
        )

        return {
            "shape_latents": shape_latents,
            "query_latents": query_latents,
            "link_latents": link_latents,
        }

    def _embed_point_tokens(
        self,
        points: Tensor,
        normals: Tensor,
        *,
        pretrained_features: Tensor | None = None,
        drop_normal_mask: Tensor | None = None,
    ) -> Tensor:
        point_tokens = self.coordinate_embedder(points)
        normal_tokens = self.normal_embedder(normals)
        if drop_normal_mask is not None:
            if drop_normal_mask.ndim != 1 or drop_normal_mask.shape[0] != points.shape[0]:
                raise ValueError(
                    "drop_normal_mask must have shape (B,), "
                    f"got {tuple(drop_normal_mask.shape)} for points {tuple(points.shape)}"
                )
            drop_normal_mask = drop_normal_mask.to(
                device=normal_tokens.device,
                dtype=torch.bool,
            )
            mask_shape = (drop_normal_mask.shape[0],) + (1,) * (normal_tokens.ndim - 1)
            normal_tokens = normal_tokens.masked_fill(drop_normal_mask.view(mask_shape), 0)
        if point_tokens.dtype != normal_tokens.dtype:
            point_tokens = point_tokens.to(dtype=normal_tokens.dtype)
        point_tokens = point_tokens + normal_tokens
        if pretrained_features is None:
            return point_tokens
        if (
            self.pretrained_feature_input_norm is None
            or self.pretrained_feature_projector is None
        ):
            raise ValueError(
                "Received pretrained_features but pretrained feature projection is disabled"
            )
        pretrained_features = pretrained_features.to(
            dtype=self.pretrained_feature_projector[0].weight.dtype
        )
        projected_pretrained_features = self.pretrained_feature_projector(
            self.pretrained_feature_input_norm(pretrained_features)
        )
        if point_tokens.dtype != projected_pretrained_features.dtype:
            point_tokens = point_tokens.to(dtype=projected_pretrained_features.dtype)
        return point_tokens + projected_pretrained_features

    def _get_partfield_feature_extractor(self) -> PartFieldFeatureExtractor:
        if self._partfield_feature_extractor is None:
            self._partfield_feature_extractor = PartFieldFeatureExtractor()
        return self._partfield_feature_extractor

    def _compute_pretrained_point_features(
        self,
        *,
        shape_points: Tensor,
        query_points: Tensor,
        link_point_prompts: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        needs_prompt_features = (
            self.use_pretrained_features_point_prompt
            and link_point_prompts is not None
        )
        if not (
            self.use_pretrained_features_shape
            or self.use_pretrained_features_query
            or needs_prompt_features
        ):
            return None, None, None
        decode_segments: list[tuple[str, Tensor]] = []
        if self.use_pretrained_features_query:
            decode_segments.append(("query", query_points))
        if needs_prompt_features:
            if link_point_prompts.shape[0] != shape_points.shape[0]:
                raise ValueError(
                    "link_point_prompts batch dimension must match shape_points when "
                    "use_pretrained_features_point_prompt=True, "
                    f"got {tuple(link_point_prompts.shape)} and {tuple(shape_points.shape)}"
                )
            decode_segments.append(("prompt", link_point_prompts))

        decode_query_points = (
            torch.cat([points for _, points in decode_segments], dim=1)
            if decode_segments
            else None
        )
        extractor = self._get_partfield_feature_extractor()
        shape_features, combined_decode_features = extractor.extract(
            encode_points=shape_points,
            decode_shape_points=shape_points if self.use_pretrained_features_shape else None,
            decode_query_points=decode_query_points,
        )
        query_features = None
        prompt_features = None
        if combined_decode_features is not None:
            if len(decode_segments) == 1:
                decoded_features = {decode_segments[0][0]: combined_decode_features}
            else:
                decoded_features: dict[str, Tensor] = {}
                offset = 0
                for name, points in decode_segments:
                    count = points.shape[1]
                    decoded_features[name] = combined_decode_features[:, offset : offset + count]
                    offset += count
            query_features = decoded_features.get("query")
            prompt_features = decoded_features.get("prompt")
        return shape_features, query_features, prompt_features

    def _sample_point_prompt_dropout_mask(
        self,
        link_valid_flag: Tensor,
        link_point_prompt_dropout_eligible: Tensor | None = None,
    ) -> Tensor:
        if not self.training or (
            self.dropout_all_point_prompts_prob == 0.0
            and self.dropout_individual_point_prompt_prob == 0.0
        ):
            return torch.zeros_like(link_valid_flag)

        batch_size = link_valid_flag.shape[0]
        device = link_valid_flag.device
        if link_point_prompt_dropout_eligible is None:
            eligible_link_mask = link_valid_flag
        else:
            if link_point_prompt_dropout_eligible.shape != link_valid_flag.shape:
                raise ValueError(
                    "link_point_prompt_dropout_eligible must match link_valid_flag shape, "
                    f"got {tuple(link_point_prompt_dropout_eligible.shape)} and "
                    f"{tuple(link_valid_flag.shape)}"
                )
            eligible_link_mask = link_valid_flag & link_point_prompt_dropout_eligible
        drop_mask = torch.zeros_like(link_valid_flag)
        drop_all_samples = torch.zeros(batch_size, device=device, dtype=torch.bool)

        if self.dropout_all_point_prompts_prob > 0.0:
            drop_all_samples = (
                torch.rand(batch_size, device=device) < self.dropout_all_point_prompts_prob
            )
            drop_mask[drop_all_samples] = eligible_link_mask[drop_all_samples]

        if self.dropout_individual_point_prompt_prob > 0.0:
            keep_individual_dropout = ~drop_all_samples
            individual_drop_mask = (
                torch.rand(link_valid_flag.shape, device=device)
                < self.dropout_individual_point_prompt_prob
            )
            drop_mask[keep_individual_dropout] = (
                individual_drop_mask[keep_individual_dropout]
                & eligible_link_mask[keep_individual_dropout]
            )

        return drop_mask

    def _sample_all_normals_dropout_mask(
        self,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if not self.training or self.dropout_all_normals_prob == 0.0:
            return torch.zeros((batch_size,), device=device, dtype=torch.bool)
        return torch.rand(batch_size, device=device) < self.dropout_all_normals_prob

    def _resolve_no_point_prompt_mask(
        self,
        *,
        link_point_prompts: Tensor | None,
        link_valid_flag: Tensor,
        link_point_prompt_dropout_eligible: Tensor | None = None,
        forced_no_point_prompt_mask: Tensor | None = None,
    ) -> Tensor:
        if forced_no_point_prompt_mask is not None:
            if forced_no_point_prompt_mask.shape != link_valid_flag.shape:
                raise ValueError(
                    "forced_no_point_prompt_mask must match link_valid_flag shape, "
                    f"got {tuple(forced_no_point_prompt_mask.shape)} and "
                    f"{tuple(link_valid_flag.shape)}"
                )
            forced_no_point_prompt_mask = (
                forced_no_point_prompt_mask.to(
                    device=link_valid_flag.device,
                    dtype=torch.bool,
                )
                & link_valid_flag
            )
        if link_point_prompts is None:
            no_point_prompt_mask = link_valid_flag.clone()
        else:
            no_point_prompt_mask = self._sample_point_prompt_dropout_mask(
                link_valid_flag,
                link_point_prompt_dropout_eligible=link_point_prompt_dropout_eligible,
            )
        if forced_no_point_prompt_mask is not None:
            no_point_prompt_mask = no_point_prompt_mask | forced_no_point_prompt_mask
        return no_point_prompt_mask & link_valid_flag

    def _sample_text_conditioning_dropout_mask(
        self,
        link_valid_flag: Tensor,
        *,
        no_point_prompt_mask: Tensor | None = None,
    ) -> Tensor:
        if (
            not self.training
            or self.dropout_individual_text_conditioning_prob == 0.0
        ):
            return torch.zeros_like(link_valid_flag)
        if no_point_prompt_mask is None:
            text_dropout_eligible_mask = link_valid_flag
        else:
            if no_point_prompt_mask.shape != link_valid_flag.shape:
                raise ValueError(
                    "no_point_prompt_mask must match link_valid_flag shape, "
                    f"got {tuple(no_point_prompt_mask.shape)} and "
                    f"{tuple(link_valid_flag.shape)}"
                )
            text_dropout_eligible_mask = link_valid_flag & ~no_point_prompt_mask.to(
                device=link_valid_flag.device,
                dtype=torch.bool,
            )
        return (
            torch.rand(link_valid_flag.shape, device=link_valid_flag.device)
            < self.dropout_individual_text_conditioning_prob
        ) & text_dropout_eligible_mask

    def _resolve_valid_link_point_features(
        self,
        link_point_prompts: Tensor | None,
        link_point_prompt_normals: Tensor | None,
        link_valid_flag: Tensor,
        no_point_prompt_mask: Tensor | None = None,
        drop_normal_mask: Tensor | None = None,
        link_point_prompt_pretrained_features: Tensor | None = None,
    ) -> Tensor:
        if no_point_prompt_mask is not None:
            if no_point_prompt_mask.shape != link_valid_flag.shape:
                raise ValueError(
                    "no_point_prompt_mask must match link_valid_flag shape, "
                    f"got {tuple(no_point_prompt_mask.shape)} and "
                    f"{tuple(link_valid_flag.shape)}"
                )
            no_point_prompt_mask = (
                no_point_prompt_mask.to(
                    device=link_valid_flag.device,
                    dtype=torch.bool,
                )
                & link_valid_flag
            )

        if link_point_prompts is None:
            if link_point_prompt_pretrained_features is not None:
                raise ValueError(
                    "link_point_prompt_pretrained_features requires link_point_prompts"
                )
            no_prompt_embedding = self.no_point_prompt_embedding.to(
                device=link_valid_flag.device,
                dtype=self.no_point_prompt_embedding.dtype,
            )
            return no_prompt_embedding.view(1, 1, -1).expand(*link_valid_flag.shape, -1)[link_valid_flag]

        point_features = self._embed_point_tokens(
            link_point_prompts,
            link_point_prompt_normals,
            pretrained_features=link_point_prompt_pretrained_features,
            drop_normal_mask=drop_normal_mask,
        )[link_valid_flag]
        if no_point_prompt_mask is None:
            no_point_prompt_mask = torch.zeros_like(link_valid_flag)
        dropped_valid_links = no_point_prompt_mask[link_valid_flag]
        return torch.where(
            dropped_valid_links.unsqueeze(-1),
            self.no_point_prompt_embedding.to(
                device=point_features.device,
                dtype=point_features.dtype,
            ).unsqueeze(0).expand_as(point_features),
            point_features,
        )

    def _resolve_valid_link_text_features(
        self,
        link_text_prompts: Optional[Sequence[Sequence[str]]],
        link_text_embeddings: Tensor | None,
        link_valid_flag: Tensor,
    ) -> Tensor:
        """Returns text features flattened in the same order as `link_valid_flag`."""
        if link_text_embeddings is not None:
            return link_text_embeddings[link_valid_flag].to(
                dtype=self.link_text_projector[0].weight.dtype
            )

        if link_text_prompts is None:
            flattened_prompts = [""] * int(link_valid_flag.sum().item())
        else:
            flattened_prompts = list(chain.from_iterable(link_text_prompts))
        return self._encode_link_text_prompts(
            flattened_prompts,
            dtype=self.link_text_projector[0].weight.dtype,
        )

    def _ensure_text_model_loaded(self) -> None:
        if self.text_model is not None and self.text_tokenizer is not None:
            return

        cache_dir = os.environ.get("HF_HOME")
        self.text_tokenizer, self.text_model = load_clip_text_encoder(
            self.clip_model_name,
            device=self.link_text_projector[0].weight.device,
            expected_embedding_dim=self.link_text_feature_dim,
            cache_dir=cache_dir,
        )

    @torch.no_grad()
    def _encode_link_text_prompts(
        self,
        prompts: Sequence[str],
        dtype: torch.dtype,
    ) -> Tensor:
        if len(prompts) == 0:
            return torch.zeros(
                (0, self.link_text_feature_dim),
                device=self.link_text_projector[0].weight.device,
                dtype=dtype,
            )
        if not self.compute_link_text_embeddings_on_the_fly:
            raise ValueError(
                "Missing link_text_embeddings but compute_link_text_embeddings_on_the_fly=False. "
                "Provide dataset-side text embeddings or enable on-the-fly CLIP text encoding."
            )
        self._ensure_text_model_loaded()
        return encode_clip_text_prompts(
            prompts,
            tokenizer=self.text_tokenizer,
            text_model=self.text_model,
            batch_size=self.clip_text_batch_size,
            output_device=self.link_text_projector[0].weight.device,
            output_dtype=dtype,
        )


class Particulate2ArticulationModel(nn.Module):
    """Encoder, decoders, and training losses for articulation prediction."""

    def __init__(
        self,
        encoder: Optional[Particulate2Encoder] = None,
        *,
        segmentation_decode_type: str = "mlp",
        segmentation_query_chunk_size: Optional[int] = None,
        joint_decode_type: str = "overparameterization",
        joint_fm_hidden_dim: Optional[int] = None,
        joint_fm_prediction_type: str = "v",
        joint_fm_time_embedding_dim: int = 256,
        joint_fm_inference_steps: int = 100,
        joint_fm_time_scale: float = 1000.0,
        joint_fm_sigma_min: float = 0.0,
        joint_fm_rescale_t: float = 1.0,
        joint_fm_cfg_scale: float = 1.0,
        revolute_joint_fm_state_mean: Optional[Sequence[float]] = None,
        revolute_joint_fm_state_std: Optional[Sequence[float]] = None,
        prismatic_joint_fm_state_mean: Optional[Sequence[float]] = None,
        prismatic_joint_fm_state_std: Optional[Sequence[float]] = None,
        joint_fm_training_time_mean: float = 0.0,
        joint_fm_training_time_std: float = 1.0,
        use_ancestor_context_for_segmentation: bool = False,
        ancestor_context_decay: float = 0.5,
        segmentation_bias: bool = True,
        joint_decoder_bias: bool = True,
        segmentation_cross_entropy_weight: float = 1.0,
        segmentation_dice_weight: float = 1.0,
        revolute_joint_axis_l1_weight: float = 1.0,
        prismatic_joint_axis_l1_weight: float = 1.0,
        revolute_joint_range_l1_weight: float = 1.0,
        prismatic_joint_range_l1_weight: float = 1.0,
        revolute_joint_fm_weight: float = 1.0,
        prismatic_joint_fm_weight: float = 1.0,
        revolute_overparam_axis_l1_weight: float = 1.0,
        revolute_overparam_point_l1_weight: float = 1.0,
        prismatic_overparam_axis_l1_weight: float = 1.0,
        prismatic_overparam_point_l1_weight: float = 1.0,
        revolute_overparam_direction_weight: float = 1.0,
        prismatic_overparam_direction_weight: float = 1.0,
        overparam_closest_axis_space: str = "world",
        dice_smoothing: float = 1e-6,
        **encoder_kwargs: Any,
    ):
        super().__init__()
        if encoder is not None and encoder_kwargs:
            raise ValueError("Pass either an encoder instance or encoder kwargs, not both")

        self.encoder = encoder if encoder is not None else Particulate2Encoder(**encoder_kwargs)
        joint_decode_type = _normalize_joint_decode_type(joint_decode_type)
        if joint_decode_type not in {*_PLAIN_JOINT_DECODE_TYPES, *_OVERPARAM_JOINT_DECODE_TYPES}:
            raise ValueError(
                "joint_decode_type must be 'plain', 'plain+fm', 'overparametrized', "
                "'overparameterized', 'overparameterization', 'overparam+dir', or "
                "'overparam+singledir', "
                f"got {joint_decode_type!r}"
            )
        self.joint_decode_type = joint_decode_type
        self.plain_flow_matching_enabled = self.joint_decode_type == "plain+fm"
        self.overparam_predicts_query_axis_direction = self.joint_decode_type == "overparam+dir"
        self.overparam_predicts_single_axis_direction = (
            self.joint_decode_type == "overparam+singledir"
        )
        self.overparam_uses_axis_direction = (
            self.overparam_predicts_query_axis_direction
            or self.overparam_predicts_single_axis_direction
        )
        overparam_closest_axis_space = _normalize_overparam_closest_axis_space(
            overparam_closest_axis_space
        )
        if overparam_closest_axis_space not in {"world", "part_aabb"}:
            raise ValueError(
                "overparam_closest_axis_space must be 'world', 'sample', 'part_aabb', or "
                f"'local_aabb', got {overparam_closest_axis_space!r}"
            )
        self.overparam_closest_axis_space = overparam_closest_axis_space
        self.overparam_closest_axis_uses_part_aabb = (
            self.overparam_closest_axis_space == "part_aabb"
        )
        self.use_ancestor_context_for_segmentation = bool(
            use_ancestor_context_for_segmentation
        )
        self.ancestor_context_decay = float(ancestor_context_decay)
        if not math.isfinite(self.ancestor_context_decay):
            raise ValueError(
                "ancestor_context_decay must be finite, "
                f"got {ancestor_context_decay!r}"
            )
        if self.ancestor_context_decay < 0.0 or self.ancestor_context_decay > 1.0:
            raise ValueError(
                "ancestor_context_decay must be in [0, 1], "
                f"got {ancestor_context_decay!r}"
            )
        self.segmentation_decoder = SegmentationDecoder(
            model_dim=self.encoder.model_dim,
            decode_type=segmentation_decode_type,
            bias=segmentation_bias,
            query_chunk_size=segmentation_query_chunk_size,
        )
        self.joint_direction_decoder: JointDecoderSingleDirection | None = None
        if self.joint_decode_type == "plain":
            self.joint_decoder = JointDecoderPlain(
                model_dim=self.encoder.model_dim,
                bias=joint_decoder_bias,
            )
        elif self.plain_flow_matching_enabled:
            self.joint_decoder = JointDecoderPlainFlowMatching(
                model_dim=self.encoder.model_dim,
                hidden_dim=joint_fm_hidden_dim,
                prediction_type=joint_fm_prediction_type,
                time_embedding_dim=joint_fm_time_embedding_dim,
                inference_steps=joint_fm_inference_steps,
                time_scale=joint_fm_time_scale,
                sigma_min=joint_fm_sigma_min,
                rescale_t=joint_fm_rescale_t,
                cfg_scale=joint_fm_cfg_scale,
                revolute_state_mean=revolute_joint_fm_state_mean,
                revolute_state_std=revolute_joint_fm_state_std,
                prismatic_state_mean=prismatic_joint_fm_state_mean,
                prismatic_state_std=prismatic_joint_fm_state_std,
                bias=joint_decoder_bias,
            )
        else:
            self.joint_decoder = JointDecoderOverParametrized(
                model_dim=self.encoder.model_dim,
                output_dim=12 if self.overparam_predicts_query_axis_direction else 9,
                bias=joint_decoder_bias,
            )
            if self.overparam_predicts_single_axis_direction:
                self.joint_direction_decoder = JointDecoderSingleDirection(
                    model_dim=self.encoder.model_dim,
                    bias=joint_decoder_bias,
                )
        if self.use_ancestor_context_for_segmentation:
            self.ancestor_context_projector = nn.Linear(
                self.encoder.model_dim,
                self.encoder.model_dim,
                bias=False,
            )
            self.ancestor_context_gate = _make_silu_mlp(
                input_dim=self.encoder.model_dim * 2,
                hidden_dim=self.encoder.model_dim,
                output_dim=1,
            )
        else:
            self.ancestor_context_projector = None
            self.ancestor_context_gate = None
        self.segmentation_cross_entropy_weight = float(segmentation_cross_entropy_weight)
        self.segmentation_dice_weight = float(segmentation_dice_weight)
        self.revolute_joint_axis_l1_weight = float(revolute_joint_axis_l1_weight)
        self.prismatic_joint_axis_l1_weight = float(prismatic_joint_axis_l1_weight)
        self.revolute_joint_range_l1_weight = float(revolute_joint_range_l1_weight)
        self.prismatic_joint_range_l1_weight = float(prismatic_joint_range_l1_weight)
        self.revolute_joint_fm_weight = float(revolute_joint_fm_weight)
        self.prismatic_joint_fm_weight = float(prismatic_joint_fm_weight)
        self.revolute_overparam_axis_l1_weight = float(revolute_overparam_axis_l1_weight)
        self.revolute_overparam_point_l1_weight = float(revolute_overparam_point_l1_weight)
        self.prismatic_overparam_axis_l1_weight = float(prismatic_overparam_axis_l1_weight)
        self.prismatic_overparam_point_l1_weight = float(prismatic_overparam_point_l1_weight)
        self.revolute_overparam_direction_weight = float(revolute_overparam_direction_weight)
        self.prismatic_overparam_direction_weight = float(prismatic_overparam_direction_weight)
        self.joint_fm_training_time_mean = float(joint_fm_training_time_mean)
        self.joint_fm_training_time_std = float(joint_fm_training_time_std)
        self.dice_smoothing = float(dice_smoothing)

    def decode_segmentation(
        self,
        query_latents: Tensor,
        link_latents: Tensor,
        link_valid_flag: Tensor,
    ) -> Tensor:
        return self.segmentation_decoder(
            query_latents=query_latents,
            link_latents=link_latents,
            link_valid_flag=link_valid_flag,
        )

    def decode_joint_parameters(
        self,
        *,
        link_latents: Tensor,
        joint_connections: Tensor,
        joint_valid_flag: Tensor,
        is_revolute: Tensor,
        is_prismatic: Tensor,
        query_latents: Tensor | None = None,
        query_points: Tensor | None = None,
        assigned_link_ids: Tensor | None = None,
        decoded_motion_points: Tuple[Tensor, Tensor] | None = None,
        decoded_axis_directions: Tuple[Tensor, Tensor] | None = None,
        decoded_motion_points_are_world: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns revolute/prismatic axis and range tensors for the configured decode mode.

        In plain mode the joint decoder predicts per-joint parameters directly from
        parent/child link latents. In plain+fm mode it samples parameters with
        a SAM3D-style flow-matching Euler sampler. In over-parameterized mode it first obtains
        query-wise motion targets and then fits a single joint axis and range for
        each child link from those targets.
        """
        if self.joint_decode_type == "plain":
            return self.joint_decoder.predict(
                link_latents=link_latents,
                joint_connections=joint_connections,
                joint_valid_flag=joint_valid_flag,
                is_revolute=is_revolute,
                is_prismatic=is_prismatic,
            )
        if self.plain_flow_matching_enabled:
            return self.joint_decoder.predict(
                link_latents=link_latents,
                joint_connections=joint_connections,
                joint_valid_flag=joint_valid_flag,
                is_revolute=is_revolute,
                is_prismatic=is_prismatic,
            )

        if query_points is None or assigned_link_ids is None:
            raise ValueError(
                "over-parameterized joint decoding requires query_points and assigned_link_ids"
            )
        if decoded_motion_points is None:
            if query_latents is None:
                raise ValueError(
                    "over-parameterized joint decoding requires query_latents when decoded_motion_points are not provided"
                )
            decoded_motion_points = self._decode_joint_motion_points(
                query_latents=query_latents,
                link_latents=link_latents,
                assigned_link_ids=assigned_link_ids,
                joint_connections=joint_connections,
            )
        if self.overparam_predicts_single_axis_direction and decoded_axis_directions is None:
            decoded_axis_directions = self._decode_joint_axis_directions(
                link_latents=link_latents,
                joint_connections=joint_connections,
                joint_valid_flag=joint_valid_flag,
                is_revolute=is_revolute,
                is_prismatic=is_prismatic,
            )
        if not decoded_motion_points_are_world:
            decoded_motion_points = (
                self._convert_overparam_motion_points_to_world_coordinates(
                    motion_points=decoded_motion_points[0],
                    query_points=query_points,
                    assigned_link_ids=assigned_link_ids,
                ),
                self._convert_overparam_motion_points_to_world_coordinates(
                    motion_points=decoded_motion_points[1],
                    query_points=query_points,
                    assigned_link_ids=assigned_link_ids,
                ),
            )
        try:
            autocast_context = torch.autocast(device_type=query_points.device.type, enabled=False)
        except (RuntimeError, TypeError, ValueError):
            autocast_context = nullcontext()
        with autocast_context:
            return self._recover_overparam_joint_parameters(
                query_points=query_points,
                assigned_link_ids=assigned_link_ids,
                joint_connections=joint_connections,
                joint_valid_flag=joint_valid_flag,
                is_revolute=is_revolute,
                is_prismatic=is_prismatic,
                revolute_motion_points=decoded_motion_points[0],
                prismatic_motion_points=decoded_motion_points[1],
                revolute_axis_directions=(
                    None if decoded_axis_directions is None else decoded_axis_directions[0]
                ),
                prismatic_axis_directions=(
                    None if decoded_axis_directions is None else decoded_axis_directions[1]
                ),
            )

    def _decode_joint_motion_points(
        self,
        *,
        query_latents: Tensor,
        link_latents: Tensor,
        assigned_link_ids: Tensor,
        joint_connections: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if self.joint_decode_type not in _OVERPARAM_JOINT_DECODE_TYPES:
            raise ValueError(
                "_decode_joint_motion_points is only available in over-parameterized joint decoding mode"
            )
        return self.joint_decoder(
            query_latents=query_latents,
            link_latents=link_latents,
            assigned_link_ids=assigned_link_ids,
            joint_connections=joint_connections,
        )

    def _decode_joint_axis_directions(
        self,
        *,
        link_latents: Tensor,
        joint_connections: Tensor,
        joint_valid_flag: Tensor,
        is_revolute: Tensor,
        is_prismatic: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if not self.overparam_predicts_single_axis_direction or self.joint_direction_decoder is None:
            raise ValueError(
                "_decode_joint_axis_directions is only available for joint_decode_type='overparam+singledir'"
            )
        return self.joint_direction_decoder.predict(
            link_latents=link_latents,
            joint_connections=joint_connections,
            joint_valid_flag=joint_valid_flag,
            is_revolute=is_revolute,
            is_prismatic=is_prismatic,
        )

    def _compute_query_link_aabb_parameters(
        self,
        *,
        query_points: Tensor,
        link_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Returns per-query AABB centers and half-extents for the assigned link ID."""
        if query_points.ndim != 3 or query_points.shape[-1] != 3:
            raise ValueError(
                f"query_points must have shape (B, Q, 3), got {tuple(query_points.shape)}"
            )
        if link_ids.shape != query_points.shape[:2]:
            raise ValueError(
                "link_ids must match query_points batch/query dims, "
                f"got {tuple(link_ids.shape)} and {tuple(query_points.shape)}"
            )
        query_points = query_points.float()
        centers = query_points.new_zeros(query_points.shape)
        half_extents = query_points.new_ones(query_points.shape)
        for batch_idx in range(query_points.shape[0]):
            batch_link_ids = link_ids[batch_idx]
            valid_mask = batch_link_ids >= 0
            if not bool(valid_mask.any().item()):
                continue
            unique_link_ids = torch.unique(batch_link_ids[valid_mask])
            for link_id in unique_link_ids.tolist():
                query_mask = batch_link_ids == int(link_id)
                link_query_points = query_points[batch_idx][query_mask]
                min_corner = link_query_points.min(dim=0).values
                max_corner = link_query_points.max(dim=0).values
                centers[batch_idx][query_mask] = 0.5 * (min_corner + max_corner)
                half_extents[batch_idx][query_mask] = (
                    0.5 * (max_corner - min_corner)
                ).clamp_min(_OVERPARAM_AXIS_AABB_HALF_EXTENT_MIN)
        return centers, half_extents

    def _normalize_overparam_axis_points(
        self,
        *,
        axis_points: Tensor,
        query_points: Tensor,
        link_ids: Tensor,
    ) -> Tensor:
        """Normalizes closest-axis points into the per-link query AABB coordinate frame."""
        if not self.overparam_closest_axis_uses_part_aabb:
            return axis_points
        centers, half_extents = self._compute_query_link_aabb_parameters(
            query_points=query_points,
            link_ids=link_ids,
        )
        return (axis_points.float() - centers) / half_extents

    def _denormalize_overparam_axis_points(
        self,
        *,
        axis_points: Tensor,
        query_points: Tensor,
        link_ids: Tensor,
    ) -> Tensor:
        """Converts normalized closest-axis points back into sample/world coordinates."""
        if not self.overparam_closest_axis_uses_part_aabb:
            return axis_points
        centers, half_extents = self._compute_query_link_aabb_parameters(
            query_points=query_points,
            link_ids=link_ids,
        )
        return axis_points.float() * half_extents + centers

    def _convert_overparam_motion_points_to_world_coordinates(
        self,
        *,
        motion_points: Tensor,
        query_points: Tensor,
        assigned_link_ids: Tensor,
    ) -> Tensor:
        """Returns motion-point predictions in sample/world coordinates."""
        if not self.overparam_closest_axis_uses_part_aabb:
            return motion_points
        closest_axis_points = self._denormalize_overparam_axis_points(
            axis_points=motion_points[..., :3],
            query_points=query_points,
            link_ids=assigned_link_ids,
        )
        return torch.cat((closest_axis_points, motion_points[..., 3:]), dim=-1)

    def _fit_revolute_joint_parameters(
        self,
        query_points: Tensor,
        revolute_motion_points: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Fits one revolute axis and low/high angles from query-wise motion targets."""
        # This solver is small and numerically sensitive; keep it in fp32 even
        # when the surrounding network runs under AMP.
        query_points = query_points.float()
        revolute_motion_points = revolute_motion_points.float()
        if query_points.numel() == 0:
            zero_axis = revolute_motion_points.new_zeros(6)
            zero_range = revolute_motion_points.new_zeros(2)
            return zero_axis, zero_range

        closest_axis_points = revolute_motion_points[..., :3]
        low_points = revolute_motion_points[..., 3:6]
        high_points = revolute_motion_points[..., 6:9]
        direction_hint = (
            torch.linalg.cross(
                query_points - closest_axis_points,
                low_points - closest_axis_points,
                dim=-1,
            )
            + torch.linalg.cross(
                query_points - closest_axis_points,
                high_points - closest_axis_points,
                dim=-1,
            )
            + torch.linalg.cross(
                low_points - closest_axis_points,
                high_points - closest_axis_points,
                dim=-1,
            )
        ).mean(dim=0)
        axis_direction, axis_point = fit_axis_to_closest_points_torch(
            query_points,
            closest_axis_points,
            direction_hint=direction_hint,
        )
        revolute_axis = axis_point_to_plucker_torch(axis_direction, axis_point)
        low_limit = estimate_revolute_limit_torch(
            query_points,
            low_points,
            axis_direction,
            axis_point,
        )
        high_limit = estimate_revolute_limit_torch(
            query_points,
            high_points,
            axis_direction,
            axis_point,
        )
        return revolute_axis, torch.stack((low_limit, high_limit))

    def _aggregate_flip_invariant_axis_direction(
        self,
        predicted_directions: Tensor,
        *,
        sign_hint: Tensor | None = None,
    ) -> Tensor:
        """Returns one averaged unit direction while treating per-query flips as equivalent."""
        predicted_directions = predicted_directions.float()
        if predicted_directions.numel() == 0:
            return predicted_directions.new_zeros(3)

        direction_norms = torch.linalg.vector_norm(predicted_directions, dim=-1)
        valid_mask = direction_norms > 1e-8
        if not bool(valid_mask.any().item()):
            return predicted_directions.new_zeros(3)

        unit_directions = predicted_directions[valid_mask] / direction_norms[valid_mask].unsqueeze(-1)
        direction_covariance = unit_directions.transpose(0, 1) @ unit_directions
        _, eigenvectors = torch.linalg.eigh(direction_covariance)
        anchor_direction = eigenvectors[:, -1]
        alignment = torch.sign(unit_directions @ anchor_direction)
        alignment = torch.where(
            alignment == 0,
            torch.ones_like(alignment),
            alignment,
        )
        aligned_mean_direction = (unit_directions * alignment.unsqueeze(-1)).mean(dim=0)
        if float(torch.linalg.vector_norm(aligned_mean_direction).item()) <= 1e-8:
            axis_direction = F.normalize(anchor_direction, dim=0, eps=1e-8)
        else:
            axis_direction = F.normalize(aligned_mean_direction, dim=0, eps=1e-8)

        if sign_hint is not None:
            sign_hint = sign_hint.float()
            if float(torch.linalg.vector_norm(sign_hint).item()) > 1e-8:
                if float(torch.dot(axis_direction, sign_hint).item()) < 0.0:
                    axis_direction = -axis_direction
        return axis_direction

    def _fit_revolute_joint_parameters_with_direction(
        self,
        query_points: Tensor,
        revolute_motion_points: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Fits one revolute axis from predicted axis directions plus point targets."""
        query_points = query_points.float()
        revolute_motion_points = revolute_motion_points.float()
        if query_points.numel() == 0:
            zero_axis = revolute_motion_points.new_zeros(6)
            zero_range = revolute_motion_points.new_zeros(2)
            return zero_axis, zero_range

        closest_axis_points = revolute_motion_points[..., :3]
        low_points = revolute_motion_points[..., 3:6]
        high_points = revolute_motion_points[..., 6:9]
        predicted_axis_directions = revolute_motion_points[..., 9:12]
        direction_sign_hint = (
            torch.linalg.cross(
                query_points - closest_axis_points,
                low_points - closest_axis_points,
                dim=-1,
            )
            + torch.linalg.cross(
                query_points - closest_axis_points,
                high_points - closest_axis_points,
                dim=-1,
            )
            + torch.linalg.cross(
                low_points - closest_axis_points,
                high_points - closest_axis_points,
                dim=-1,
            )
        ).mean(dim=0)
        axis_direction = self._aggregate_flip_invariant_axis_direction(
            predicted_axis_directions,
            sign_hint=direction_sign_hint,
        )
        if float(torch.linalg.vector_norm(axis_direction).item()) <= 1e-8:
            return self._fit_revolute_joint_parameters(
                query_points=query_points,
                revolute_motion_points=revolute_motion_points[..., :9],
            )

        axis_point = torch.quantile(closest_axis_points, 0.5, dim=0)
        low_limit = estimate_revolute_limit_torch(
            query_points,
            low_points,
            axis_direction,
            axis_point,
        )
        high_limit = estimate_revolute_limit_torch(
            query_points,
            high_points,
            axis_direction,
            axis_point,
        )
        if float(low_limit.item()) > float(high_limit.item()):
            axis_direction = -axis_direction
            low_limit = estimate_revolute_limit_torch(
                query_points,
                low_points,
                axis_direction,
                axis_point,
            )
            high_limit = estimate_revolute_limit_torch(
                query_points,
                high_points,
                axis_direction,
                axis_point,
            )
        revolute_axis = axis_point_to_plucker_torch(axis_direction, axis_point)
        return revolute_axis, torch.stack((low_limit, high_limit))

    def _fit_prismatic_joint_parameters(
        self,
        query_points: Tensor,
        prismatic_motion_points: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Fits one prismatic axis and low/high displacements from query-wise targets."""
        # This solver is small and numerically sensitive; keep it in fp32 even
        # when the surrounding network runs under AMP.
        query_points = query_points.float()
        prismatic_motion_points = prismatic_motion_points.float()
        if query_points.numel() == 0:
            zero_axis = prismatic_motion_points.new_zeros(6)
            zero_range = prismatic_motion_points.new_zeros(2)
            return zero_axis, zero_range

        closest_axis_points = prismatic_motion_points[..., :3]
        low_points = prismatic_motion_points[..., 3:6]
        high_points = prismatic_motion_points[..., 6:9]
        direction_hint = (high_points - low_points).mean(dim=0)
        axis_direction, axis_point = fit_axis_to_closest_points_torch(
            query_points,
            closest_axis_points,
            direction_hint=direction_hint,
        )
        prismatic_axis = axis_point_to_plucker_torch(axis_direction, axis_point)
        low_limit = estimate_prismatic_limit_torch(
            query_points,
            low_points,
            axis_direction,
        )
        high_limit = estimate_prismatic_limit_torch(
            query_points,
            high_points,
            axis_direction,
        )
        return prismatic_axis, torch.stack((low_limit, high_limit))

    def _fit_prismatic_joint_parameters_with_direction(
        self,
        query_points: Tensor,
        prismatic_motion_points: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Fits one prismatic axis from predicted axis directions plus point targets."""
        query_points = query_points.float()
        prismatic_motion_points = prismatic_motion_points.float()
        if query_points.numel() == 0:
            zero_axis = prismatic_motion_points.new_zeros(6)
            zero_range = prismatic_motion_points.new_zeros(2)
            return zero_axis, zero_range

        closest_axis_points = prismatic_motion_points[..., :3]
        low_points = prismatic_motion_points[..., 3:6]
        high_points = prismatic_motion_points[..., 6:9]
        predicted_axis_directions = prismatic_motion_points[..., 9:12]
        axis_direction = self._aggregate_flip_invariant_axis_direction(
            predicted_axis_directions,
            sign_hint=(high_points - low_points).mean(dim=0),
        )
        if float(torch.linalg.vector_norm(axis_direction).item()) <= 1e-8:
            return self._fit_prismatic_joint_parameters(
                query_points=query_points,
                prismatic_motion_points=prismatic_motion_points[..., :9],
            )

        axis_point = torch.quantile(closest_axis_points, 0.5, dim=0)
        low_limit = estimate_prismatic_limit_torch(
            query_points,
            low_points,
            axis_direction,
        )
        high_limit = estimate_prismatic_limit_torch(
            query_points,
            high_points,
            axis_direction,
        )
        if float(low_limit.item()) > float(high_limit.item()):
            axis_direction = -axis_direction
            low_limit = estimate_prismatic_limit_torch(
                query_points,
                low_points,
                axis_direction,
            )
            high_limit = estimate_prismatic_limit_torch(
                query_points,
                high_points,
                axis_direction,
            )
        prismatic_axis = axis_point_to_plucker_torch(axis_direction, axis_point)
        return prismatic_axis, torch.stack((low_limit, high_limit))

    def _fit_revolute_joint_parameters_with_single_direction(
        self,
        query_points: Tensor,
        revolute_motion_points: Tensor,
        axis_direction: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Fits one revolute axis from a single predicted axis direction plus point targets."""
        query_points = query_points.float()
        revolute_motion_points = revolute_motion_points.float()
        axis_direction = axis_direction.float()
        if query_points.numel() == 0:
            zero_axis = revolute_motion_points.new_zeros(6)
            zero_range = revolute_motion_points.new_zeros(2)
            return zero_axis, zero_range

        if float(torch.linalg.vector_norm(axis_direction).item()) <= 1e-8:
            return self._fit_revolute_joint_parameters(
                query_points=query_points,
                revolute_motion_points=revolute_motion_points,
            )
        axis_direction = F.normalize(axis_direction, dim=0, eps=1e-8)

        closest_axis_points = revolute_motion_points[..., :3]
        low_points = revolute_motion_points[..., 3:6]
        high_points = revolute_motion_points[..., 6:9]
        axis_point = torch.quantile(closest_axis_points, 0.5, dim=0)
        low_limit = estimate_revolute_limit_torch(
            query_points,
            low_points,
            axis_direction,
            axis_point,
        )
        high_limit = estimate_revolute_limit_torch(
            query_points,
            high_points,
            axis_direction,
            axis_point,
        )
        if float(low_limit.item()) > float(high_limit.item()):
            axis_direction = -axis_direction
            low_limit = estimate_revolute_limit_torch(
                query_points,
                low_points,
                axis_direction,
                axis_point,
            )
            high_limit = estimate_revolute_limit_torch(
                query_points,
                high_points,
                axis_direction,
                axis_point,
            )
        revolute_axis = axis_point_to_plucker_torch(axis_direction, axis_point)
        return revolute_axis, torch.stack((low_limit, high_limit))

    def _fit_prismatic_joint_parameters_with_single_direction(
        self,
        query_points: Tensor,
        prismatic_motion_points: Tensor,
        axis_direction: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Fits one prismatic axis from a single predicted axis direction plus point targets."""
        query_points = query_points.float()
        prismatic_motion_points = prismatic_motion_points.float()
        axis_direction = axis_direction.float()
        if query_points.numel() == 0:
            zero_axis = prismatic_motion_points.new_zeros(6)
            zero_range = prismatic_motion_points.new_zeros(2)
            return zero_axis, zero_range

        if float(torch.linalg.vector_norm(axis_direction).item()) <= 1e-8:
            return self._fit_prismatic_joint_parameters(
                query_points=query_points,
                prismatic_motion_points=prismatic_motion_points,
            )
        axis_direction = F.normalize(axis_direction, dim=0, eps=1e-8)

        closest_axis_points = prismatic_motion_points[..., :3]
        low_points = prismatic_motion_points[..., 3:6]
        high_points = prismatic_motion_points[..., 6:9]
        axis_point = torch.quantile(closest_axis_points, 0.5, dim=0)
        low_limit = estimate_prismatic_limit_torch(
            query_points,
            low_points,
            axis_direction,
        )
        high_limit = estimate_prismatic_limit_torch(
            query_points,
            high_points,
            axis_direction,
        )
        if float(low_limit.item()) > float(high_limit.item()):
            axis_direction = -axis_direction
            low_limit = estimate_prismatic_limit_torch(
                query_points,
                low_points,
                axis_direction,
            )
            high_limit = estimate_prismatic_limit_torch(
                query_points,
                high_points,
                axis_direction,
            )
        prismatic_axis = axis_point_to_plucker_torch(axis_direction, axis_point)
        return prismatic_axis, torch.stack((low_limit, high_limit))

    def _recover_overparam_joint_parameters(
        self,
        *,
        query_points: Tensor,
        assigned_link_ids: Tensor,
        joint_connections: Tensor,
        joint_valid_flag: Tensor,
        is_revolute: Tensor,
        is_prismatic: Tensor,
        revolute_motion_points: Tensor,
        prismatic_motion_points: Tensor,
        revolute_axis_directions: Tensor | None = None,
        prismatic_axis_directions: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Recovers per-joint parameters by fitting each child link's query-wise targets."""
        query_points = query_points.float()
        revolute_motion_points = revolute_motion_points.float()
        prismatic_motion_points = prismatic_motion_points.float()
        batch_size, max_joints = joint_connections.shape[:2]
        revolute_axis = query_points.new_zeros((batch_size, max_joints, 6))
        prismatic_axis = query_points.new_zeros((batch_size, max_joints, 6))
        revolute_range = query_points.new_zeros((batch_size, max_joints, 2))
        prismatic_range = query_points.new_zeros((batch_size, max_joints, 2))

        child_link_ids = joint_connections[..., 1]
        for batch_idx in range(batch_size):
            for joint_idx in range(max_joints):
                query_mask = assigned_link_ids[batch_idx] == child_link_ids[batch_idx, joint_idx]
                joint_query_points = query_points[batch_idx][query_mask]
                if self.overparam_uses_axis_direction:
                    if self.joint_decode_type == "overparam+dir":
                        (
                            revolute_axis[batch_idx, joint_idx],
                            revolute_range[batch_idx, joint_idx],
                        ) = self._fit_revolute_joint_parameters_with_direction(
                            joint_query_points,
                            revolute_motion_points[batch_idx][query_mask],
                        )
                        (
                            prismatic_axis[batch_idx, joint_idx],
                            prismatic_range[batch_idx, joint_idx],
                        ) = self._fit_prismatic_joint_parameters_with_direction(
                            joint_query_points,
                            prismatic_motion_points[batch_idx][query_mask],
                        )
                    else:
                        if revolute_axis_directions is None or prismatic_axis_directions is None:
                            raise ValueError(
                                "overparam+singledir recovery requires per-joint axis directions"
                            )
                        (
                            revolute_axis[batch_idx, joint_idx],
                            revolute_range[batch_idx, joint_idx],
                        ) = self._fit_revolute_joint_parameters_with_single_direction(
                            joint_query_points,
                            revolute_motion_points[batch_idx][query_mask],
                            revolute_axis_directions[batch_idx, joint_idx],
                        )
                        (
                            prismatic_axis[batch_idx, joint_idx],
                            prismatic_range[batch_idx, joint_idx],
                        ) = self._fit_prismatic_joint_parameters_with_single_direction(
                            joint_query_points,
                            prismatic_motion_points[batch_idx][query_mask],
                            prismatic_axis_directions[batch_idx, joint_idx],
                        )
                else:
                    (
                        revolute_axis[batch_idx, joint_idx],
                        revolute_range[batch_idx, joint_idx],
                    ) = self._fit_revolute_joint_parameters(
                        joint_query_points,
                        revolute_motion_points[batch_idx][query_mask],
                    )
                    (
                        prismatic_axis[batch_idx, joint_idx],
                        prismatic_range[batch_idx, joint_idx],
                    ) = self._fit_prismatic_joint_parameters(
                        joint_query_points,
                        prismatic_motion_points[batch_idx][query_mask],
                    )

        revolute_mask = (joint_valid_flag & is_revolute).unsqueeze(-1)
        prismatic_mask = (joint_valid_flag & is_prismatic).unsqueeze(-1)
        return (
            revolute_axis.masked_fill(~revolute_mask, 0),
            prismatic_axis.masked_fill(~prismatic_mask, 0),
            revolute_range.masked_fill(~revolute_mask[..., :1], 0),
            prismatic_range.masked_fill(~prismatic_mask[..., :1], 0),
        )

    def _resolve_joint_decoding_link_ids(
        self,
        *,
        segmentation_logits: Tensor,
        link_ids: Tensor | None,
    ) -> Tensor:
        if link_ids is not None:
            return link_ids
        if self.training:
            raise ValueError(
                "forward requires link_ids when joint_decode_type uses over-parameterized decoding during training"
            )
        return segmentation_logits.argmax(dim=-1)

    def ancestor_context_parameters(self) -> Tuple[nn.Parameter, ...]:
        """Returns all parameters in the ancestor-context segmentation branch."""
        parameters: list[nn.Parameter] = []
        for module in (self.ancestor_context_projector, self.ancestor_context_gate):
            if module is not None:
                parameters.extend(tuple(module.parameters()))
        return tuple(parameters)

    def set_ancestor_context_gate_trainable(self, trainable: bool) -> bool:
        """Toggles gradient updates for the ancestor-context segmentation branch."""
        ancestor_context_parameters = self.ancestor_context_parameters()
        if len(ancestor_context_parameters) == 0:
            return False
        trainable = bool(trainable)
        if all(bool(parameter.requires_grad) == trainable for parameter in ancestor_context_parameters):
            return False
        for parameter in ancestor_context_parameters:
            parameter.requires_grad_(trainable)
        return True

    def _build_parent_index(
        self,
        *,
        joint_connections: Tensor,
        joint_valid_flag: Tensor,
        num_links: int,
    ) -> Tensor:
        """Returns one parent-link index per link, using `-1` for roots/padding."""
        device = joint_connections.device
        parent_index = torch.full(
            (joint_connections.shape[0], num_links),
            fill_value=-1,
            dtype=joint_connections.dtype,
            device=device,
        )
        if joint_connections.shape[1] == 0 or num_links == 0:
            return parent_index

        parent_indices = joint_connections[..., 0]
        child_indices = joint_connections[..., 1]
        valid_joint_mask = (
            joint_valid_flag
            & (parent_indices >= 0)
            & (parent_indices < num_links)
            & (child_indices >= 0)
            & (child_indices < num_links)
        )
        if not bool(valid_joint_mask.any().item()):
            return parent_index

        valid_child_mask = torch.zeros(
            (joint_connections.shape[0], num_links),
            dtype=torch.bool,
            device=device,
        )
        valid_child_mask.scatter_(
            dim=1,
            index=child_indices.clamp_min(0),
            src=valid_joint_mask,
        )
        parent_values = torch.where(valid_joint_mask, parent_indices, torch.zeros_like(parent_indices))
        parent_index.scatter_(
            dim=1,
            index=child_indices.clamp_min(0),
            src=parent_values,
        )
        return parent_index.masked_fill(~valid_child_mask, -1)

    def _build_depth_decayed_ancestor_context(
        self,
        *,
        link_latents: Tensor,
        joint_connections: Tensor,
        joint_valid_flag: Tensor,
    ) -> Tensor:
        """Returns a depth-decayed ancestor average for each link latent."""
        if not self.use_ancestor_context_for_segmentation:
            return link_latents

        batch_size, num_links, _ = link_latents.shape
        if num_links == 0:
            return torch.zeros_like(link_latents)

        parent_index = self._build_parent_index(
            joint_connections=joint_connections,
            joint_valid_flag=joint_valid_flag,
            num_links=num_links,
        )
        valid_parent_mask = parent_index >= 0
        if self.ancestor_context_decay == 0.0:
            parent_latents = link_latents.gather(
                dim=1,
                index=parent_index.clamp_min(0).unsqueeze(-1).expand_as(link_latents),
            )
            return parent_latents * valid_parent_mask.unsqueeze(-1).to(dtype=link_latents.dtype)
        if not bool(valid_parent_mask.any().item()):
            return torch.zeros_like(link_latents)
        solve_dtype = (
            torch.float64
            if link_latents.device.type == "cpu"
            else torch.float32
        )
        parent_adjacency = torch.zeros(
            (batch_size, num_links, num_links),
            device=link_latents.device,
            dtype=solve_dtype,
        )
        parent_adjacency.scatter_(
            dim=2,
            index=parent_index.clamp_min(0).unsqueeze(-1),
            src=valid_parent_mask.unsqueeze(-1).to(dtype=solve_dtype),
        )
        system_matrix = (
            torch.eye(num_links, device=link_latents.device, dtype=solve_dtype).unsqueeze(0)
            - self.ancestor_context_decay * parent_adjacency
        )
        weighted_ancestor_matrix = torch.linalg.solve(system_matrix, parent_adjacency)
        normalization = weighted_ancestor_matrix.sum(dim=-1, keepdim=True)
        normalized_ancestor_matrix = weighted_ancestor_matrix / normalization.clamp_min(1.0)
        normalized_ancestor_matrix = normalized_ancestor_matrix.masked_fill(
            normalization <= 0.0,
            0.0,
        )
        return torch.matmul(
            normalized_ancestor_matrix.to(dtype=link_latents.dtype),
            link_latents,
        )

    def build_segmentation_link_latents(
        self,
        *,
        link_latents: Tensor,
        joint_connections: Tensor,
        joint_valid_flag: Tensor,
    ) -> Tensor:
        """Applies optional ancestor-context refinement for segmentation only."""
        if not self.use_ancestor_context_for_segmentation:
            return link_latents
        if self.ancestor_context_gate is None or self.ancestor_context_projector is None:
            raise RuntimeError(
                "ancestor-context projector and gate must exist when "
                "use_ancestor_context_for_segmentation=True"
            )
        ancestor_context = self._build_depth_decayed_ancestor_context(
            link_latents=link_latents,
            joint_connections=joint_connections,
            joint_valid_flag=joint_valid_flag,
        )
        projected_ancestor_context = self.ancestor_context_projector(ancestor_context)
        gate = torch.sigmoid(
            self.ancestor_context_gate(
                torch.cat((link_latents, projected_ancestor_context), dim=-1)
            )
        )
        return link_latents + gate * projected_ancestor_context

    def decode(
        self,
        *,
        query_latents: Tensor,
        query_points: Tensor,
        link_latents: Tensor,
        link_valid_flag: Tensor,
        joint_connections: Tensor,
        joint_valid_flag: Tensor,
        is_revolute: Tensor,
        is_prismatic: Tensor,
        link_ids: Tensor | None = None,
    ) -> Dict[str, Any]:
        """Decodes predictions while preserving the caller-provided joint layout."""
        link_latents = link_latents.masked_fill(~link_valid_flag.unsqueeze(-1), 0)
        segmentation_link_latents = self.build_segmentation_link_latents(
            link_latents=link_latents,
            joint_connections=joint_connections,
            joint_valid_flag=joint_valid_flag,
        )
        segmentation_logits = self.decode_segmentation(
            query_latents=query_latents,
            link_latents=segmentation_link_latents,
            link_valid_flag=link_valid_flag,
        )
        revolute_axis = None
        prismatic_axis = None
        revolute_range = None
        prismatic_range = None
        revolute_closest_axis_points = None
        revolute_low_points = None
        revolute_high_points = None
        revolute_axis_directions = None
        revolute_closest_axis_points_decoder = None
        prismatic_closest_axis_points = None
        prismatic_low_points = None
        prismatic_high_points = None
        prismatic_axis_directions = None
        prismatic_closest_axis_points_decoder = None
        joint_decoding_link_ids = None
        decoded_motion_points = None
        if self.joint_decode_type in _PLAIN_JOINT_DECODE_TYPES:
            if not self.training:
                revolute_axis, prismatic_axis, revolute_range, prismatic_range = (
                    self.decode_joint_parameters(
                        link_latents=link_latents,
                        joint_connections=joint_connections,
                        joint_valid_flag=joint_valid_flag,
                        is_revolute=is_revolute,
                        is_prismatic=is_prismatic,
                    )
                )
        else:
            joint_decoding_link_ids = self._resolve_joint_decoding_link_ids(
                segmentation_logits=segmentation_logits,
                link_ids=link_ids,
            )
            decoded_motion_points = self._decode_joint_motion_points(
                query_latents=query_latents,
                link_latents=link_latents,
                assigned_link_ids=joint_decoding_link_ids,
                joint_connections=joint_connections,
            )
            revolute_motion_points_decoder_raw, prismatic_motion_points_decoder_raw = (
                decoded_motion_points
            )
            revolute_joint_axis_directions = None
            prismatic_joint_axis_directions = None
            if self.overparam_predicts_single_axis_direction:
                (
                    revolute_joint_axis_directions,
                    prismatic_joint_axis_directions,
                ) = self._decode_joint_axis_directions(
                    link_latents=link_latents,
                    joint_connections=joint_connections,
                    joint_valid_flag=joint_valid_flag,
                    is_revolute=is_revolute,
                    is_prismatic=is_prismatic,
                )
            revolute_motion_points_decoder, prismatic_motion_points_decoder = decoded_motion_points
            revolute_motion_points = self._convert_overparam_motion_points_to_world_coordinates(
                motion_points=revolute_motion_points_decoder,
                query_points=query_points,
                assigned_link_ids=joint_decoding_link_ids,
            )
            prismatic_motion_points = self._convert_overparam_motion_points_to_world_coordinates(
                motion_points=prismatic_motion_points_decoder,
                query_points=query_points,
                assigned_link_ids=joint_decoding_link_ids,
            )
            revolute_axis, prismatic_axis, revolute_range, prismatic_range = (
                self.decode_joint_parameters(
                    link_latents=link_latents,
                    joint_connections=joint_connections,
                    joint_valid_flag=joint_valid_flag,
                    is_revolute=is_revolute,
                    is_prismatic=is_prismatic,
                    query_points=query_points,
                    assigned_link_ids=joint_decoding_link_ids,
                    decoded_motion_points=(revolute_motion_points, prismatic_motion_points),
                    decoded_axis_directions=(
                        None
                        if revolute_joint_axis_directions is None
                        or prismatic_joint_axis_directions is None
                        else (
                            revolute_joint_axis_directions,
                            prismatic_joint_axis_directions,
                        )
                    ),
                    decoded_motion_points_are_world=True,
                )
            )
            revolute_closest_axis_points_decoder = revolute_motion_points_decoder_raw[..., :3]
            revolute_closest_axis_points = revolute_motion_points[..., :3]
            revolute_low_points = revolute_motion_points[..., 3:6]
            revolute_high_points = revolute_motion_points[..., 6:9]
            if self.overparam_predicts_query_axis_direction:
                revolute_axis_directions = revolute_motion_points[..., 9:12]
            elif self.overparam_predicts_single_axis_direction:
                revolute_axis_directions = revolute_joint_axis_directions
            prismatic_closest_axis_points_decoder = prismatic_motion_points_decoder_raw[..., :3]
            prismatic_closest_axis_points = prismatic_motion_points[..., :3]
            prismatic_low_points = prismatic_motion_points[..., 3:6]
            prismatic_high_points = prismatic_motion_points[..., 6:9]
            if self.overparam_predicts_query_axis_direction:
                prismatic_axis_directions = prismatic_motion_points[..., 9:12]
            elif self.overparam_predicts_single_axis_direction:
                prismatic_axis_directions = prismatic_joint_axis_directions
        return {
            "segmentation_logits": segmentation_logits,
            "revolute_axis": revolute_axis,
            "prismatic_axis": prismatic_axis,
            "revolute_range": revolute_range,
            "prismatic_range": prismatic_range,
            "revolute_closest_axis_points": revolute_closest_axis_points,
            "revolute_closest_axis_points_decoder": revolute_closest_axis_points_decoder,
            "revolute_low_points": revolute_low_points,
            "revolute_high_points": revolute_high_points,
            "revolute_axis_directions": revolute_axis_directions,
            "prismatic_closest_axis_points": prismatic_closest_axis_points,
            "prismatic_closest_axis_points_decoder": prismatic_closest_axis_points_decoder,
            "prismatic_low_points": prismatic_low_points,
            "prismatic_high_points": prismatic_high_points,
            "prismatic_axis_directions": prismatic_axis_directions,
            "joint_decoding_link_ids": joint_decoding_link_ids,
            "joint_connections": joint_connections,
            "joint_valid_flag": joint_valid_flag,
            "is_revolute": is_revolute,
            "is_prismatic": is_prismatic,
            "query_points": query_points,
        }

    def forward(
        self,
        shape_points: Tensor,
        shape_point_normals: Tensor,
        query_points: Tensor,
        query_point_normals: Tensor,
        link_point_prompts: Tensor | None,
        link_point_prompt_normals: Tensor | None,
        link_valid_flag: Tensor,
        joint_connections: Tensor,
        joint_valid_flag: Tensor,
        is_revolute: Tensor,
        is_prismatic: Tensor,
        link_point_prompt_dropout_eligible: Tensor | None = None,
        link_text_prompts: Sequence[Sequence[str]] | None = None,
        link_text_embeddings: Tensor | None = None,
        link_ids: Tensor | None = None,
    ) -> Dict[str, Any]:
        """Encodes inputs and decodes using padded joint tensors supplied by the caller."""
        encoder_output = self.encoder(
            shape_points=shape_points,
            shape_point_normals=shape_point_normals,
            query_points=query_points,
            query_point_normals=query_point_normals,
            link_point_prompts=link_point_prompts,
            link_point_prompt_normals=link_point_prompt_normals,
            link_valid_flag=link_valid_flag,
            link_point_prompt_dropout_eligible=link_point_prompt_dropout_eligible,
            link_text_prompts=link_text_prompts,
            link_text_embeddings=link_text_embeddings,
        )
        decoded_output = self.decode(
            query_latents=encoder_output["query_latents"],
            query_points=query_points,
            link_latents=encoder_output["link_latents"],
            link_valid_flag=link_valid_flag,
            joint_connections=joint_connections,
            joint_valid_flag=joint_valid_flag,
            is_revolute=is_revolute,
            is_prismatic=is_prismatic,
            link_ids=link_ids,
        )
        return {**encoder_output, **decoded_output}

    def compute_losses(
        self,
        output: Dict[str, Any],
        *,
        link_ids: Tensor,
        fm_batch_size: int = 1024,
        cfg_dropout: float = 0.0,
        link_scales: Tensor | None = None,
        revolute_axis: Tensor | None = None,
        prismatic_axis: Tensor | None = None,
        revolute_range: Tensor | None = None,
        prismatic_range: Tensor | None = None,
        revolute_closest_axis_points: Tensor | None = None,
        revolute_low_points: Tensor | None = None,
        revolute_high_points: Tensor | None = None,
        revolute_axis_directions: Tensor | None = None,
        prismatic_closest_axis_points: Tensor | None = None,
        prismatic_low_points: Tensor | None = None,
        prismatic_high_points: Tensor | None = None,
        prismatic_axis_directions: Tensor | None = None,
        revolute_axis_supervision_valid: Tensor | None = None,
        prismatic_axis_supervision_valid: Tensor | None = None,
        revolute_range_supervision_valid: Tensor | None = None,
        prismatic_range_supervision_valid: Tensor | None = None,
    ) -> LossComputationResult:
        """Computes losses plus joint-loss support counts used for logging."""
        segmentation_cross_entropy_loss = self.compute_segmentation_cross_entropy_loss(
            segmentation_logits=output["segmentation_logits"],
            link_ids=link_ids,
        )
        segmentation_dice_loss = self.compute_segmentation_dice_loss(
            segmentation_logits=output["segmentation_logits"],
            link_ids=link_ids,
            smoothing=self.dice_smoothing,
        )
        zero_loss = self._zero_loss(segmentation_cross_entropy_loss)
        revolute_axis_l1_loss = zero_loss
        prismatic_axis_l1_loss = zero_loss
        revolute_point_l1_loss = zero_loss
        prismatic_point_l1_loss = zero_loss
        revolute_direction_loss = zero_loss
        prismatic_direction_loss = zero_loss
        revolute_range_l1_loss = zero_loss
        prismatic_range_l1_loss = zero_loss
        revolute_fm_loss = zero_loss
        prismatic_fm_loss = zero_loss
        joint_loss_support_counts = self._initialize_joint_loss_support_counts(
            reference=segmentation_cross_entropy_loss,
        )

        total_loss = (
            self.segmentation_cross_entropy_weight * segmentation_cross_entropy_loss
            + self.segmentation_dice_weight * segmentation_dice_loss
        )
        if self.joint_decode_type == "plain":
            joint_decoder_losses = self.joint_decoder.compute_loss(
                link_latents=output["link_latents"],
                joint_connections=output["joint_connections"],
                joint_valid_flag=output["joint_valid_flag"],
                is_revolute=output["is_revolute"],
                is_prismatic=output["is_prismatic"],
                revolute_axis=revolute_axis,
                prismatic_axis=prismatic_axis,
                revolute_range=revolute_range,
                prismatic_range=prismatic_range,
                revolute_axis_supervision_valid=revolute_axis_supervision_valid,
                prismatic_axis_supervision_valid=prismatic_axis_supervision_valid,
                revolute_range_supervision_valid=revolute_range_supervision_valid,
                prismatic_range_supervision_valid=prismatic_range_supervision_valid,
            )
            revolute_axis_l1_loss = joint_decoder_losses["revolute_joint_axis_l1_loss"]
            prismatic_axis_l1_loss = joint_decoder_losses["prismatic_joint_axis_l1_loss"]
            revolute_range_l1_loss = joint_decoder_losses["revolute_joint_range_l1_loss"]
            prismatic_range_l1_loss = joint_decoder_losses["prismatic_joint_range_l1_loss"]
            joint_loss_support_counts = self._build_plain_joint_loss_support_counts(
                output=output,
                revolute_axis_supervision_valid=revolute_axis_supervision_valid,
                prismatic_axis_supervision_valid=prismatic_axis_supervision_valid,
                revolute_range_supervision_valid=revolute_range_supervision_valid,
                prismatic_range_supervision_valid=prismatic_range_supervision_valid,
                reference=segmentation_cross_entropy_loss,
            )
            total_loss = (
                total_loss
                + self.revolute_joint_axis_l1_weight * revolute_axis_l1_loss
                + self.prismatic_joint_axis_l1_weight * prismatic_axis_l1_loss
                + self.revolute_joint_range_l1_weight * revolute_range_l1_loss
                + self.prismatic_joint_range_l1_weight * prismatic_range_l1_loss
            )
        elif self.plain_flow_matching_enabled:
            joint_decoder_losses = self.joint_decoder.compute_loss(
                link_latents=output["link_latents"],
                joint_connections=output["joint_connections"],
                joint_valid_flag=output["joint_valid_flag"],
                is_revolute=output["is_revolute"],
                is_prismatic=output["is_prismatic"],
                revolute_axis=revolute_axis,
                prismatic_axis=prismatic_axis,
                revolute_range=revolute_range,
                prismatic_range=prismatic_range,
                revolute_axis_supervision_valid=revolute_axis_supervision_valid,
                prismatic_axis_supervision_valid=prismatic_axis_supervision_valid,
                revolute_range_supervision_valid=revolute_range_supervision_valid,
                prismatic_range_supervision_valid=prismatic_range_supervision_valid,
                fm_batch_size=fm_batch_size,
                training_time_mean=self.joint_fm_training_time_mean,
                training_time_std=self.joint_fm_training_time_std,
                cfg_dropout=cfg_dropout,
            )
            revolute_fm_loss = joint_decoder_losses["revolute_joint_fm_loss"]
            prismatic_fm_loss = joint_decoder_losses["prismatic_joint_fm_loss"]
            total_loss = (
                total_loss
                + self.revolute_joint_fm_weight * revolute_fm_loss
                + self.prismatic_joint_fm_weight * prismatic_fm_loss
            )
        else:
            query_masks = self._build_overparam_query_masks(
                output,
                link_ids=link_ids,
                revolute_axis_supervision_valid=revolute_axis_supervision_valid,
                prismatic_axis_supervision_valid=prismatic_axis_supervision_valid,
                revolute_range_supervision_valid=revolute_range_supervision_valid,
                prismatic_range_supervision_valid=prismatic_range_supervision_valid,
            )
            query_groups = self._build_overparam_query_groups(
                query_masks=query_masks,
                link_ids=link_ids,
            )
            (
                revolute_axis_l1_loss,
                revolute_point_l1_loss,
                prismatic_axis_l1_loss,
                prismatic_point_l1_loss,
            ) = self.compute_joint_overparam_l1_losses(
                output=output,
                link_ids=link_ids,
                link_scales=link_scales,
                revolute_closest_axis_points=revolute_closest_axis_points,
                revolute_low_points=revolute_low_points,
                revolute_high_points=revolute_high_points,
                prismatic_closest_axis_points=prismatic_closest_axis_points,
                prismatic_low_points=prismatic_low_points,
                prismatic_high_points=prismatic_high_points,
                query_masks=query_masks,
                query_groups=query_groups,
            )
            if self.overparam_uses_axis_direction:
                if self.overparam_predicts_single_axis_direction:
                    if revolute_axis is None or prismatic_axis is None:
                        raise ValueError(
                            "overparam+singledir joint decoding requires joint axis targets "
                            "for direction supervision"
                        )
                    (
                        revolute_direction_loss,
                        prismatic_direction_loss,
                    ) = self.compute_joint_overparam_single_direction_losses(
                        output=output,
                        revolute_axis=revolute_axis,
                        prismatic_axis=prismatic_axis,
                        revolute_axis_supervision_valid=revolute_axis_supervision_valid,
                        prismatic_axis_supervision_valid=prismatic_axis_supervision_valid,
                    )
                else:
                    if revolute_axis_directions is None or prismatic_axis_directions is None:
                        raise ValueError(
                            "direction-assisted overparam joint decoding requires per-query "
                            "axis direction targets"
                        )
                    (
                        revolute_direction_loss,
                        prismatic_direction_loss,
                    ) = self.compute_joint_overparam_direction_losses(
                        output=output,
                        link_ids=link_ids,
                        revolute_axis_directions=revolute_axis_directions,
                        prismatic_axis_directions=prismatic_axis_directions,
                        query_masks=query_masks,
                        query_groups=query_groups,
                    )
            joint_loss_support_counts = self._build_overparam_joint_loss_support_counts(
                query_groups=query_groups,
                reference=segmentation_cross_entropy_loss,
                include_direction=self.overparam_uses_axis_direction,
                output=output,
                revolute_axis_supervision_valid=revolute_axis_supervision_valid,
                prismatic_axis_supervision_valid=prismatic_axis_supervision_valid,
                use_joint_direction_counts=self.overparam_predicts_single_axis_direction,
            )
            total_loss = (
                total_loss
                + self.revolute_overparam_axis_l1_weight * revolute_axis_l1_loss
                + self.revolute_overparam_point_l1_weight * revolute_point_l1_loss
                + self.prismatic_overparam_axis_l1_weight * prismatic_axis_l1_loss
                + self.prismatic_overparam_point_l1_weight * prismatic_point_l1_loss
                + self.revolute_overparam_direction_weight * revolute_direction_loss
                + self.prismatic_overparam_direction_weight * prismatic_direction_loss
            )
        loss_terms: Dict[str, Tensor] = {
            "loss": total_loss,
            "segmentation_cross_entropy_loss": segmentation_cross_entropy_loss,
            "segmentation_dice_loss": segmentation_dice_loss,
        }
        if self.joint_decode_type == "plain":
            loss_terms.update(
                {
                    "revolute_axis_l1_loss": revolute_axis_l1_loss,
                    "prismatic_axis_l1_loss": prismatic_axis_l1_loss,
                    "revolute_range_l1_loss": revolute_range_l1_loss,
                    "prismatic_range_l1_loss": prismatic_range_l1_loss,
                }
            )
        elif self.plain_flow_matching_enabled:
            loss_terms.update(
                {
                    "revolute_fm_loss": revolute_fm_loss,
                    "prismatic_fm_loss": prismatic_fm_loss,
                }
            )
        else:
            loss_terms.update(
                {
                    "revolute_overparam_axis_l1_loss": revolute_axis_l1_loss,
                    "prismatic_overparam_axis_l1_loss": prismatic_axis_l1_loss,
                    "revolute_overparam_point_l1_loss": revolute_point_l1_loss,
                    "prismatic_overparam_point_l1_loss": prismatic_point_l1_loss,
                    "revolute_overparam_direction_loss": revolute_direction_loss,
                    "prismatic_overparam_direction_loss": prismatic_direction_loss,
                    "revolute_axis_l1_loss": revolute_axis_l1_loss,
                    "prismatic_axis_l1_loss": prismatic_axis_l1_loss,
                    "revolute_point_l1_loss": revolute_point_l1_loss,
                    "prismatic_point_l1_loss": prismatic_point_l1_loss,
                    "revolute_direction_loss": revolute_direction_loss,
                    "prismatic_direction_loss": prismatic_direction_loss,
                    "revolute_range_l1_loss": revolute_range_l1_loss,
                    "prismatic_range_l1_loss": prismatic_range_l1_loss,
                }
            )
        return LossComputationResult(
            losses=loss_terms,
            joint_loss_support_counts=joint_loss_support_counts,
        )

    def _initialize_joint_loss_support_counts(
        self,
        *,
        reference: Tensor,
    ) -> Dict[str, Tensor]:
        count_reference = reference.detach().float()
        zero_count = count_reference.new_zeros(())
        return {
            name: zero_count.clone()
            for name in JOINT_LOSS_LOGGING_KEYS
        }

    def _build_plain_joint_loss_support_counts(
        self,
        *,
        output: Dict[str, Any],
        revolute_axis_supervision_valid: Tensor | None,
        prismatic_axis_supervision_valid: Tensor | None,
        revolute_range_supervision_valid: Tensor | None,
        prismatic_range_supervision_valid: Tensor | None,
        reference: Tensor,
    ) -> Dict[str, Tensor]:
        support_counts = self._initialize_joint_loss_support_counts(reference=reference)
        joint_valid_flag = output["joint_valid_flag"]
        support_counts["revolute_axis_l1_loss"] = self._build_joint_supervision_mask(
            joint_valid_flag=joint_valid_flag,
            motion_flag=output["is_revolute"],
            supervision_valid=revolute_axis_supervision_valid,
        ).sum().to(dtype=support_counts["revolute_axis_l1_loss"].dtype)
        support_counts["prismatic_axis_l1_loss"] = self._build_joint_supervision_mask(
            joint_valid_flag=joint_valid_flag,
            motion_flag=output["is_prismatic"],
            supervision_valid=prismatic_axis_supervision_valid,
        ).sum().to(dtype=support_counts["prismatic_axis_l1_loss"].dtype)
        support_counts["revolute_range_l1_loss"] = self._build_joint_supervision_mask(
            joint_valid_flag=joint_valid_flag,
            motion_flag=output["is_revolute"],
            supervision_valid=revolute_range_supervision_valid,
        ).sum().to(dtype=support_counts["revolute_range_l1_loss"].dtype)
        support_counts["prismatic_range_l1_loss"] = self._build_joint_supervision_mask(
            joint_valid_flag=joint_valid_flag,
            motion_flag=output["is_prismatic"],
            supervision_valid=prismatic_range_supervision_valid,
        ).sum().to(dtype=support_counts["prismatic_range_l1_loss"].dtype)
        return support_counts

    def _build_overparam_joint_loss_support_counts(
        self,
        *,
        query_groups: _OverparamQueryGroups,
        reference: Tensor,
        include_direction: bool = False,
        output: Dict[str, Any] | None = None,
        revolute_axis_supervision_valid: Tensor | None = None,
        prismatic_axis_supervision_valid: Tensor | None = None,
        use_joint_direction_counts: bool = False,
    ) -> Dict[str, Tensor]:
        support_counts = self._initialize_joint_loss_support_counts(reference=reference)
        support_counts["revolute_axis_l1_loss"] = reference.new_tensor(
            float(query_groups.revolute_axis.num_groups),
        ).to(dtype=support_counts["revolute_axis_l1_loss"].dtype)
        support_counts["revolute_point_l1_loss"] = reference.new_tensor(
            float(query_groups.revolute_point.num_groups),
        ).to(dtype=support_counts["revolute_point_l1_loss"].dtype)
        support_counts["prismatic_axis_l1_loss"] = reference.new_tensor(
            float(query_groups.prismatic_axis.num_groups),
        ).to(dtype=support_counts["prismatic_axis_l1_loss"].dtype)
        support_counts["prismatic_point_l1_loss"] = reference.new_tensor(
            float(query_groups.prismatic_point.num_groups),
        ).to(dtype=support_counts["prismatic_point_l1_loss"].dtype)
        if include_direction:
            if use_joint_direction_counts:
                if output is None:
                    raise ValueError(
                        "output is required when use_joint_direction_counts=True"
                    )
                support_counts["revolute_direction_loss"] = self._build_joint_supervision_mask(
                    joint_valid_flag=output["joint_valid_flag"],
                    motion_flag=output["is_revolute"],
                    supervision_valid=revolute_axis_supervision_valid,
                ).sum().to(dtype=support_counts["revolute_direction_loss"].dtype)
                support_counts["prismatic_direction_loss"] = self._build_joint_supervision_mask(
                    joint_valid_flag=output["joint_valid_flag"],
                    motion_flag=output["is_prismatic"],
                    supervision_valid=prismatic_axis_supervision_valid,
                ).sum().to(dtype=support_counts["prismatic_direction_loss"].dtype)
            else:
                support_counts["revolute_direction_loss"] = reference.new_tensor(
                    float(query_groups.revolute_axis.num_groups),
                ).to(dtype=support_counts["revolute_direction_loss"].dtype)
                support_counts["prismatic_direction_loss"] = reference.new_tensor(
                    float(query_groups.prismatic_axis.num_groups),
                ).to(dtype=support_counts["prismatic_direction_loss"].dtype)
        return support_counts

    def _build_overparam_query_groups(
        self,
        *,
        query_masks: _OverparamQueryMasks,
        link_ids: Tensor,
    ) -> _OverparamQueryGroups:
        return _OverparamQueryGroups(
            revolute_axis=self._vectorize_active_link_groups(
                mask=query_masks.revolute_axis,
                link_ids=link_ids,
            ),
            revolute_point=self._vectorize_active_link_groups(
                mask=query_masks.revolute_point,
                link_ids=link_ids,
            ),
            prismatic_axis=self._vectorize_active_link_groups(
                mask=query_masks.prismatic_axis,
                link_ids=link_ids,
            ),
            prismatic_point=self._vectorize_active_link_groups(
                mask=query_masks.prismatic_point,
                link_ids=link_ids,
            ),
        )

    def compute_segmentation_cross_entropy_loss(
        self,
        segmentation_logits: Tensor,
        link_ids: Tensor,
    ) -> Tensor:
        """Computes per-query cross entropy over padded link logits."""
        return F.cross_entropy(
            segmentation_logits.transpose(1, 2),
            link_ids,
            reduction="mean",
        )

    def compute_segmentation_dice_loss(
        self,
        segmentation_logits: Tensor,
        link_ids: Tensor,
        smoothing: Optional[float] = None,
    ) -> Tensor:
        """Computes a multiclass Dice loss over padded link logits."""
        smoothing = self.dice_smoothing if smoothing is None else float(smoothing)
        probabilities = segmentation_logits.softmax(dim=-1)
        target_one_hot = F.one_hot(
            link_ids,
            num_classes=segmentation_logits.shape[-1],
        ).to(probabilities.dtype)
        target_support = target_one_hot.sum(dim=1)
        present_classes = target_support > 0
        intersection = (probabilities * target_one_hot).sum(dim=1)
        cardinality = probabilities.sum(dim=1) + target_support
        dice_per_class = (2.0 * intersection + smoothing) / (cardinality + smoothing)
        present_class_weights = present_classes.to(dtype=dice_per_class.dtype)
        dice_error = (1.0 - dice_per_class) * present_class_weights
        normalizer = present_class_weights.sum().clamp_min(1.0)
        return dice_error.sum() / normalizer

    def compute_joint_plain_fm_losses(
        self,
        output: Dict[str, Any],
        *,
        revolute_axis: Tensor,
        prismatic_axis: Tensor,
        revolute_range: Tensor,
        prismatic_range: Tensor,
        revolute_axis_supervision_valid: Tensor | None = None,
        prismatic_axis_supervision_valid: Tensor | None = None,
        revolute_range_supervision_valid: Tensor | None = None,
        prismatic_range_supervision_valid: Tensor | None = None,
        fm_batch_size: int = 1024,
        cfg_dropout: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        if not self.plain_flow_matching_enabled:
            raise ValueError(
                "compute_joint_plain_fm_losses requires joint_decode_type='plain+fm'"
            )
        decoder_losses = self.joint_decoder.compute_loss(
            link_latents=output["link_latents"],
            joint_connections=output["joint_connections"],
            joint_valid_flag=output["joint_valid_flag"],
            is_revolute=output["is_revolute"],
            is_prismatic=output["is_prismatic"],
            revolute_axis=revolute_axis,
            prismatic_axis=prismatic_axis,
            revolute_range=revolute_range,
            prismatic_range=prismatic_range,
            revolute_axis_supervision_valid=revolute_axis_supervision_valid,
            prismatic_axis_supervision_valid=prismatic_axis_supervision_valid,
            revolute_range_supervision_valid=revolute_range_supervision_valid,
            prismatic_range_supervision_valid=prismatic_range_supervision_valid,
            fm_batch_size=fm_batch_size,
            training_time_mean=self.joint_fm_training_time_mean,
            training_time_std=self.joint_fm_training_time_std,
            cfg_dropout=cfg_dropout,
        )
        return (
            decoder_losses["revolute_joint_fm_loss"],
            decoder_losses["prismatic_joint_fm_loss"],
        )

    def compute_joint_l1_losses(
        self,
        output: Dict[str, Any],
        *,
        revolute_axis: Tensor,
        prismatic_axis: Tensor,
        revolute_range: Tensor,
        prismatic_range: Tensor,
        revolute_axis_supervision_valid: Tensor | None = None,
        prismatic_axis_supervision_valid: Tensor | None = None,
        revolute_range_supervision_valid: Tensor | None = None,
        prismatic_range_supervision_valid: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Computes L1 losses while ignoring padded joints and missing targets."""
        if self.joint_decode_type not in _PLAIN_JOINT_DECODE_TYPES:
            raise ValueError(
                "compute_joint_l1_losses is only available for plain joint decoders"
            )
        joint_valid_flag = output["joint_valid_flag"]
        revolute_axis_mask = self._build_joint_supervision_mask(
            joint_valid_flag=joint_valid_flag,
            motion_flag=output["is_revolute"],
            supervision_valid=revolute_axis_supervision_valid,
        )
        prismatic_axis_mask = self._build_joint_supervision_mask(
            joint_valid_flag=joint_valid_flag,
            motion_flag=output["is_prismatic"],
            supervision_valid=prismatic_axis_supervision_valid,
        )
        revolute_range_mask = self._build_joint_supervision_mask(
            joint_valid_flag=joint_valid_flag,
            motion_flag=output["is_revolute"],
            supervision_valid=revolute_range_supervision_valid,
        )
        prismatic_range_mask = self._build_joint_supervision_mask(
            joint_valid_flag=joint_valid_flag,
            motion_flag=output["is_prismatic"],
            supervision_valid=prismatic_range_supervision_valid,
        )
        (
            revolute_axis_l1_loss,
            revolute_range_l1_loss,
        ) = _masked_flip_invariant_joint_parameter_losses(
            predicted_axis=output["revolute_axis"],
            target_axis=revolute_axis,
            axis_mask=revolute_axis_mask,
            predicted_range=output["revolute_range"],
            target_range=revolute_range,
            range_mask=revolute_range_mask,
        )
        (
            prismatic_axis_l1_loss,
            prismatic_range_l1_loss,
        ) = _masked_flip_invariant_joint_parameter_losses(
            predicted_axis=output["prismatic_axis"],
            target_axis=prismatic_axis,
            axis_mask=prismatic_axis_mask,
            predicted_range=output["prismatic_range"],
            target_range=prismatic_range,
            range_mask=prismatic_range_mask,
        )
        return (
            revolute_axis_l1_loss,
            prismatic_axis_l1_loss,
            revolute_range_l1_loss,
            prismatic_range_l1_loss,
        )

    def _build_joint_supervision_mask(
        self,
        *,
        joint_valid_flag: Tensor,
        motion_flag: Tensor,
        supervision_valid: Tensor | None,
    ) -> Tensor:
        """Builds a joint-level supervision mask for one target component."""
        return _build_joint_supervision_mask(
            joint_valid_flag=joint_valid_flag,
            motion_flag=motion_flag,
            supervision_valid=supervision_valid,
        )

    def _build_overparam_query_masks(
        self,
        output: Dict[str, Any],
        *,
        link_ids: Tensor,
        revolute_axis_supervision_valid: Tensor | None = None,
        prismatic_axis_supervision_valid: Tensor | None = None,
        revolute_range_supervision_valid: Tensor | None = None,
        prismatic_range_supervision_valid: Tensor | None = None,
    ) -> _OverparamQueryMasks:
        """Builds all per-query supervision masks used by over-parameterized losses."""
        revolute_axis_mask, prismatic_axis_mask = self._build_query_component_masks(
            output,
            link_ids=link_ids,
            revolute_supervision_valid=revolute_axis_supervision_valid,
            prismatic_supervision_valid=prismatic_axis_supervision_valid,
        )
        revolute_point_mask, prismatic_point_mask = self._build_query_component_masks(
            output,
            link_ids=link_ids,
            revolute_supervision_valid=revolute_range_supervision_valid,
            prismatic_supervision_valid=prismatic_range_supervision_valid,
        )
        return _OverparamQueryMasks(
            revolute_axis=revolute_axis_mask,
            revolute_point=revolute_point_mask,
            prismatic_axis=prismatic_axis_mask,
            prismatic_point=prismatic_point_mask,
        )

    def _build_query_component_masks(
        self,
        output: Dict[str, Any],
        *,
        link_ids: Tensor,
        revolute_supervision_valid: Tensor | None = None,
        prismatic_supervision_valid: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Builds per-query revolute/prismatic supervision masks from joint metadata."""
        batch_size = link_ids.shape[0]
        num_links = output["segmentation_logits"].shape[-1]
        link_revolute_mask = torch.zeros(
            (batch_size, num_links),
            device=link_ids.device,
            dtype=torch.bool,
        )
        link_prismatic_mask = torch.zeros_like(link_revolute_mask)
        revolute_joint_mask = self._build_joint_supervision_mask(
            joint_valid_flag=output["joint_valid_flag"],
            motion_flag=output["is_revolute"],
            supervision_valid=revolute_supervision_valid,
        )
        prismatic_joint_mask = self._build_joint_supervision_mask(
            joint_valid_flag=output["joint_valid_flag"],
            motion_flag=output["is_prismatic"],
            supervision_valid=prismatic_supervision_valid,
        )

        valid_child_mask = output["joint_connections"][..., 1] >= 0
        valid_revolute_joint_mask = revolute_joint_mask & valid_child_mask
        valid_prismatic_joint_mask = prismatic_joint_mask & valid_child_mask
        child_indices = output["joint_connections"][..., 1].to(
            device=link_ids.device,
            dtype=torch.long,
        )
        batch_indices = torch.arange(batch_size, device=link_ids.device).unsqueeze(1).expand_as(
            child_indices
        )
        link_revolute_mask[
            batch_indices[valid_revolute_joint_mask],
            child_indices[valid_revolute_joint_mask],
        ] = True
        link_prismatic_mask[
            batch_indices[valid_prismatic_joint_mask],
            child_indices[valid_prismatic_joint_mask],
        ] = True

        return (
            link_revolute_mask.gather(dim=1, index=link_ids),
            link_prismatic_mask.gather(dim=1, index=link_ids),
        )

    def compute_joint_overparam_l1_losses(
        self,
        output: Dict[str, Any],
        *,
        link_ids: Tensor,
        link_scales: Tensor | None = None,
        revolute_closest_axis_points: Tensor,
        revolute_low_points: Tensor,
        revolute_high_points: Tensor,
        prismatic_closest_axis_points: Tensor,
        prismatic_low_points: Tensor,
        prismatic_high_points: Tensor,
        query_masks: _OverparamQueryMasks,
        query_groups: _OverparamQueryGroups | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Computes masked losses for motion-specific over-parameterized targets."""
        if query_groups is None:
            query_groups = self._build_overparam_query_groups(
                query_masks=query_masks,
                link_ids=link_ids,
            )
        revolute_axis_predictions = output.get(
            "revolute_closest_axis_points_decoder",
            output["revolute_closest_axis_points"],
        )
        prismatic_axis_predictions = output.get(
            "prismatic_closest_axis_points_decoder",
            output["prismatic_closest_axis_points"],
        )
        revolute_axis_targets = revolute_closest_axis_points
        prismatic_axis_targets = prismatic_closest_axis_points
        query_link_scale_normalizer = None
        if self.overparam_closest_axis_uses_part_aabb:
            query_points = output.get("query_points")
            if query_points is None:
                raise ValueError(
                    "part_aabb closest-axis supervision requires query_points in the model output"
                )
            revolute_axis_targets = self._normalize_overparam_axis_points(
                axis_points=revolute_closest_axis_points,
                query_points=query_points,
                link_ids=link_ids,
            )
            prismatic_axis_targets = self._normalize_overparam_axis_points(
                axis_points=prismatic_closest_axis_points,
                query_points=query_points,
                link_ids=link_ids,
            )
            if "revolute_closest_axis_points_decoder" not in output:
                revolute_axis_predictions = self._normalize_overparam_axis_points(
                    axis_points=output["revolute_closest_axis_points"],
                    query_points=query_points,
                    link_ids=link_ids,
                )
            if "prismatic_closest_axis_points_decoder" not in output:
                prismatic_axis_predictions = self._normalize_overparam_axis_points(
                    axis_points=output["prismatic_closest_axis_points"],
                    query_points=query_points,
                    link_ids=link_ids,
                )
        else:
            query_link_scale_normalizer = self._gather_query_link_scale_normalizer(
                link_ids=link_ids,
                link_scales=link_scales,
            )
        revolute_axis_l1_loss = self._masked_l1_loss_grouped_by_link(
            predicted=revolute_axis_predictions,
            target=revolute_axis_targets,
            mask=query_masks.revolute_axis,
            link_ids=link_ids,
            query_normalizer=query_link_scale_normalizer,
            active_link_groups=query_groups.revolute_axis,
        )
        revolute_point_l1_loss = self._masked_swappable_pair_l1_loss(
            link_ids=link_ids,
            predicted_first=output["revolute_low_points"],
            predicted_second=output["revolute_high_points"],
            target_first=revolute_low_points,
            target_second=revolute_high_points,
            mask=query_masks.revolute_point,
            active_link_groups=query_groups.revolute_point,
        )
        prismatic_axis_l1_loss = self._masked_l1_loss_grouped_by_link(
            predicted=prismatic_axis_predictions,
            target=prismatic_axis_targets,
            mask=query_masks.prismatic_axis,
            link_ids=link_ids,
            query_normalizer=query_link_scale_normalizer,
            active_link_groups=query_groups.prismatic_axis,
        )
        prismatic_point_l1_loss = self._masked_swappable_pair_l1_loss(
            link_ids=link_ids,
            predicted_first=output["prismatic_low_points"],
            predicted_second=output["prismatic_high_points"],
            target_first=prismatic_low_points,
            target_second=prismatic_high_points,
            mask=query_masks.prismatic_point,
            active_link_groups=query_groups.prismatic_point,
        )
        return (
            revolute_axis_l1_loss,
            revolute_point_l1_loss,
            prismatic_axis_l1_loss,
            prismatic_point_l1_loss,
        )

    def compute_joint_overparam_direction_losses(
        self,
        output: Dict[str, Any],
        *,
        link_ids: Tensor,
        revolute_axis_directions: Tensor,
        prismatic_axis_directions: Tensor,
        query_masks: _OverparamQueryMasks,
        query_groups: _OverparamQueryGroups | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Computes flip-invariant per-query direction losses for direction-assisted overparam modes."""
        if query_groups is None:
            query_groups = self._build_overparam_query_groups(
                query_masks=query_masks,
                link_ids=link_ids,
            )
        if output["revolute_axis_directions"] is None or output["prismatic_axis_directions"] is None:
            raise ValueError(
                "Direction losses require per-query predicted axis directions in the model output"
            )
        revolute_direction_loss = self._masked_flip_invariant_direction_loss_grouped_by_link(
            predicted=output["revolute_axis_directions"],
            target=revolute_axis_directions,
            mask=query_masks.revolute_axis,
            link_ids=link_ids,
            active_link_groups=query_groups.revolute_axis,
        )
        prismatic_direction_loss = self._masked_flip_invariant_direction_loss_grouped_by_link(
            predicted=output["prismatic_axis_directions"],
            target=prismatic_axis_directions,
            mask=query_masks.prismatic_axis,
            link_ids=link_ids,
            active_link_groups=query_groups.prismatic_axis,
        )
        return revolute_direction_loss, prismatic_direction_loss

    def compute_joint_overparam_single_direction_losses(
        self,
        output: Dict[str, Any],
        *,
        revolute_axis: Tensor,
        prismatic_axis: Tensor,
        revolute_axis_supervision_valid: Tensor | None = None,
        prismatic_axis_supervision_valid: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Computes flip-invariant joint-level direction losses for overparam+singledir."""
        if output["revolute_axis_directions"] is None or output["prismatic_axis_directions"] is None:
            raise ValueError(
                "Single-direction losses require per-joint predicted axis directions in the model output"
            )
        revolute_mask = self._build_joint_supervision_mask(
            joint_valid_flag=output["joint_valid_flag"],
            motion_flag=output["is_revolute"],
            supervision_valid=revolute_axis_supervision_valid,
        )
        prismatic_mask = self._build_joint_supervision_mask(
            joint_valid_flag=output["joint_valid_flag"],
            motion_flag=output["is_prismatic"],
            supervision_valid=prismatic_axis_supervision_valid,
        )
        return (
            self._masked_flip_invariant_direction_loss(
                predicted=output["revolute_axis_directions"],
                target=revolute_axis[..., :3],
                mask=revolute_mask,
            ),
            self._masked_flip_invariant_direction_loss(
                predicted=output["prismatic_axis_directions"],
                target=prismatic_axis[..., :3],
                mask=prismatic_mask,
            ),
        )

    def _masked_l1_loss_grouped_by_link(
        self,
        *,
        predicted: Tensor,
        target: Tensor,
        mask: Tensor,
        link_ids: Tensor,
        query_normalizer: Tensor | None = None,
        active_link_groups: _ActiveLinkGroups | None = None,
    ) -> Tensor:
        """Returns a masked L1 loss averaged uniformly across active links."""
        l1_error = self._reduce_l1_error_to_mask_shape(
            F.l1_loss(predicted, target, reduction="none"),
            mask,
        )
        group_query_weight = None
        if query_normalizer is not None:
            if query_normalizer.shape != mask.shape:
                raise ValueError(
                    "query_normalizer must match the mask shape, "
                    f"got {tuple(query_normalizer.shape)} and {tuple(mask.shape)}"
                )
            group_query_weight = query_normalizer.to(dtype=l1_error.dtype).reciprocal()
        return self._mean_query_error_over_active_links(
            query_error=l1_error,
            mask=mask,
            link_ids=link_ids,
            group_query_weight=group_query_weight,
            active_link_groups=active_link_groups,
        )

    def _gather_query_link_scale_normalizer(
        self,
        *,
        link_ids: Tensor,
        link_scales: Tensor | None,
    ) -> Tensor | None:
        """Returns a per-query link-scale tensor for loss normalization."""
        if link_scales is None:
            return None
        if link_scales.ndim != 2:
            raise ValueError(
                f"link_scales must have shape (B, L), got {tuple(link_scales.shape)}"
            )
        if link_scales.shape[0] != link_ids.shape[0]:
            raise ValueError(
                "link_scales batch dim must match link_ids, "
                f"got {tuple(link_scales.shape)} and {tuple(link_ids.shape)}"
            )
        safe_link_ids = link_ids.clamp_min(0)
        if safe_link_ids.numel() > 0 and int(safe_link_ids.max().item()) >= link_scales.shape[1]:
            raise ValueError(
                "link_ids reference a link outside link_scales, "
                f"got max link_id={int(safe_link_ids.max().item())} for {link_scales.shape[1]} scales"
            )
        query_link_scales = link_scales.gather(dim=1, index=safe_link_ids)
        if torch.any(link_ids < 0):
            query_link_scales = torch.where(
                link_ids >= 0,
                query_link_scales,
                torch.ones_like(query_link_scales),
            )
        return query_link_scales.clamp_min(_OVERPARAM_LINK_SCALE_MIN)

    def _masked_swappable_pair_l1_loss(
        self,
        *,
        link_ids: Tensor,
        predicted_first: Tensor,
        predicted_second: Tensor,
        target_first: Tensor,
        target_second: Tensor,
        mask: Tensor,
        active_link_groups: _ActiveLinkGroups | None = None,
    ) -> Tensor:
        """Returns a masked L1 loss that is invariant to swapping the target pair."""
        if predicted_first.shape != predicted_second.shape:
            raise ValueError(
                "predicted_first and predicted_second must share the same shape, "
                f"got {tuple(predicted_first.shape)} and {tuple(predicted_second.shape)}"
            )
        if target_first.shape != target_second.shape:
            raise ValueError(
                "target_first and target_second must share the same shape, "
                f"got {tuple(target_first.shape)} and {tuple(target_second.shape)}"
            )
        if predicted_first.shape != target_first.shape:
            raise ValueError(
                "Predicted and target point pairs must share the same shape, "
                f"got {tuple(predicted_first.shape)} and {tuple(target_first.shape)}"
            )
        direct_error = 0.5 * (
            self._reduce_l1_error_to_mask_shape(
                F.l1_loss(predicted_first, target_first, reduction="none"),
                mask,
            )
            + self._reduce_l1_error_to_mask_shape(
                F.l1_loss(predicted_second, target_second, reduction="none"),
                mask,
            )
        )
        swapped_error = 0.5 * (
            self._reduce_l1_error_to_mask_shape(
                F.l1_loss(predicted_first, target_second, reduction="none"),
                mask,
            )
            + self._reduce_l1_error_to_mask_shape(
                F.l1_loss(predicted_second, target_first, reduction="none"),
                mask,
            )
        )
        return self._mean_query_error_over_active_links(
            query_error=direct_error,
            alternate_query_error=swapped_error,
            mask=mask,
            link_ids=link_ids,
            active_link_groups=active_link_groups,
        )

    def _masked_flip_invariant_direction_loss_grouped_by_link(
        self,
        *,
        predicted: Tensor,
        target: Tensor,
        mask: Tensor,
        link_ids: Tensor,
        active_link_groups: _ActiveLinkGroups | None = None,
    ) -> Tensor:
        """Returns a per-query `1 - |cos|` loss averaged uniformly across active links."""
        predicted_unit = F.normalize(predicted, dim=-1, eps=1e-8)
        target_unit = F.normalize(target, dim=-1, eps=1e-8)
        cosine_similarity = (predicted_unit * target_unit).sum(dim=-1).abs().clamp(max=1.0)
        return self._mean_query_error_over_active_links(
            query_error=1.0 - cosine_similarity,
            mask=mask,
            link_ids=link_ids,
            active_link_groups=active_link_groups,
        )

    def _masked_flip_invariant_direction_loss(
        self,
        *,
        predicted: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Returns a masked `1 - |cos|` loss over a joint-shaped mask."""
        predicted_unit = F.normalize(predicted, dim=-1, eps=1e-8)
        target_unit = F.normalize(target, dim=-1, eps=1e-8)
        cosine_similarity = (predicted_unit * target_unit).sum(dim=-1).abs().clamp(max=1.0)
        if not bool(mask.any().item()):
            return cosine_similarity.sum() * 0.0
        return self._masked_mean_error(1.0 - cosine_similarity, mask)

    def _masked_flip_invariant_joint_parameter_losses(
        self,
        *,
        predicted_axis: Tensor,
        target_axis: Tensor,
        axis_mask: Tensor,
        predicted_range: Tensor,
        target_range: Tensor,
        range_mask: Tensor,
        axis_weight: float = 1.0,
        range_weight: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """Returns plain-mode axis/range losses while treating global sign flips as equivalent."""
        return _masked_flip_invariant_joint_parameter_losses(
            predicted_axis=predicted_axis,
            target_axis=target_axis,
            axis_mask=axis_mask,
            predicted_range=predicted_range,
            target_range=target_range,
            range_mask=range_mask,
            axis_weight=axis_weight,
            range_weight=range_weight,
        )

    def _mean_query_error_over_active_links(
        self,
        *,
        query_error: Tensor,
        mask: Tensor,
        link_ids: Tensor,
        alternate_query_error: Tensor | None = None,
        group_query_weight: Tensor | None = None,
        active_link_groups: _ActiveLinkGroups | None = None,
    ) -> Tensor:
        """Averages per-query errors uniformly across active `(batch item, link)` groups."""
        if query_error.shape != mask.shape:
            raise ValueError(
                "query_error must match the mask shape, "
                f"got {tuple(query_error.shape)} and {tuple(mask.shape)}"
            )
        if alternate_query_error is not None and alternate_query_error.shape != query_error.shape:
            raise ValueError(
                "alternate_query_error must match query_error shape, "
                f"got {tuple(alternate_query_error.shape)} and {tuple(query_error.shape)}"
            )
        if group_query_weight is not None and group_query_weight.shape != query_error.shape:
            raise ValueError(
                "group_query_weight must match query_error shape, "
                f"got {tuple(group_query_weight.shape)} and {tuple(query_error.shape)}"
            )
        if active_link_groups is None:
            active_link_groups = self._vectorize_active_link_groups(
                mask=mask,
                link_ids=link_ids,
            )
        if active_link_groups.num_groups == 0:
            # Keep the zero loss connected to the decoder graph so DDP sees the
            # same parameter usage even when a rank has no supervised groups.
            return query_error.sum() * 0.0

        active_query_error = query_error.reshape(-1)[active_link_groups.active_flat_mask]
        group_sums = active_query_error.new_zeros((active_link_groups.num_groups,))
        group_sums.scatter_add_(0, active_link_groups.group_inverse, active_query_error)
        group_counts = active_query_error.new_zeros((active_link_groups.num_groups,))
        group_counts.scatter_add_(
            0,
            active_link_groups.group_inverse,
            torch.ones_like(active_query_error),
        )
        group_error = group_sums / group_counts.clamp_min(1.0)
        if alternate_query_error is not None:
            active_alternate_query_error = alternate_query_error.reshape(-1)[
                active_link_groups.active_flat_mask
            ]
            alternate_group_sums = active_alternate_query_error.new_zeros(
                (active_link_groups.num_groups,)
            )
            alternate_group_sums.scatter_add_(
                0,
                active_link_groups.group_inverse,
                active_alternate_query_error,
            )
            alternate_group_error = alternate_group_sums / group_counts.clamp_min(1.0)
            group_error = torch.minimum(group_error, alternate_group_error)
        if group_query_weight is None:
            return group_error.mean()

        active_group_query_weight = group_query_weight.reshape(-1)[
            active_link_groups.active_flat_mask
        ]
        group_weight_sums = active_group_query_weight.new_zeros((active_link_groups.num_groups,))
        group_weight_sums.scatter_add_(
            0,
            active_link_groups.group_inverse,
            active_group_query_weight,
        )
        group_weight = group_weight_sums / group_counts.clamp_min(1.0)
        return (group_error * group_weight).sum() / group_weight.sum().clamp_min(1.0)

    def _vectorize_active_link_groups(
        self,
        *,
        mask: Tensor,
        link_ids: Tensor,
    ) -> _ActiveLinkGroups:
        """Returns flat active-query masks plus inverse ids for `(batch item, link)` groups."""
        if link_ids.shape != mask.shape:
            raise ValueError(
                "link_ids must match the mask shape, "
                f"got {tuple(link_ids.shape)} and {tuple(mask.shape)}"
            )
        active_flat_mask = mask.reshape(-1)
        if not torch.any(active_flat_mask):
            empty_inverse = link_ids.new_empty((0,), dtype=torch.long)
            return _ActiveLinkGroups(
                active_flat_mask=active_flat_mask,
                group_inverse=empty_inverse,
                num_groups=0,
            )

        batch_indices = torch.arange(
            mask.shape[0],
            device=link_ids.device,
            dtype=link_ids.dtype,
        ).unsqueeze(1).expand_as(link_ids)
        active_group_pairs = torch.stack(
            (
                batch_indices.reshape(-1)[active_flat_mask],
                link_ids.reshape(-1)[active_flat_mask],
            ),
            dim=1,
        )
        unique_group_pairs, group_inverse = torch.unique(
            active_group_pairs,
            dim=0,
            return_inverse=True,
        )
        return _ActiveLinkGroups(
            active_flat_mask=active_flat_mask,
            group_inverse=group_inverse,
            num_groups=unique_group_pairs.shape[0],
        )

    def _masked_l1_loss(
        self,
        predicted: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Returns the mean L1 loss over `mask`, or zero when no entries are active."""
        l1_error = self._reduce_l1_error_to_mask_shape(
            F.l1_loss(predicted, target, reduction="none"),
            mask,
        )
        mask_weights = mask.to(dtype=l1_error.dtype)
        masked_error = l1_error * mask_weights
        normalizer = mask_weights.sum()
        return masked_error.sum() / normalizer.clamp_min(1.0)

    def _masked_mean_error(self, error: Tensor, mask: Tensor) -> Tensor:
        """Returns the mean of a precomputed per-entry error tensor over `mask`."""
        return _masked_mean_error(error, mask)

    def _reduce_l1_error_to_mask_shape(
        self,
        l1_error: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Reduces trailing target dimensions so the loss matches `mask` shape."""
        return _reduce_l1_error_to_mask_shape(l1_error, mask)

    def _zero_loss(self, reference: Tensor) -> Tensor:
        return reference.new_zeros(())


__all__ = [
    "Particulate2ArticulationModel",
    "Particulate2Encoder",
    "JointDecoderPlainFlowMatching",
    "JointDecoderOverParametrized",
    "JointDecoderSingleDirection",
    "JointDecoderPlain",
    "SegmentationDecoder",
]
