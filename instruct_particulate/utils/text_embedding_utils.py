from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F


def load_clip_text_encoder(
    clip_model_name: str,
    *,
    device: torch.device,
    expected_embedding_dim: int | None,
    cache_dir: str | None,
    local_files_only: bool = False,
):
    """Load the CLIP tokenizer plus text encoder with the nested text config."""
    from transformers import AutoConfig, AutoTokenizer, CLIPTextModel

    tokenizer = AutoTokenizer.from_pretrained(
        clip_model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    config = AutoConfig.from_pretrained(
        clip_model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    text_config = getattr(config, "text_config", config)
    hidden_size = int(text_config.hidden_size)
    if expected_embedding_dim is not None and hidden_size != expected_embedding_dim:
        raise ValueError(
            f"CLIP hidden size mismatch for {clip_model_name}: "
            f"expected {expected_embedding_dim}, got {hidden_size}"
        )

    text_model = CLIPTextModel.from_pretrained(
        clip_model_name,
        cache_dir=cache_dir,
        config=text_config,
        use_safetensors=True,
        local_files_only=local_files_only,
    )
    text_model.eval()
    text_model.requires_grad_(False)
    text_model.to(device=device)
    return tokenizer, text_model


@torch.no_grad()
def encode_clip_text_prompts(
    prompts: Sequence[str],
    *,
    tokenizer: Any,
    text_model: torch.nn.Module,
    batch_size: int,
    output_device: torch.device | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Encode text prompts the same way as dataset-side CLIP embedding caching."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    model_device = next(text_model.parameters()).device
    if output_device is None:
        output_device = model_device

    hidden_size = int(text_model.config.hidden_size)
    if len(prompts) == 0:
        return torch.zeros((0, hidden_size), device=output_device, dtype=output_dtype)

    encoded_batches: list[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            batch_prompts,
            max_length=tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model_device)
        text_outputs = text_model(**encoded)
        encoded_batches.append(
            text_outputs.pooler_output.detach().to(device=output_device, dtype=output_dtype)
        )
    return torch.cat(encoded_batches, dim=0)


def load_qwen3_text_encoder(
    model_name: str,
    *,
    device: torch.device,
    expected_embedding_dim: int | None,
    cache_dir: str | None,
    local_files_only: bool = False,
):
    """Load the tokenizer and base model used by Qwen3 embedding extraction."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        padding_side="left",
    )
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    hidden_size = int(config.hidden_size)
    if expected_embedding_dim is not None and hidden_size != expected_embedding_dim:
        raise ValueError(
            f"Qwen hidden size mismatch for {model_name}: "
            f"expected {expected_embedding_dim}, got {hidden_size}"
        )

    text_model = AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    text_model.eval()
    text_model.requires_grad_(False)
    text_model.to(device=device)
    return tokenizer, text_model


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = bool(torch.all(attention_mask[:, -1] == 1).item())
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


@torch.no_grad()
def encode_qwen3_text_prompts(
    prompts: Sequence[str],
    *,
    tokenizer: Any,
    text_model: torch.nn.Module,
    batch_size: int,
    output_device: torch.device | None = None,
    output_dtype: torch.dtype = torch.float32,
    max_length: int = 8192,
) -> torch.Tensor:
    """Encode text prompts using the Qwen3 embedding extraction recipe."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    model_device = next(text_model.parameters()).device
    if output_device is None:
        output_device = model_device

    hidden_size = int(text_model.config.hidden_size)
    if len(prompts) == 0:
        return torch.zeros((0, hidden_size), device=output_device, dtype=output_dtype)

    encoded_batches: list[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model_device)
        text_outputs = text_model(**encoded)
        embeddings = _last_token_pool(
            text_outputs.last_hidden_state,
            encoded["attention_mask"],
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        encoded_batches.append(
            embeddings.detach().to(device=output_device, dtype=output_dtype)
        )
    return torch.cat(encoded_batches, dim=0)
