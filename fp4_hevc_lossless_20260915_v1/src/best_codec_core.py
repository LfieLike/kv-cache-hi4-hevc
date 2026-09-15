#!/usr/bin/env python3
"""Minimal evaluation core for the selected FP4 + lossless-HEVC method."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open

from codec_aware_rotation import (
    codec_rate_proxy,
    fp4_e2m1_monotonic_ste,
    int2_symmetric_monotonic_qdq,
    int2_turboquant_qdq,
    int4_turboquant_qdq,
    int8_symmetric_monotonic_qdq,
    int_symmetric_monotonic_qdq,
    key_nmse_terms,
    qk_error_terms,
)
from codec_rotation_data import CodecSlab


@dataclass(frozen=True)
class QuantizerConfig:
    """Exact storage-scale configuration used by one evaluation arm."""

    label: str
    scale_granularity: str
    quantizer_kind: str = "fp4-e2m1"
    scale_block_tokens: int = 256
    scale_group_elements: int = 32
    scale_multiplier: float = 1.0
    int8_deadzone: int = 0
    int8_step: int = 1
    int_bits: int = 8


def load_final_rotations(
    path: Path,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Load the final per-layer/per-head orthogonal matrices."""

    path = path.expanduser().resolve()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = set(handle.keys())
        required = {"final_rotations", "layer_indices", "head_indices"}
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"{path} is missing tensors: {missing}")
        rotations = handle.get_tensor("final_rotations").float().to(device)
        layers = [int(value) for value in handle.get_tensor("layer_indices")]
        heads = [int(value) for value in handle.get_tensor("head_indices")]
    return rotations, layers, heads


def load_key_means(
    path: Path,
    *,
    device: torch.device,
    tensor_name: str = "key_means",
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Load the frozen per-(layer, head) Key mean table fitted on Split A."""

    path = path.expanduser().resolve()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = set(handle.keys())
        required = {tensor_name, "layer_indices", "head_indices"}
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"{path} is missing tensors: {missing}")
        means = handle.get_tensor(tensor_name).float().to(device)
        layers = [int(value) for value in handle.get_tensor("layer_indices")]
        heads = [int(value) for value in handle.get_tensor("head_indices")]
    return means, layers, heads


def _rotate(keys: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    return torch.einsum("lthd,lhde->lthe", keys, rotations)


def _inverse(values: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        "lthe,lhed->lthd", values, rotations.transpose(-1, -2)
    )


def broadcast_key_means(
    key_means: torch.Tensor | None,
    *,
    keys: torch.Tensor,
) -> torch.Tensor | None:
    """Shape a ``[layer, head, channel]`` mean table onto ``[L, T, H, D]`` keys."""

    if key_means is None:
        return None
    layers, _, heads, dim = keys.shape
    if tuple(key_means.shape) != (layers, heads, dim):
        raise ValueError(
            f"key_means must be [{layers}, {heads}, {dim}], "
            f"got {tuple(key_means.shape)}"
        )
    return key_means.to(device=keys.device, dtype=keys.dtype).unsqueeze(1)


@torch.no_grad()
def evaluate_one_slab(
    slab: CodecSlab,
    rotations: torch.Tensor,
    *,
    config: QuantizerConfig,
    temperature: float,
    proxy_block_height: int,
    proxy_block_width: int,
    device: torch.device,
    need_codes: bool,
    key_means: torch.Tensor | None = None,
) -> tuple[dict, torch.Tensor | None]:
    """Rotate, hard-quantize, inverse-rotate, and score one synchronized slab.

    When ``key_means`` is given the quantizer only ever sees ``K - mean`` and
    the mean is added back on reconstruction, so the reported QK and Key errors
    stay comparisons against the untouched original Key.

    ``slab.cache_kind='value'`` encodes Value activations stored in
    ``slab.keys``. QK scoring is skipped; reconstruction NMSE still uses the
    Key NMSE formula on that tensor.
    """

    keys = slab.keys.to(device=device, dtype=torch.float32)
    covariances = slab.query_covariances.to(device=device, dtype=torch.float32)
    counts = slab.query_counts.to(device=device)

    means = broadcast_key_means(key_means, keys=keys)
    rotated = _rotate(keys if means is None else keys - means, rotations)
    if config.quantizer_kind == "fp4-e2m1":
        quantized = fp4_e2m1_monotonic_ste(
            rotated,
            temperature=temperature,
            scale_granularity=config.scale_granularity,
            scale_block_tokens=config.scale_block_tokens,
            scale_group_elements=config.scale_group_elements,
            scale_multiplier=config.scale_multiplier,
        )
        proxy_residual_scale = 1.0
        neutral_code = 8.0
    elif config.quantizer_kind == "int8-symmetric":
        quantized = int8_symmetric_monotonic_qdq(
            rotated,
            scale_granularity=config.scale_granularity,
            scale_block_tokens=config.scale_block_tokens,
            scale_multiplier=config.scale_multiplier,
            deadzone=config.int8_deadzone,
            step=config.int8_step,
        )
        # Normalize the 256-symbol residual magnitude to roughly the same
        # range as the 16-symbol FP4 proxy. Exact x265 bytes remain decisive.
        proxy_residual_scale = 16.0
        neutral_code = 128.0
    elif config.quantizer_kind == "int-symmetric":
        quantized = int_symmetric_monotonic_qdq(
            rotated,
            bits=config.int_bits,
            scale_granularity=config.scale_granularity,
            scale_block_tokens=config.scale_block_tokens,
            scale_multiplier=config.scale_multiplier,
        )
        proxy_residual_scale = 256.0 / float(1 << config.int_bits)
        neutral_code = float(1 << (config.int_bits - 1))
    elif config.quantizer_kind == "int2-symmetric":
        quantized = int2_symmetric_monotonic_qdq(
            rotated,
            scale_granularity=config.scale_granularity,
            scale_group_elements=config.scale_group_elements,
            scale_multiplier=config.scale_multiplier,
        )
        proxy_residual_scale = 1.0
        neutral_code = 2.0
    elif config.quantizer_kind == "int2-turboquant":
        quantized = int2_turboquant_qdq(
            rotated,
            scale_multiplier=config.scale_multiplier,
        )
        proxy_residual_scale = 1.0
        # Mid-riser codebook: codes 1 and 2 straddle zero.
        neutral_code = 1.5
    elif config.quantizer_kind == "int4-turboquant":
        quantized = int4_turboquant_qdq(
            rotated,
            scale_multiplier=config.scale_multiplier,
            scale_granularity=config.scale_granularity,
            scale_block_tokens=config.scale_block_tokens,
        )
        proxy_residual_scale = 1.0
        # Mid-riser codebook: codes 7 and 8 straddle zero.
        neutral_code = 7.5
    else:
        raise ValueError(f"unknown quantizer kind: {config.quantizer_kind!r}")
    restored = _inverse(quantized.restored, rotations)
    if means is not None:
        restored = restored + means

    key_error, key_reference = key_nmse_terms(keys, restored)
    if getattr(slab, "cache_kind", "key") == "value":
        qk_error = torch.zeros((), device=keys.device, dtype=torch.float32)
        qk_reference = torch.ones((), device=keys.device, dtype=torch.float32)
    else:
        qk_error, qk_reference = qk_error_terms(
            keys, restored, covariances, counts
        )
    proxy = codec_rate_proxy(
        quantized.hard_codes.float(),
        block_height=proxy_block_height,
        block_width=proxy_block_width,
        mode_temperature=1.0,
        residual_scale=proxy_residual_scale,
        neutral_code=neutral_code,
        bidirectional=False,
    )
    metrics = {
        "proxy_sum": float(proxy.total_bits_per_value.item()) * keys.numel(),
        "qk_error": float(qk_error.item()),
        "qk_reference": float(qk_reference.item()),
        "key_error": float(key_error.item()),
        "key_reference": float(key_reference.item()),
        "values": keys.numel(),
        "tokens": keys.shape[1],
        "scale_count": quantized.scale_count,
        "scale_bytes": quantized.scale_bytes,
    }
    codes = quantized.hard_codes.cpu() if need_codes else None
    return metrics, codes


def aggregate_metrics(items: Iterable[dict]) -> dict:
    items = list(items)
    if not items:
        raise ValueError("at least one slab metric is required")
    names = (
        "proxy_sum",
        "qk_error",
        "qk_reference",
        "key_error",
        "key_reference",
        "values",
        "tokens",
        "scale_count",
        "scale_bytes",
    )
    totals = {name: sum(item[name] for item in items) for name in names}
    return {
        **totals,
        "proxy_bpv": totals["proxy_sum"] / totals["values"],
        "qk_nmse": totals["qk_error"] / max(totals["qk_reference"], 1e-30),
        "key_nmse": totals["key_error"] / max(totals["key_reference"], 1e-30),
    }


@torch.no_grad()
def evaluate_slabs(
    slabs: list[CodecSlab],
    rotations: torch.Tensor,
    *,
    config: QuantizerConfig,
    temperature: float,
    proxy_block_height: int,
    proxy_block_width: int,
    device: torch.device,
    need_codes: bool,
    key_means: torch.Tensor | None = None,
) -> tuple[dict, list[torch.Tensor]]:
    metrics = []
    codes = []
    for index, slab in enumerate(slabs, start=1):
        item, item_codes = evaluate_one_slab(
            slab,
            rotations,
            config=config,
            temperature=temperature,
            proxy_block_height=proxy_block_height,
            proxy_block_width=proxy_block_width,
            device=device,
            need_codes=need_codes,
            key_means=key_means,
        )
        metrics.append(item)
        if item_codes is not None:
            codes.append(item_codes)
        print(
            f"quantized {config.label}: slab={index}/{len(slabs)} "
            f"kind={getattr(slab, 'cache_kind', 'key')} "
            f"tokens={slab.keys.shape[1]}",
            flush=True,
        )
    return aggregate_metrics(metrics), codes
