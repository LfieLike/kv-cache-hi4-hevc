#!/usr/bin/env python3
"""Evaluate held-out cross-layer alignment and mean/shuffle controls."""

from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path

import torch
from safetensors import safe_open

from alignment_math import normalize_rows, orthogonality_max_error
from cross_layer_data import iter_synchronized_key_chunks, resolve_layer_sources


E2M1_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
QUANTIZER_LEVELS = {
    "fp8_e4m3fn": 256,
    "fp4_e2m1": 16,
    "affine_int4": 16,
    "affine_int2": 4,
}
QUANTIZER_STORAGE = {
    "fp8_e4m3fn": {"bits_per_code": 8, "metadata_bytes_per_token_layer": 0},
    "fp4_e2m1": {"bits_per_code": 4, "metadata_bytes_per_token_layer": 2},
    "affine_int4": {"bits_per_code": 4, "metadata_bytes_per_token_layer": 4},
    "affine_int2": {"bits_per_code": 2, "metadata_bytes_per_token_layer": 4},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--rotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--token-stride", type=int, default=1)
    parser.add_argument("--shuffle-offset", type=int, default=997)
    parser.add_argument(
        "--quantizers",
        nargs="+",
        choices=tuple(QUANTIZER_LEVELS),
        default=list(QUANTIZER_LEVELS),
    )
    parser.add_argument(
        "--zip-level",
        type=int,
        choices=range(10),
        default=6,
        help="zlib/DEFLATE level used for the generic entropy-coding proxy",
    )
    parser.add_argument(
        "--zip-max-tokens",
        type=int,
        default=16384,
        help="contiguous held-out tokens used by ZIP proxies; 0 means all tokens",
    )
    parser.add_argument("--device", default="cuda")
    return parser


def load_alignment(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        result = {
            "layer_indices": handle.get_tensor("layer_indices").long(),
            "head_indices": handle.get_tensor("head_indices").long(),
            "head_rotations": handle.get_tensor("head_rotations").float(),
            "layer_rotations": handle.get_tensor("layer_rotations").float(),
            "final_rotations": handle.get_tensor("final_rotations").float(),
            "key_means": handle.get_tensor("key_means").float(),
        }
    for name in ("head_rotations", "layer_rotations", "final_rotations", "key_means"):
        result[name] = result[name].to(device)
    return result


def add_metric(state: dict, name: str, values: torch.Tensor) -> None:
    item = state.setdefault(name, {"sum": 0.0, "count": 0})
    item["sum"] += float(values.sum().item())
    item["count"] += values.numel()


def mean_metric(state: dict, name: str) -> float:
    item = state[name]
    return item["sum"] / max(item["count"], 1)


def entropy_from_counts(counts: torch.Tensor) -> float:
    probabilities = counts.double() / counts.sum().clamp_min(1)
    nonzero = probabilities > 0
    return float(-(probabilities[nonzero] * probabilities[nonzero].log2()).sum().item())


def fp4_e2m1_qdq(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E2M1 with one BF16 amax scale per token and layer."""
    amax = values.abs().amax(dim=(1, 2), keepdim=True)
    scale = torch.where(amax > 0, amax / 6.0, torch.ones_like(amax))
    scale = scale.to(torch.bfloat16).float()
    normalized = values / scale
    boundaries = E2M1_BOUNDARIES.to(values.device)
    magnitudes = E2M1_MAGNITUDES.to(values.device)
    magnitude_codes = torch.bucketize(normalized.abs(), boundaries)
    negative = normalized < 0
    codes = magnitude_codes + negative.to(torch.long) * 8
    restored_magnitude = magnitudes.index_select(
        0, magnitude_codes.reshape(-1)
    ).reshape_as(values)
    restored = torch.where(negative, -restored_magnitude, restored_magnitude) * scale
    return restored, codes.to(torch.uint8)


def fp8_e4m3fn_qdq(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct native E4M3FN bytes without an external scale or zero-point."""
    clipped = values.clamp(min=-448.0, max=448.0)
    encoded = clipped.to(torch.float8_e4m3fn).contiguous()
    codes = encoded.view(torch.uint8)
    return encoded.float(), codes


def affine_qdq(values: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Affine QDQ with one BF16 min/scale per token and layer."""
    levels = (1 << bits) - 1
    minimum = values.amin(dim=(1, 2), keepdim=True).to(torch.bfloat16).float()
    maximum = values.amax(dim=(1, 2), keepdim=True)
    scale = torch.where(
        maximum > minimum,
        (maximum - minimum) / levels,
        torch.ones_like(maximum),
    ).to(torch.bfloat16).float()
    codes = torch.round((values - minimum) / scale).clamp(0, levels)
    return codes * scale + minimum, codes.to(torch.uint8)


def quantize(values: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if name == "fp8_e4m3fn":
        return fp8_e4m3fn_qdq(values)
    if name == "fp4_e2m1":
        return fp4_e2m1_qdq(values)
    if name == "affine_int4":
        return affine_qdq(values, bits=4)
    if name == "affine_int2":
        return affine_qdq(values, bits=2)
    raise ValueError(f"unknown quantizer: {name}")


def init_quant_state(
    levels: int,
    *,
    bits_per_code: int | None = None,
    metadata_bytes_per_token_layer: int = 0,
    layer_count: int = 0,
    zip_level: int = 6,
    zip_max_tokens: int = 0,
) -> dict:
    state = {
        "error_square": 0.0,
        "reference_square": 0.0,
        "value_count": 0,
        "code_counts": torch.zeros(levels, dtype=torch.int64),
        "metrics": {},
    }
    if bits_per_code is not None:
        state["zip"] = {
            "bits_per_code": bits_per_code,
            "metadata_bytes_per_token_layer": metadata_bytes_per_token_layer,
            "layer_count": layer_count,
            "max_tokens": zip_max_tokens,
            "tokens": 0,
            "raw_code_bytes": 0,
            "metadata_bytes": 0,
            "layer_major_code_bytes": 0,
            "token_interleaved_code_bytes": 0,
            "layer_compressors": [
                zlib.compressobj(level=zip_level) for _ in range(layer_count)
            ],
            "token_interleaved_compressor": zlib.compressobj(level=zip_level),
            "zip_level": zip_level,
        }
    return state


def pack_codes(codes: torch.Tensor, bits_per_code: int) -> bytes:
    """Pack uint codes in flat order using little sub-byte lanes."""
    flat = codes.detach().to(device="cpu", dtype=torch.uint8).contiguous().view(-1)
    values_per_byte = 8 // bits_per_code
    padding = (-flat.numel()) % values_per_byte
    if padding:
        flat = torch.cat([flat, torch.zeros(padding, dtype=torch.uint8)])
    packed = torch.zeros(flat.numel() // values_per_byte, dtype=torch.uint8)
    mask = (1 << bits_per_code) - 1
    for lane in range(values_per_byte):
        packed |= (flat[lane::values_per_byte] & mask) << (lane * bits_per_code)
    return packed.numpy().tobytes()


def update_zip_proxy(state: dict, codes: list[torch.Tensor]) -> None:
    if "zip" not in state:
        return
    item = state["zip"]
    available = codes[0].shape[0]
    if item["max_tokens"] > 0:
        available = min(available, item["max_tokens"] - item["tokens"])
    if available <= 0:
        return

    selected = [layer_codes[:available] for layer_codes in codes]
    layer_raw_bytes = 0
    for layer_index, layer_codes in enumerate(selected):
        plane = layer_codes.permute(0, 2, 1).contiguous()
        payload = pack_codes(plane, item["bits_per_code"])
        layer_raw_bytes += len(payload)
        item["layer_major_code_bytes"] += len(
            item["layer_compressors"][layer_index].compress(payload)
        )

    token_interleaved = torch.stack(selected, dim=1).permute(0, 1, 3, 2).contiguous()
    token_payload = pack_codes(token_interleaved, item["bits_per_code"])
    if len(token_payload) != layer_raw_bytes:
        raise AssertionError("packed layouts must contain the same number of code bytes")
    item["token_interleaved_code_bytes"] += len(
        item["token_interleaved_compressor"].compress(token_payload)
    )
    item["raw_code_bytes"] += layer_raw_bytes
    item["metadata_bytes"] += (
        available * item["layer_count"] * item["metadata_bytes_per_token_layer"]
    )
    item["tokens"] += available


def update_quant_metrics(
    state: dict,
    *,
    raw: list[torch.Tensor],
    aligned: list[torch.Tensor],
    restored_aligned: list[torch.Tensor],
    codes: list[torch.Tensor],
    rotations: torch.Tensor,
    levels: int,
) -> None:
    for layer_index in range(len(raw)):
        restored = torch.einsum(
            "nhe,hde->nhd", restored_aligned[layer_index], rotations[layer_index]
        )
        residual = restored - raw[layer_index]
        state["error_square"] += float(residual.square().sum().item())
        state["reference_square"] += float(raw[layer_index].square().sum().item())
        state["value_count"] += residual.numel()
        state["code_counts"] += torch.bincount(
            codes[layer_index].detach().reshape(-1).long(),
            minlength=levels,
        ).cpu()

        rms = aligned[layer_index].square().mean(dim=(1, 2)).sqrt()
        amax = aligned[layer_index].abs().amax(dim=(1, 2))
        add_metric(state["metrics"], "rotated_amax_over_rms", amax / rms.clamp_min(1e-12))

        for left in range(codes[layer_index].shape[1]):
            for right in range(left + 1, codes[layer_index].shape[1]):
                add_metric(
                    state["metrics"],
                    "within_layer_head_code_agreement",
                    codes[layer_index][:, left] == codes[layer_index][:, right],
                )

        plane = codes[layer_index].permute(0, 2, 1).reshape(codes[layer_index].shape[0], -1).float()
        if plane.shape[1] > 1:
            add_metric(
                state["metrics"],
                "h265_horizontal_code_mae_normalized",
                (plane[:, 1:] - plane[:, :-1]).abs() / max(levels - 1, 1),
            )
        if codes[layer_index].shape[0] > 1:
            add_metric(
                state["metrics"],
                "h265_vertical_code_mae_normalized",
                (codes[layer_index][1:].float() - codes[layer_index][:-1].float()).abs()
                / max(levels - 1, 1),
            )

    for layer_index in range(len(codes) - 1):
        left_codes = codes[layer_index]
        right_codes = codes[layer_index + 1]
        add_metric(
            state["metrics"],
            "adjacent_layer_same_head_code_agreement",
            left_codes == right_codes,
        )
        add_metric(
            state["metrics"],
            "h265_inter_frame_code_mae_normalized",
            (left_codes.float() - right_codes.float()).abs() / max(levels - 1, 1),
        )
        for left in range(left_codes.shape[1]):
            for right in range(right_codes.shape[1]):
                add_metric(
                    state["metrics"],
                    "adjacent_layer_all_head_code_agreement",
                    left_codes[:, left] == right_codes[:, right],
                )
    update_zip_proxy(state, codes)


def finalize_zip_proxy(state: dict) -> dict | None:
    if "zip" not in state:
        return None
    item = state["zip"]
    for compressor in item["layer_compressors"]:
        item["layer_major_code_bytes"] += len(compressor.flush())
    item["token_interleaved_code_bytes"] += len(
        item["token_interleaved_compressor"].flush()
    )
    raw_codes = item["raw_code_bytes"]
    metadata = item["metadata_bytes"]
    # value_count may cover more tokens than the ZIP sample. The exact sampled
    # source size follows directly from packed codes and bits-per-code.
    source_values = raw_codes * 8 // item["bits_per_code"]
    source_bf16_bytes = source_values * 2

    def layout_summary(compressed_codes: int) -> dict:
        compressed_total = compressed_codes + metadata
        raw_total = raw_codes + metadata
        return {
            "compressed_code_bytes": compressed_codes,
            "compressed_total_bytes_with_raw_metadata": compressed_total,
            "code_only_ratio": raw_codes / max(compressed_codes, 1),
            "packed_cache_ratio": raw_total / max(compressed_total, 1),
            "compression_ratio_vs_bf16": source_bf16_bytes
            / max(compressed_total, 1),
            "effective_bits_per_original_value": compressed_total
            * 8
            / max(source_values, 1),
        }

    return {
        "codec": "zlib-deflate",
        "level": item["zip_level"],
        "sampled_tokens": item["tokens"],
        "bits_per_packed_code": item["bits_per_code"],
        "raw_packed_code_bytes": raw_codes,
        "raw_metadata_bytes": metadata,
        "source_bf16_bytes": source_bf16_bytes,
        "metadata_policy": "stored-raw-conservative",
        "layer_major_separate_streams": layout_summary(
            item["layer_major_code_bytes"]
        ),
        "token_interleaved_stream": layout_summary(
            item["token_interleaved_code_bytes"]
        ),
    }


def finalize_quant_state(state: dict) -> dict:
    probabilities = state["code_counts"].double() / state["code_counts"].sum().clamp_min(1)
    metrics = state["metrics"]
    result = {
        "scale_scope": "shared-per-token-per-layer-across-all-heads-and-channels",
        "nmse": state["error_square"] / max(state["reference_square"], 1e-30),
        "rmse": math.sqrt(state["error_square"] / max(state["value_count"], 1)),
        "symbol_entropy_bits": entropy_from_counts(state["code_counts"]),
        "code_probabilities": probabilities.tolist(),
        "rotated_amax_over_rms": mean_metric(metrics, "rotated_amax_over_rms"),
        "within_layer_head_code_agreement": mean_metric(
            metrics, "within_layer_head_code_agreement"
        ),
        "adjacent_layer_same_head_code_agreement": mean_metric(
            metrics, "adjacent_layer_same_head_code_agreement"
        ),
        "adjacent_layer_all_head_code_agreement": mean_metric(
            metrics, "adjacent_layer_all_head_code_agreement"
        ),
        "h265_horizontal_code_mae_normalized": mean_metric(
            metrics, "h265_horizontal_code_mae_normalized"
        ),
        "h265_vertical_code_mae_normalized": mean_metric(
            metrics, "h265_vertical_code_mae_normalized"
        ),
        "h265_inter_frame_code_mae_normalized": mean_metric(
            metrics, "h265_inter_frame_code_mae_normalized"
        ),
    }
    zip_proxy = finalize_zip_proxy(state)
    if zip_proxy is not None:
        result["zip_proxy"] = zip_proxy
    return result


def within_layer_head_cosine(values: list[torch.Tensor]) -> torch.Tensor:
    scores = []
    for layer_values in values:
        for left in range(layer_values.shape[1]):
            for right in range(left + 1, layer_values.shape[1]):
                scores.append((layer_values[:, left] * layer_values[:, right]).sum(dim=-1))
    return torch.stack(scores, dim=0)


def adjacent_layer_same_head(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(
        [
            (values[index] * values[index + 1]).sum(dim=-1)
            for index in range(len(values) - 1)
        ],
        dim=0,
    )


def adjacent_layer_all_heads(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(
        [
            torch.einsum("nhd,njd->nhj", values[index], values[index + 1])
            for index in range(len(values) - 1)
        ],
        dim=0,
    )


def layer_templates(values: list[torch.Tensor]) -> list[torch.Tensor]:
    return [normalize_rows(layer_values.mean(dim=1)) for layer_values in values]


def adjacent_template_cosine(templates: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(
        [
            (templates[index] * templates[index + 1]).sum(dim=-1)
            for index in range(len(templates) - 1)
        ],
        dim=0,
    )


def all_template_pairwise_cosine(templates: list[torch.Tensor]) -> torch.Tensor:
    scores = []
    for left in range(len(templates)):
        for right in range(left + 1, len(templates)):
            scores.append((templates[left] * templates[right]).sum(dim=-1))
    return torch.stack(scores, dim=0)


def transformed_values(
    normalized: list[torch.Tensor], rotations: torch.Tensor
) -> list[torch.Tensor]:
    return [
        torch.einsum("nhd,hde->nhe", values, rotations[layer_index])
        for layer_index, values in enumerate(normalized)
    ]


def shuffled_adjacent_metrics(
    values: list[torch.Tensor], offset: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    same_head = []
    all_heads = []
    templates = layer_templates(values)
    template_scores = []
    for index in range(len(values) - 1):
        n = values[index].shape[0]
        shift = offset % max(n, 1)
        if shift == 0 and n > 1:
            shift = 1
        right = torch.roll(values[index + 1], shifts=shift, dims=0)
        same_head.append((values[index] * right).sum(dim=-1))
        all_heads.append(torch.einsum("nhd,njd->nhj", values[index], right))
        right_template = torch.roll(templates[index + 1], shifts=shift, dims=0)
        template_scores.append((templates[index] * right_template).sum(dim=-1))
    return (
        torch.stack(same_head, dim=0),
        torch.stack(all_heads, dim=0),
        torch.stack(template_scores, dim=0),
    )


def pair_details(
    accumulators: dict[str, list[dict[str, float]]],
    method: str,
    layers: list[int],
    values: list[torch.Tensor],
) -> None:
    rows = accumulators.setdefault(
        method,
        [
            {
                "left_layer": layers[index],
                "right_layer": layers[index + 1],
                "same_head_sum": 0.0,
                "same_head_count": 0,
                "all_head_sum": 0.0,
                "all_head_count": 0,
                "template_sum": 0.0,
                "template_count": 0,
            }
            for index in range(len(layers) - 1)
        ],
    )
    templates = layer_templates(values)
    for index, row in enumerate(rows):
        same = (values[index] * values[index + 1]).sum(dim=-1)
        all_heads = torch.einsum("nhd,njd->nhj", values[index], values[index + 1])
        template = (templates[index] * templates[index + 1]).sum(dim=-1)
        row["same_head_sum"] += float(same.sum().item())
        row["same_head_count"] += same.numel()
        row["all_head_sum"] += float(all_heads.sum().item())
        row["all_head_count"] += all_heads.numel()
        row["template_sum"] += float(template.sum().item())
        row["template_count"] += template.numel()


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    if args.token_stride <= 0:
        raise ValueError("token-stride must be positive")
    if args.zip_max_tokens < 0:
        raise ValueError("zip-max-tokens must be non-negative")
    args.rotations = args.rotations.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    sources = resolve_layer_sources(args.manifests, args.layers)
    alignment = load_alignment(args.rotations, device)
    saved_layers = alignment["layer_indices"].tolist()
    saved_heads = alignment["head_indices"].tolist()
    if saved_layers != args.layers:
        raise ValueError(f"layer mismatch: saved={saved_layers}, requested={args.layers}")
    if saved_heads != list(sources[0].manifest.head_indices):
        raise ValueError("head-index mismatch between rotations and manifests")
    layer_count = len(args.layers)
    heads = len(saved_heads)
    dim = sources[0].manifest.head_dim
    identity = torch.eye(dim, device=device).expand(layer_count, heads, dim, dim).clone()
    methods_rotations = {
        "identity": identity,
        "head_only": alignment["head_rotations"],
        "head_layer": alignment["final_rotations"],
    }
    states = {name: {} for name in methods_rotations}
    quant_states = {
        method: {
            quantizer: init_quant_state(
                QUANTIZER_LEVELS[quantizer],
                bits_per_code=QUANTIZER_STORAGE[quantizer]["bits_per_code"],
                metadata_bytes_per_token_layer=QUANTIZER_STORAGE[quantizer][
                    "metadata_bytes_per_token_layer"
                ],
                layer_count=layer_count,
                zip_level=args.zip_level,
                zip_max_tokens=args.zip_max_tokens,
            )
            for quantizer in args.quantizers
        }
        for method in methods_rotations
    }
    controls: dict = {}
    pair_accumulators: dict[str, list[dict[str, float]]] = {}
    chunk_count = 0
    token_count = 0

    for sequence, _, chunks_cpu in iter_synchronized_key_chunks(
        sources,
        max_sequences=args.max_sequences,
        token_stride=args.token_stride,
    ):
        raw = [values.to(device=device, dtype=torch.float32) for values in chunks_cpu]
        normalized = [normalize_rows(values) for values in raw]
        token_count += raw[0].shape[0]
        chunk_count += 1
        for method, rotations in methods_rotations.items():
            values = transformed_values(normalized, rotations)
            raw_aligned = transformed_values(raw, rotations)
            add_metric(states[method], "within_layer_cross_head", within_layer_head_cosine(values))
            add_metric(states[method], "adjacent_layer_same_head", adjacent_layer_same_head(values))
            add_metric(states[method], "adjacent_layer_all_heads", adjacent_layer_all_heads(values))
            templates = layer_templates(values)
            add_metric(states[method], "adjacent_layer_template", adjacent_template_cosine(templates))
            add_metric(states[method], "all_layer_template", all_template_pairwise_cosine(templates))
            if raw[0].shape[0] > 1:
                before = torch.stack(
                    [
                        torch.nn.functional.cosine_similarity(item[1:], item[:-1], dim=-1)
                        for item in raw
                    ],
                    dim=0,
                )
                after = torch.stack(
                    [
                        torch.nn.functional.cosine_similarity(item[1:], item[:-1], dim=-1)
                        for item in values
                    ],
                    dim=0,
                )
                add_metric(states[method], "adjacent_token_before", before)
                add_metric(states[method], "adjacent_token_after", after)
            pair_details(pair_accumulators, method, args.layers, values)
            for quantizer in args.quantizers:
                restored_aligned = []
                codes = []
                for layer_values in raw_aligned:
                    restored, layer_codes = quantize(layer_values, quantizer)
                    restored_aligned.append(restored)
                    codes.append(layer_codes)
                update_quant_metrics(
                    quant_states[method][quantizer],
                    raw=raw,
                    aligned=raw_aligned,
                    restored_aligned=restored_aligned,
                    codes=codes,
                    rotations=rotations,
                    levels=QUANTIZER_LEVELS[quantizer],
                )

        final_values = transformed_values(normalized, alignment["final_rotations"])
        shuffled = shuffled_adjacent_metrics(final_values, args.shuffle_offset)
        add_metric(controls, "raw_shuffled_same_head", shuffled[0])
        add_metric(controls, "raw_shuffled_all_heads", shuffled[1])
        add_metric(controls, "raw_shuffled_template", shuffled[2])

        centered_normalized = [
            normalize_rows(raw[index] - alignment["key_means"][index].unsqueeze(0))
            for index in range(layer_count)
        ]
        centered_values = transformed_values(centered_normalized, alignment["final_rotations"])
        add_metric(controls, "centered_same_head", adjacent_layer_same_head(centered_values))
        add_metric(controls, "centered_all_heads", adjacent_layer_all_heads(centered_values))
        centered_templates = layer_templates(centered_values)
        add_metric(controls, "centered_template", adjacent_template_cosine(centered_templates))
        centered_shuffled = shuffled_adjacent_metrics(centered_values, args.shuffle_offset)
        add_metric(controls, "centered_shuffled_same_head", centered_shuffled[0])
        add_metric(controls, "centered_shuffled_all_heads", centered_shuffled[1])
        add_metric(controls, "centered_shuffled_template", centered_shuffled[2])
        if chunk_count % 16 == 0:
            print(
                f"evaluated chunks={chunk_count} tokens={token_count} sequence={sequence}",
                flush=True,
            )

    methods = {}
    for method, rotations in methods_rotations.items():
        state = states[method]
        before = mean_metric(state, "adjacent_token_before")
        after = mean_metric(state, "adjacent_token_after")
        methods[method] = {
            "within_layer_cross_head_cosine": mean_metric(state, "within_layer_cross_head"),
            "adjacent_layer_same_head_cosine": mean_metric(state, "adjacent_layer_same_head"),
            "adjacent_layer_all_heads_cosine": mean_metric(state, "adjacent_layer_all_heads"),
            "adjacent_layer_template_cosine": mean_metric(state, "adjacent_layer_template"),
            "all_layer_template_pairwise_cosine": mean_metric(state, "all_layer_template"),
            "within_head_adjacent_token_cosine_before": before,
            "within_head_adjacent_token_cosine_after": after,
            "adjacent_token_cosine_invariance_error": abs(after - before),
            "orthogonality_max_error": orthogonality_max_error(rotations),
            "quantization": {
                quantizer: finalize_quant_state(quant_states[method][quantizer])
                for quantizer in args.quantizers
            },
        }
    control_summary = {name: mean_metric(controls, name) for name in controls}
    final_same = methods["head_layer"]["adjacent_layer_same_head_cosine"]
    final_all = methods["head_layer"]["adjacent_layer_all_heads_cosine"]
    final_template = methods["head_layer"]["adjacent_layer_template_cosine"]
    control_summary.update(
        {
            "raw_same_token_excess_over_shuffle_same_head": final_same - control_summary["raw_shuffled_same_head"],
            "raw_same_token_excess_over_shuffle_all_heads": final_all - control_summary["raw_shuffled_all_heads"],
            "raw_same_token_excess_over_shuffle_template": final_template - control_summary["raw_shuffled_template"],
            "centered_same_token_excess_over_shuffle_same_head": control_summary["centered_same_head"] - control_summary["centered_shuffled_same_head"],
            "centered_same_token_excess_over_shuffle_all_heads": control_summary["centered_all_heads"] - control_summary["centered_shuffled_all_heads"],
            "centered_same_token_excess_over_shuffle_template": control_summary["centered_template"] - control_summary["centered_shuffled_template"],
        }
    )
    pair_details_summary = {}
    for method, rows in pair_accumulators.items():
        pair_details_summary[method] = [
            {
                "left_layer": row["left_layer"],
                "right_layer": row["right_layer"],
                "same_head_cosine": row["same_head_sum"] / max(row["same_head_count"], 1),
                "all_head_cosine": row["all_head_sum"] / max(row["all_head_count"], 1),
                "template_cosine": row["template_sum"] / max(row["template_count"], 1),
            }
            for row in rows
        ]
    summary = {
        "format": "qwen35-cross-layer-orthogonal-alignment-evaluation-v2",
        "manifests": [str(source.manifest.path) for source in sources],
        "rotations": str(args.rotations),
        "layers": args.layers,
        "head_indices": saved_heads,
        "head_dim": dim,
        "token_count": token_count,
        "quantizers": args.quantizers,
        "zip_level": args.zip_level,
        "zip_max_tokens": args.zip_max_tokens,
        "methods": methods,
        "controls": control_summary,
        "adjacent_layer_pair_details": pair_details_summary,
    }
    output = args.output_dir / "cross_layer_alignment_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
