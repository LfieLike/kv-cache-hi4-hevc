#!/usr/bin/env python3
"""Evaluate the selected 32-view FP4 + lossless-H.265 KV-cache codec."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch

from best_codec_core import QuantizerConfig, evaluate_slabs, load_final_rotations
from codec_exact_h265 import evaluate_lossless_x265
from codec_rotation_data import available_sequence_indices, collect_codec_slabs
from cross_layer_data import resolve_layer_sources
from evaluate_h265_cross_layer_codec import require_ffmpeg
from x265_defaults import add_x265_cli


LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--rotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    parser.add_argument("--sequence-indices", type=int, nargs="*")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--frame-height", type=int, default=2048)
    parser.add_argument(
        "--max-slabs",
        type=int,
        default=0,
        help="global slab cap for a smoke test; 0 evaluates all selected data",
    )
    parser.add_argument(
        "--include-aggressive-block256",
        action="store_true",
        help="also test one BF16 scale per layer and 256-token block",
    )
    parser.add_argument("--quantizer-temperature", type=float, default=0.02)
    parser.add_argument("--proxy-block-height", type=int, default=16)
    parser.add_argument("--proxy-block-width", type=int, default=16)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    add_x265_cli(parser)
    parser.add_argument(
        "--x265-extra-params",
        nargs="*",
        default=[],
        help=(
            "optional raw x265 key=value items, e.g. pools=12 "
            "frame-threads=4 wpp=1"
        ),
    )
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.frame_height <= 0:
        raise ValueError("frame-height must be positive")
    if args.max_sequences < 0 or args.max_slabs < 0:
        raise ValueError("max-sequences and max-slabs must be non-negative")
    require_ffmpeg(args.ffmpeg)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")

    sources = resolve_layer_sources(args.manifests, args.layers)
    sequences = available_sequence_indices(sources)
    if args.sequence_indices is not None:
        sequences = list(dict.fromkeys(args.sequence_indices))
    elif args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise ValueError("no evaluation sequences selected")

    rotations, saved_layers, saved_heads = load_final_rotations(
        args.rotations, device=device
    )
    expected_heads = list(sources[0].manifest.head_indices)
    if saved_layers != args.layers:
        raise ValueError(
            f"rotation layer mismatch: saved={saved_layers}, requested={args.layers}"
        )
    if saved_heads != expected_heads:
        raise ValueError(
            f"rotation head mismatch: saved={saved_heads}, data={expected_heads}"
        )

    slabs = collect_codec_slabs(
        sources,
        sequence_indices=sequences,
        frame_height=args.frame_height,
        max_slabs=args.max_slabs,
    )
    configurations = [
        QuantizerConfig(
            label="balanced_token_layer",
            scale_granularity="token-layer",
        )
    ]
    if args.include_aggressive_block256:
        configurations.append(
            QuantizerConfig(
                label="aggressive_block256_layer",
                scale_granularity="block-layer",
                scale_block_tokens=256,
            )
        )

    rows = []
    for config in configurations:
        metrics, codes = evaluate_slabs(
            slabs,
            rotations,
            config=config,
            temperature=args.quantizer_temperature,
            proxy_block_height=args.proxy_block_height,
            proxy_block_width=args.proxy_block_width,
            device=device,
            need_codes=True,
        )
        exact = evaluate_lossless_x265(
            label=config.label,
            code_slabs=codes,
            output_dir=args.output_dir / "exact" / config.label,
            metadata_bytes=int(metrics["scale_bytes"]),
            ffmpeg=args.ffmpeg,
            preset=args.x265_preset,
            bframes=args.x265_bframes,
            b_adapt=args.x265_b_adapt,
            frame_layout="head-view",
            extra_x265_params=tuple(args.x265_extra_params),
            keep_raw=args.keep_raw,
        )
        row = {
            "label": config.label,
            "scale_granularity": config.scale_granularity,
            "scale_block_tokens": config.scale_block_tokens,
            "scale_count": int(metrics["scale_count"]),
            "proxy_bpv": metrics["proxy_bpv"],
            "qk_nmse": metrics["qk_nmse"],
            "key_nmse": metrics["key_nmse"],
            **exact.as_dict(),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del codes
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    balanced = rows[0]
    for row in rows:
        row["total_byte_reduction_vs_balanced"] = (
            balanced["total_bytes"] - row["total_bytes"]
        ) / balanced["total_bytes"]
        row["qk_nmse_change_vs_balanced"] = (
            row["qk_nmse"] - balanced["qk_nmse"]
        ) / balanced["qk_nmse"]
        row["key_nmse_change_vs_balanced"] = (
            row["key_nmse"] - balanced["key_nmse"]
        ) / balanced["key_nmse"]

    summary = {
        "format": "qwen35-best-h265-fp4-32view-evaluation-v1",
        "method": {
            "rotation": "hierarchical-head-layer-orthogonal-procrustes",
            "quantizer": "fp4-e2m1-hard-forward",
            "symbol_mapping": "monotonic-reversible",
            "frame_layout": "canonical-layer-major-head-minor-32-view",
            "codec": "libx265-lossless-gray8-inter-gop",
            "gop": "one complete 32-view slab",
        },
        "manifests": [str(path.expanduser().resolve()) for path in args.manifests],
        "rotation_file": str(args.rotations.expanduser().resolve()),
        "layers": args.layers,
        "heads": expected_heads,
        "sequences": sequences,
        "frame_height": args.frame_height,
        "slabs": len(slabs),
        "x265": {
            "preset": args.x265_preset,
            "bframes": args.x265_bframes,
            "b_adapt": args.x265_b_adapt,
            "extra_params": args.x265_extra_params,
        },
        "rows": rows,
    }
    summary_path = args.output_dir / "best_h265_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_path = args.output_dir / "best_h265_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        "\nlabel total_MiB bpv BF16_ratio QK_NMSE Key_NMSE "
        "bytes_vs_balanced QK_change Key_change"
    )
    for row in rows:
        print(
            f'{row["label"]:28s} '
            f'{row["total_bytes"] / (1 << 20):9.3f} '
            f'{row["bits_per_value"]:6.3f} '
            f'{row["ratio_vs_bf16"]:8.3f}x '
            f'{row["qk_nmse"]:.8f} '
            f'{row["key_nmse"]:.8f} '
            f'{100 * row["total_byte_reduction_vs_balanced"]:9.3f}% '
            f'{100 * row["qk_nmse_change_vs_balanced"]:9.3f}% '
            f'{100 * row["key_nmse_change_vs_balanced"]:9.3f}%'
        )
    print(f"\nsummary: {summary_path}")
    print(f"csv:     {csv_path}")


if __name__ == "__main__":
    main()
