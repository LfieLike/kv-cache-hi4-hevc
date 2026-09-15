#!/usr/bin/env python3
"""Exact lossless-x265 validation helpers for codec-aware rotations."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from evaluate_h265_cross_layer_codec import encode_and_verify, require_ffmpeg
from x265_defaults import X265_DEFAULT_B_ADAPT, X265_DEFAULT_BFRAMES, X265_DEFAULT_PRESET


FRAME_LAYOUTS = ("head-view", "layer-block-head-concat")


@dataclass
class ExactCodecResult:
    label: str
    bitstream_bytes: int
    metadata_bytes: int
    total_bytes: int
    bits_per_value: float
    ratio_vs_bf16: float
    frames: int
    tokens: int
    values: int
    roundtrip_exact: bool
    frame_layout: str
    frame_height: int
    frame_width: int
    gop_size: int
    encode_workers: int = 1
    encode_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "bitstream_bytes": self.bitstream_bytes,
            "metadata_bytes": self.metadata_bytes,
            "total_bytes": self.total_bytes,
            "bits_per_value": self.bits_per_value,
            "ratio_vs_bf16": self.ratio_vs_bf16,
            "frames": self.frames,
            "tokens": self.tokens,
            "values": self.values,
            "roundtrip_exact": self.roundtrip_exact,
            "frame_layout": self.frame_layout,
            "frame_height": self.frame_height,
            "frame_width": self.frame_width,
            "gop_size": self.gop_size,
            "encode_workers": self.encode_workers,
            "encode_seconds": self.encode_seconds,
        }


def write_canonical_views(
    path: Path,
    code_slabs: Iterable[torch.Tensor],
    view_order: Sequence[int] | None = None,
) -> tuple[int, int, int, int, int]:
    """Write `[L,T,H,D]` monotonic codes as L*H gray8 views per slab.

    The temporal order is layer-major then head-minor.  Within each gray frame,
    rows are token positions and columns are channels.

    ``view_order`` replaces that default sequence. IPPP prices one predecessor;
    the default cell (bframes=3) also reads bidirectional refs inside the GOP.
    Which views end up adjacent is still priced by the encoder even though
    every global alignment objective is blind to it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    tokens = 0
    values = 0
    expected_shape = None
    with path.open("wb") as handle:
        for codes in code_slabs:
            if codes.ndim != 4:
                raise ValueError("code slabs must be [layers,tokens,heads,dim]")
            layers, slab_tokens, heads, dim = codes.shape
            shape = (layers, slab_tokens, heads, dim)
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(
                    f"all exact-codec slabs must share one shape: {shape} vs {expected_shape}"
                )
            planes = codes.permute(0, 2, 1, 3).reshape(
                layers * heads, slab_tokens, dim
            )
            if view_order is not None:
                if sorted(view_order) != list(range(layers * heads)):
                    raise ValueError(
                        f"view_order must be a permutation of 0..{layers * heads - 1}"
                    )
                planes = planes[torch.tensor(list(view_order))]
            planes = planes.contiguous().to(device="cpu", dtype=torch.uint8)
            handle.write(planes.numpy().tobytes())
            frames += layers * heads
            tokens += slab_tokens
            values += layers * slab_tokens * heads * dim
    if expected_shape is None:
        raise ValueError("at least one code slab is required")
    layers, frame_height, heads, dim = expected_shape
    return frames, tokens, values, frame_height, dim


def canonical_gray8_payload(codes: torch.Tensor) -> tuple[bytes, int, int, int]:
    """Pack one ``[L,T,H,D]`` INT8 GOP the same way ``write_canonical_views`` does."""
    if codes.ndim != 4:
        raise ValueError("code slabs must be [layers,tokens,heads,dim]")
    planes = codes.permute(0, 2, 1, 3).reshape(
        codes.shape[0] * codes.shape[2], codes.shape[1], codes.shape[3]
    )
    payload = planes.contiguous().to(device="cpu", dtype=torch.uint8).numpy().tobytes()
    _layers, height, _heads, width = codes.shape
    frames = _layers * _heads
    return payload, height, width, frames


def write_layer_block_head_concat(
    path: Path,
    code_slabs: Iterable[torch.Tensor],
    *,
    block_tokens: int,
) -> tuple[int, int, int, int, int, int]:
    """Write block-major, layer-minor ``[token, head*channel]`` frames.

    Each input slab is ``[L,T,H,D]``.  A physical gray8 frame contains one
    layer and one contiguous token block and has shape ``[block_tokens,H*D]``.
    Within a row, every head is contiguous: ``head0[D], head1[D], ...``.
    Frames are emitted as ``block0/layer0..L-1, block1/layer0..L-1, ...`` so
    the complete set of layers for one token block is one causal GOP.
    """
    if block_tokens <= 0:
        raise ValueError("block_tokens must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    tokens = 0
    values = 0
    expected_lhd = None
    with path.open("wb") as handle:
        for codes in code_slabs:
            if codes.ndim != 4:
                raise ValueError("code slabs must be [layers,tokens,heads,dim]")
            layers, slab_tokens, heads, dim = codes.shape
            if slab_tokens % block_tokens:
                raise ValueError(
                    f"slab token count {slab_tokens} must be divisible by "
                    f"block_tokens={block_tokens}"
                )
            lhd = (layers, heads, dim)
            if expected_lhd is None:
                expected_lhd = lhd
            elif lhd != expected_lhd:
                raise ValueError(
                    f"all exact-codec slabs must share [L,H,D]: {lhd} "
                    f"vs {expected_lhd}"
                )
            for start in range(0, slab_tokens, block_tokens):
                stop = start + block_tokens
                # [L,B,H,D] -> [L,B,H*D], preserving head-major rows.
                layer_frames = codes[:, start:stop].contiguous().reshape(
                    layers, block_tokens, heads * dim
                ).to(device="cpu", dtype=torch.uint8)
                handle.write(layer_frames.numpy().tobytes())
                frames += layers
            tokens += slab_tokens
            values += layers * slab_tokens * heads * dim
    if expected_lhd is None:
        raise ValueError("at least one code slab is required")
    layers, heads, dim = expected_lhd
    return frames, tokens, values, block_tokens, heads * dim, layers


def evaluate_lossless_x265(
    *,
    label: str,
    code_slabs: Iterable[torch.Tensor],
    output_dir: Path,
    metadata_bytes: int,
    ffmpeg: str = "ffmpeg",
    preset: str = X265_DEFAULT_PRESET,
    bframes: int = X265_DEFAULT_BFRAMES,
    b_adapt: int = X265_DEFAULT_B_ADAPT,
    frame_layout: str = "head-view",
    frame_block_tokens: int = 256,
    extra_x265_params: tuple[str, ...] = (),
    keep_raw: bool = False,
    view_order: Sequence[int] | None = None,
    encode_workers: int = 0,
    mode: str = "inter_gop",
) -> ExactCodecResult:
    require_ffmpeg(ffmpeg)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{label}.gray8.raw"
    bitstream_path = output_dir / f"{label}.hevc"
    decoded_path = output_dir / f"{label}.decoded.gray8.raw"
    if frame_layout == "head-view":
        frames, tokens, values, height, width = write_canonical_views(
            raw_path, code_slabs, view_order=view_order
        )
        # One GOP is the complete ordered set of layer/head views for one slab.
        gop_size = values // (tokens * width)
    elif frame_layout == "layer-block-head-concat":
        if view_order is not None:
            raise ValueError(
                "view_order only applies to head-view; this layout concatenates "
                "heads inside a frame, so there is no view axis to reorder"
            )
        (
            frames,
            tokens,
            values,
            height,
            width,
            gop_size,
        ) = write_layer_block_head_concat(
            raw_path,
            code_slabs,
            block_tokens=frame_block_tokens,
        )
    else:
        raise ValueError(
            f"unknown frame_layout={frame_layout!r}; choices={FRAME_LAYOUTS}"
        )
    if frames <= 0:
        raise ValueError("no frames were written")
    raw_bytes = raw_path.stat().st_size
    values_per_frame = height * width
    if raw_bytes != frames * values_per_frame:
        raise RuntimeError("raw gray8 byte count does not match frame geometry")
    metrics = encode_and_verify(
        ffmpeg=ffmpeg,
        raw_path=raw_path,
        bitstream_path=bitstream_path,
        decoded_path=decoded_path,
        width=width,
        height=height,
        frames=frames,
        preset=preset,
        mode=mode,
        gop_size=gop_size,
        bframes=bframes,
        b_adapt=b_adapt,
        pixel_format="gray",
        extra_x265_params=extra_x265_params,
        encode_workers=encode_workers,
    )
    total_bytes = int(metrics["bitstream_bytes"]) + int(metadata_bytes)
    result = ExactCodecResult(
        label=label,
        bitstream_bytes=int(metrics["bitstream_bytes"]),
        metadata_bytes=int(metadata_bytes),
        total_bytes=total_bytes,
        bits_per_value=8.0 * total_bytes / values,
        ratio_vs_bf16=2.0 * values / total_bytes,
        frames=frames,
        tokens=tokens,
        values=values,
        roundtrip_exact=bool(metrics["roundtrip_exact"]),
        frame_layout=frame_layout,
        frame_height=height,
        frame_width=width,
        gop_size=gop_size,
        encode_workers=int(metrics.get("encode_workers", 1)),
        encode_seconds=float(metrics.get("encode_seconds", 0.0)),
    )
    if not keep_raw:
        raw_path.unlink(missing_ok=True)
        decoded_path.unlink(missing_ok=True)
    return result


def remove_exact_directory(path: Path) -> None:
    """Remove a disposable exact-validation directory between checkpoints."""
    if path.exists():
        shutil.rmtree(path)
