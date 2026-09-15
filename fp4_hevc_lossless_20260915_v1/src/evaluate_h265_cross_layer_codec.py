#!/usr/bin/env python3
"""Encode rotated low-bit Key planes with lossless H.265 and verify round trips."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from cross_layer_data import iter_synchronized_key_chunks, resolve_layer_sources
from evaluate_cross_layer_alignment import (
    QUANTIZER_STORAGE,
    load_alignment,
    quantize,
    transformed_values,
)
from fp8_byte_layouts import fp8_monotonic_encode
from x265_defaults import (
    X265_DEFAULT_B_ADAPT,
    X265_DEFAULT_BFRAMES,
    X265_DEFAULT_PRESET,
    X265_MAX_BFRAMES,
    ensure_lookahead,
    resolve_b_adapt,
)


E2M1_MONOTONIC_REMAP = torch.tensor(
    [8, 9, 10, 11, 12, 13, 14, 15, 7, 6, 5, 4, 3, 2, 1, 0],
    dtype=torch.uint8,
)

VARIANTS = {
    "fp8_native": {"quantizer": "fp8_e4m3fn", "remap": "identity"},
    "fp8_monotonic": {
        "quantizer": "fp8_e4m3fn",
        "remap": "fp8_monotonic",
    },
    "fp4_native": {"quantizer": "fp4_e2m1", "remap": "identity"},
    "fp4_monotonic": {"quantizer": "fp4_e2m1", "remap": "e2m1_monotonic"},
    "affine_int4": {"quantizer": "affine_int4", "remap": "identity"},
    "affine_int2": {"quantizer": "affine_int2", "remap": "identity"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--rotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("identity", "head_only", "head_layer"),
        default=["identity", "head_only", "head_layer"],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=["fp4_native", "fp4_monotonic", "affine_int4"],
    )
    parser.add_argument(
        "--controls",
        nargs="+",
        choices=("aligned", "layer_shift"),
        default=["aligned"],
        help=(
            "aligned keeps same-token layer correspondence; layer_shift applies "
            "a different cyclic token-row shift to every layer inside each frame"
        ),
    )
    parser.add_argument("--max-sequences", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--frame-height", type=int, default=1024)
    parser.add_argument(
        "--row-layout",
        choices=("channel-interleaved", "head-concat"),
        default="channel-interleaved",
        help=(
            "channel-interleaved writes ch0/head0..H then ch1/...; "
            "head-concat writes all D channels of head0, then head1, ..."
        ),
    )
    parser.add_argument("--preset", default=X265_DEFAULT_PRESET)
    parser.add_argument("--bframes", type=int, default=X265_DEFAULT_BFRAMES)
    parser.add_argument("--b-adapt", type=int, default=X265_DEFAULT_B_ADAPT)
    parser.add_argument("--shuffle-seed", type=int, default=1234)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def remap_codes(codes: torch.Tensor, remap: str) -> torch.Tensor:
    if remap == "identity":
        return codes
    if remap == "e2m1_monotonic":
        table = E2M1_MONOTONIC_REMAP.to(codes.device)
        return table[codes.long()]
    if remap == "fp8_monotonic":
        return fp8_monotonic_encode(codes)
    raise ValueError(f"unknown code remap: {remap}")


def write_frame(handle, codes: torch.Tensor, *, row_layout: str) -> None:
    """Write one gray8 frame with one token per row."""
    if codes.ndim != 3:
        raise ValueError("codes must be [token,head,channel]")
    if row_layout == "channel-interleaved":
        plane = codes.permute(0, 2, 1).contiguous()
    elif row_layout == "head-concat":
        plane = codes.contiguous()
    else:
        raise ValueError(f"unknown row_layout={row_layout!r}")
    plane = plane.reshape(codes.shape[0], -1).to(
        device="cpu", dtype=torch.uint8
    )
    handle.write(plane.numpy().tobytes())


def make_layer_shifts(layer_count: int, frame_height: int, seed: int) -> list[int]:
    if layer_count > frame_height:
        raise ValueError("layer count cannot exceed frame height for unique controls")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sampled = torch.randperm(frame_height, generator=generator)[:layer_count]
    sampled = (sampled - sampled[0]) % frame_height
    return [int(value) for value in sampled]


def control_codes(
    codes: torch.Tensor,
    *,
    control: str,
    layer_index: int,
    layer_shifts: list[int],
) -> torch.Tensor:
    if control == "aligned":
        return codes
    if control == "layer_shift":
        return torch.roll(codes, shifts=layer_shifts[layer_index], dims=0)
    raise ValueError(control)


_FFMPEG_OK: set[str] = set()


def require_ffmpeg(ffmpeg: str) -> None:
    if ffmpeg in _FFMPEG_OK:
        return
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise RuntimeError(f"ffmpeg executable not found: {ffmpeg}")
    result = subprocess.run(
        [executable, "-hide_banner", "-encoders"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if "libx265" not in result.stdout:
        raise RuntimeError("this ffmpeg build does not expose the libx265 encoder")
    _FFMPEG_OK.add(ffmpeg)


def ffmpeg_raw_pix_fmt(pixel_format: str) -> str:
    """ffmpeg ``yuv420p`` is MPEG TV range (16-235).

    INT8 codes use 0-255. JPEG ``yuvj420p`` is the same I420 layout without
    that conversion. ``range=full`` on x265 makes a ``yuv420p`` dump worse.
    """
    if pixel_format in {"yuv420p", "yuvj420p"}:
        return "yuvj420p"
    return pixel_format


def _color_range_args(pixel_format: str) -> list[str]:
    if ffmpeg_raw_pix_fmt(pixel_format) == "yuvj420p":
        return ["-color_range", "pc"]
    return []


def _i420_mismatch_detail(
    raw: bytes,
    decoded: bytes,
    *,
    width: int,
    height: int,
) -> str:
    if len(raw) != len(decoded):
        return f"raw={len(raw)} decoded={len(decoded)}"
    y_bytes = width * height
    uv_bytes = (width // 2) * (height // 2)
    frame_bytes = y_bytes + 2 * uv_bytes
    first = -1
    n_diff = 0
    max_abs = 0
    plane_hits = {"Y": 0, "U": 0, "V": 0}
    for index, (left, right) in enumerate(zip(raw, decoded)):
        if left == right:
            continue
        n_diff += 1
        delta = abs(left - right)
        if delta > max_abs:
            max_abs = delta
        if first < 0:
            first = index
        offset = index % frame_bytes
        if offset < y_bytes:
            plane_hits["Y"] += 1
        elif offset < y_bytes + uv_bytes:
            plane_hits["U"] += 1
        else:
            plane_hits["V"] += 1
    return (
        f"raw={len(raw)} decoded={len(decoded)} differ={n_diff} "
        f"first={first} max_abs={max_abs} planes={plane_hits}"
    )


def x265_params(
    mode: str,
    gop_size: int,
    *,
    bframes: int = X265_DEFAULT_BFRAMES,
    b_adapt: int = X265_DEFAULT_B_ADAPT,
    extra_params: tuple[str, ...] = (),
) -> str:
    if bframes < 0:
        raise ValueError("bframes must be non-negative")
    if bframes > X265_MAX_BFRAMES:
        raise ValueError(
            f"x265 --bframes is 0..{X265_MAX_BFRAMES} consecutive B-frames, "
            f"not GOP length; got {bframes}"
        )
    b_adapt = resolve_b_adapt(bframes, b_adapt)
    extra = ensure_lookahead(tuple(extra_params), bframes)
    common = [
        "lossless=1",
        "scenecut=0",
        f"bframes={bframes}",
        f"b-adapt={b_adapt}",
        "open-gop=0",
        "log-level=error",
    ]
    if mode == "all_intra":
        common.extend(["keyint=1", "min-keyint=1"])
    elif mode == "inter_gop":
        common.extend([f"keyint={gop_size}", f"min-keyint={gop_size}"])
    else:
        raise ValueError(mode)
    common.extend(extra)
    return ":".join(common)


_PARALLEL_DROP_KEYS = {"pools", "frame-threads", "numa-pools"}


def sanitize_x265_params_for_parallel(
    extra_params: tuple[str, ...],
) -> tuple[str, ...]:
    """One ffmpeg per closed GOP: do not also spawn x265 thread pools."""
    kept: list[str] = []
    for item in extra_params:
        key = item.split("=", 1)[0]
        if key not in _PARALLEL_DROP_KEYS:
            kept.append(item)
    return tuple(kept) + ("pools=1", "frame-threads=1")


def resolve_encode_workers(requested: int, gop_count: int) -> int:
    if gop_count <= 1:
        return 1
    if requested == 0:
        requested = os.cpu_count() or 8
    if requested < 0:
        raise ValueError("encode_workers must be >= 0")
    return max(1, min(int(requested), gop_count))


def _run_one_gop(
    *,
    ffmpeg: str,
    payload: bytes,
    hevc_path: Path,
    width: int,
    height: int,
    gop_size: int,
    preset: str,
    mode: str,
    bframes: int,
    b_adapt: int,
    pixel_format: str,
    extra_x265_params: tuple[str, ...],
    attempts: int = 3,
) -> int:
    """Encode one GOP to memory, then verify the file round-trip.

    Writing ``-f hevc`` straight to a path can return 0 with an empty file
    under parallel ffmpeg load. The training encode-only path already uses
    ``pipe:1``; eval now does the same and retries empty or truncated GOPs.
    """
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    pix = ffmpeg_raw_pix_fmt(pixel_format)
    range_args = _color_range_args(pixel_format)
    encode_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix,
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        "1",
        *range_args,
        "-i",
        "pipe:0",
        "-frames:v",
        str(gop_size),
        "-an",
        "-c:v",
        "libx265",
        "-preset",
        preset,
        "-x265-params",
        x265_params(
            mode,
            gop_size,
            bframes=bframes,
            b_adapt=b_adapt,
            extra_params=extra_x265_params,
        ),
        "-pix_fmt",
        pix,
        *range_args,
        "-f",
        "hevc",
        "pipe:1",
    ]
    decode_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "hevc",
        *range_args,
        "-i",
        str(hevc_path),
        "-frames:v",
        str(gop_size),
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix,
        *range_args,
        "pipe:1",
    ]
    last_error = "x265 GOP encode failed"
    for attempt in range(1, attempts + 1):
        hevc_path.unlink(missing_ok=True)
        encode_result = subprocess.run(
            encode_command,
            check=False,
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        bitstream = encode_result.stdout
        if encode_result.returncode != 0 or not bitstream:
            last_error = (
                f"x265 GOP encode failed for {hevc_path.name} "
                f"(attempt {attempt}/{attempts}): "
                f"returncode={encode_result.returncode} stdout={len(bitstream)} bytes\n"
                f"command={shlex.join(encode_command)}\n"
                f"{encode_result.stderr.decode('utf-8', errors='replace') or '<empty>'}"
            )
            continue
        hevc_path.write_bytes(bitstream)
        decode_result = subprocess.run(
            decode_command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if decode_result.returncode != 0:
            last_error = (
                f"HEVC GOP decode failed for {hevc_path.name} "
                f"(attempt {attempt}/{attempts}): "
                f"returncode={decode_result.returncode} hevc={len(bitstream)} bytes\n"
                f"{decode_result.stderr.decode('utf-8', errors='replace') or '<empty>'}"
            )
            continue
        if decode_result.stdout != payload:
            detail = (
                f"raw={len(payload)} decoded={len(decode_result.stdout)} "
                f"hevc={len(bitstream)}"
            )
            if ffmpeg_raw_pix_fmt(pixel_format) == "yuvj420p":
                detail = _i420_mismatch_detail(
                    payload,
                    decode_result.stdout,
                    width=width,
                    height=height,
                ) + f" hevc={len(bitstream)} pix={pix}"
            encode_err = encode_result.stderr.decode("utf-8", errors="replace")
            decode_err = decode_result.stderr.decode("utf-8", errors="replace")
            last_error = (
                f"lossless round-trip failed for {hevc_path.name} "
                f"(attempt {attempt}/{attempts}): {detail}\n"
                f"command={shlex.join(encode_command)}\n"
                f"encode_stderr={encode_err or '<empty>'}\n"
                f"decode_stderr={decode_err or '<empty>'}"
            )
            continue
        return len(bitstream)
    raise RuntimeError(last_error)


def encode_gray8_payload_size(
    payload: bytes,
    *,
    ffmpeg: str,
    width: int,
    height: int,
    frames: int,
    preset: str,
    mode: str,
    gop_size: int,
    bframes: int = X265_DEFAULT_BFRAMES,
    b_adapt: int = X265_DEFAULT_B_ADAPT,
    extra_x265_params: tuple[str, ...] = (),
    pixel_format: str = "gray",
) -> int:
    """Encode one GOP from memory and return bitstream bytes.

    Training calls this thousands of times. It does not decode: lossless
    round-trip stays on the Split-B eval path.
    """
    if frames <= 0 or width <= 0 or height <= 0:
        raise ValueError("frame geometry must be positive")
    if gop_size <= 0:
        raise ValueError("gop_size must be positive")
    if pixel_format in ("gray", "gray8"):
        expected = frames * width * height
    elif pixel_format in ("yuv420p", "yuvj420p"):
        if width % 2 or height % 2:
            raise ValueError("yuv420p needs even width and height")
        expected = frames * (width * height + 2 * (width // 2) * (height // 2))
    elif pixel_format == "yuv444p":
        expected = frames * width * height * 3
    else:
        raise ValueError(f"unsupported pixel_format {pixel_format}")
    if len(payload) != expected:
        raise ValueError(
            f"payload is {len(payload)} bytes, expected {expected} "
            f"for {frames} {width}x{height} {pixel_format} frames"
        )
    extra = sanitize_x265_params_for_parallel(extra_x265_params)
    pix = ffmpeg_raw_pix_fmt(pixel_format)
    range_args = _color_range_args(pixel_format)
    encode_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix,
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        "1",
        *range_args,
        "-i",
        "pipe:0",
        "-frames:v",
        str(frames),
        "-an",
        "-c:v",
        "libx265",
        "-preset",
        preset,
        "-x265-params",
        x265_params(
            mode,
            gop_size,
            bframes=bframes,
            b_adapt=b_adapt,
            extra_params=extra,
        ),
        "-pix_fmt",
        pix,
        *range_args,
        "-f",
        "hevc",
        "pipe:1",
    ]
    result = subprocess.run(
        encode_command,
        check=False,
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace") or "<empty>"
        raise RuntimeError(
            f"x265 encode-only failed: returncode={result.returncode} "
            f"stdout={len(result.stdout)} bytes\n"
            f"command={shlex.join(encode_command)}\n{detail}"
        )
    return len(result.stdout)


def encode_and_verify_parallel_gops(
    *,
    ffmpeg: str,
    raw_path: Path,
    bitstream_path: Path,
    decoded_path: Path,
    width: int,
    height: int,
    frames: int,
    preset: str,
    mode: str,
    gop_size: int,
    bframes: int = X265_DEFAULT_BFRAMES,
    b_adapt: int = X265_DEFAULT_B_ADAPT,
    pixel_format: str = "gray",
    extra_x265_params: tuple[str, ...] = (),
    encode_workers: int = 0,
    keep_decoded: bool = False,
) -> dict:
    """Encode each closed GOP as its own ffmpeg job.

    Independent GOPs are the producer-consumer unit (32 views / 256 tokens in
    gray8, or 768 tokens in the YUV444 3-block layout). CABAC does not warm
    across ``open-gop=0``, so the summed payload matches one long file aside
    from repeated headers. x265 thread pools are stripped so N workers do not
    oversubscribe.
    """
    if gop_size <= 0 or frames % gop_size:
        raise ValueError(
            f"parallel GOP encode needs frames={frames} divisible by gop_size={gop_size}"
        )
    gop_count = frames // gop_size
    workers = resolve_encode_workers(encode_workers, gop_count)
    extra = sanitize_x265_params_for_parallel(extra_x265_params)
    b_adapt = resolve_b_adapt(bframes, b_adapt)
    if pixel_format not in {"gray", "gray8"}:
        raise ValueError(
            f"parallel GOP encode currently supports gray8, not {pixel_format!r}"
        )
    frame_bytes = width * height
    gop_bytes = gop_size * frame_bytes
    raw_size = raw_path.stat().st_size
    if raw_size != frames * frame_bytes:
        raise RuntimeError("raw gray8 byte count does not match frame geometry")
    gop_dir = bitstream_path.parent / f"{bitstream_path.stem}.gops"
    gop_dir.mkdir(parents=True, exist_ok=True)

    def job(index: int) -> tuple[int, Path]:
        hevc_path = gop_dir / f"gop{index:04d}.hevc"
        with raw_path.open("rb") as handle:
            handle.seek(index * gop_bytes)
            payload = handle.read(gop_bytes)
        if len(payload) != gop_bytes:
            raise RuntimeError(f"short GOP read at index={index}")
        size = _run_one_gop(
            ffmpeg=ffmpeg,
            payload=payload,
            hevc_path=hevc_path,
            width=width,
            height=height,
            gop_size=gop_size,
            preset=preset,
            mode=mode,
            bframes=bframes,
            b_adapt=b_adapt,
            pixel_format=pixel_format,
            extra_x265_params=extra,
        )
        return size, hevc_path

    print(
        f"parallel GOP encode: {gop_count} gops, {workers} workers, "
        f"{gop_size} frames/gop, {width}x{height} {pixel_format}",
        flush=True,
    )
    encode_started = time.perf_counter()
    sizes = [0] * gop_count
    paths: list[Path] = [Path()] * gop_count
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, index): index for index in range(gop_count)}
        for future in as_completed(futures):
            index = futures[future]
            size, path = future.result()
            sizes[index] = size
            paths[index] = path
    encode_seconds = time.perf_counter() - encode_started
    bitstream_path.parent.mkdir(parents=True, exist_ok=True)
    with bitstream_path.open("wb") as handle:
        for path in paths:
            handle.write(path.read_bytes())
            path.unlink(missing_ok=True)
    shutil.rmtree(gop_dir, ignore_errors=True)
    raw_hash = file_sha256(raw_path)
    decoded_hash = raw_hash
    if keep_decoded:
        shutil.copyfile(raw_path, decoded_path)
        decoded_hash = file_sha256(decoded_path)
    print(
        f"parallel GOP encode done in {encode_seconds:.1f}s "
        f"({gop_count / max(encode_seconds, 1e-9):.2f} gop/s)",
        flush=True,
    )
    return {
        "mode": mode,
        "codec": "libx265-lossless",
        "pixel_format": pixel_format,
        "preset": preset,
        "bframes": bframes,
        "b_adapt": b_adapt,
        "x265_params": x265_params(
            mode,
            gop_size,
            bframes=bframes,
            b_adapt=b_adapt,
            extra_params=extra,
        ),
        "encode_seconds": encode_seconds,
        "decode_seconds": 0.0,
        "bitstream_bytes": int(sum(sizes)),
        "raw_gray8_bytes": raw_size,
        "roundtrip_exact": True,
        "raw_sha256": raw_hash,
        "decoded_sha256": decoded_hash,
        "encode_workers": workers,
        "gop_count": gop_count,
    }


def encode_and_verify(
    *,
    ffmpeg: str,
    raw_path: Path,
    bitstream_path: Path,
    decoded_path: Path,
    width: int,
    height: int,
    frames: int,
    preset: str,
    mode: str,
    gop_size: int,
    bframes: int = X265_DEFAULT_BFRAMES,
    b_adapt: int = X265_DEFAULT_B_ADAPT,
    pixel_format: str = "gray",
    extra_x265_params: tuple[str, ...] = (),
    encode_workers: int = 0,
) -> dict:
    b_adapt = resolve_b_adapt(bframes, b_adapt)
    gop_count = frames // gop_size if gop_size > 0 else 1
    workers = resolve_encode_workers(
        encode_workers, gop_count if frames >= 2 * gop_size else 1
    )
    if (
        workers > 1
        and pixel_format in {"gray", "gray8"}
        and frames >= 2 * gop_size
        and frames % gop_size == 0
    ):
        return encode_and_verify_parallel_gops(
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
            pixel_format=pixel_format,
            extra_x265_params=extra_x265_params,
            encode_workers=workers,
            keep_decoded=False,
        )
    encode_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pixel_format,
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        "1",
        "-i",
        str(raw_path),
        "-frames:v",
        str(frames),
        "-an",
        "-c:v",
        "libx265",
        "-preset",
        preset,
        "-x265-params",
        x265_params(
            mode,
            gop_size,
            bframes=bframes,
            b_adapt=b_adapt,
            extra_params=extra_x265_params,
        ),
        "-pix_fmt",
        pixel_format,
        "-f",
        "hevc",
        str(bitstream_path),
    ]
    encode_started = time.perf_counter()
    encode_result = subprocess.run(
        encode_command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    encode_seconds = time.perf_counter() - encode_started
    if encode_result.returncode != 0:
        raise RuntimeError(
            f"x265 encode failed for {bitstream_path.name}: "
            f"returncode={encode_result.returncode}\n"
            f"command={shlex.join(encode_command)}\n"
            f"raw_bytes={raw_path.stat().st_size if raw_path.exists() else 'missing'} "
            f"width={width} height={height} frames={frames} "
            f"mode={mode} gop={gop_size}\n"
            f"output={encode_result.stdout or '<empty>'}"
        )
    decode_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(bitstream_path),
        "-frames:v",
        str(frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        pixel_format,
        str(decoded_path),
    ]
    decode_started = time.perf_counter()
    decode_result = subprocess.run(
        decode_command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    decode_seconds = time.perf_counter() - decode_started
    if decode_result.returncode != 0:
        raise RuntimeError(
            f"HEVC decode failed for {bitstream_path.name}: "
            f"returncode={decode_result.returncode}\n"
            f"command={shlex.join(decode_command)}\n"
            f"output={decode_result.stdout or '<empty>'}"
        )
    raw_size = raw_path.stat().st_size
    decoded_size = decoded_path.stat().st_size
    raw_hash = file_sha256(raw_path)
    decoded_hash = file_sha256(decoded_path)
    if raw_size != decoded_size or raw_hash != decoded_hash:
        raise RuntimeError(
            f"lossless round-trip failed for {bitstream_path.name}: "
            f"raw={raw_size}/{raw_hash}, decoded={decoded_size}/{decoded_hash}"
        )
    return {
        "mode": mode,
        "codec": "libx265-lossless",
        "pixel_format": pixel_format,
        "preset": preset,
        "bframes": bframes,
        "b_adapt": b_adapt,
        "x265_params": x265_params(
            mode,
            gop_size,
            bframes=bframes,
            b_adapt=b_adapt,
            extra_params=extra_x265_params,
        ),
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "bitstream_bytes": bitstream_path.stat().st_size,
        "raw_gray8_bytes": raw_size,
        "roundtrip_exact": True,
        "raw_sha256": raw_hash,
        "decoded_sha256": decoded_hash,
        "encode_workers": 1,
        "gop_count": gop_count,
    }


def method_rotations(alignment: dict[str, torch.Tensor], layers: int, heads: int, dim: int, device: torch.device) -> dict[str, torch.Tensor]:
    identity = torch.eye(dim, device=device).expand(layers, heads, dim, dim).clone()
    return {
        "identity": identity,
        "head_only": alignment["head_rotations"],
        "head_layer": alignment["final_rotations"],
    }


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be positive")
    if args.frame_height <= 0:
        raise ValueError("frame-height must be positive")
    if args.max_tokens % args.frame_height != 0:
        raise ValueError("max-tokens must be divisible by frame-height")
    require_ffmpeg(args.ffmpeg)
    args.output_dir = args.output_dir.expanduser().resolve()
    raw_dir = args.output_dir / "raw"
    bitstream_dir = args.output_dir / "bitstreams"
    decoded_dir = args.output_dir / "decoded"
    for directory in (args.output_dir, raw_dir, bitstream_dir, decoded_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    sources = resolve_layer_sources(args.manifests, args.layers)
    alignment = load_alignment(args.rotations.expanduser().resolve(), device)
    saved_layers = alignment["layer_indices"].tolist()
    saved_heads = alignment["head_indices"].tolist()
    if saved_layers != args.layers:
        raise ValueError(f"layer mismatch: saved={saved_layers}, requested={args.layers}")
    if saved_heads != list(sources[0].manifest.head_indices):
        raise ValueError("head-index mismatch between rotations and manifests")

    layer_count = len(args.layers)
    heads = len(saved_heads)
    dim = sources[0].manifest.head_dim
    width = heads * dim
    rotations = method_rotations(alignment, layer_count, heads, dim, device)
    selected_rotations = {name: rotations[name] for name in args.methods}
    layer_shifts = make_layer_shifts(
        layer_count, args.frame_height, args.shuffle_seed
    )
    configurations = [
        (method, variant, control)
        for method in args.methods
        for variant in args.variants
        for control in args.controls
    ]
    raw_paths = {
        (method, variant, control): raw_dir
        / f"{method}__{variant}__{control}.gray8.raw"
        for method, variant, control in configurations
    }
    handles = {key: path.open("wb") for key, path in raw_paths.items()}

    pending: list[torch.Tensor] | None = None
    sampled_tokens = 0
    frame_blocks = 0
    try:
        for _, _, chunks_cpu in iter_synchronized_key_chunks(
            sources,
            max_sequences=args.max_sequences,
            token_stride=1,
        ):
            remaining = args.max_tokens - sampled_tokens
            if remaining <= 0:
                break
            chunks_cpu = [chunk[:remaining] for chunk in chunks_cpu]
            if pending is not None:
                chunks_cpu = [
                    torch.cat([pending[index], chunk], dim=0)
                    for index, chunk in enumerate(chunks_cpu)
                ]
            cursor = 0
            while chunks_cpu[0].shape[0] - cursor >= args.frame_height:
                raw = [
                    chunk[cursor : cursor + args.frame_height]
                    .to(device=device, dtype=torch.float32)
                    for chunk in chunks_cpu
                ]
                for method, method_matrix in selected_rotations.items():
                    aligned = transformed_values(raw, method_matrix)
                    quantized_by_name: dict[str, list[torch.Tensor]] = {}
                    for variant in args.variants:
                        quantizer = VARIANTS[variant]["quantizer"]
                        if quantizer not in quantized_by_name:
                            quantized_by_name[quantizer] = [
                                quantize(layer_values, quantizer)[1]
                                for layer_values in aligned
                            ]
                        remap = VARIANTS[variant]["remap"]
                        for layer_index, layer_codes in enumerate(
                            quantized_by_name[quantizer]
                        ):
                            remapped = remap_codes(layer_codes, remap)
                            for control in args.controls:
                                write_frame(
                                    handles[(method, variant, control)],
                                    control_codes(
                                        remapped,
                                        control=control,
                                        layer_index=layer_index,
                                        layer_shifts=layer_shifts,
                                    ),
                                    row_layout=args.row_layout,
                                )
                cursor += args.frame_height
                sampled_tokens += args.frame_height
                frame_blocks += 1
                print(
                    f"prepared tokens={sampled_tokens}/{args.max_tokens} "
                    f"frame_blocks={frame_blocks}",
                    flush=True,
                )
                if sampled_tokens >= args.max_tokens:
                    break
            if sampled_tokens >= args.max_tokens:
                pending = None
                break
            pending = [chunk[cursor:].clone() for chunk in chunks_cpu]
    finally:
        for handle in handles.values():
            handle.close()

    if sampled_tokens != args.max_tokens:
        raise RuntimeError(
            f"requested {args.max_tokens} tokens but prepared {sampled_tokens}; "
            "choose a smaller frame-aligned max-tokens value"
        )
    frames = frame_blocks * layer_count
    expected_raw_size = frames * args.frame_height * width
    for path in raw_paths.values():
        if path.stat().st_size != expected_raw_size:
            raise RuntimeError(
                f"raw frame size mismatch for {path}: "
                f"{path.stat().st_size} != {expected_raw_size}"
            )

    results = []
    for method, variant, control in configurations:
        raw_path = raw_paths[(method, variant, control)]
        variant_spec = VARIANTS[variant]
        quantizer = variant_spec["quantizer"]
        storage = QUANTIZER_STORAGE[quantizer]
        values = sampled_tokens * layer_count * heads * dim
        packed_code_bytes = values * storage["bits_per_code"] // 8
        metadata_bytes = (
            sampled_tokens
            * layer_count
            * storage["metadata_bytes_per_token_layer"]
        )
        source_bf16_bytes = values * 2
        encodings = {}
        for mode in ("all_intra", "inter_gop"):
            stem = f"{method}__{variant}__{control}__{mode}"
            bitstream_path = bitstream_dir / f"{stem}.hevc"
            decoded_path = decoded_dir / f"{stem}.gray8.raw"
            print(f"encoding {stem}", flush=True)
            encoded = encode_and_verify(
                ffmpeg=args.ffmpeg,
                raw_path=raw_path,
                bitstream_path=bitstream_path,
                decoded_path=decoded_path,
                width=width,
                height=args.frame_height,
                frames=frames,
                preset=args.preset,
                mode=mode,
                gop_size=layer_count,
                bframes=args.bframes,
                b_adapt=args.b_adapt,
                encode_workers=1,
            )
            total_bytes = encoded["bitstream_bytes"] + metadata_bytes
            encoded.update(
                {
                    "metadata_bytes_raw": metadata_bytes,
                    "total_bytes_with_raw_metadata": total_bytes,
                    "compression_ratio_vs_bf16": source_bf16_bytes
                    / total_bytes,
                    "effective_bits_per_original_value": total_bytes
                    * 8
                    / values,
                    "gain_vs_packed_quantized_cache": 1.0
                    - total_bytes / (packed_code_bytes + metadata_bytes),
                }
            )
            encodings[mode] = encoded
            if not args.keep_raw:
                decoded_path.unlink()
        inter_bytes = encodings["inter_gop"]["total_bytes_with_raw_metadata"]
        intra_bytes = encodings["all_intra"]["total_bytes_with_raw_metadata"]
        results.append(
            {
                "method": method,
                "variant": variant,
                "control": control,
                "quantizer": quantizer,
                "code_remap": variant_spec["remap"],
                "packed_code_bytes": packed_code_bytes,
                "metadata_bytes_raw": metadata_bytes,
                "source_bf16_bytes": source_bf16_bytes,
                "encodings": encodings,
                "inter_gain_vs_all_intra": 1.0 - inter_bytes / intra_bytes,
            }
        )
        if not args.keep_raw:
            raw_path.unlink()

    summary = {
        "format": "qwen35-cross-layer-lossless-h265-evaluation-v2",
        "manifests": [str(source.manifest.path) for source in sources],
        "rotations": str(args.rotations.expanduser().resolve()),
        "layers": args.layers,
        "head_indices": saved_heads,
        "head_dim": dim,
        "sampled_tokens": sampled_tokens,
        "frame_layout": (
            "block-major,layer-frame,[token,head,channel]"
            if args.row_layout == "head-concat"
            else "block-major,layer-frame,[token,channel,head]"
        ),
        "row_layout": args.row_layout,
        "frame_width": width,
        "frame_height": args.frame_height,
        "frame_blocks": frame_blocks,
        "frame_count": frames,
        "gop_size": layer_count,
        "controls": args.controls,
        "layer_shift_control": {
            "kind": "within-frame-cyclic-token-row-shift-per-layer",
            "seed": args.shuffle_seed,
            "shifts": layer_shifts,
            "preserves": [
                "per-layer-code-histogram",
                "within-frame-cyclic-token-adjacency",
            ],
            "destroys": "same-token-cross-layer-correspondence",
        },
        "pixel_format": "gray8",
        "lossless_roundtrip_required": True,
        "results": results,
    }
    summary_path = args.output_dir / "h265_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = args.output_dir / "h265_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "variant",
                "control",
                "mode",
                "bitstream_bytes",
                "metadata_bytes",
                "total_bytes",
                "ratio_vs_bf16",
                "bits_per_value",
                "gain_vs_packed_cache",
                "inter_gain_vs_intra",
                "roundtrip_exact",
            ]
        )
        for result in results:
            for mode, encoded in result["encodings"].items():
                writer.writerow(
                    [
                        result["method"],
                        result["variant"],
                        result["control"],
                        mode,
                        encoded["bitstream_bytes"],
                        encoded["metadata_bytes_raw"],
                        encoded["total_bytes_with_raw_metadata"],
                        encoded["compression_ratio_vs_bf16"],
                        encoded["effective_bits_per_original_value"],
                        encoded["gain_vs_packed_quantized_cache"],
                        result["inter_gain_vs_all_intra"],
                        encoded["roundtrip_exact"],
                    ]
                )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
