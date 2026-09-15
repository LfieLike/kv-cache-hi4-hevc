"""No-rotation INT8 cache coding: HEVC high nibbles + raw packed low nibbles.

The quantizer and token/view geometry match ``hevc_token``. Only the high
four bits are sent through lossless HEVC; low nibbles are packed two per byte
and stored verbatim in the packet.
"""
import json
import struct

import numpy as np

from paper_bench import hevc_token as base

TOKENS_PER_FRAME = base.TOKENS_PER_FRAME
CODEC_NAME = "uint8 INT8 codes split into HEVC high4 + raw packed low4"
QUANTIZATION_DESCRIPTION = (
    "symmetric INT8; per-view BF16 scales rounded upward; "
    "max(sqrt(12)*epsilon*RMS, amax/127); no clipping"
)
CODEC_DESCRIPTION = (
    "lossless libx265 on high-nibble Y samples; low nibbles packed two per "
    "byte and stored raw; 8 token positions per frame"
)


def quantize(xs, epsilon):
    return base.quantize(xs, epsilon)


def parameters(n, h):
    return base.parameters(n, h)


def run(cmd, data):
    return base.run(cmd, data)


def pack_nibbles(values):
    """Pack uint8 low nibbles; even-index symbol occupies the low half-byte."""
    flat = np.asarray(values, dtype=np.uint8).reshape(-1) & 0x0F
    if flat.size & 1:
        flat = np.pad(flat, (0, 1))
    return (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)


def unpack_nibbles(packed, count):
    packed = np.frombuffer(packed, dtype=np.uint8)
    if packed.size != (count + 1) // 2:
        raise ValueError("Low-nibble payload has the wrong length")
    values = np.empty(packed.size * 2, dtype=np.uint8)
    values[0::2] = packed & 0x0F
    values[1::2] = packed >> 4
    return values[:count]


def encode(xs, model, epsilon, ffmpeg):
    q, scales, floors = quantize(xs, epsilon)
    tokens = q.shape[1]
    high = base.layout(q >> 4, model)
    frames, height, width = high.shape
    raw = np.concatenate(
        (high.reshape(frames, -1), np.full((frames, height * width // 2), 128, np.uint8)),
        axis=1,
    ).tobytes()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "1",
        "-filter_threads", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "-video_size", f"{width}x{height}", "-framerate", "1", "-i", "pipe:0",
        "-frames:v", str(frames), "-an", "-c:v", "libx265", "-preset", "veryfast",
        "-x265-params", parameters(frames, height), "-threads", "1",
        "-pix_fmt", "yuv420p", "-f", "hevc", "pipe:1",
    ]
    hevc = run(cmd, raw)
    low = pack_nibbles(q)
    head = dict(
        format="ours-hi4-token-hevc-v1", model=model, frames=frames,
        height=height, width=width, views=len(scales), groups=[len(x) for x in xs],
        tokens=tokens, tokens_per_frame=TOKENS_PER_FRAME,
        values=int(q.size), low4_bytes=len(low),
        low4_order="first_symbol_in_low_nibble", range_floor_views=floors,
    )
    header = json.dumps(head, separators=(",", ":")).encode()
    scale_bytes = (scales.view(np.uint32) >> 16).astype("<u2").tobytes()
    return struct.pack("<I", len(header)) + header + scale_bytes + low + hevc


def parse(packet):
    header_len, = struct.unpack_from("<I", packet)
    head = json.loads(packet[4:4 + header_len])
    if head.get("format") != "ours-hi4-token-hevc-v1":
        raise ValueError("Wrong high4/HEVC packet format")
    pos = 4 + header_len
    scales = (
        np.frombuffer(packet, dtype="<u2", count=head["views"], offset=pos)
        .astype(np.uint32) << 16
    ).view(np.float32)
    pos += head["views"] * 2
    metadata_end = pos
    low_end = pos + head["low4_bytes"]
    if low_end > len(packet):
        raise ValueError("Truncated low-nibble payload")
    return head, scales, packet[low_end:], metadata_end


def decode(packet, ffmpeg):
    head, scales, hevc, metadata_end = parse(packet)
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "1",
        "-filter_threads", "1", "-f", "hevc", "-i", "pipe:0", "-frames:v",
        str(head["frames"]), "-vf", "extractplanes=y", "-pix_fmt", "gray",
        "-threads", "1", "-f", "rawvideo", "pipe:1",
    ]
    raw = run(cmd, hevc)
    y = np.frombuffer(raw, dtype=np.uint8).reshape(
        head["frames"], head["height"], head["width"]
    )
    high = base.inverse(
        y, head["model"], head["views"], head["tokens"], head["tokens_per_frame"]
    )
    low_start = metadata_end
    low_end = low_start + head["low4_bytes"]
    low = unpack_nibbles(packet[low_start:low_end], head["values"]).reshape(high.shape)
    q = ((high << 4) | low).astype(np.uint8)
    return q, scales
