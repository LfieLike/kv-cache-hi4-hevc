#!/usr/bin/env python3
"""Reversible byte representations and entropy diagnostics for native FP8."""

from __future__ import annotations

import torch


REPRESENTATIONS = (
    "fp8_native",
    "fp8_monotonic",
    "fp8_random_permutation",
    "fp8_bitplane",
)
RANDOM_PERMUTATION_SEED = 20260818


def _random_tables() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(RANDOM_PERMUTATION_SEED)
    forward = torch.randperm(256, generator=generator).to(torch.uint8)
    inverse = torch.empty(256, dtype=torch.uint8)
    inverse[forward.long()] = torch.arange(256, dtype=torch.uint8)
    return forward, inverse


FP8_RANDOM_FORWARD, FP8_RANDOM_INVERSE = _random_tables()


def fp8_monotonic_encode(codes: torch.Tensor) -> torch.Tensor:
    """IEEE-style float flip: numerical order becomes unsigned-byte order.

    E4M3FN finite positive codes are sign-bit flipped.  Negative codes are
    bitwise inverted, reversing their sign-magnitude order.  The transform is
    a bijection over all 256 byte values, including the unused NaN patterns.
    """
    if codes.dtype != torch.uint8:
        raise TypeError("FP8 codes must be uint8")
    sign = (codes & 0x80) != 0
    return torch.where(sign, torch.bitwise_not(codes), codes ^ 0x80)


def fp8_monotonic_decode(encoded: torch.Tensor) -> torch.Tensor:
    if encoded.dtype != torch.uint8:
        raise TypeError("encoded FP8 bytes must be uint8")
    positive_original = (encoded & 0x80) != 0
    return torch.where(
        positive_original,
        encoded ^ 0x80,
        torch.bitwise_not(encoded),
    )


def split_e4m3_sm_exp(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split IEEE E4M3 bytes into sign+M3 (0..15) and exponent (0..15).

    bit7 = sign, bits6-3 = exponent, bits2-0 = mantissa.
    """

    if codes.dtype != torch.uint8:
        raise TypeError("E4M3 codes must be uint8")
    sign = (codes >> 7) & 1
    exp = (codes >> 3) & 0x0F
    mantissa = codes & 0x07
    sm = (sign << 3) | mantissa
    return sm.to(torch.uint8), exp.to(torch.uint8)


def join_e4m3_sm_exp(sm: torch.Tensor, exp: torch.Tensor) -> torch.Tensor:
    if sm.shape != exp.shape:
        raise ValueError("sign+M3 and exponent shapes must match")
    sm = sm.to(torch.uint8)
    exp = exp.to(torch.uint8)
    sign = (sm >> 3) & 1
    mantissa = sm & 0x07
    return ((sign << 7) | ((exp & 0x0F) << 3) | mantissa).to(torch.uint8)


def split_e4m3_high_low(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """IEEE high nibble S|E3|E2|E1 and low nibble E0|M2|M1|M0."""

    if codes.dtype != torch.uint8:
        raise TypeError("E4M3 codes must be uint8")
    return ((codes >> 4) & 0x0F).to(torch.uint8), (codes & 0x0F).to(torch.uint8)


def join_e4m3_high_low(high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    if high.shape != low.shape:
        raise ValueError("high and low nibble shapes must match")
    return (((high.to(torch.uint8) & 0x0F) << 4) | (low.to(torch.uint8) & 0x0F)).to(
        torch.uint8
    )


def high4_monotonic_encode(high: torch.Tensor) -> torch.Tensor:
    """4-bit analog of ``fp8_monotonic_encode`` on S|E3|E2|E1.

    Positives XOR 0x8; negatives are 4-bit bitwise not. Rank 0 is the
    most negative coarse bin, 15 the most positive — the HEVC-friendly
    order of this 16-level field.
    """

    if high.dtype != torch.uint8:
        raise TypeError("high nibble must be uint8")
    sign = (high & 0x08) != 0
    return torch.where(sign, torch.bitwise_not(high) & 0x0F, high ^ 0x08).to(
        torch.uint8
    )


def high4_monotonic_decode(mapped: torch.Tensor) -> torch.Tensor:
    if mapped.dtype != torch.uint8:
        raise TypeError("mapped high nibble must be uint8")
    positive_mapped = (mapped & 0x08) != 0
    return torch.where(
        positive_mapped,
        mapped ^ 0x08,
        torch.bitwise_not(mapped) & 0x0F,
    ).to(torch.uint8)


def high4_to_gray(high: torch.Tensor) -> torch.Tensor:
    """16-level gray8 as values 0..15 (INT8 high-plane, not masked).

    Script 85: x265 on 0–15 is the high4 surface (qs_r0 1.612 bpv).
    ``rank << 4`` (0, 16, …, 240) is ``msb4-masked`` and cost +57–61%
    on INT8; on a less smooth FP8 S|E field the ×16 residual saturates
    lossless x265 near 8 bpv.
    """

    return high4_monotonic_encode(high)


def gray_to_high4(gray: torch.Tensor) -> torch.Tensor:
    return high4_monotonic_decode(gray.to(torch.uint8) & 0x0F)


def sm_to_gray(sm: torch.Tensor) -> torch.Tensor:
    """Map sign+M3 to a 16-level gray8 plane for HEVC.

    Rank 0 = most negative mantissa, 15 = most positive. Stored in the high
    nibble (``rank << 4``), same convention as INT8 MSB-only gray8.
    """

    sm = sm.to(torch.uint8)
    sign = (sm >> 3) & 1
    mantissa = sm & 0x07
    rank = torch.where(sign.bool(), 7 - mantissa, 8 + mantissa)
    return (rank.to(torch.uint8) << 4)


def gray_to_sm(gray: torch.Tensor) -> torch.Tensor:
    rank = (gray.to(torch.uint8) >> 4).to(torch.int16)
    negative = rank < 8
    mantissa = torch.where(negative, 7 - rank, rank - 8)
    sign = negative.to(torch.uint8)
    return ((sign << 3) | (mantissa.to(torch.uint8) & 7)).to(torch.uint8)


def pack_nibbles(codes_0_15: torch.Tensor) -> bytes:
    """Two 4-bit symbols per byte. Same packing as INT8 low-nibble raw."""

    from codec_nvcomp_lz import pack_quantizer_codes

    return pack_quantizer_codes(codes_0_15, "int4-turboquant")


def nibble_bytes(count: int) -> int:
    return (int(count) + 1) // 2


def _bitplane_encode_last_dim(codes: torch.Tensor) -> torch.Tensor:
    """Bit-shuffle the last dimension without changing byte count or shape.

    For a D-byte row, produce eight consecutive bit planes of D/8 packed
    bytes.  This is the standard reversible bit-transpose idea used by
    bitshuffle-style preprocessors, adapted to the codec-visible channel row.
    """
    if codes.dtype != torch.uint8:
        raise TypeError("FP8 codes must be uint8")
    dim = codes.shape[-1]
    if dim % 8:
        raise ValueError(f"last dimension must be divisible by 8, got {dim}")
    shifts = torch.arange(8, device=codes.device, dtype=torch.uint8)
    bits = (codes.unsqueeze(-1) >> shifts) & 1  # [..., D, bit]
    plane_bits = bits.transpose(-2, -1).contiguous()  # [..., bit, D]
    plane_bits = plane_bits.reshape(*codes.shape[:-1], 8, dim // 8, 8)
    weights = (1 << shifts).reshape(*([1] * (plane_bits.ndim - 1)), 8)
    packed = (plane_bits * weights).sum(dim=-1).to(torch.uint8)
    return packed.reshape_as(codes)


def _bitplane_decode_last_dim(encoded: torch.Tensor) -> torch.Tensor:
    if encoded.dtype != torch.uint8:
        raise TypeError("encoded FP8 bytes must be uint8")
    dim = encoded.shape[-1]
    if dim % 8:
        raise ValueError(f"last dimension must be divisible by 8, got {dim}")
    shifts = torch.arange(8, device=encoded.device, dtype=torch.uint8)
    packed = encoded.reshape(*encoded.shape[:-1], 8, dim // 8)
    element_bits = (packed.unsqueeze(-1) >> shifts) & 1
    # [..., source_bit, packed_group, source_element]
    element_bits = element_bits.reshape(*encoded.shape[:-1], 8, dim)
    bits = element_bits.transpose(-2, -1).contiguous()  # [..., D, source_bit]
    weights = (1 << shifts).reshape(*([1] * (bits.ndim - 1)), 8)
    return (bits * weights).sum(dim=-1).to(torch.uint8)


def encode_fp8_bytes(codes: torch.Tensor, representation: str) -> torch.Tensor:
    if representation == "fp8_native":
        return codes
    if representation == "fp8_monotonic":
        return fp8_monotonic_encode(codes)
    if representation == "fp8_random_permutation":
        return FP8_RANDOM_FORWARD.to(codes.device)[codes.long()]
    if representation == "fp8_bitplane":
        return _bitplane_encode_last_dim(codes)
    raise ValueError(f"unknown FP8 byte representation: {representation}")


def decode_fp8_bytes(encoded: torch.Tensor, representation: str) -> torch.Tensor:
    if representation == "fp8_native":
        return encoded
    if representation == "fp8_monotonic":
        return fp8_monotonic_decode(encoded)
    if representation == "fp8_random_permutation":
        return FP8_RANDOM_INVERSE.to(encoded.device)[encoded.long()]
    if representation == "fp8_bitplane":
        return _bitplane_decode_last_dim(encoded)
    raise ValueError(f"unknown FP8 byte representation: {representation}")


def init_byte_diagnostics() -> dict:
    return {
        "symbol_counts": torch.zeros(256, dtype=torch.int64),
        "axes": {
            axis: {
                "joint_counts": torch.zeros(256 * 256, dtype=torch.int64),
                "residual_counts": torch.zeros(511, dtype=torch.int64),
                "absolute_residual_sum": 0,
                "zero_residual_count": 0,
                "pair_count": 0,
            }
            for axis in ("token", "view", "channel")
        },
    }


def _update_pairs(state: dict, left: torch.Tensor, right: torch.Tensor) -> None:
    left_flat = left.reshape(-1).long()
    right_flat = right.reshape(-1).long()
    joint = left_flat * 256 + right_flat
    residual = right_flat - left_flat
    state["joint_counts"] += torch.bincount(
        joint, minlength=256 * 256
    ).cpu()
    state["residual_counts"] += torch.bincount(
        residual + 255, minlength=511
    ).cpu()
    state["absolute_residual_sum"] += int(residual.abs().sum().item())
    state["zero_residual_count"] += int((residual == 0).sum().item())
    state["pair_count"] += residual.numel()


def update_byte_diagnostics(
    state: dict,
    codes_by_layer: list[torch.Tensor],
) -> None:
    """Update diagnostics in canonical layer-major/head-major frame order."""
    if not codes_by_layer:
        raise ValueError("codes_by_layer cannot be empty")
    head_count = codes_by_layer[0].shape[1]
    frames = torch.stack(
        [
            layer_codes[:, head_index]
            for layer_codes in codes_by_layer
            for head_index in range(head_count)
        ],
        dim=0,
    )  # [view, token, channel]
    state["symbol_counts"] += torch.bincount(
        frames.reshape(-1).long(), minlength=256
    ).cpu()
    _update_pairs(state["axes"]["token"], frames[:, :-1], frames[:, 1:])
    _update_pairs(state["axes"]["view"], frames[:-1], frames[1:])
    _update_pairs(state["axes"]["channel"], frames[:, :, :-1], frames[:, :, 1:])


def _entropy(counts: torch.Tensor) -> float:
    total = int(counts.sum().item())
    if total == 0:
        return 0.0
    probabilities = counts.double() / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * probabilities.log2()).sum().item())


def _conditional_entropy(joint_counts: torch.Tensor) -> float:
    joint = joint_counts.reshape(256, 256).double()
    total = float(joint.sum().item())
    if total == 0:
        return 0.0
    row_totals = joint.sum(dim=1, keepdim=True)
    nonzero = joint > 0
    ratios = torch.zeros_like(joint)
    ratios[nonzero] = joint[nonzero] / row_totals.expand_as(joint)[nonzero]
    probabilities = joint[nonzero] / total
    return float(-(probabilities * ratios[nonzero].log2()).sum().item())


def finalize_byte_diagnostics(state: dict) -> dict:
    result = {
        "symbol_entropy_bits": _entropy(state["symbol_counts"]),
        "unique_symbol_count": int((state["symbol_counts"] > 0).sum().item()),
        "sample_count": int(state["symbol_counts"].sum().item()),
        "axes": {},
    }
    for axis, axis_state in state["axes"].items():
        pairs = axis_state["pair_count"]
        result["axes"][axis] = {
            "conditional_entropy_bits": _conditional_entropy(
                axis_state["joint_counts"]
            ),
            "signed_residual_entropy_bits": _entropy(
                axis_state["residual_counts"]
            ),
            "absolute_residual_mean": (
                axis_state["absolute_residual_sum"] / max(pairs, 1)
            ),
            "zero_residual_fraction": (
                axis_state["zero_residual_count"] / max(pairs, 1)
            ),
            "pair_count": pairs,
        }
    return result
