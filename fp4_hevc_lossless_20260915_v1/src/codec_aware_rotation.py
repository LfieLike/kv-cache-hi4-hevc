#!/usr/bin/env python3
"""Differentiable, exactly orthogonal rotations for codec-aware KV training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


E2M1_MONOTONIC_VALUES = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0,
      0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0],
    dtype=torch.float32,
)
E2M1_BOUNDARIES = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32
)
E2M1_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
E2M1_MONOTONIC_REMAP = torch.tensor(
    [8, 9, 10, 11, 12, 13, 14, 15, 7, 6, 5, 4, 3, 2, 1, 0],
    dtype=torch.uint8,
)

# Two's-complement INT2: q in {-2, -1, 0, +1}. The gray8 storage byte is q+2,
# so code 2 is exact zero and byte order matches signed-value order, the same
# convention as INT8's q+128. Scale is amax/2 so q=-2 reconstructs -amax; the
# positive peak reconstructs at +amax/2 (the usual 2-bit two's-complement tax).
INT2_QMIN = -2
INT2_QMAX = 1
INT2_CODE_ZERO = 2
INT2_SCALE_DIVISOR = 2.0
INT2_ALPHABET = 4

# J. Max, "Quantizing for minimum distortion", IRE 1960, unit Gaussian.
# PolarQuant / TurboQuant stage-1 uses these Lloyd-Max tables on RMS-normalized
# coordinates after a Hadamard. Mid-riser: no exact zero reconstruction.
# N=4 distortion 0.1175; N=16 distortion 0.009497.
GAUSSIAN_LLOYD_MAX_2BIT_LEVELS = torch.tensor(
    [-1.5104, -0.4528, 0.4528, 1.5104], dtype=torch.float32
)
GAUSSIAN_LLOYD_MAX_2BIT_BOUNDARIES = torch.tensor(
    [-0.9816, 0.0, 0.9816], dtype=torch.float32
)
GAUSSIAN_LLOYD_MAX_4BIT_LEVELS = torch.tensor(
    [
        -2.7326,
        -2.0690,
        -1.6180,
        -1.2560,
        -0.9424,
        -0.6568,
        -0.3881,
        -0.1284,
        0.1284,
        0.3881,
        0.6568,
        0.9424,
        1.2560,
        1.6180,
        2.0690,
        2.7326,
    ],
    dtype=torch.float32,
)
GAUSSIAN_LLOYD_MAX_4BIT_BOUNDARIES = torch.tensor(
    [
        -2.4008,
        -1.8435,
        -1.4371,
        -1.0993,
        -0.7996,
        -0.5224,
        -0.2582,
        0.0,
        0.2582,
        0.5224,
        0.7996,
        1.0993,
        1.4371,
        1.8435,
        2.4008,
    ],
    dtype=torch.float32,
)


def _require_power_of_two(dim: int) -> int:
    if dim <= 0 or dim & (dim - 1):
        raise ValueError(f"dimension must be a positive power of two, got {dim}")
    return int(math.log2(dim))


def normalized_hadamard(dim: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    _require_power_of_two(dim)
    matrix = torch.ones(1, 1, device=device, dtype=dtype)
    while matrix.shape[0] < dim:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)],
            dim=0,
        )
    return matrix / math.sqrt(dim)


def project_orthogonal(matrix: torch.Tensor) -> torch.Tensor:
    left, _, right = torch.linalg.svd(matrix.float(), full_matrices=False)
    return (left @ right).to(matrix.dtype)


def orthogonality_max_error(matrix: torch.Tensor) -> float:
    dim = matrix.shape[-1]
    identity = torch.eye(dim, device=matrix.device, dtype=matrix.dtype)
    gram = matrix.transpose(-1, -2) @ matrix
    return float((gram - identity).abs().amax().item())


def apply_butterfly(values: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply row-vector Givens butterfly stages to ``values[..., D]``.

    ``angles`` may be ``[stages,D/2]`` (one shared transform) or
    ``[views,stages,D/2]`` while values are ``[tokens,views,D]``.
    """

    dim = values.shape[-1]
    stages = _require_power_of_two(dim)
    if angles.shape[-2:] != (stages, dim // 2):
        raise ValueError(
            f"expected angle tail {(stages, dim // 2)}, got {tuple(angles.shape)}"
        )
    result = values
    for stage in range(stages):
        stride = 1 << stage
        groups = dim // (2 * stride)
        shaped = result.reshape(*result.shape[:-1], groups, 2, stride)
        left = shaped[..., 0, :]
        right = shaped[..., 1, :]
        theta = angles[..., stage, :].reshape(*angles.shape[:-2], groups, stride)
        # Broadcast shared [groups,stride] angles or view-specific angles.
        while theta.ndim < left.ndim:
            theta = theta.unsqueeze(0)
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        mixed_left = cosine * left + sine * right
        mixed_right = -sine * left + cosine * right
        result = torch.stack([mixed_left, mixed_right], dim=-2).reshape_as(result)
    return result


def cayley_from_raw(raw: torch.Tensor) -> torch.Tensor:
    """Map an unconstrained generator to a Cayley orthogonal factor.

    Accepts ``[D,D]`` or a batch ``[...,D,D]``.
    """
    if raw.ndim < 2 or raw.shape[-1] != raw.shape[-2]:
        raise ValueError("Cayley generator must be square on the last two axes")
    skew = raw - raw.transpose(-1, -2)
    eye = torch.eye(raw.shape[-1], device=raw.device, dtype=raw.dtype)
    return torch.linalg.solve(eye + skew, eye - skew)


class CommonRightFactor(nn.Module):
    """Cayley generator for a shared right factor V in SO(D).

    Stage-1 Procrustes leaves this gauge unidentified. SVD projection of a
    free matrix produces NaNs in the backward; Cayley stays on the manifold
    with a stable inverse.

    ``initial`` is held verbatim in a frozen ``base`` and the generator starts
    at zero, so the factor is exactly ``initial`` at step 0. Inverting Cayley to
    seed the generator instead cannot be exact: the map only covers SO(D) and
    blows up at a rotation by pi. This also makes ``max_generator_f`` a trust
    region around the init rather than a leash pulling it back to identity.
    """

    def __init__(self, dim: int, initial: torch.Tensor | None = None):
        super().__init__()
        if dim < 2:
            raise ValueError("common-right factor dimension must be >= 2")
        self.dim = dim
        base = torch.eye(dim) if initial is None else initial.float().clone()
        self.raw = nn.Parameter(torch.zeros(dim, dim, device=base.device))
        self.register_buffer("base", base)
        self.init_orthogonality_error = orthogonality_max_error(base)

    def skew(self) -> torch.Tensor:
        return self.raw - self.raw.transpose(-1, -2)

    def factor(self) -> torch.Tensor:
        return torch.matmul(cayley_from_raw(self.raw), self.base)

    def orthogonality_error(self) -> float:
        return orthogonality_max_error(self.factor())


class ViewCayleyFactor(nn.Module):
    """Per-(layer, head) Cayley orthogonal. View 0 is pinned (gauge).

    GPA left a right-multiply gauge unidentified. Pinning view 0 keeps A from
    absorbing the shared V that extra-KLT / codec-trained V already own, so
    only the 31 rotations *relative* to view 0 are searched.

    ``initial`` is held verbatim in a frozen ``base`` and the generator starts
    at zero, so ``rotations()`` is ``base @ I`` at step 0 -- the incumbent
    including whatever orthogonality defect it was saved with, which is what
    makes its byte numbers reproducible. The generator then twists the aligned
    coordinates, so a shared ``Cayley(W)`` is exactly the shared right factor
    the ``right`` space searches, and pinning view 0 is what stops this space
    from absorbing it.

    Three earlier defects made a warm start silently land elsewhere: the
    reference view was divided out and never multiplied back, so GPA began at
    ``A_0^T A_v`` (adjacent-view cosine 0.384 instead of 0.629); the inverse
    Cayley map returned a skew that ``cayley_from_raw`` re-skewed into twice
    itself; and dividing the reference out only to multiply it back turned the
    init's own non-orthogonality into a 2.9e-4 reconstruction error. Seeding a
    generator cannot be made exact in any case -- Cayley misses reflections and
    is singular at a rotation by pi.
    """

    def __init__(
        self,
        layers: int,
        heads: int,
        dim: int,
        initial: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if layers < 1 or heads < 1 or dim < 2:
            raise ValueError("layers, heads, dim must be positive (dim >= 2)")
        self.layers = layers
        self.heads = heads
        self.dim = dim
        self.views = layers * heads
        # Everything is built on ``initial``'s device so the round-trip check
        # below can run before the caller gets a chance to ``.to(device)``.
        device = torch.device("cpu") if initial is None else initial.device
        if initial is None:
            base = torch.eye(dim).expand(self.views, dim, dim).contiguous()
        else:
            if tuple(initial.shape) != (layers, heads, dim, dim):
                raise ValueError(
                    "initial rotations must be "
                    f"[layers,heads,dim,dim], got {tuple(initial.shape)}"
                )
            base = initial.float().reshape(self.views, dim, dim).clone()
        self.raw = nn.Parameter(torch.zeros(self.views, dim, dim, device=device))
        mask = torch.ones(self.views, 1, 1, device=device)
        mask[0] = 0
        self.register_buffer("trainable_mask", mask)
        self.register_buffer("base", base)
        self.init_orthogonality_error = orthogonality_max_error(base)
        self.init_max_error = 0.0
        if initial is not None:
            with torch.no_grad():
                self.init_max_error = float(
                    (self.rotations() - initial.float()).abs().max()
                )

    def rotations(self) -> torch.Tensor:
        factors = cayley_from_raw(self.raw * self.trainable_mask)
        composed = torch.matmul(self.base, factors)
        return composed.reshape(self.layers, self.heads, self.dim, self.dim)

    def orthogonality_error(self) -> float:
        return orthogonality_max_error(self.rotations())


def adjacent_residual_bits(
    codes: torch.Tensor,
    *,
    axis: str,
    residual_scale: float = 16.0,
) -> torch.Tensor:
    """Differentiable ``log1p(|Δ|)`` proxy for adjacent INT8 residuals.

    ``view`` is layer-major then head-minor, matching the canonical GOP.
    ``token`` is the frame-row neighbour.
    """
    if codes.ndim != 4:
        raise ValueError("codes must be [layers,tokens,heads,dim]")
    if residual_scale <= 0:
        raise ValueError("residual_scale must be positive")
    layers, tokens, heads, dim = codes.shape
    if axis == "view":
        frames = codes.permute(1, 0, 2, 3).reshape(tokens, layers * heads, dim)
        if frames.shape[1] < 2:
            raise ValueError("view residual needs at least two views")
        residual = frames[:, 1:] - frames[:, :-1]
    elif axis == "token":
        if tokens < 2:
            raise ValueError("token residual needs at least two tokens")
        residual = codes[:, 1:] - codes[:, :-1]
    else:
        raise ValueError(f"axis must be 'view' or 'token', got {axis!r}")
    return torch.log1p(residual.abs() / residual_scale).mean() / math.log(2.0)


def high_nibble_ste(
    proxy_codes: torch.Tensor,
    hard_codes: torch.Tensor,
) -> torch.Tensor:
    """Forward: ``hard >> 4``. Backward: ``d(proxy)/16``.

    The STE zeros the low nibble in the value the rate proxy sees, so V is
    not trained to model INT8 LSBs. The backward pass still goes through the
    quantizer, at 1/16 the gray8 scale.
    """
    hard_msb = (hard_codes.to(torch.int16) >> 4).to(dtype=proxy_codes.dtype)
    soft_msb = proxy_codes / 16.0
    return soft_msb + (hard_msb - soft_msb).detach()


def rate_surface_codes(
    hard_codes: torch.Tensor,
    *,
    surface: str,
) -> torch.Tensor:
    if surface == "full":
        return hard_codes
    if surface == "msb4":
        return (hard_codes.to(torch.uint8) >> 4).contiguous()
    raise ValueError(f"rate surface must be 'full' or 'msb4', got {surface!r}")


class SharedButterfly(nn.Module):
    def __init__(self, dim: int, *, initialization: str = "identity", seed: int = 0):
        super().__init__()
        stages = _require_power_of_two(dim)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        if initialization == "identity":
            initial = torch.zeros(stages, dim // 2)
        elif initialization == "hadamard":
            initial = torch.full((stages, dim // 2), math.pi / 4)
        elif initialization == "random":
            initial = (torch.rand(stages, dim // 2, generator=generator) - 0.5) * 0.2
        else:
            raise ValueError(f"unknown butterfly initialization: {initialization}")
        self.dim = dim
        self.initialization = initialization
        self.angles = nn.Parameter(initial)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return apply_butterfly(values, self.angles)

    def matrix(self) -> torch.Tensor:
        identity = torch.eye(self.dim, device=self.angles.device, dtype=self.angles.dtype)
        return self(identity)


class ViewButterfly(nn.Module):
    def __init__(self, views: int, dim: int, *, anchor_view: int = 0):
        super().__init__()
        stages = _require_power_of_two(dim)
        if not 0 <= anchor_view < views:
            raise ValueError("anchor_view is outside the view range")
        self.views = views
        self.dim = dim
        self.anchor_view = anchor_view
        self.angles = nn.Parameter(torch.zeros(views, stages, dim // 2))
        mask = torch.ones(views, 1, 1)
        mask[anchor_view] = 0
        self.register_buffer("trainable_mask", mask)

    def effective_angles(self) -> torch.Tensor:
        return self.angles * self.trainable_mask

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim < 3 or values.shape[-2] != self.views:
            raise ValueError(
                f"view butterfly expects [...,{self.views},{self.dim}], got {values.shape}"
            )
        return apply_butterfly(values, self.effective_angles())

    def matrices(self) -> torch.Tensor:
        identity = torch.eye(
            self.dim, device=self.angles.device, dtype=self.angles.dtype
        ).expand(self.views, self.dim, self.dim)
        # Treat matrix rows as the leading sample dimension for each view.
        values = identity.permute(1, 0, 2).contiguous()
        return self(values).permute(1, 0, 2).contiguous()


class CodecAwareRotationModel(nn.Module):
    """Base Procrustes rotations plus view-relative and common butterflies."""

    def __init__(
        self,
        base_rotations: torch.Tensor,
        *,
        common_initialization: str = "hadamard",
        seed: int = 0,
    ) -> None:
        super().__init__()
        if base_rotations.ndim != 4 or base_rotations.shape[-1] != base_rotations.shape[-2]:
            raise ValueError("base_rotations must have shape [layers,heads,D,D]")
        layers, heads, dim, _ = base_rotations.shape
        self.layers = layers
        self.heads = heads
        self.views = layers * heads
        self.dim = dim
        self.register_buffer("base_rotations", base_rotations.float().contiguous())
        self.view_adapter = ViewButterfly(self.views, dim, anchor_view=0)
        self.common_adapter = SharedButterfly(
            dim, initialization=common_initialization, seed=seed
        )

    def final_rotations(self, *, use_common: bool = True) -> torch.Tensor:
        relative = self.view_adapter.matrices().reshape(
            self.layers, self.heads, self.dim, self.dim
        )
        result = self.base_rotations @ relative
        if use_common:
            result = result @ self.common_adapter.matrix()
        return result

    def rotate(self, keys: torch.Tensor, *, use_common: bool = True) -> torch.Tensor:
        if tuple(keys.shape[:1] + keys.shape[2:]) != (
            self.layers, self.heads, self.dim
        ):
            raise ValueError(
                f"keys must be [layers,tokens,heads,dim], got {tuple(keys.shape)}"
            )
        rotations = self.final_rotations(use_common=use_common)
        return torch.einsum("lthd,lhde->lthe", keys, rotations)

    def inverse(self, values: torch.Tensor, *, use_common: bool = True) -> torch.Tensor:
        rotations = self.final_rotations(use_common=use_common)
        return torch.einsum(
            "lthe,lhed->lthd", values, rotations.transpose(-1, -2)
        )

    def configure_stage(self, stage: int) -> bool:
        """Configure trainable parameters and return whether common U is active."""
        if stage == 1:
            self.view_adapter.angles.requires_grad_(True)
            self.common_adapter.angles.requires_grad_(False)
            return False
        if stage == 2:
            self.view_adapter.angles.requires_grad_(False)
            self.common_adapter.angles.requires_grad_(True)
            return True
        if stage == 3:
            self.view_adapter.angles.requires_grad_(True)
            self.common_adapter.angles.requires_grad_(True)
            return True
        raise ValueError("stage must be 1, 2, or 3")


@dataclass
class QuantizedCodes:
    restored: torch.Tensor
    hard_codes: torch.Tensor
    proxy_codes: torch.Tensor
    scales: torch.Tensor
    scale_count: int
    scale_bytes: int


# One BF16 amax per GOP view: (layer, head) over the 256-token block.
# ``block-layer`` shares one scale across the four heads of a layer
# (8 scales on a 32-view GOP). ``gop`` shares one scale across all 32 views
# of a 256-token package (1 scale per GOP).
INT8_SCALE_GRANULARITY = "block-view"
BLOCK_SCALE_GRANULARITIES = (
    "token-layer",
    "block-layer",
    "block-view",
    "gop",
)


def amax_broadcast_scale(
    values: torch.Tensor,
    *,
    scale_granularity: str,
    scale_block_tokens: int,
    scale_multiplier: float,
    divisor: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """BF16 amax scale for a ``[layer, token, head, channel]`` tensor.

    Returns ``(broadcast scale, compact metadata, scale count)``.
    ``block-view`` is one scale per ``(layer, head, token-block)``.
    ``block-layer`` additionally shares that scale across heads.
    ``gop`` shares one scale across every layer and head of a token-block
    (one BF16 amax per 32-view GOP when the block is 256 tokens).
    """
    if scale_multiplier <= 0:
        raise ValueError("scale_multiplier must be positive")
    if divisor <= 0:
        raise ValueError("scale divisor must be positive")
    if values.ndim != 4:
        raise ValueError(
            "values must have shape [layer, token, head, channel], "
            f"got {tuple(values.shape)}"
        )
    layers, tokens, heads, channels = values.shape
    if scale_granularity == "token-layer":
        amax = values.abs().amax(dim=(2, 3), keepdim=True) * scale_multiplier
        compact = torch.where(
            amax > 0, amax / divisor, torch.ones_like(amax)
        ).to(torch.bfloat16).float()
        return compact, compact, layers * tokens
    if scale_granularity not in ("block-layer", "block-view", "gop"):
        raise ValueError(
            "scale_granularity must be 'token-layer', 'block-layer', "
            f"'block-view', or 'gop', got {scale_granularity!r}"
        )
    if scale_block_tokens <= 0:
        raise ValueError("scale_block_tokens must be positive")
    block_count = (tokens + scale_block_tokens - 1) // scale_block_tokens
    padded_tokens = block_count * scale_block_tokens
    padded = (
        values
        if padded_tokens == tokens
        else F.pad(values, (0, 0, 0, 0, 0, padded_tokens - tokens))
    )
    blocked = padded.reshape(
        layers, block_count, scale_block_tokens, heads, channels
    )
    if scale_granularity == "gop":
        amax = blocked.abs().amax(dim=(0, 2, 3, 4), keepdim=True) * scale_multiplier
        compact = torch.where(
            amax > 0, amax / divisor, torch.ones_like(amax)
        ).to(torch.bfloat16).float()
        scale = (
            compact.expand(layers, block_count, scale_block_tokens, 1, 1)
            .reshape(layers, padded_tokens, 1, 1)[:, :tokens]
        )
        return scale, compact, block_count
    if scale_granularity == "block-layer":
        amax = blocked.abs().amax(dim=(2, 3, 4), keepdim=True) * scale_multiplier
        compact = torch.where(
            amax > 0, amax / divisor, torch.ones_like(amax)
        ).to(torch.bfloat16).float()
        scale = (
            compact.expand(layers, block_count, scale_block_tokens, 1, 1)
            .reshape(layers, padded_tokens, 1, 1)[:, :tokens]
        )
        return scale, compact, layers * block_count
    amax = blocked.abs().amax(dim=(2, 4), keepdim=True) * scale_multiplier
    compact = torch.where(
        amax > 0, amax / divisor, torch.ones_like(amax)
    ).to(torch.bfloat16).float()
    scale = (
        compact.expand(layers, block_count, scale_block_tokens, heads, 1)
        .reshape(layers, padded_tokens, heads, 1)[:, :tokens]
    )
    return scale, compact, layers * block_count * heads


def fp4_e2m1_monotonic_ste(
    values: torch.Tensor,
    *,
    temperature: float,
    scale_granularity: str = "token-layer",
    scale_block_tokens: int = 256,
    scale_group_elements: int = 32,
    scale_multiplier: float = 1.0,
) -> QuantizedCodes:
    """Deployment-exact hard FP4 forward with a soft assignment backward.

    ``scale_multiplier`` widens the quantizer step beyond the block amax. On
    uncentered Keys that would clip the DC-inflated extremes, but once a per-head
    mean is removed the extremes come in and the headroom becomes usable: it
    trades the accuracy that centering bought back for a coarser, cheaper code
    surface. E2M1 bucketizes, so anything past the top boundary saturates at 6
    rather than wrapping.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if values.ndim != 4:
        raise ValueError(
            "values must have shape [layer, token, head, channel], "
            f"got {tuple(values.shape)}"
        )
    if scale_multiplier <= 0:
        raise ValueError("scale_multiplier must be positive")
    layers, tokens, heads, channels = values.shape
    if scale_granularity in BLOCK_SCALE_GRANULARITIES:
        scale, compact_scale, scale_count = amax_broadcast_scale(
            values,
            scale_granularity=scale_granularity,
            scale_block_tokens=scale_block_tokens,
            scale_multiplier=scale_multiplier,
            divisor=6.0,
        )
        scale_bytes = scale_count * 2
    elif scale_granularity in ("channel-group", "channel-group-ue8m0"):
        if scale_group_elements <= 0:
            raise ValueError("scale_group_elements must be positive")
        if channels % scale_group_elements:
            raise ValueError(
                f"channels={channels} must be divisible by "
                f"scale_group_elements={scale_group_elements}"
            )
        group_count = channels // scale_group_elements
        grouped = values.reshape(
            layers, tokens, heads, group_count, scale_group_elements
        )
        amax = grouped.abs().amax(dim=4, keepdim=True) * scale_multiplier
        raw_scale = torch.where(
            amax > 0, amax / 6.0, torch.ones_like(amax)
        )
        if scale_granularity == "channel-group":
            compact_scale = raw_scale.to(torch.bfloat16).float()
            bytes_per_scale = 2
        else:
            # MXFP4-style UE8M0 scale: one exact power-of-two amax scale
            # for every contiguous channel group. Exponent 255/NaN is not used.
            exponent = torch.ceil(torch.log2(raw_scale.clamp_min(2.0**-127)))
            exponent = exponent.clamp(min=-127, max=127)
            compact_scale = torch.pow(2.0, exponent)
            compact_scale = torch.where(
                amax > 0, compact_scale, torch.ones_like(compact_scale)
            )
            bytes_per_scale = 1
        scale = compact_scale.expand(
            layers,
            tokens,
            heads,
            group_count,
            scale_group_elements,
        ).reshape_as(values)
        scale_count = layers * tokens * heads * group_count
        scale_bytes = scale_count * bytes_per_scale
    else:
        raise ValueError(
            "scale_granularity must be 'token-layer', 'block-layer', "
            "'block-view', 'gop', 'channel-group', or 'channel-group-ue8m0', "
            f"got {scale_granularity!r}"
        )
    normalized = values / scale

    boundaries = E2M1_BOUNDARIES.to(values.device)
    magnitudes = E2M1_MAGNITUDES.to(values.device)
    magnitude_codes = torch.bucketize(normalized.abs().contiguous(), boundaries)
    negative = normalized < 0
    native_codes = magnitude_codes + negative.to(torch.long) * 8
    remap = E2M1_MONOTONIC_REMAP.to(values.device)
    hard_codes = remap[native_codes]
    hard_magnitude = magnitudes[native_codes.remainder(8)]
    hard_values = torch.where(negative, -hard_magnitude, hard_magnitude)

    levels = E2M1_MONOTONIC_VALUES.to(values.device, values.dtype)
    logits = -(
        normalized.unsqueeze(-1) - levels.reshape(*([1] * normalized.ndim), -1)
    ).square() / temperature
    probabilities = torch.softmax(logits, dim=-1)
    code_indices = torch.arange(16, device=values.device, dtype=values.dtype)
    soft_codes = (probabilities * code_indices).sum(dim=-1)
    soft_values = (probabilities * levels).sum(dim=-1)

    proxy_codes = soft_codes + (hard_codes.float() - soft_codes).detach()
    restored_normalized = soft_values + (hard_values - soft_values).detach()
    restored = restored_normalized * scale
    return QuantizedCodes(
        restored=restored,
        hard_codes=hard_codes.to(torch.uint8),
        proxy_codes=proxy_codes,
        scales=compact_scale,
        scale_count=scale_count,
        scale_bytes=scale_bytes,
    )


def int8_symmetric_monotonic_qdq(
    values: torch.Tensor,
    *,
    scale_granularity: str = INT8_SCALE_GRANULARITY,
    scale_block_tokens: int = 256,
    scale_multiplier: float = 1.0,
    deadzone: int = 0,
    step: int = 1,
    detach_scale: bool = True,
) -> QuantizedCodes:
    """Symmetric INT8 QDQ with exact monotonic uint8 storage codes.

    ``scale_multiplier`` widens the step past the block amax; the existing
    clamp to [-127, 127] saturates anything that overshoots.

    The deployment integer is ``q = clamp(round(x / scale), -127, 127)`` and
    the byte written to the codec surface is ``q + 128``.  Code 128 therefore
    denotes exact zero and byte order preserves signed-value order.  A scale
    is stored as BF16, matching the FP4 comparison arms.

    Default ``block-view`` is one BF16 amax per GOP view: every
    ``(layer, head, token-block)`` keeps its own scale.  On the 8×4 GOP that
    is 32 scales.  ``block-layer`` shares across the four heads of a layer
    (eight scales per GOP).  ``gop`` shares one scale across all 32 views
    of a 256-token package.
    """
    if deadzone < 0 or step <= 0:
        raise ValueError("deadzone must be >= 0 and step must be positive")
    if scale_granularity not in BLOCK_SCALE_GRANULARITIES:
        raise ValueError(
            "INT8 scale_granularity must be 'token-layer', 'block-layer', "
            f"'block-view', or 'gop', got {scale_granularity!r}"
        )
    scale, compact_scale, scale_count = amax_broadcast_scale(
        values,
        scale_granularity=scale_granularity,
        scale_block_tokens=scale_block_tokens,
        scale_multiplier=scale_multiplier,
        divisor=127.0,
    )

    # Default: detach the scale so a shared-right V cannot cheat by
    # shrinking amax. HEVC-A training passes detach_scale=False so the
    # block-layer step stays in the graph: an A that only thins amax
    # (GPA on Values) raises |Δq| and is rejected.
    scale_used = scale.detach() if detach_scale else scale
    normalized = values / scale_used
    rounded = torch.round(normalized)
    signed_ste = (normalized + (rounded - normalized).detach()).clamp(-127, 127)
    hard_signed = rounded.clamp(-127, 127)
    hard_codes = (hard_signed.to(torch.int16) + 128).to(torch.uint8)
    if deadzone == 0 and step == 1:
        restored = signed_ste * scale_used
        proxy_codes = signed_ste + 128.0
    else:
        from int8_deadzone import apply_map, requant_map

        mapping, centroid = requant_map(deadzone, step)
        hard_codes, restored = apply_map(hard_codes, scale_used, mapping, centroid)
        proxy_codes = hard_codes.float()
    return QuantizedCodes(
        restored=restored,
        hard_codes=hard_codes,
        proxy_codes=proxy_codes,
        scales=compact_scale,
        scale_count=scale_count,
        scale_bytes=scale_count * 2,
    )


def int_symmetric_levels(bits: int) -> tuple[int, int, int]:
    """Mid-tread signed range that matches INT8: q in [-qmax, qmax], zero at 2^{n-1}."""
    if bits < 2 or bits > 8:
        raise ValueError(f"int-symmetric bits must be in 2..8, got {bits}")
    qmax = (1 << (bits - 1)) - 1
    zero = 1 << (bits - 1)
    return -qmax, qmax, zero


def int_symmetric_monotonic_qdq(
    values: torch.Tensor,
    *,
    bits: int,
    scale_granularity: str = INT8_SCALE_GRANULARITY,
    scale_block_tokens: int = 256,
    scale_multiplier: float = 1.0,
) -> QuantizedCodes:
    """Uniform mid-tread INT-n, one BF16 amax per GOP view by default.

    This is the bit-width sweep sibling of INT8: no rotation, no deadzone, no
    PolarQuant, no per-channel groups. ``bits=8`` is the current INT8 map.
    ``bits=2`` is three levels {-1,0,1}, not the four-level two's-complement
    INT2 already measured under ``int2-symmetric``.
    """
    qmin, qmax, zero = int_symmetric_levels(bits)
    if scale_granularity not in BLOCK_SCALE_GRANULARITIES:
        raise ValueError(
            "int-symmetric scale_granularity must be 'token-layer', "
            f"'block-layer', 'block-view', or 'gop', got {scale_granularity!r}"
        )
    scale, compact_scale, scale_count = amax_broadcast_scale(
        values,
        scale_granularity=scale_granularity,
        scale_block_tokens=scale_block_tokens,
        scale_multiplier=scale_multiplier,
        divisor=float(qmax),
    )
    scale_used = scale.detach()
    normalized = values / scale_used
    rounded = torch.round(normalized)
    signed_ste = (normalized + (rounded - normalized).detach()).clamp(qmin, qmax)
    hard_signed = rounded.clamp(qmin, qmax)
    hard_codes = (hard_signed.to(torch.int16) + zero).to(torch.uint8)
    restored = signed_ste * scale_used
    return QuantizedCodes(
        restored=restored,
        hard_codes=hard_codes,
        proxy_codes=signed_ste + float(zero),
        scales=compact_scale,
        scale_count=scale_count,
        scale_bytes=scale_count * 2,
    )


def int2_symmetric_monotonic_qdq(
    values: torch.Tensor,
    *,
    scale_granularity: str = "channel-group",
    scale_group_elements: int = 64,
    scale_multiplier: float = 1.0,
) -> QuantizedCodes:
    """Symmetric INT2 QDQ with one BF16 amax scale per 64 consecutive channels.

    ``q = clamp(round(x / scale), -2, 1)`` and the byte written to the codec
    surface is ``q + 2``.  Code 2 is exact zero.  ``channel-group`` is the only
    granularity: each ``(layer, token, head)`` keeps its own scales, 256/64 = 4
    of them along the channel axis, so a large activation in head 0 cannot
    inflate head 1's step.  The scale is detached, matching INT8, so a rotation
    cannot cheat by shrinking amax.
    """
    if scale_multiplier <= 0:
        raise ValueError("scale_multiplier must be positive")
    if values.ndim != 4:
        raise ValueError(
            "values must have shape [layer, token, head, channel], "
            f"got {tuple(values.shape)}"
        )
    if scale_granularity != "channel-group":
        raise ValueError(
            "INT2 scale_granularity must be 'channel-group', "
            f"got {scale_granularity!r}"
        )
    if scale_group_elements <= 0:
        raise ValueError("scale_group_elements must be positive")
    layers, tokens, heads, channels = values.shape
    if channels % scale_group_elements:
        raise ValueError(
            f"channels={channels} must be divisible by "
            f"scale_group_elements={scale_group_elements}"
        )
    group_count = channels // scale_group_elements
    grouped = values.reshape(
        layers, tokens, heads, group_count, scale_group_elements
    )
    amax = grouped.abs().amax(dim=4, keepdim=True) * scale_multiplier
    compact_scale = torch.where(
        amax > 0, amax / INT2_SCALE_DIVISOR, torch.ones_like(amax)
    ).to(torch.bfloat16).float()
    scale = compact_scale.expand(
        layers,
        tokens,
        heads,
        group_count,
        scale_group_elements,
    ).reshape_as(values)
    scale_count = layers * tokens * heads * group_count

    scale_used = scale.detach()
    normalized = values / scale_used
    rounded = torch.round(normalized)
    signed_ste = (normalized + (rounded - normalized).detach()).clamp(
        INT2_QMIN, INT2_QMAX
    )
    hard_signed = rounded.clamp(INT2_QMIN, INT2_QMAX)
    hard_codes = (hard_signed.to(torch.int16) + INT2_CODE_ZERO).to(torch.uint8)
    restored = signed_ste * scale_used
    proxy_codes = signed_ste + float(INT2_CODE_ZERO)
    return QuantizedCodes(
        restored=restored,
        hard_codes=hard_codes,
        proxy_codes=proxy_codes,
        scales=compact_scale,
        scale_count=scale_count,
        scale_bytes=scale_count * 2,
    )


def _block_layer_rms(
    values: torch.Tensor,
    *,
    scale_block_tokens: int,
    scale_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """One RMS per ``(layer, token-block)``, shared across heads and channels.

    Partial trailing blocks use only the real tokens. Zero-padding would pull
    the RMS down; amax-based FP4/INT8 can pad, this cannot.
    """
    if scale_block_tokens <= 0:
        raise ValueError("scale_block_tokens must be positive")
    layers, tokens, _heads, _channels = values.shape
    block_count = (tokens + scale_block_tokens - 1) // scale_block_tokens
    rms_blocks = values.new_empty(layers, block_count, 1, 1)
    for block in range(block_count):
        start = block * scale_block_tokens
        end = min(start + scale_block_tokens, tokens)
        chunk = values[:, start:end]
        rms = chunk.square().mean(dim=(1, 2, 3)).clamp_min(0.0).sqrt()
        rms_blocks[:, block, 0, 0] = rms * scale_multiplier
    compact_scale = torch.where(
        rms_blocks > 0, rms_blocks, torch.ones_like(rms_blocks)
    ).to(torch.bfloat16).float()
    token_index = torch.arange(tokens, device=values.device) // scale_block_tokens
    scale = compact_scale[:, token_index]
    return compact_scale, scale, layers * block_count


def _lloyd_max_vector_qdq(
    values: torch.Tensor,
    *,
    levels: torch.Tensor,
    boundaries: torch.Tensor,
    scale_multiplier: float,
    scale_granularity: str = "vector",
    scale_block_tokens: int = 256,
) -> QuantizedCodes:
    """Lloyd-Max on RMS-normalized coordinates.

    ``vector`` is one RMS per ``(layer, token, head)`` row. ``block-layer`` is
    one RMS per ``(layer, token-block)``, shared over the four heads, matching
    the FP4/INT8 scale geometry.
    """
    if scale_multiplier <= 0:
        raise ValueError("scale_multiplier must be positive")
    if values.ndim != 4:
        raise ValueError(
            "values must have shape [layer, token, head, channel], "
            f"got {tuple(values.shape)}"
        )
    layers, tokens, heads, _channels = values.shape
    if scale_granularity == "vector":
        rms = values.square().mean(dim=-1, keepdim=True).clamp_min(0.0).sqrt()
        rms = rms * scale_multiplier
        compact_scale = torch.where(
            rms > 0, rms, torch.ones_like(rms)
        ).to(torch.bfloat16).float()
        scale = compact_scale
        scale_count = layers * tokens * heads
    elif scale_granularity == "block-layer":
        compact_scale, scale, scale_count = _block_layer_rms(
            values,
            scale_block_tokens=scale_block_tokens,
            scale_multiplier=scale_multiplier,
        )
    else:
        raise ValueError(
            "Lloyd-Max scale_granularity must be 'vector' or 'block-layer', "
            f"got {scale_granularity!r}"
        )
    scale_used = scale.detach()
    normalized = values / scale_used
    cuts = boundaries.to(device=values.device, dtype=normalized.dtype)
    codebook = levels.to(device=values.device, dtype=values.dtype)
    hard_codes = torch.bucketize(normalized.contiguous(), cuts)
    hard_values = codebook[hard_codes]
    signed_ste = normalized + (hard_values - normalized).detach()
    restored = signed_ste * scale_used
    return QuantizedCodes(
        restored=restored,
        hard_codes=hard_codes.to(torch.uint8),
        proxy_codes=hard_codes.float(),
        scales=compact_scale,
        scale_count=scale_count,
        scale_bytes=scale_count * 2,
    )


def int2_turboquant_qdq(
    values: torch.Tensor,
    *,
    scale_multiplier: float = 1.0,
) -> QuantizedCodes:
    """PolarQuant INT2: one RMS scale per cache vector, Lloyd-Max 2-bit codebook.

    A cache vector is one ``(layer, token, head)`` row of 256 channels. After a
    naive Hadamard the coordinates of an RMS-normalized vector are close to
    ``N(0, 1)``, so the Max 1960 4-level table is the data-free codebook.
    QJL is omitted: the HEVC surface is the 2-bit index image, not an inner-
    product sketch. ``scale_multiplier`` widens the RMS the same way INT8
    widens amax; 1.0 is the published PolarQuant operating point.
    """
    return _lloyd_max_vector_qdq(
        values,
        levels=GAUSSIAN_LLOYD_MAX_2BIT_LEVELS,
        boundaries=GAUSSIAN_LLOYD_MAX_2BIT_BOUNDARIES,
        scale_multiplier=scale_multiplier,
    )


def int4_turboquant_qdq(
    values: torch.Tensor,
    *,
    scale_multiplier: float = 1.0,
    scale_granularity: str = "block-layer",
    scale_block_tokens: int = 256,
) -> QuantizedCodes:
    """PolarQuant INT4: Max 1960 16-level codebook, one RMS per token block.

    Default ``block-layer`` matches FP4: eight BF16 scales on a 256-token
    slab (one per full-attention layer), shared across heads and channels.
    Packed is then 4 bits plus a negligible scale. ``vector`` keeps the older
    per-``(layer, token, head)`` RMS (packed 4.0625) for ablations.
    """
    return _lloyd_max_vector_qdq(
        values,
        levels=GAUSSIAN_LLOYD_MAX_4BIT_LEVELS,
        boundaries=GAUSSIAN_LLOYD_MAX_4BIT_BOUNDARIES,
        scale_multiplier=scale_multiplier,
        scale_granularity=scale_granularity,
        scale_block_tokens=scale_block_tokens,
    )


@dataclass
class CodecProxyResult:
    total_bits_per_value: torch.Tensor
    intra_bits_per_value: torch.Tensor
    inter_only_bits_per_value: torch.Tensor
    mode_probabilities: torch.Tensor


def _predictor_frames(
    frames: torch.Tensor,
    *,
    bidirectional: bool,
    neutral_code: float,
) -> tuple[torch.Tensor, list[str]]:
    midpoint = torch.full_like(frames[:, :1, :], neutral_code)
    left = torch.cat(
        [torch.full_like(frames[:, :, :1], neutral_code), frames[:, :, :-1]],
        dim=2,
    )
    up = torch.cat([midpoint, frames[:, :-1, :]], dim=1)
    planar = 0.5 * (left + up)
    previous = torch.cat(
        [torch.full_like(frames[:1], neutral_code), frames[:-1]], dim=0
    )
    predictors = [left, up, planar, previous]
    names = ["left", "up", "planar", "previous_view"]
    if bidirectional:
        following = torch.cat(
            [frames[1:], torch.full_like(frames[:1], neutral_code)], dim=0
        )
        predictors.append(0.5 * (previous + following))
        names.append("bidirectional_view")
    return torch.stack(predictors, dim=0), names


def codec_rate_proxy(
    monotonic_codes: torch.Tensor,
    *,
    block_height: int = 16,
    block_width: int = 16,
    mode_temperature: float = 1.0,
    residual_scale: float = 1.0,
    neutral_code: float = 8.0,
    bidirectional: bool = False,
) -> CodecProxyResult:
    """Blockwise H.265-inspired residual rate on canonical 32-view frames."""
    if monotonic_codes.ndim != 4:
        raise ValueError("codes must be [layers,tokens,heads,dim]")
    if min(block_height, block_width) <= 0:
        raise ValueError("block dimensions must be positive")
    if mode_temperature <= 0 or residual_scale <= 0:
        raise ValueError("temperatures and scales must be positive")
    layers, tokens, heads, dim = monotonic_codes.shape
    frames = monotonic_codes.permute(0, 2, 1, 3).reshape(
        layers * heads, tokens, dim
    )
    predictors, names = _predictor_frames(
        frames,
        bidirectional=bidirectional,
        neutral_code=neutral_code,
    )
    residuals = frames.unsqueeze(0) - predictors
    pixel_cost = torch.log1p(residuals.abs() / residual_scale) / math.log(2.0)

    pad_h = (-tokens) % block_height
    pad_w = (-dim) % block_width
    if pad_h or pad_w:
        pixel_cost = F.pad(pixel_cost, (0, pad_w, 0, pad_h))
    padded_h, padded_w = pixel_cost.shape[-2:]
    block_cost = pixel_cost.reshape(
        len(names),
        layers * heads,
        padded_h // block_height,
        block_height,
        padded_w // block_width,
        block_width,
    ).sum(dim=(3, 5))

    # A GOP has no previous-frame reference for its first view.
    invalid_cost = block_height * block_width * 32.0
    block_cost[3, 0] = invalid_cost
    if bidirectional:
        block_cost[4, 0] = invalid_cost
        block_cost[4, -1] = invalid_cost

    intra_cost = -mode_temperature * torch.logsumexp(
        -block_cost[:3] / mode_temperature, dim=0
    )
    all_cost = -mode_temperature * torch.logsumexp(
        -block_cost / mode_temperature, dim=0
    )
    probabilities = torch.softmax(-block_cost / mode_temperature, dim=0).mean(
        dim=(1, 2, 3)
    )
    denominator = float(layers * heads * tokens * dim)
    return CodecProxyResult(
        total_bits_per_value=all_cost.sum() / denominator,
        intra_bits_per_value=intra_cost.sum() / denominator,
        inter_only_bits_per_value=block_cost[3:].amin(dim=0).sum() / denominator,
        mode_probabilities=probabilities,
    )


def qk_error_terms(
    keys: torch.Tensor,
    restored_keys: torch.Tensor,
    query_covariances: torch.Tensor,
    query_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return query-weighted squared error and reference energy."""
    if query_covariances.shape != (
        keys.shape[0], keys.shape[2], keys.shape[3], keys.shape[3]
    ):
        raise ValueError("query covariance shape does not match keys")
    if query_counts.shape != (keys.shape[0], keys.shape[2]):
        raise ValueError("query count shape does not match keys")
    error = keys - restored_keys
    per_head_error = torch.einsum(
        "lthd,lhde,lthe->lh", error, query_covariances, error
    )
    per_head_reference = torch.einsum(
        "lthd,lhde,lthe->lh", keys, query_covariances, keys
    )
    weights = query_counts.to(per_head_error.dtype)
    return (per_head_error * weights).sum(), (per_head_reference * weights).sum()


def key_nmse_terms(keys: torch.Tensor, restored_keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    error = keys - restored_keys
    return error.square().sum(), keys.square().sum()
