#!/usr/bin/env python3
"""Monotone INT8 deadzone and step maps.

The 244/256 occupied codes are not a bug: a shared block amax leaves the
extreme rungs empty, and those empty rungs sit at the far tail where almost
no residual ever crosses them. Collapsing a hole there buys nothing. What
can buy bits is throwing away low-order structure the encoder cannot predict:

  deadzone t   |q| <= t becomes 0
  step s       the survivors are rebinned s original levels to one

The map is required to be monotone and gap-free. The FP4 collapse-vs-compact
experiment already showed what a hole in the ladder costs: identical order-0
entropy and almost none of the byte saving, because HEVC prices a residual
by its magnitude.

Reconstruction uses the centroid of the original signed levels that landed
in each new bin. That is MSE-optimal for a fixed partition and is the number
QK has to be computed against.
"""

from __future__ import annotations

import torch

CODES = 256
ZERO = 128


def requant_map(deadzone: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (code -> compact code, compact code -> signed centroid)."""
    if deadzone < 0 or step <= 0:
        raise ValueError("deadzone must be >= 0 and step must be positive")
    levels = torch.arange(CODES) - ZERO
    survivors = (levels.abs() - deadzone).clamp(min=0)
    # Floor toward zero so the map is symmetric and monotone.
    folded = levels.sign() * torch.div(survivors, step, rounding_mode="floor")
    unique, inverse = torch.unique(folded, return_inverse=True)
    mapping = inverse.to(torch.int64)
    width = int(unique.numel())
    mass = torch.zeros(width, dtype=torch.float64)
    weighted = torch.zeros(width, dtype=torch.float64)
    mass.scatter_add_(0, mapping, torch.ones(CODES, dtype=torch.float64))
    weighted.scatter_add_(0, mapping, levels.double())
    centroid = torch.zeros(width, dtype=torch.float32)
    seen = mass > 0
    centroid[seen] = (weighted[seen] / mass[seen]).float()
    return mapping, centroid


def apply_map(
    hard_codes: torch.Tensor,
    scale: torch.Tensor,
    mapping: torch.Tensor,
    centroid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remap storage codes and rebuild the reconstruction from centroids."""
    compact = mapping.to(device=hard_codes.device)[hard_codes.long()]
    restored_levels = centroid.to(device=scale.device)[compact]
    while scale.ndim < hard_codes.ndim:
        scale = scale.unsqueeze(-1)
    restored = restored_levels * scale.expand_as(hard_codes)
    return compact.to(torch.uint8), restored


def identity_is_lossless(mapping: torch.Tensor, centroid: torch.Tensor) -> bool:
    levels = (torch.arange(CODES) - ZERO).float()
    return torch.allclose(centroid[mapping], levels, atol=1e-5)
