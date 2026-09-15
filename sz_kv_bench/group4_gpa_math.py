#!/usr/bin/env python3
"""Small-matrix routines for hierarchical orthogonal alignment."""

from __future__ import annotations

import torch


def normalize_rows(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    values = values.float()
    return values / values.norm(dim=-1, keepdim=True).clamp_min(eps)


def identity_rotations(count: int, dim: int, device: torch.device) -> torch.Tensor:
    return torch.eye(dim, device=device, dtype=torch.float32).expand(count, dim, dim).clone()


def project_orthogonal(matrix: torch.Tensor) -> torch.Tensor:
    left, _, right = torch.linalg.svd(matrix, full_matrices=False)
    return left @ right


def anchor_first(rotations: torch.Tensor) -> torch.Tensor:
    """Remove the common right-rotation ambiguity by anchoring item zero."""
    common = rotations[0].transpose(-1, -2)
    return torch.matmul(rotations, common)


def pair_weight_matrix(
    count: int,
    pairs: list[tuple[int, int]] | None,
    *,
    device: torch.device,
    include_self: bool = True,
) -> torch.Tensor:
    """Build the GPA weight matrix. ``pairs is None`` recovers all-ones."""
    if pairs is None:
        return torch.ones(count, count, device=device, dtype=torch.float32)
    weights = torch.zeros(count, count, device=device, dtype=torch.float32)
    for left, right in pairs:
        if not (0 <= left < count and 0 <= right < count):
            raise ValueError(f"pair {(left, right)} is outside 0..{count - 1}")
        weights[left, right] = 1.0
        weights[right, left] = 1.0
    if include_self:
        weights.fill_diagonal_(1.0)
    return weights


def path_pairs(order: list[int]) -> list[tuple[int, int]]:
    """Undirected edges of a GOP path. The encoder only ever pays these."""
    return [(order[index], order[index + 1]) for index in range(len(order) - 1)]


def generalized_procrustes_update(
    cross_covariance: torch.Tensor,
    rotations: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    project=None,
    anchor=None,
) -> torch.Tensor:
    count = rotations.shape[0]
    if weights is None:
        weights = torch.ones(
            count, count, device=rotations.device, dtype=rotations.dtype
        )
    elif tuple(weights.shape) != (count, count):
        raise ValueError(
            f"weights must be [{count}, {count}], got {tuple(weights.shape)}"
        )
    if project is None:
        project = project_orthogonal
    if anchor is None:
        anchor = anchor_first
    updates = []
    for item in range(count):
        target = torch.zeros_like(rotations[item])
        mass = 0.0
        for other in range(count):
            weight = float(weights[item, other].item())
            if weight == 0.0:
                continue
            target.add_(cross_covariance[item, other] @ rotations[other], alpha=weight)
            mass += weight
        if mass <= 0.0:
            updates.append(rotations[item].clone())
            continue
        target.div_(mass)
        updates.append(project(target))
    return anchor(torch.stack(updates, dim=0))


def fit_generalized_procrustes(
    cross_covariance: torch.Tensor,
    *,
    iterations: int,
    tolerance: float,
    weights: torch.Tensor | None = None,
    project=None,
    anchor=None,
    init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    count, _, dim, _ = cross_covariance.shape
    rotations = (
        init.clone()
        if init is not None
        else identity_rotations(count, dim, cross_covariance.device)
    )
    all_pairs = pairwise_cosines_from_cross_covariance(cross_covariance, rotations)
    objective = (
        all_pairs
        if weights is None
        else weighted_cosines_from_cross_covariance(
            cross_covariance, rotations, weights
        )
    )
    previous = float(objective.mean().item()) if objective.numel() else 1.0
    pair_mean = float(all_pairs.mean().item()) if all_pairs.numel() else 1.0
    pair_min = float(all_pairs.min().item()) if all_pairs.numel() else 1.0
    pair_max = float(all_pairs.max().item()) if all_pairs.numel() else 1.0
    history = [
        {
            "iteration": 0,
            "mean_objective_cosine": previous,
            "mean_pairwise_cosine": pair_mean,
            "min_pairwise_cosine": pair_min,
            "max_pairwise_cosine": pair_max,
            "improvement": 0.0,
            "orthogonality_max_error": 0.0,
        }
    ]
    for iteration in range(1, iterations + 1):
        rotations = generalized_procrustes_update(
            cross_covariance,
            rotations,
            weights=weights,
            project=project,
            anchor=anchor,
        )
        all_pairs = pairwise_cosines_from_cross_covariance(cross_covariance, rotations)
        objective = (
            all_pairs
            if weights is None
            else weighted_cosines_from_cross_covariance(
                cross_covariance, rotations, weights
            )
        )
        current = float(objective.mean().item()) if objective.numel() else 1.0
        history.append(
            {
                "iteration": iteration,
                "mean_objective_cosine": current,
                "mean_pairwise_cosine": float(all_pairs.mean().item()) if all_pairs.numel() else 1.0,
                "min_pairwise_cosine": float(all_pairs.min().item()) if all_pairs.numel() else 1.0,
                "max_pairwise_cosine": float(all_pairs.max().item()) if all_pairs.numel() else 1.0,
                "improvement": current - previous,
                "orthogonality_max_error": orthogonality_max_error(rotations),
            }
        )
        if abs(current - previous) < tolerance:
            break
        previous = current
    return rotations, history


def pairwise_cloud_cosines(
    cross_covariance: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """tr(R_i^T C_ij R_j) / sqrt(tr C_ii tr C_jj). Unit-row GPA reduces to cosine."""
    values = []
    for left in range(rotations.shape[0]):
        left_energy = torch.trace(cross_covariance[left, left]).clamp_min(1e-8)
        for right in range(left + 1, rotations.shape[0]):
            right_energy = torch.trace(cross_covariance[right, right]).clamp_min(1e-8)
            aligned = (
                rotations[left].transpose(-1, -2)
                @ cross_covariance[left, right]
                @ rotations[right]
            )
            values.append(torch.trace(aligned) / torch.sqrt(left_energy * right_energy))
    if not values:
        return torch.zeros(0, device=rotations.device, dtype=rotations.dtype)
    return torch.stack(values)


def pairwise_cosines_from_cross_covariance(
    cross_covariance: torch.Tensor,
    rotations: torch.Tensor,
    sample_count: int | None = None,
) -> torch.Tensor:
    if sample_count is None:
        # Each input row is normalized, hence trace(C_ii) equals N.
        sample_count = int(round(float(torch.trace(cross_covariance[0, 0]).item())))
    values = []
    for left in range(rotations.shape[0]):
        for right in range(left + 1, rotations.shape[0]):
            aligned = (
                rotations[left].transpose(-1, -2)
                @ cross_covariance[left, right]
                @ rotations[right]
            )
            values.append(torch.trace(aligned) / max(sample_count, 1))
    if not values:
        return torch.zeros(0, device=rotations.device, dtype=rotations.dtype)
    return torch.stack(values)


def weighted_cosines_from_cross_covariance(
    cross_covariance: torch.Tensor,
    rotations: torch.Tensor,
    weights: torch.Tensor,
    sample_count: int | None = None,
) -> torch.Tensor:
    """Cosines of the pairs the weight matrix actually pays for.

    Self-weights are ignored: they stabilize the polar factor but are not a
    pair the encoder sees. If every off-diagonal weight is zero the tensor is
    empty and the caller has to treat that as a degenerate graph.
    """
    pairs = [
        (left, right)
        for left in range(rotations.shape[0])
        for right in range(left + 1, rotations.shape[0])
        if float(weights[left, right].item()) > 0.0
    ]
    if not pairs:
        return torch.zeros(0, device=rotations.device, dtype=torch.float32)
    return torch.tensor(
        selected_pair_cosines_from_cross_covariance(
            cross_covariance, rotations, pairs, sample_count
            if sample_count is not None
            else int(round(float(torch.trace(cross_covariance[0, 0]).item())))
        ),
        device=rotations.device,
        dtype=torch.float32,
    )


def selected_pair_cosines_from_cross_covariance(
    cross_covariance: torch.Tensor,
    rotations: torch.Tensor,
    pairs: list[tuple[int, int]],
    sample_count: int,
) -> list[float]:
    result = []
    for left, right in pairs:
        aligned = (
            rotations[left].transpose(-1, -2)
            @ cross_covariance[left, right]
            @ rotations[right]
        )
        result.append(float(torch.trace(aligned).item() / max(sample_count, 1)))
    return result


def orthogonality_max_error(rotations: torch.Tensor) -> float:
    dim = rotations.shape[-1]
    identity = torch.eye(dim, device=rotations.device, dtype=rotations.dtype)
    gram = rotations.transpose(-1, -2) @ rotations
    return float((gram - identity).abs().max().item())
