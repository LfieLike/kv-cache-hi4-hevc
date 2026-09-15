#!/usr/bin/env python3
"""Sequence-disjoint slab loader for codec-aware rotation training."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from safetensors import safe_open

from cross_layer_data import LayerSource, normalize_cache_kind, raw_sample_tensor_name


@dataclass
class CodecSlab:
    sequence_index: int
    base_index: int
    token_offset: int
    positions: torch.Tensor
    keys: torch.Tensor
    query_covariances: torch.Tensor
    query_counts: torch.Tensor
    cache_kind: str = "key"


def available_sequence_indices(sources: Sequence[LayerSource]) -> list[int]:
    if not sources:
        raise ValueError("at least one layer source is required")
    reference = [entry.sequence_index for entry in sources[0].manifest.entries]
    for source in sources[1:]:
        current = [entry.sequence_index for entry in source.manifest.entries]
        if current != reference:
            raise ValueError("layer manifests do not contain the same sequence order")
    return reference


def _entry_by_sequence(source: LayerSource) -> dict[int, object]:
    return {entry.sequence_index: entry for entry in source.manifest.entries}


def iter_codec_slabs(
    sources: Sequence[LayerSource],
    *,
    sequence_indices: Sequence[int],
    frame_height: int,
    max_slabs: int = 0,
    cache_kind: str = "key",
) -> Iterator[CodecSlab]:
    """Yield synchronized `[layer,token,head,dim]` slabs.

    ``cache_kind='key'`` loads ``raw_key_samples`` (the historical path).
    ``cache_kind='value'`` loads ``raw_value_samples`` into ``CodecSlab.keys``
    so the INT8+HEVC encoder is unchanged; QK scoring is skipped downstream.
    Future-Query moments remain at the collection base-segment granularity and
    are attached to every slab cut from the same base segment.
    """
    cache_kind = normalize_cache_kind(cache_kind)
    sample_name = raw_sample_tensor_name(cache_kind)
    if not sources:
        raise ValueError("at least one layer source is required")
    if frame_height <= 0:
        raise ValueError("frame_height must be positive")
    base_size = sources[0].manifest.base_segment_size
    if frame_height > base_size or base_size % frame_height:
        raise ValueError(
            f"frame_height={frame_height} must divide base segment size={base_size}"
        )
    requested = list(dict.fromkeys(int(index) for index in sequence_indices))
    if not requested:
        raise ValueError("sequence_indices cannot be empty")
    entries = [_entry_by_sequence(source) for source in sources]
    available = set(entries[0])
    missing = [index for index in requested if index not in available]
    if missing:
        raise ValueError(f"sequence indices are not present in manifests: {missing}")

    produced = 0
    for sequence_index in requested:
        with ExitStack() as stack:
            handles = [
                stack.enter_context(
                    safe_open(
                        str(entry_map[sequence_index].path),
                        framework="pt",
                        device="cpu",
                    )
                )
                for entry_map in entries
            ]
            first_layer = sources[0].layer
            first_tensor = f"layers.{first_layer}.{sample_name}"
            first_names = set(handles[0].keys())
            if first_tensor not in first_names:
                hint = ""
                if cache_kind == "value":
                    hint = "; re-collect with scripts/50_collect_value_banks.sh"
                raise ValueError(f"shard lacks {first_tensor}{hint}")
            raw_shape = handles[0].get_slice(first_tensor).get_shape()
            heads, bases, samples, dim = raw_shape
            if samples != base_size:
                raise ValueError(
                    "codec-aware training requires all cache tokens in every base: "
                    f"raw samples={samples}, base size={base_size}"
                )

            for base_index in range(bases):
                keys_by_layer = []
                covariances_by_layer = []
                counts_by_layer = []
                positions_reference = None
                for source, handle in zip(sources, handles):
                    prefix = f"layers.{source.layer}"
                    names = set(handle.keys())
                    required = {
                        f"{prefix}.{sample_name}",
                        f"{prefix}.raw_sample_valid",
                        f"{prefix}.raw_sample_positions",
                        f"{prefix}.future_query_second_moment",
                        f"{prefix}.future_query_count",
                    }
                    missing_names = sorted(required - names)
                    if missing_names:
                        hint = ""
                        if cache_kind == "value" and any(
                            name.endswith(".raw_value_samples") for name in missing_names
                        ):
                            hint = (
                                "; re-collect with scripts/50_collect_value_banks.sh"
                            )
                        raise ValueError(
                            f"shard lacks {cache_kind} codec tensors: "
                            f"{missing_names}{hint}"
                        )
                    valid = handle.get_slice(f"{prefix}.raw_sample_valid")[
                        base_index
                    ].bool()
                    positions = handle.get_slice(f"{prefix}.raw_sample_positions")[
                        base_index
                    ].long()
                    if not bool(valid.all()):
                        # Ignore a short final base; training frames must be dense.
                        keys_by_layer = []
                        break
                    order = torch.argsort(positions)
                    positions = positions.index_select(0, order)
                    if positions_reference is None:
                        positions_reference = positions
                    elif not torch.equal(positions_reference, positions):
                        raise ValueError(
                            f"position mismatch at sequence={sequence_index}, base={base_index}"
                        )
                    raw = handle.get_slice(f"{prefix}.{sample_name}")[
                        :, base_index
                    ].index_select(1, order)
                    keys_by_layer.append(raw.permute(1, 0, 2).contiguous())
                    covariance = handle.get_slice(
                        f"{prefix}.future_query_second_moment"
                    )[:, base_index].float()
                    count = handle.get_slice(f"{prefix}.future_query_count")[
                        :, base_index
                    ].long()
                    if tuple(covariance.shape) != (heads, dim, dim):
                        raise ValueError(
                            f"unexpected Query covariance shape: {covariance.shape}"
                        )
                    if bool((count <= 0).any()):
                        raise ValueError(
                            f"no future Queries at sequence={sequence_index}, base={base_index}"
                        )
                    covariances_by_layer.append(covariance)
                    counts_by_layer.append(count)
                if not keys_by_layer:
                    continue

                keys = torch.stack(keys_by_layer, dim=0)
                covariances = torch.stack(covariances_by_layer, dim=0)
                counts = torch.stack(counts_by_layer, dim=0)
                for token_offset in range(0, base_size, frame_height):
                    stop = token_offset + frame_height
                    yield CodecSlab(
                        sequence_index=sequence_index,
                        base_index=base_index,
                        token_offset=token_offset,
                        positions=positions_reference[token_offset:stop].clone(),
                        keys=keys[:, token_offset:stop].clone(),
                        query_covariances=covariances.clone(),
                        query_counts=counts.clone(),
                        cache_kind=cache_kind,
                    )
                    produced += 1
                    if max_slabs > 0 and produced >= max_slabs:
                        return


def collect_codec_slabs(
    sources: Sequence[LayerSource],
    *,
    sequence_indices: Sequence[int],
    frame_height: int,
    max_slabs: int,
    cache_kind: str = "key",
) -> list[CodecSlab]:
    slabs = list(
        iter_codec_slabs(
            sources,
            sequence_indices=sequence_indices,
            frame_height=frame_height,
            max_slabs=max_slabs,
            cache_kind=cache_kind,
        )
    )
    if not slabs:
        raise ValueError("no complete codec slabs were loaded")
    return slabs


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    """Spread ``count`` picks over ``[0, length)``, including both ends."""
    if length <= 0:
        return []
    if count <= 0 or count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    raw = [
        int(round(index * (length - 1) / (count - 1)))
        for index in range(count)
    ]
    unique: list[int] = []
    for value in raw:
        if value not in unique:
            unique.append(value)
    return unique


def collect_evenly_spaced_slabs(
    sources: Sequence[LayerSource],
    *,
    sequence_indices: Sequence[int],
    frame_height: int,
    slabs_per_sequence: int,
    cache_kind: str = "key",
) -> list[CodecSlab]:
    """Take GOP slabs evenly along each sequence, not just the prefix.

    ``collect_codec_slabs(..., max_slabs=N)`` stops after the first N frames
    (the first 4096 tokens when N=16 and frame_height=256). Training then
    never sees the rest of the 98k sequence.
    """
    selected: list[CodecSlab] = []
    for sequence in sequence_indices:
        count = sum(
            1
            for _ in iter_codec_slabs(
                sources,
                sequence_indices=[sequence],
                frame_height=frame_height,
                max_slabs=0,
                cache_kind=cache_kind,
            )
        )
        if count <= 0:
            continue
        wanted = set(evenly_spaced_indices(count, slabs_per_sequence))
        taken = 0
        for index, slab in enumerate(
            iter_codec_slabs(
                sources,
                sequence_indices=[sequence],
                frame_height=frame_height,
                max_slabs=0,
                cache_kind=cache_kind,
            )
        ):
            if index in wanted:
                selected.append(slab)
                taken += 1
                if taken >= len(wanted):
                    break
    if not selected:
        raise ValueError("no complete codec slabs were loaded")
    return selected
