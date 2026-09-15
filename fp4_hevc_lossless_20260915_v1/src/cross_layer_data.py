#!/usr/bin/env python3
"""Synchronized streaming of full-token Key/Value shards from multiple layers."""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from safetensors import safe_open


CACHE_KINDS = ("key", "value")
RAW_SAMPLE_TENSORS = {
    "key": "raw_key_samples",
    "value": "raw_value_samples",
}


def normalize_cache_kind(cache_kind: str) -> str:
    kind = str(cache_kind).strip().lower()
    if kind in {"key", "k", "keys"}:
        return "key"
    if kind in {"value", "v", "values"}:
        return "value"
    raise ValueError(f"cache_kind must be 'key' or 'value', got {cache_kind!r}")


def raw_sample_tensor_name(cache_kind: str) -> str:
    return RAW_SAMPLE_TENSORS[normalize_cache_kind(cache_kind)]


@dataclass(frozen=True)
class ShardEntry:
    sequence_index: int
    path: Path
    token_fingerprint: str


@dataclass(frozen=True)
class Manifest:
    path: Path
    entries: tuple[ShardEntry, ...]
    layers: tuple[int, ...]
    head_indices: tuple[int, ...]
    head_dim: int
    sequence_length: int
    base_segment_size: int
    has_raw_values: bool


@dataclass(frozen=True)
class LayerSource:
    layer: int
    manifest: Manifest


def _scalar(handle, name: str) -> int:
    return int(handle.get_tensor(name).reshape(-1)[0].item())


def _resolve_shard(manifest_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    if candidate.is_file():
        return candidate.resolve()
    local = manifest_path.parent / Path(raw_path).name
    if local.is_file():
        return local.resolve()
    matches = list(manifest_path.parent.glob(f"**/{Path(raw_path).name}"))
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"cannot resolve {raw_path!r} from {manifest_path}")


def load_manifest(path: Path | str) -> Manifest:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "qwen35-qk-sequence-shards-v1":
        raise ValueError(f"unsupported manifest format: {path}")
    entries = []
    for item in payload.get("shards", []):
        worker_files = item.get("worker_files")
        if not isinstance(worker_files, list) or len(worker_files) != 1:
            raise ValueError("cross-layer alignment currently requires TP=1 shards")
        entries.append(
            ShardEntry(
                sequence_index=int(item["sequence_index"]),
                path=_resolve_shard(path, str(worker_files[0])),
                token_fingerprint=str(
                    item.get("key_token_sha256")
                    or item.get("prompt_token_sha256")
                    or ""
                ),
            )
        )
    if not entries:
        raise ValueError(f"manifest contains no shards: {path}")
    with safe_open(str(entries[0].path), framework="pt", device="cpu") as handle:
        names = set(handle.keys())
        layers = tuple(
            sorted(
                {
                    int(match.group(1))
                    for name in names
                    if (match := re.fullmatch(r"layers\.(\d+)\.raw_key_samples", name))
                }
            )
        )
        if not layers:
            raise ValueError(f"no raw Key tensors in {entries[0].path}")
        has_raw_values = any(
            re.fullmatch(r"layers\.\d+\.raw_value_samples", name) for name in names
        )
        heads = tuple(int(value) for value in handle.get_tensor("global_kv_head_indices"))
        head_dim = _scalar(handle, "head_dim")
        base_segment_size = _scalar(handle, "segment_size")
        raw_count = _scalar(handle, "raw_samples_per_segment")
        sequence_length = _scalar(handle, "max_model_len")
    if raw_count != base_segment_size:
        raise ValueError(
            "cross-layer alignment requires full-token shards: "
            f"raw_count={raw_count}, base_segment_size={base_segment_size}, manifest={path}"
        )
    return Manifest(
        path=path,
        entries=tuple(entries),
        layers=layers,
        head_indices=heads,
        head_dim=head_dim,
        sequence_length=sequence_length,
        base_segment_size=base_segment_size,
        has_raw_values=has_raw_values,
    )


def resolve_layer_sources(
    manifest_paths: Sequence[Path | str],
    requested_layers: Sequence[int],
) -> tuple[LayerSource, ...]:
    manifests = tuple(load_manifest(path) for path in manifest_paths)
    if not manifests:
        raise ValueError("at least one manifest is required")
    reference = manifests[0]
    reference_sequences = tuple(entry.sequence_index for entry in reference.entries)
    reference_fingerprints = tuple(entry.token_fingerprint for entry in reference.entries)
    for manifest in manifests[1:]:
        fields = (
            manifest.head_indices == reference.head_indices,
            manifest.head_dim == reference.head_dim,
            manifest.sequence_length == reference.sequence_length,
            manifest.base_segment_size == reference.base_segment_size,
            tuple(entry.sequence_index for entry in manifest.entries) == reference_sequences,
        )
        if not all(fields):
            raise ValueError(f"incompatible manifest: {manifest.path}")
        fingerprints = tuple(entry.token_fingerprint for entry in manifest.entries)
        if all(reference_fingerprints) and all(fingerprints):
            if fingerprints != reference_fingerprints:
                raise ValueError(
                    "manifests contain different token sequences: "
                    f"{reference.path} vs {manifest.path}"
                )
    sources = []
    for layer in requested_layers:
        candidates = [manifest for manifest in manifests if layer in manifest.layers]
        unique_paths = {candidate.path for candidate in candidates}
        if len(unique_paths) != 1:
            raise ValueError(
                f"layer {layer} must occur in exactly one manifest; "
                f"found {[str(path) for path in sorted(unique_paths)]}"
            )
        sources.append(LayerSource(layer=int(layer), manifest=candidates[0]))
    return tuple(sources)


def iter_key_chunks(
    manifest: Manifest,
    *,
    layer: int,
    max_sequences: int,
    token_stride: int,
    cache_kind: str = "key",
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    cache_kind = normalize_cache_kind(cache_kind)
    sample_name = raw_sample_tensor_name(cache_kind)
    entries = list(manifest.entries)
    if max_sequences > 0:
        entries = entries[:max_sequences]
    prefix = f"layers.{layer}"
    for entry in entries:
        with safe_open(str(entry.path), framework="pt", device="cpu") as handle:
            names = set(handle.keys())
            tensor_key = f"{prefix}.{sample_name}"
            if tensor_key not in names:
                if cache_kind == "value":
                    raise ValueError(
                        f"{entry.path} has no {tensor_key}; re-collect with "
                        "scripts/50_collect_value_banks.sh"
                    )
                raise ValueError(f"{entry.path} has no {tensor_key}")
            raw = handle.get_slice(tensor_key)
            valid = handle.get_tensor(f"{prefix}.raw_sample_valid").bool()
            positions = handle.get_tensor(f"{prefix}.raw_sample_positions").long()
            shape = raw.get_shape()
            if shape[0] != len(manifest.head_indices) or shape[-1] != manifest.head_dim:
                raise ValueError(f"unexpected raw {cache_kind} shape {shape} in {entry.path}")
            for segment in range(shape[1]):
                selected = valid[segment]
                if not bool(selected.any()):
                    continue
                values = raw[:, segment].clone()[:, selected]
                chunk_positions = positions[segment][selected]
                order = torch.argsort(chunk_positions)
                values = values.index_select(1, order).permute(1, 0, 2).contiguous()
                chunk_positions = chunk_positions.index_select(0, order).contiguous()
                if token_stride > 1:
                    values = values[::token_stride]
                    chunk_positions = chunk_positions[::token_stride]
                yield entry.sequence_index, chunk_positions, values


def iter_synchronized_key_chunks(
    sources: Sequence[LayerSource],
    *,
    max_sequences: int = 0,
    token_stride: int = 1,
    cache_kind: str = "key",
) -> Iterator[tuple[int, torch.Tensor, tuple[torch.Tensor, ...]]]:
    cache_kind = normalize_cache_kind(cache_kind)
    generators = [
        iter_key_chunks(
            source.manifest,
            layer=source.layer,
            max_sequences=max_sequences,
            token_stride=token_stride,
            cache_kind=cache_kind,
        )
        for source in sources
    ]
    missing = object()
    for items in itertools.zip_longest(*generators, fillvalue=missing):
        if any(item is missing for item in items):
            raise ValueError("layer streams have different numbers of chunks")
        sequence_indices = [item[0] for item in items]
        if len(set(sequence_indices)) != 1:
            raise ValueError(f"sequence mismatch across layers: {sequence_indices}")
        positions = items[0][1]
        for source, item in zip(sources[1:], items[1:]):
            if not torch.equal(positions, item[1]):
                raise ValueError(
                    f"token-position mismatch at layer {source.layer}, sequence {item[0]}"
                )
        yield sequence_indices[0], positions, tuple(item[2] for item in items)
