"""Fit the repository's Split-A group-GPA rotations from a collected dump."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sz_kv_bench"))

from fresh_group4_fit import fit_one
from fresh_qwen_dump import digest, write_json


def fit_dump(dump_root, output, iterations=100, tolerance=1e-6, device="cpu"):
    dump_root = Path(dump_root)
    output = Path(output)
    manifest_path = dump_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Input dump manifest must be complete")
    calibration = [r for r in manifest["records"] if r.get("split") == "a"]
    if not calibration:
        raise ValueError("No Split-A calibration records in input manifest")
    if not any(r.get("split") == "b" for r in manifest["records"]):
        raise ValueError("Manifest must identify its held-out Split-B records")

    output.mkdir(parents=True, exist_ok=False)
    groups_out = []
    total_energy = 0.0
    total_fp8_sse = 0.0
    fit_source_hashes = []
    sample_counts = []
    for group_index, entry in enumerate(manifest["groups"]):
        kind, layers, views, dim = entry
        chunks = []
        for record in calibration:
            group = record["groups"][group_index]
            if digest(group["path"]) != group["sha256"]:
                raise ValueError(f"Source checksum mismatch: {group['path']}")
            with safe_open(group["path"], framework="pt", device="cpu") as handle:
                x = handle.get_tensor("x")
            if x.dtype != torch.float16 or tuple(x.shape[::2]) != (views, dim):
                raise ValueError(f"Unexpected FP16 source shape in {group['path']}: {tuple(x.shape)}")
            chunks.append(x)
            fit_source_hashes.append(group["sha256"])
            x32 = x.float()
            fp8 = x32.clamp(-448, 448).to(torch.float8_e4m3fn).float()
            total_energy += float(x.double().square().sum())
            total_fp8_sse += float((fp8.double() - x.double()).square().sum())

        fit_values = torch.cat(chunks, dim=1).contiguous()
        sample_counts.append(int(fit_values.shape[1]))
        rotations, means, stats = fit_one(
            fit_values.numpy(), iterations, tolerance, device, expected_views=views
        )
        path = output / f"group{group_index:02d}.npz"
        np.savez(
            path,
            rotations=rotations,
            means=means,
            layers=np.asarray(layers),
            heads=np.arange(views),
        )
        groups_out.append(
            dict(
                index=group_index,
                kind=kind,
                layers=layers,
                views=views,
                dim=dim,
                path=str(path.resolve()),
                sha256=digest(path),
                **stats,
            )
        )
        print(
            f"fit group={group_index} kind={kind} layers={layers}: "
            f"pairwise cosine {stats['history'][0]['mean_pairwise_cosine']:.5f} -> "
            f"{stats['history'][-1]['mean_pairwise_cosine']:.5f}",
            flush=True,
        )

    report = dict(
        status="complete",
        source_manifest_sha256=digest(manifest_path),
        source_group_sha256=fit_source_hashes,
        calibration_split="A",
        heldout_split="B",
        group_size="as declared by source manifest",
        objective="center each view; normalize token rows; all-pairs generalized Procrustes",
        gauge="first view anchored to identity per group",
        iterations=iterations,
        tolerance=tolerance,
        device=str(device),
        samples_per_group=sample_counts,
        epsilon_fp8=float(np.sqrt(total_fp8_sse / total_energy)) if total_energy else None,
        groups=groups_out,
    )
    write_json(output / "fit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.iterations < 1 or args.tolerance <= 0 or args.threads < 1:
        parser.error("iterations, tolerance, and threads must be positive")
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    fit_dump(args.dump_root, args.output, args.iterations, args.tolerance, device)


if __name__ == "__main__":
    main()
