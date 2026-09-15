#!/usr/bin/env python3
"""Print the compact rate/distortion table from a best-codec run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    path = args.result
    if path.is_dir():
        path = path / "best_h265_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    print("label total_MiB bpv BF16_ratio QK_NMSE Key_NMSE exact")
    for row in payload["rows"]:
        print(
            f'{row["label"]:28s} '
            f'{row["total_bytes"] / (1 << 20):9.3f} '
            f'{row["bits_per_value"]:6.3f} '
            f'{row["ratio_vs_bf16"]:8.3f}x '
            f'{row["qk_nmse"]:.8f} '
            f'{row["key_nmse"]:.8f} '
            f'{str(row["roundtrip_exact"]):>5s}'
        )
    if len(payload["rows"]) > 1:
        aggressive = payload["rows"][1]
        print()
        print(
            "aggressive byte reduction: "
            f'{100 * aggressive["total_byte_reduction_vs_balanced"]:.3f}%'
        )
        print(
            "aggressive QK-NMSE increase: "
            f'{100 * aggressive["qk_nmse_change_vs_balanced"]:.3f}%'
        )
        print(
            "aggressive Key-NMSE increase: "
            f'{100 * aggressive["key_nmse_change_vs_balanced"]:.3f}%'
        )


if __name__ == "__main__":
    main()
