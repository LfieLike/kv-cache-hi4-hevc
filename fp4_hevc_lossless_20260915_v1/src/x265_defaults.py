"""Default lossless x265 cell for new work.

New default (script 88, 2026-08-23): veryfast / bframes=3 / b-adapt=2.
``bframes`` is consecutive B-frames between I/P (x265 0..16), not GOP length.
``rc-lookahead`` must be strictly greater than ``bframes``; veryfast_b3 was
measured with ``rc-lookahead=4``.

Published IPPP table (RULER 5.081 / 4.216, book 5.606) used bframes=0.
Replay that cell with ``--x265-bframes 0``. Do not overwrite those trees.
"""

from __future__ import annotations

import argparse

X265_DEFAULT_PRESET = "veryfast"
X265_DEFAULT_BFRAMES = 3
X265_DEFAULT_B_ADAPT = 2
X265_MAX_BFRAMES = 16
X265_IPPP_BFRAMES = 0
X265_IPPP_B_ADAPT = 0
X265_DEFAULT_EXTRA = ("pools=12", "frame-threads=4", "wpp=1", "pmode=0")


def lookahead_for_bframes(bframes: int) -> int:
    """x265: rc-lookahead must be strictly greater than max consecutive B-frames."""
    if bframes <= 0:
        return 0
    return bframes + 1


def merge_x265_extra(base: tuple[str, ...], override: tuple[str, ...]) -> tuple[str, ...]:
    """Later entries win on the same key. Preserves first-seen key order."""
    order: list[str] = []
    by_key: dict[str, str] = {}
    for item in tuple(base) + tuple(override):
        if not item:
            continue
        key = item.split("=", 1)[0]
        if key not in by_key:
            order.append(key)
        by_key[key] = item
    return tuple(by_key[key] for key in order)


def ensure_lookahead(extra: tuple[str, ...], bframes: int) -> tuple[str, ...]:
    """veryfast's default lookahead is ~10, so b=16 fails without this."""
    if bframes <= 0:
        return extra
    need = lookahead_for_bframes(bframes)
    current = None
    for item in extra:
        if item.startswith("rc-lookahead="):
            current = int(item.split("=", 1)[1])
    if current is not None and current > bframes:
        return extra
    return merge_x265_extra(extra, (f"rc-lookahead={need}",))


def resolve_b_adapt(bframes: int, b_adapt: int | None = None) -> int:
    """IPPP forces b-adapt=0. B-frames with omitted/0 adapt use 1 (b=1) or 2."""
    if bframes <= 0:
        return 0
    if b_adapt not in (None, 0, 1, 2):
        raise ValueError("b-adapt must be 0, 1, or 2")
    if b_adapt in (1, 2):
        return b_adapt
    return 1 if bframes == 1 else X265_DEFAULT_B_ADAPT


def add_x265_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--x265-preset", default=X265_DEFAULT_PRESET)
    parser.add_argument("--x265-bframes", type=int, default=X265_DEFAULT_BFRAMES)
    parser.add_argument("--x265-b-adapt", type=int, default=X265_DEFAULT_B_ADAPT)
