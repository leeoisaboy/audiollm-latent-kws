"""Character-grid snapping for the production Stage-2 left context."""

from __future__ import annotations

import json
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from common.config import FRAME_S, GRID_SNAP_TOL, READ_FRAME_S, is_cjk


def load_reading(path: str | Path) -> tuple[dict[str, list[tuple[int, str]]], float]:
    readings: dict[str, list[tuple[int, str]]] = {}
    frame_s = READ_FRAME_S
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            readings[row["utt_id"]] = [
                (int(frame), str(symbol)) for frame, symbol in row["tokens"]
            ]
            frame_s = float(row["frame_time"])
    return readings, frame_s


@lru_cache(maxsize=4096)
def phonetic_key(symbol: str) -> str:
    """Return a compact Mandarin pinyin key when pypinyin is available."""
    if not symbol:
        return ""
    try:
        from pypinyin import Style, lazy_pinyin

        values = lazy_pinyin(symbol, style=Style.NORMAL, errors="ignore")
        return "".join(values).lower()
    except ImportError:
        return symbol


def phoneme_distance(left: str, right: str) -> int:
    """Levenshtein distance between two compact phonetic keys."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for row, left_symbol in enumerate(left, 1):
        current = [row]
        for column, right_symbol in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(left_symbol != right_symbol),
                )
            )
        previous = current
    return previous[-1]


def _matches(
    symbol: str, target: str, phonetic: bool, max_phoneme_distance: int
) -> bool:
    if symbol == target:
        return True
    if not phonetic:
        return False
    return (
        phoneme_distance(phonetic_key(symbol), phonetic_key(target))
        <= max_phoneme_distance
    )


def snap_index(
    chars: Sequence[tuple[int, str]],
    first_char: str,
    seconds: float,
    read_frame_s: float,
    tolerance: int = GRID_SNAP_TOL,
    phonetic: bool = False,
    max_phoneme_distance: int = 1,
) -> int | None:
    """Find the nearest exact or phonetically compatible reading occurrence."""
    if not chars:
        return None
    target = seconds / read_frame_s
    matching = [
        position
        for position, (frame, symbol) in enumerate(chars)
        if _matches(symbol, first_char, phonetic, max_phoneme_distance)
        and abs(frame - target) <= tolerance
    ]
    if matching:
        return min(
            matching,
            key=lambda position: (
                phoneme_distance(
                    phonetic_key(chars[position][1]), phonetic_key(first_char)
                ),
                abs(chars[position][0] - target),
            ),
        )
    frames = [frame for frame, _ in chars]
    if phonetic:
        return max(0, bisect_right(frames, target) - 1)
    return min(range(len(chars)), key=lambda position: abs(chars[position][0] - target))


def snapped_cut_text(
    chars: Sequence[tuple[int, str]],
    word: str,
    seconds: float,
    read_frame_s: float,
    tolerance: int = GRID_SNAP_TOL,
    phonetic: bool = False,
    max_phoneme_distance: int = 1,
) -> str:
    """Snap the boundary into the reading grid, then cut the safe prefix."""
    if not chars or seconds < 0 or not word:
        return ""
    index = snap_index(
        chars,
        word[0],
        seconds,
        read_frame_s,
        tolerance,
        phonetic,
        max_phoneme_distance,
    )
    if index is None:
        return ""
    return "".join(symbol for _, symbol in chars[:index] if is_cjk(symbol))


def build_prefixes(
    words: Sequence[str],
    starts: Sequence[int],
    chars: Sequence[tuple[int, str]],
    read_frame_s: float,
    frame_s: float = FRAME_S,
    phonetic: bool = False,
    max_phoneme_distance: int = 1,
) -> list[str]:
    """Build one aligned left context per candidate word."""
    seconds = [float(start) * frame_s for start in starts]
    return [
        snapped_cut_text(
            chars,
            word,
            seconds[position],
            read_frame_s,
            phonetic=phonetic,
            max_phoneme_distance=max_phoneme_distance,
        )
        for position, word in enumerate(words)
    ]
