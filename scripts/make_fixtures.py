from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import FRAME_S, READ_FRAME_S


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic synthetic latent fixtures")
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--utterances", type=int, default=8)
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    root = Path(args.output)
    (root / "search").mkdir(parents=True, exist_ok=True)
    (root / "enroll").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    words = ["开灯", "关窗", "拉帘", "调温"]
    template_rows, search_rows = [], []
    centers = {word: rng.normal(size=args.dimensions) for word in words}

    for word_index, word in enumerate(words):
        for take in range(2):
            clip_id = f"enroll-{word_index:02d}-{take}"
            frames = rng.normal(scale=0.05, size=(6, args.dimensions))
            frames[1:4] += centers[word] + rng.normal(scale=0.02, size=(3, args.dimensions))
            np.save(root / "enroll" / f"{clip_id}.npy", frames.astype(np.float16))
            template_rows.append(
                {
                    "clip_id": clip_id,
                    "word": word,
                    "kw_t0": 0.08,
                    "kw_t1": 0.40,
                    "source": "synthetic-neutral-command",
                }
            )

    for utterance_index in range(args.utterances):
        utterance = f"synthetic-{utterance_index:04d}"
        word = words[utterance_index % len(words)]
        transcript = f"请帮我{word}好吗"
        frames = rng.normal(scale=0.05, size=(18, args.dimensions))
        onset = 3 + (utterance_index % 6)
        frames[onset : onset + 5] += centers[word] + rng.normal(scale=0.02, size=(5, args.dimensions))
        np.save(root / "search" / f"{utterance}.npy", frames.astype(np.float16))
        token_start = onset * 2 - 3
        search_rows.append(
            {
                "utt": utterance,
                "text": transcript,
                "gold": [{"word": word, "kw_t0": onset * FRAME_S, "kw_t1": (onset + 5) * FRAME_S}],
                "tokens": [
                    (position + token_start, symbol)
                    for position, symbol in enumerate(transcript)
                ],
            }
        )

    with open(root / "templates.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for row in template_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(root / "search_manifest.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for row in search_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(root / "readings.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for row in search_rows:
            handle.write(
                json.dumps(
                    {"utt_id": row["utt"], "frame_time": READ_FRAME_S, "tokens": row["tokens"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote synthetic fixtures to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
