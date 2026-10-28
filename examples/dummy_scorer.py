from __future__ import annotations

import hashlib

import numpy as np


class DeterministicDummyScorer:
    """Stable pseudo-NLL used only to exercise CLI and fusion code paths."""

    def score(self, utterance: str, wav_path: str, prefixes, candidates):
        if len(prefixes) != len(candidates):
            raise ValueError("one prefix is required for every candidate")
        values = []
        for prefix, candidate in zip(prefixes, candidates):
            payload = f"{utterance}\0{wav_path}\0{prefix}\0{candidate}".encode("utf-8")
            digest = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            values.append(-1.0 - (digest % 10_000) / 10_000.0)
        return np.asarray(values, dtype=np.float64)

def create_scorer(model: str = "", device: str = "", adapter: str = "", **_: object) -> DeterministicDummyScorer:
    if model or device or adapter:
        print("dummy scorer ignores --model, --device, and --adapter", flush=True)
    return DeterministicDummyScorer()
