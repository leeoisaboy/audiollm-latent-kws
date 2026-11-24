"""Unit tests for the converged language-agnostic 5-step core pipeline and release gates."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from analysis.gates import audit_core_language_agnostic, evaluate_release_gates
from analysis.open_set import (
    check_prefix_leak,
    compute_roc_metrics,
    evaluate_speaker_cv,
)
from core import (
    compute_auto_window,
    enroll,
    kv_rerank,
    locate,
    safe_prefix_cut,
    threshold,
)
from integration.backend import AudioLLMBackend, PrefixKVState


class _MockBackend(AudioLLMBackend):
    name = "mock_audio_llm"
    frame_s = 0.08
    default_layer = 2
    hidden_dim = 16

    def __init__(self, target_word: str):
        self.target_word = target_word
        self.prefill_calls = 0
        self.score_calls = 0

    def extract_shallow(self, wav_path: str, layer: int | None = None) -> np.ndarray:
        return np.zeros((20, self.hidden_dim), dtype=np.float32)

    def prefill_prefix_kv(self, wav_path: str, prefix_text: str) -> PrefixKVState:
        self.prefill_calls += 1
        return PrefixKVState(
            wav_path=wav_path,
            prefix_text=prefix_text,
            cache={"prefix": prefix_text},
            last_hidden=np.zeros((1, 1, self.hidden_dim)),
        )

    def score_suffixes(
        self, prefix_state: PrefixKVState, suffixes: Sequence[str]
    ) -> np.ndarray:
        self.score_calls += 1
        return np.array(
            [-0.5 if s == self.target_word else -4.5 for s in suffixes],
            dtype=np.float64,
        )

    def transcribe_prefix(self, wav_path: str, cut_time_s: float) -> str:
        return "please turn on " if cut_time_s > 0 else ""


def test_auto_window_from_median_duration():
    frames_80ms, med = compute_auto_window([0.4, 0.64, 0.8], frame_s=0.08)
    assert frames_80ms == 8
    assert np.isclose(med, 0.64)
    # 40ms backend (e.g. Qwen2-Audio) automatically doubles frame count
    frames_40ms, _ = compute_auto_window([0.4, 0.64, 0.8], frame_s=0.04)
    assert frames_40ms == 16


def test_five_step_core_pipeline_end_to_end():
    rng = np.random.default_rng(7)
    dim = 16
    w_a = rng.normal(size=dim)
    w_b = rng.normal(size=dim)
    feats = {
        "e_a": np.tile(w_a, (6, 1)),
        "e_b": np.tile(w_b, (6, 1)),
    }
    items = [
        {"word": "hello world", "clip_id": "e_a", "kw_t0": 0.0, "kw_t1": 0.48},
        {"word": "goodbye", "clip_id": "e_b", "kw_t0": 0.0, "kw_t1": 0.48},
    ]
    bank = enroll(items, lambda cid: feats[cid], frame_s=0.08)
    assert bank.auto_window_frames == 6

    # Build a search utterance containing w_a around frames 10..16
    utt = rng.normal(scale=0.1, size=(24, dim))
    utt[10:16] = w_a
    pool = locate("utt_1", utt, bank, pool_top=2)
    assert pool.window == 6
    assert pool.words[0] == "hello world"

    backend = _MockBackend(target_word="hello world")
    cuts = safe_prefix_cut(
        pool,
        backoff_frames=2,
        mode="acoustic_backoff",
        prefix_decoder=lambda t: backend.transcribe_prefix("fake.wav", t),
    )
    assert cuts[0].cut_frame == max(0, int(pool.starts[0]) - 2)

    reranked = kv_rerank(pool, "fake.wav", backend, prefix_cuts=cuts, lambda_lm=4.0)
    assert reranked.top_word == "hello world"

    det_pos = threshold(reranked, threshold_value=-1.0)
    assert det_pos.triggered is True
    assert det_pos.word == "hello world"

    det_neg = threshold(reranked, threshold_value=10.0)
    assert det_neg.triggered is False
    assert det_neg.word is None


def test_language_agnostic_prefix_leak_and_open_set_metrics():
    # CJK
    assert check_prefix_leak("外面风大了请帮我把关", "关窗") == (True, True)
    assert check_prefix_leak("外面风大了请帮我把", "关窗") == (False, False)
    # English (LibriPhrase)
    assert check_prefix_leak("he said good", "good morning") == (True, True)
    assert check_prefix_leak("he said", "good morning") == (False, False)
    # Substring non-word shouldn't false-trigger in English
    assert check_prefix_leak("the goodness of", "good morning") == (False, False)

    roc = compute_roc_metrics([1, 1, 1, 0, 0, 0], [0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert roc["auc"] == 1.0
    assert roc["eer"] == 0.0
    assert roc["recall_at_far_5pct"] == 1.0


def test_core_has_zero_language_specific_deps_and_release_gates():
    repo_root = Path(__file__).resolve().parents[1]
    audit = audit_core_language_agnostic(repo_root / "core")
    assert audit["passed"] is True
    assert audit["violations"] == []

    # Synthetic passing report for 2 backends
    tier_ok = {
        "cv": {
            "pool_recall": 0.92,
            "leak_rate": 0.02,
            "bootstrap": {"mean_gain_pp": 6.5, "ci95_low_pp": 3.2},
        },
        "open_set": {
            "stage1": {"auc": 0.82, "recall_at_far_5pct": 0.55},
            "fused": {"auc": 0.95, "recall_at_far_5pct": 0.85},
        },
    }
    backend_ok = {
        "datasets": {"L1": tier_ok, "L2": tier_ok, "L3": tier_ok},
        "kv_cache": {"correlation_r": 0.999, "speedup": 1.65},
    }
    gate_res = evaluate_release_gates(
        {"step_audio2": backend_ok, "qwen2_audio": backend_ok},
        core_dir=repo_root / "core",
    )
    assert gate_res["release_passed"] is True
    assert len(gate_res["passing_backends"]) == 2

    # Speaker CV helper sanity check
    cv = evaluate_speaker_cv(
        [
            {"speaker": f"s{i}", "in_pool": True, "s1_top1": 0.0, "s2_top1": 1.0, "leaked": False, "ends_leaked": False}
            for i in range(10)
        ]
    )
    assert cv["bootstrap"]["ci95_low_pp"] > 0.0
