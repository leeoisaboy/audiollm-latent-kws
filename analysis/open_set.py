"""Open-set KWS metrics (ROC-AUC, EER, Recall@FAR), 5-fold speaker CV, and
language-agnostic prefix leakage auditing.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def compute_roc_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    far_targets: Sequence[float] = (0.01, 0.05, 0.10),
) -> dict[str, float]:
    """Compute ROC-AUC, EER, and Recall@FAR using pure NumPy (no sklearn required)."""
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        out: dict[str, float] = {"auc": 0.0, "eer": 1.0, "n_pos": n_pos, "n_neg": n_neg}
        for far in far_targets:
            out[f"recall_at_far_{int(round(far * 100))}pct"] = 0.0
        return out

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]

    # Build TPR/FPR at unique score thresholds (handles ties cleanly)
    distinct_idx = np.where(np.diff(sorted_scores))[0]
    threshold_idxs = np.r_[distinct_idx, len(sorted_scores) - 1]
    tps = np.cumsum(sorted_labels)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    tpr = np.r_[0.0, tps / float(n_pos)]
    fpr = np.r_[0.0, fps / float(n_neg)]
    fnr = 1.0 - tpr

    # Trapezoidal ROC-AUC (NumPy 1.x & 2.x compatible)
    auc = float(np.sum(np.diff(fpr) * 0.5 * (tpr[:-1] + tpr[1:])))

    # EER: point where FPR == FNR (linear interpolation on closest crossing)
    diff = fpr - fnr
    cross_idx = int(np.argmin(np.abs(diff)))
    eer = float(0.5 * (fpr[cross_idx] + fnr[cross_idx]))

    out = {
        "auc": auc,
        "eer": eer,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
    for far in far_targets:
        valid = np.where(fpr <= far + 1e-12)[0]
        rec = float(tpr[valid[-1]]) if len(valid) else 0.0
        out[f"recall_at_far_{int(round(far * 100))}pct"] = rec
    return out


def _first_unit(word: str) -> str:
    """Return the language-agnostic first unit (first CJK char or first Latin word)."""
    text = word.strip().lower()
    if not text:
        return ""
    if ord(text[0]) > 127:
        return text[0]
    tokens = re.findall(r"[a-z0-9']+", text)
    return tokens[0] if tokens else text[0]


def check_prefix_leak(prefix: str, target_word: str) -> tuple[bool, bool]:
    """Check whether ``prefix`` leaks ``target_word``'s first unit (language-agnostic).

    Returns ``(contains_first_unit, ends_with_first_unit)``.
    For CJK scripts (e.g. Shanghainese, Mandarin, Cantonese), checks the first
    character. For space-delimited scripts (e.g. English LibriPhrase), checks
    whether the first word appears as a whole word in the prefix (or at its end).
    """
    if not prefix or not target_word:
        return False, False
    unit = _first_unit(target_word)
    if not unit:
        return False, False
    pref_norm = prefix.strip().lower()
    if not pref_norm:
        return False, False
    if ord(unit[0]) > 127:
        return (unit in pref_norm), pref_norm.endswith(unit)
    pref_tokens = re.findall(r"[a-z0-9']+", pref_norm)
    if not pref_tokens:
        return False, False
    return (unit in pref_tokens), (pref_tokens[-1] == unit)


def speaker_kfold_split(
    speakers: Sequence[str], n_folds: int = 5
) -> list[int]:
    """Deterministic speaker-disjoint fold assignment balanced by utterance count."""
    counts: dict[str, int] = defaultdict(int)
    for spk in speakers:
        counts[str(spk)] += 1
    sorted_spks = sorted(counts.keys(), key=lambda s: (-counts[s], s))
    fold_totals = [0] * n_folds
    spk_to_fold: dict[str, int] = {}
    for spk in sorted_spks:
        best_fold = int(np.argmin(fold_totals))
        spk_to_fold[spk] = best_fold
        fold_totals[best_fold] += counts[spk]
    return [spk_to_fold[str(spk)] for spk in speakers]


def paired_speaker_bootstrap_ci(
    stage1_hits: Sequence[float],
    stage2_hits: Sequence[float],
    speakers: Sequence[str],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired cluster-bootstrap (by speaker) 95% CI for Stage-2 minus Stage-1 Top-1 gain."""
    s1 = np.asarray(stage1_hits, dtype=np.float64)
    s2 = np.asarray(stage2_hits, dtype=np.float64)
    diff = s2 - s1
    by_spk: dict[str, list[int]] = defaultdict(list)
    for idx, spk in enumerate(speakers):
        by_spk[str(spk)].append(idx)
    spk_keys = sorted(by_spk.keys())
    n_spks = len(spk_keys)
    mean_gain = float(np.mean(diff)) if len(diff) else 0.0
    if n_spks < 2:
        return {
            "mean_gain": mean_gain,
            "mean_gain_pp": mean_gain * 100.0,
            "ci95_low": mean_gain,
            "ci95_high": mean_gain,
            "ci95_low_pp": mean_gain * 100.0,
            "ci95_high_pp": mean_gain * 100.0,
            "n_speakers": n_spks,
        }
    rng = np.random.default_rng(seed)
    boot_gains = np.empty(n_boot, dtype=np.float64)
    spk_arrays = [diff[by_spk[k]] for k in spk_keys]
    for b in range(n_boot):
        chosen = rng.integers(0, n_spks, size=n_spks)
        vals = np.concatenate([spk_arrays[i] for i in chosen])
        boot_gains[b] = float(np.mean(vals))
    low, high = np.percentile(boot_gains, [2.5, 97.5])
    return {
        "mean_gain": mean_gain,
        "mean_gain_pp": mean_gain * 100.0,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "ci95_low_pp": float(low) * 100.0,
        "ci95_high_pp": float(high) * 100.0,
        "n_speakers": n_spks,
    }


def evaluate_speaker_cv(
    records: Sequence[Mapping[str, Any]],
    n_folds: int = 5,
) -> dict[str, Any]:
    """Summarize 5-fold speaker-disjoint CV metrics + paired bootstrap 95% CI.

    Each record should contain: ``speaker``, ``in_pool``, ``s1_top1``, ``s2_top1``,
    ``leaked`` (bool), ``ends_leaked`` (bool).
    """
    if not records:
        return {"n": 0, "folds": []}
    speakers = [str(r.get("speaker", "spk0")) for r in records]
    folds = speaker_kfold_split(speakers, n_folds=n_folds)
    fold_reports = []
    for f in range(n_folds):
        sub = [r for r, fold in zip(records, folds) if fold == f]
        if not sub:
            continue
        fold_reports.append(
            {
                "fold": f,
                "n": len(sub),
                "pool_recall": float(np.mean([r["in_pool"] for r in sub])),
                "s1_top1": float(np.mean([r["s1_top1"] for r in sub])),
                "s2_top1": float(np.mean([r["s2_top1"] for r in sub])),
                "gain_pp": float(np.mean([r["s2_top1"] - r["s1_top1"] for r in sub])) * 100.0,
                "leak_rate": float(np.mean([r.get("leaked", False) for r in sub])),
            }
        )
    s1_hits = [float(r["s1_top1"]) for r in records]
    s2_hits = [float(r["s2_top1"]) for r in records]
    boot = paired_speaker_bootstrap_ci(s1_hits, s2_hits, speakers)
    return {
        "n": len(records),
        "n_folds": len(fold_reports),
        "pool_recall": float(np.mean([r["in_pool"] for r in records])),
        "s1_top1": float(np.mean(s1_hits)),
        "s2_top1": float(np.mean(s2_hits)),
        "leak_rate": float(np.mean([r.get("leaked", False) for r in records])),
        "ends_leak_rate": float(np.mean([r.get("ends_leaked", False) for r in records])),
        "bootstrap": boot,
        "folds": fold_reports,
    }
