"""Automated quantitative Release Gate assertions (Pass/Fail).

Encodes the 6 release gates for the converged universal Latent-KWS engine:
  1. Acoustic recall floor:   Stage-1 Top-20 recall >= 85% on L1, L2, L3
  2. Prefix leak compliance:  First-unit prefix leak rate < 5%
  3. Rerank net gain:         Stage-2 vs Stage-1 5-fold CV 95% CI lower bound > 0 on L1, L2, L3
  4. Open-set usability:      Open-set AUC > 0.90 (1:1 negatives) & beats Stage-1 at FAR=5%
  5. KV-Cache consistency:    Cached vs full correlation r >= 0.995 & H20 speedup > 1.3x
  6. Cross-model & minimal:   Zero pinyin/language-specific deps in core/ & >= 2 frozen Audio-LLMs pass
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping, Sequence

FORBIDDEN_CORE_IMPORTS = frozenset(
    {"pypinyin", "jieba", "pkuseg", "opencc", "cn2an", "g2pM", "g2p_en", "nltk"}
)


def audit_core_language_agnostic(core_dir: str | Path) -> dict[str, Any]:
    """AST-scan ``core/`` to verify zero language-specific or external reader deps."""
    root = Path(core_dir)
    violations: list[dict[str, str]] = []
    scanned: list[str] = []
    for py_file in sorted(root.rglob("*.py")):
        scanned.append(str(py_file.name))
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in FORBIDDEN_CORE_IMPORTS:
                        violations.append({"file": py_file.name, "import": alias.name})
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top in FORBIDDEN_CORE_IMPORTS:
                    violations.append({"file": py_file.name, "import": node.module})
    return {
        "passed": len(violations) == 0,
        "scanned_files": scanned,
        "violations": violations,
    }


def evaluate_backend_gates(
    backend_report: Mapping[str, Any],
    tiers: Sequence[str] = ("L1", "L2", "L3"),
) -> dict[str, Any]:
    """Evaluate gates 1-5 for a single Audio-LLM backend report."""
    datasets = backend_report.get("datasets", {})
    kv = backend_report.get("kv_cache", {})

    # Gate 1: Stage-1 Top-20 recall >= 0.85 on all tiers
    recall_details = {}
    recall_pass = True
    for tier in tiers:
        rec = float(datasets.get(tier, {}).get("cv", {}).get("pool_recall", 0.0))
        ok = rec >= 0.85
        recall_details[tier] = {"pool_recall": rec, "threshold": 0.85, "passed": ok}
        recall_pass = recall_pass and ok

    # Gate 2: Prefix leak < 0.05 on all tiers
    leak_details = {}
    leak_pass = True
    for tier in tiers:
        leak = float(datasets.get(tier, {}).get("cv", {}).get("leak_rate", 1.0))
        ok = leak < 0.05
        leak_details[tier] = {"leak_rate": leak, "threshold": 0.05, "passed": ok}
        leak_pass = leak_pass and ok

    # Gate 3: Stage-2 vs Stage-1 5-fold CV 95% CI lower bound > 0 on all tiers
    gain_details = {}
    gain_pass = True
    for tier in tiers:
        boot = datasets.get(tier, {}).get("cv", {}).get("bootstrap", {})
        ci_low = float(boot.get("ci95_low_pp", -999.0))
        mean_pp = float(boot.get("mean_gain_pp", -999.0))
        ok = ci_low > 0.0
        gain_details[tier] = {
            "mean_gain_pp": mean_pp,
            "ci95_low_pp": ci_low,
            "threshold": 0.0,
            "passed": ok,
        }
        gain_pass = gain_pass and ok

    # Gate 4: Open-set AUC > 0.90 and beats pure Stage-1 at FAR=5% (on all tiers with negatives)
    open_details = {}
    open_pass = True
    for tier in tiers:
        os_rep = datasets.get(tier, {}).get("open_set", {})
        fused_auc = float(os_rep.get("fused", {}).get("auc", 0.0))
        s1_auc = float(os_rep.get("stage1", {}).get("auc", 0.0))
        fused_r5 = float(os_rep.get("fused", {}).get("recall_at_far_5pct", 0.0))
        s1_r5 = float(os_rep.get("stage1", {}).get("recall_at_far_5pct", 0.0))
        ok = (fused_auc > 0.90) and (fused_r5 >= s1_r5)
        open_details[tier] = {
            "fused_auc": fused_auc,
            "stage1_auc": s1_auc,
            "fused_recall_at_far_5pct": fused_r5,
            "stage1_recall_at_far_5pct": s1_r5,
            "passed": ok,
        }
        open_pass = open_pass and ok

    # Gate 5: KV-cache correlation >= 0.995 and speedup > 1.3x
    corr = float(kv.get("correlation_r", 0.0))
    speedup = float(kv.get("speedup", 0.0))
    kv_pass = (corr >= 0.995) and (speedup > 1.3)
    kv_details = {
        "correlation_r": corr,
        "speedup": speedup,
        "min_r": 0.995,
        "min_speedup": 1.3,
        "passed": kv_pass,
    }

    all_passed = recall_pass and leak_pass and gain_pass and open_pass and kv_pass
    return {
        "passed": all_passed,
        "gates": {
            "acoustic_recall": {"passed": recall_pass, "details": recall_details},
            "prefix_leak": {"passed": leak_pass, "details": leak_details},
            "rerank_gain": {"passed": gain_pass, "details": gain_details},
            "open_set": {"passed": open_pass, "details": open_details},
            "kv_cache": {"passed": kv_pass, "details": kv_details},
        },
    }


def evaluate_release_gates(
    backends_reports: Mapping[str, Mapping[str, Any]],
    core_dir: str | Path,
    tiers: Sequence[str] = ("L1", "L2", "L3"),
) -> dict[str, Any]:
    """Evaluate all 6 release gates across multiple frozen Audio-LLM backends."""
    core_audit = audit_core_language_agnostic(core_dir)
    per_backend = {
        name: evaluate_backend_gates(rep, tiers=tiers)
        for name, rep in backends_reports.items()
    }
    passing_backends = [name for name, res in per_backend.items() if res["passed"]]
    gate6_pass = core_audit["passed"] and (len(passing_backends) >= 2)
    return {
        "release_passed": gate6_pass,
        "passing_backends": passing_backends,
        "core_language_agnostic": core_audit,
        "per_backend": per_backend,
    }
