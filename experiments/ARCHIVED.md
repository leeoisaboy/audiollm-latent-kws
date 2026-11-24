# Archived Research & Falsified Experiments

This directory is excluded from the core production package (`pyproject.toml` ships
only `core`, `common`, `locator`, `reranker`, `analysis`, `integration`, `examples`).

The following 7 experimental branches were evaluated with speaker-disjoint cross-validation
and retired from the default production path because they showed either negative or
statistically non-significant gains:

1. **Duration-adaptive windows & peak growing** (`experiments/legacy/search.py`):
   No gain over auto-derived fixed window + enrollment mean centering.
2. **Multi-prototype / K-means sub-templates** (`experiments/legacy/templates.py`):
   Degraded Top-1 and Top-20 recall under few-shot enrollment.
3. **ABTT / PCA whitening** (`experiments/legacy/calibration.py`):
   Severe Top-20 pool recall drop; shallow Audio-LLM principal directions carry phonetic energy.
4. **Score-level multi-layer fusion** (`experiments/legacy/layers.py`):
   No significant gain over single shallow decoder layer (`layer=2`).
5. **PMI text-prior calibration** (`experiments/text_prior.py`):
   Consistent negative gain (-3.45 pp at λ=4.0).
6. **Temporal-shape & entropy-gated fusion** (`experiments/shape.py`):
   95% CI crossed zero after Stage-2 Audio-LLM reranking.
7. **Mandarin Pinyin fuzzy grid snapping** (`reranker/align.py --phonetic-snap`):
   Language-specific (+0.80 pp on Wu dialect only); stripped from the language-agnostic
   `core/` engine so the core pipeline runs identically across dialects and languages.
