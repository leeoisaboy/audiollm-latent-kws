# Universal Convergence & Public Benchmark Results (H20 GPU0)

This report documents the architectural convergence of `audiollm-latent-kws` from a
single-model, Wu-dialect prototype into a **language-agnostic, reader-free, 5-step
core engine** (`enroll -> locate -> safe_prefix_cut -> kv_rerank -> threshold`)
verified across **3 public dataset tiers** and **2 frozen Audio-LLM backends** on
NVIDIA H20 (GPU0, fp16).

## 1. Architecture Subtraction & Decoupling

| Coupling in Prototype | Converged Solution | Verification |
|---|---|---|
| External 40 ms character reader (`readings.jsonl`) + dual-grid snap | Removed from core path; replaced by **Shared-Audio-KV Binary Yes/No** and **Listwise MCQ (`A..T + U. NONE`)** with **0.00% prefix leakage** | L1 ablation below (`+10.62 pp` vs Arm C `-3.42 pp`) |
| Mandarin Pinyin fuzzy snap (`pypinyin`, `is_cjk`) | Stripped from `core/`; AST-verified zero language-specific imports | `analysis/gates.py` (`audit_core_language_agnostic` PASS) |
| Hardcoded `Step-Audio-2`, `layer=2`, `window=8`, `λ=4.0` | Auto window from median enrollment duration; per-backend optimal shallow layer (`layer=2` for Step-Audio-2, `audio_tower` for Qwen2-Audio); per-pool z-score fusion; 3-method `AudioLLMBackend` (`extract_shallow`, `prefill_prefix_kv`, `score_suffixes`) | Validated on `Step-Audio-2-mini` & `Qwen2-Audio-7B-Instruct` |
| 7 falsified experiments mixed in main package | Excluded from `pyproject.toml` production packages and archived in `experiments/ARCHIVED.md` | 26 unit tests pass (`pytest -q`) |

## 2. Three-Tier Public Benchmark Matrix (Speaker-Disjoint + 1:1 Negatives)

All three pre-segmented evaluation benchmarks (with word-level `[kw_t0, kw_t1]`
boundaries and 1:1 negative utterances) are hosted as a reproducible public
Hugging Face dataset (`audiollm-latent-kws-bench`, see dataset card on Hugging
Face for full provenance and boundary-annotation details). This GitHub
repository contains **only** the core training-free Latent-KWS algorithm and
evaluation harness.

| Tier | Source Public Corpus | Keywords | Enroll Takes | Pos / Neg Test Utts | Held-out Speakers |
|---|---|---:|---:|---:|---:|
| **L1** (Wu / Shanghainese) | `Shanghai_Dialect_Conversational_Speech_Corpus` | 40 | 120 | 180 / 180 | 14 |
| **L2** (Cross-Chinese Dialect) | `KeSpeech` (`test-00000`) | 40 | 120 | 180 / 180 | 96 |
| **L3** (Cross-Lingual Open-Set) | `LibriPhrase` (2-word, 50% hardneg + 50% easyneg) | 40 | 120 | 180 / 180 | 160 |

## 3. L1 Subtractive Prefix Ablation (`reports/l1_prefix_ablation.json`)

| Variant (Step-Audio-2, L1 Shanghainese) | External 40ms Reader | R@20 | S1 Top-1 | S2 Top-1 | Gain (pp) | 95% CI (pp) | Prefix Leak |
|---|---|---:|---:|---:|---:|---:|---:|
| Arm C (external 40ms + snap) | Yes | 89.4% | 23.6% | 20.2% | -3.42 | [-6.97, +0.20] | 3.08% |
| Arm C + Pinyin | Yes + `pypinyin` | 89.4% | 23.6% | 19.5% | -4.11 | [-8.36, +0.15] | 3.08% |
| Generative Empty Prefix | **No** | 89.4% | 23.6% | 18.8% | -4.79 | [-9.12, -0.45] | **0.00%** |
| Shared-Audio-KV Binary Yes/No (`λ=1.5`) | **No** | 89.4% | 23.6% | 30.8% | **+7.19** | **[+3.44, +11.07]** | **0.00%** |
| **Shared-Audio-KV Listwise MCQ + `NONE` (`λ=3.0`)** | **No** | 89.4% | 23.6% | **34.2%** | **+10.62** | **[+6.00, +16.77]** | **0.00%** |

## 4. Dual-Backend Results on Pre-Segmented L1 / L2 / L3 (`reports/listwise_mcq_final.json`, `reports/l1_bench_listwise.json`, `reports/qwen2_layer_sweep.json`)

Sweeping `Qwen2-Audio-7B-Instruct` across 11 layers (`audio_tower(-2)`, `projector(-1)`,
decoder `0..28`) revealed that its bidirectional `audio_tower` output (`layer=-2`,
before causal text-prompt mixing) lifts Stage-1 `R@20` from `67.1% / 51.9% / 75.0%`
(`layer=2`) to **`87.7%–91.4% / 94.4% / 97.8%`** while running 5× faster.

| Backend / Baseline | Tier | Latency/utt | R@20 | S1 Top-1 | Final Top-1 | Gain vs S1 (95% CI) | Open-Set AUC | Recall@5%FAR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Whisper-tiny ASR + Fuzzy | L1 | 0.697s | — | — | 3.3% | — | 0.583 | 4.4% |
| **Step-Audio-2 (`layer=2` + Listwise, λ=1.5)** | **L1** | **0.124s** | **96.2%** | 34.9% | **40.4%** | **+5.48 pp `[+2.63, +8.43]`** | **0.847** | **31.7%** |
| **Qwen2-Audio (`audio_tower` + Listwise, λ=1.5)** | **L1** | **0.168s** | **87.7%** | 20.5% | **26.7%** | **+6.16 pp `[+2.01, +10.36]`** | 0.700 | 16.7% |
| Whisper-tiny ASR + Fuzzy | L2 | 0.125s | — | — | 44.4% | — | 0.703 | 40.0% |
| **Step-Audio-2 (`layer=2` + Listwise)** | **L2** | **0.121s** | **99.1%** | **72.4%** | **72.4% / 71.5%** | — | **0.973** | **90.6%** |
| **Qwen2-Audio (`audio_tower` + Listwise)** | **L2** | **0.174s** | **94.4%** | 55.1% | **61.7%** | **+6.54 pp `[+0.00, +12.86]`** | **0.828** | **50.6%** |
| Whisper-tiny ASR + Fuzzy | L3 | 0.160s | — | — | 73.9% | — | 0.885 | 73.3% |
| **Step-Audio-2 (`layer=2` + Listwise)** | **L3** | **0.114s** | **100.0%** | 94.4% | **97.8%** | **+3.33 pp `[+0.00, +6.86]`** | **0.987** | **91.7%** |
| **Qwen2-Audio (`audio_tower` + Listwise)** | **L3** | **0.151s** | **97.8%** | 36.7% | **90.6%** | **+53.89 pp `[+45.76, +62.01]`** | **0.942** | **57.2%** |

### Upgraded Multi-Keyword Metrics on L1 (Utterance `Hit@1` + Word `mAP / F1`)

Because 35% of L1 conversational utterances contain 2–7 target keywords (capping
single-winner occurrence Top-1 at `180/292 = 61.6%` even for an oracle), we also
report utterance-level `Utt-Hit@1` and word-level `mAP / Macro-F1`
(`reports/l1_wu_asr_s1_plus_wu_und_s2.json`):

| L1 Configuration (0.00% leak) | `R@20` | `Utt-Hit@1` | `Utt-Hit@5` | `word_mAP` | `Macro-F1` | Open-Set `AUC` | `R@5%FAR` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Step-Audio-2 Stage-1 Mean | 96.2% | 56.7% | 85.6% | 56.7% | 62.0% | 0.846 | 31.7% |
| Base Step-Audio-2 Stage-1 3-Part Pyramid | 93.5% | 60.0% | 83.9% | 55.8% | 60.9% | 0.835 | 40.0% |
| Base Step-Audio-2 Pyramid + Pairwise A2A+Text (`λ=0.5`) | 93.5% | 61.7% | 83.9% | 37.0% | 44.3% | 0.828 | 37.2% |
| Public Wu-ASR Adapter Stage-1 Mean (`layer=2`) | 99.0% | 68.3% | 91.7% | **68.8%** | **73.9%** | **0.896** | **62.8%** |
| **Public Wu-ASR S1 Pyramid + Wu-Und S2 Listwise (`λ=1.0`, `0.123s`)** | **99.3%** | **72.2%** | **93.3%** | 54.0% | 59.8% | 0.852 | 54.4% |

## 5. Prefix KV-Cache Consistency & Speedup (H20 GPU0, 20 candidates)

| Backend | Pearson `r` | Spearman `r` | Max Abs Diff | KV Median | Full Median | H20 Speedup | Gate (`r≥0.995, >1.3×`) |
|---|---:|---:|---:|---:|---:|---:|---|
| `Step-Audio-2-mini` | **0.99993** | **0.99992** | 0.032 nats | 0.140s | 0.429s | **3.07×** | **PASS** |
| `Qwen2-Audio-7B-Instruct` | **1.00000** | **1.00000** | 0.011 nats | 0.202s | 1.605s | **7.93×** | **PASS** |

## 6. Release Gate Summary

- **Acoustic Recall Floor (R@20 ≥ 85% on L1/L2/L3)**: **PASS on BOTH backends** (`Step-Audio-2`: 96.2% / 99.1% / 100.0%; `Qwen2-Audio`: 87.7% / 94.4% / 97.8%).
- **Prefix Leak Compliance (< 5%)**: **PASS — 0.00%** across all tiers & both backends.
- **KV-Cache Consistency (r ≥ 0.995, > 1.3×)**: **PASS on BOTH backends** (`Step-Audio-2`: `r=0.99993`, `3.07×`; `Qwen2-Audio`: `r=1.00000`, `7.93×`).
- **Open-Set & Near-100% Top-1**: **L3 PASS on BOTH backends** (`Step-Audio-2`: **97.8% Top-1 / 0.987 AUC**; `Qwen2-Audio`: **90.6% Top-1 / 0.942 AUC**); **L2 Step-Audio-2**: **0.973 AUC / 90.6% R@5%FAR**.
- **Cross-Model & Minimalism**: **PASS** — 0 language/pinyin dependencies in `core/` (AST-verified), unified 3-method backend interface across 2 frozen Audio-LLMs.
