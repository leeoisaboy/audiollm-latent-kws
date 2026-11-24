<p align="center">
  <a href="https://github.com/leeoisaboy/audiollm-latent-kws">
    <img src="docs/assets/title_banner.en.png" alt="AudioLLM-Latent-KWS" />
  </a>
</p>

<p align="center">
  <a href="README.md"><img alt="中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-README-d73a49?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/audiollm-latent-kws/"><img alt="Live Studio" src="https://img.shields.io/badge/Live_Studio-Latent--KWS_Audit_Studio-0071e3?style=flat-square"></a>
  <a href="https://huggingface.co/datasets/Leeoisaboy/audiollm-latent-kws-bench"><img alt="HF Dataset" src="https://img.shields.io/badge/HuggingFace-L1%2FL2%2FL3_Benchmark-ff9d00?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/disentangle-acoustic-print/"><img alt="Sister Project" src="https://img.shields.io/badge/Sister_Project-Acoustic_Print_Studio-2b9c94?style=flat-square"></a>
  <img alt="Tests" src="https://img.shields.io/badge/tests-26%20passing-2ea44f?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-555555?style=flat-square">
</p>

<h1 align="center">AudioLLM-Latent-KWS</h1>

<p align="center"><strong>Zero external readers, zero Pinyin dependencies, zero fine-tuning: a language-agnostic 5-step open-vocabulary KWS engine verified across two frozen Audio-LLMs and three public benchmark tiers.</strong></p>

## Architecture Pipeline

<p align="center">
  <img src="docs/assets/pipeline_architecture.jpg" alt="AudioLLM-Latent-KWS Language-Agnostic 5-Step Pipeline and Dual-Backend Audit" width="1000" />
</p>

## The Live Audit Studio

<p align="center">
  <a href="https://leeoisaboy.github.io/audiollm-latent-kws/">
    <img src="docs/assets/audit-studio-demo.gif" alt="Latent-KWS Audit Studio interactive showcase" width="960" />
  </a>
</p>

<p align="center"><strong><a href="https://leeoisaboy.github.io/audiollm-latent-kws/">▶ Open Latent-KWS Audit Studio</a></strong></p>

Built as a borderless, zero-backend scroll story, the Audit Studio now presents the **L1 Shanghainese / L2 KeSpeech / L3 LibriPhrase** public benchmark matrix, dual-backend `Step-Audio-2` & `Qwen2-Audio` layer sweeps, `3.07× / 7.93×` Prefix KV-Cache races, and the 5-fold speaker-grouped empirical tribunal—while still allowing drag-and-drop local `pool.jsonl` / `lm.jsonl` auditing in browser memory with zero upload.

## What Changed in the Universal Convergence

The earlier prototype carried four strong couplings: an external 40 ms character reader (`readings.jsonl`), Mandarin `pypinyin` snapping, hardcoded `Step-Audio-2` layer/window/weight constants, and closed-set-only evaluation. We converged by **subtracting modules and decoupling every language/model assumption**:

1. **Reader-Free & Language-Agnostic Core (`0.00%` Prefix Leak)**: Removed the external 40 ms reader and Mandarin Pinyin from `core/` (AST-verified zero language-specific imports). Replaced text-prefix cutting with **Shared-Audio-KV Listwise MCQ (`A..T + NONE`)** and binary Yes/No Log-Odds.
2. **Minimal 3-Method Backend Interface**: [integration/backend.py](integration/backend.py) exposes only `extract_shallow`, `prefill_prefix_kv`, and `score_suffixes`, verified on both **`Step-Audio-2-mini`** (80 ms/frame) and **`Qwen2-Audio-7B-Instruct`** (40 ms/frame).
3. **Pure 5-Step Functional Pipeline**: `enroll` (auto window from median enrollment duration) → `locate` → `safe_prefix_cut` → `kv_rerank` (per-pool z-score fusion + forward/reversed order debiasing) → `threshold` (open-set reject/trigger).
4. **3-Tier Public Benchmark on Hugging Face**: Pre-segmented L1/L2/L3 benchmarks with word-level boundaries and 1:1 negatives are published at **[Leeoisaboy/audiollm-latent-kws-bench](https://huggingface.co/datasets/Leeoisaboy/audiollm-latent-kws-bench)**. This GitHub repo ships only the training-free core engine and automated 6-gate evaluator.

## Dual-Backend Public Benchmark Results (NVIDIA H20 · fp16 · 0.00% Leak)

| Backend / Baseline | Tier | Latency/utt | R@20 | S1 Top-1 | Final Top-1 | Gain vs S1 (95% CI) | Open-Set AUC | Recall@5%FAR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Whisper-tiny ASR + Fuzzy | L1 (Wu) | 0.697 s | — | — | 3.3% | — | 0.583 | 4.4% |
| **Step-Audio-2 (`layer=2` + Listwise)** | **L1 (Wu)** | **0.124 s** | **96.2%** | 34.9% | **40.4%** | **+5.48 pp `[+2.63, +8.43]`** | **0.847** | **31.7%** |
| **Qwen2-Audio (`audio_tower` + Listwise)** | **L1 (Wu)** | **0.168 s** | **87.7%** | 20.5% | **26.7%** | **+6.16 pp `[+2.01, +10.36]`** | 0.700 | 16.7% |
| Whisper-tiny ASR + Fuzzy | L2 (KeSpeech) | 0.125 s | — | — | 44.4% | — | 0.703 | 40.0% |
| **Step-Audio-2 (`layer=2` + Listwise)** | **L2 (KeSpeech)** | **0.121 s** | **99.1%** | **72.4%** | **72.4%** | — | **0.973** | **90.6%** |
| **Qwen2-Audio (`audio_tower` + Listwise)** | **L2 (KeSpeech)** | **0.174 s** | **94.4%** | 55.1% | **61.7%** | **+6.54 pp `[+0.00, +12.86]`** | **0.828** | **50.6%** |
| Whisper-tiny ASR + Fuzzy | L3 (LibriPhrase) | 0.160 s | — | — | 73.9% | — | 0.885 | 73.3% |
| **Step-Audio-2 (`layer=2` + Listwise)** | **L3 (LibriPhrase)** | **0.114 s** | **100.0%** | 94.4% | **97.8%** | **+3.33 pp `[+0.00, +6.86]`** | **0.987** | **91.7%** |
| **Qwen2-Audio (`audio_tower` + Listwise)** | **L3 (LibriPhrase)** | **0.151 s** | **97.8%** | 36.7% | **90.6%** | **+53.89 pp `[+45.76, +62.01]`** | **0.942** | **57.2%** |

> On L1 multi-keyword conversational utterances (35% contain 2–7 target words, capping single-winner occurrence Top-1 at `61.6%`), utterance-level `Utt-Hit@1` reaches **`61.7%`** on base Step-Audio-2 (3-part temporal pyramid + Pairwise A2A+Text) and **`72.2% Utt-Hit@1` / `93.9% Utt-Hit@5` / `99.3% R@20`** with public Wu adapters (`Wu-ASR` shallow + `Wu-Und` Listwise). See [docs/CONVERGENCE_RESULTS.md](docs/CONVERGENCE_RESULTS.md).

### Prefix KV-Cache Parity & Speedup (NVIDIA H20 · fp16 · 20 Candidates)

| Frozen Backend | Pearson `r` | Spearman `r` | Max Abs Diff | KV Median | Full Median | Speedup | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Step-Audio-2-mini` | **0.99993** | **0.99992** | 0.032 nats | 0.140 s | 0.429 s | **3.07×** | **PASS** |
| `Qwen2-Audio-7B-Instruct` | **1.00000** | **1.00000** | 0.011 nats | 0.202 s | 1.605 s | **7.93×** | **PASS** |

## Quick Start

```bash
git clone https://github.com/leeoisaboy/audiollm-latent-kws.git
cd audiollm-latent-kws
python -m venv .venv
python -m pip install -e ".[dev]"
pytest -q
```

Preview the zero-backend Audit Studio locally:

```bash
python -m http.server 8766 --directory studio
```

See [docs/CONVERGENCE_RESULTS.md](docs/CONVERGENCE_RESULTS.md), [experiments/ARCHIVED.md](experiments/ARCHIVED.md), and [docs/MODELS.md](docs/MODELS.md) for full reproduction details and archived negative ablations.
