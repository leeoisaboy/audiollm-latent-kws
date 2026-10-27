# Architecture

## Stage boundaries

| Stage | Input | Output | Uses gold |
| --- | --- | --- | --- |
| Feature extraction | WAV + frozen Audio-LLM | float16 `[frames, 3584]` | no |
| Locator | Layer-2 latents + templates | Top-20 candidate pool | no |
| Reranker | pool + static character readings + WAV | per-candidate NLL per arm | only O/T diagnostics |
| Fusion | acoustic and LM columns | fused rank | no |
| Audit | pool, LM, reading | prefix-pollution report | eval-only |

## Frozen constants

- Acoustic latent frame: `0.08 s`
- Character reading frame: `0.04 s`
- Locator window: `5` frames by default; `8` frames plus enrollment mean
  centering is the validated real-corpus setting
- Candidate pool: `20`
- Grid-snapping tolerance: `6` reading frames
- LM fusion weight: `4.0`; the validated plateau is `4.0` to `4.75`. The old
  `1.5` was inherited from an earlier experiment and was not tuned.

The deployable reranking arms are A, P, X, S, and C; C is the default. O and T
require transcript timestamps and exist only to expose the remaining headroom.

## Optional optimization paths

The published baseline remains the default. The following paths are explicit
opt-in ablations:

- `locate --adaptive-window --multi-scale R`: one window length per keyword from
  the enrollment mean duration, evaluated at $k_w\pm R$ frames.
- `locate --prototypes M --nms-candidates T`: deterministic per-word k-means
  prototypes and separated alternative starts (`ps_alt` in pool rows).
- `locate --whitening center|abtt|whiten`: calibrator fitted on enrollment or a
  separate unlabeled background manifest, never on labels.
- `compare-fusions`: weighted layer pooling (`0:0.7,2:0.3`) or concatenation
  (`concat:0,2`) for Stage 1.
- `rerank --phonetic-snap`: pinyin-key edit-distance matching before the 40 ms
  grid snap; `--alternate-starts` re-scores NMS start candidates.
- `rerank --pmi` + `fuse --calibration pmi`: subtract the text-only prior from
  the audio-conditional candidate log probability inside each pool.
- `analyze-shape --gating`: enable the first-difference shape term only when the
  LM pool entropy exceeds the configured threshold.

Real-corpus outcome (see `docs/REAL_CORPUS_RESULTS.md`):

| Path | Result |
| --- | --- |
| k8 + mean centering | speaker-disjoint CV +3.70 Top-1 points |
| duration-adaptive / grow-span | no gain over k8+center |
| multi-prototype | no gain; Top-2 NMS becomes positive at LM weight 4.0 but trails phonetic snapping |
| ABTT / PCA whitening | large pool-recall loss |
| layer late fusion | no gain over `llm2` |
| phonetic snap | +0.80 point over recomputed C at LM weight 4.0 |
| LM weight 1.5 -> 4.0 | +1.05 point on the same recomputed runner; nested CV +0.80 |
| PMI calibration | negative at gamma >= 0.75 |
| DTW-gated shape | positive on LM-only; zero within noise at LM weight 4.0, no deployable gain |

## KV-cache protocol

`StepAudio2Scorer` encodes the instruction + audio prompt once per WAV, keeps
that `DynamicCache`, and clones it for each unique prefix group. Prefix tokens
are prefilled once per group; candidates with the same token length are then
scored in one batched incremental forward against a batch-expanded copy of the
prefix cache. Rows are padded to a 40-row floor because the installed
cuBLAS/torch build aborts on some tiny bf16/fp16 GEMMs (see below). The
full-sequence audit path groups by both prefix-token length and candidate-token
length and passes position ids derived from the attention mask; omitting either
changes RoPE positions when prefixes have unequal length.

Measured on an H20 with 20 candidates and one shared prefix (median of five
warm runs, fp16):

| carrier audio | full-sequence | cached | speed-up | peak extra memory |
|---|---|---|---|---|
| 1.0 s | 0.283 s | 0.160 s | 1.77x | 174 -> 154 MB |
| 3.6 s | 0.239 s | 0.183 s | 1.30x | 127 -> 123 MB |
| 6.4 s | 0.301 s | 0.141 s | 2.14x | 192 -> 163 MB |

Per-candidate score parity is not bit-exact. On the 200-utterance Stage-2 audit
the largest difference is 0.047 nats, minimum Spearman is 0.997, C-arm Top-1 is
identical on 199/200 utterances, and all four reported metrics are exactly the
same. A shared-prefix microbenchmark is tighter (0.017 nats). The speed-up is
modest because the baseline already batches same-length candidates and the
candidate strings are only 1-4 tokens long; caching pays off mainly for long
shared prefixes and many candidates. Claims of "5-10x" or "80% memory" are not
supported by these measurements and must not be quoted.

Runtime hazard: on torch 2.3.1+cu121 with the H20 driver, bf16 bias-epilogue
GEMMs abort with SIGFPE for some row counts (for example 884 rows) and plain
bf16 GEMMs abort below 17 rows. fp16 is stable for those shapes but aborts at
exactly 32 rows when `K=18944`. `load_model_and_adapter` therefore defaults to
fp16 on CUDA, `install_bf16_linear_guard` removes the fused bias epilogue and
pads tiny bf16 GEMMs, and all incremental prefills are padded to 40 rows.
Pass `--dtype bfloat16` only on a runtime that has been re-tested.

## Why grid snapping is not post-hoc tuning

P was the pre-registered naive policy. The audit showed why it failed, and C
changed only the boundary choice using data already present in the pipeline. No
new model, feature, annotation, parameter, or fitted threshold is introduced.
The X control remains essential: a wrong-sentence prefix reaches 0.3597, while
the correctly snapped prefix reaches 0.6172 on the same pool.
