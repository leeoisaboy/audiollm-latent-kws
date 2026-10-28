# Evidence index

These small JSON files support the numbers in the public README. Raw LM score
dumps, per-utterance shape dumps, generated pools, and local smoke output are
intentionally not committed.

| File | What it supports |
| --- | --- |
| `stage1_cv_speaker.json` | Primary speaker-grouped nested CV for Stage-1 window and calibration |
| `stage1_cv.json` | Utterance-grouped counterpart |
| `layer_comparison.json` | Small layer probe; layer 0 is promising but not full-split validated |
| `stage1_grow.json`, `stage1_kmeans.json`, `stage1_layer_fusion*.json` | Rejected Stage-1 variants |
| `stage2_lambda_sweep.json` | Full-set LM-weight curve, PMI arms, and paired bootstrap |
| `lambda_cv_*.json` | Grouped cross-checks that select arm and weight on training utterances |
| `stage2_real_ablation.json` | Stage-2 C/P/phonetic/Top-2/PMI comparison at the inherited weight |
| `prefix_audit_real.json` | 3244-occurrence first-character leakage audit |
| `shape_gate_cv_fused_l4*.json` | Negative end-to-end temporal-shape result at LM weight 4.0 |
| `stage2_cache_parity.json` | Cached versus full-sequence score and ranking parity |
| `cache_benchmark.json` | Warm latency and peak-memory benchmark |

Group assignments use the corpus filename rule `rec_sXX`; this is not an
official speaker release. See `docs/REAL_CORPUS_RESULTS.md` for the full
evidence boundaries.
