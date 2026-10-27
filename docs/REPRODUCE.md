# Reproduction checklist

1. Create the private test manifest as `dialect_kws_testset/manifest.jsonl`.
   Keep one row per utterance and record dataset revision and license outside
   the repository.
2. Extract frozen audio hidden states with your model runner. Save each clip as
   `<features-root>/layer_<n>/enroll/<clip_id>.npy` and
   `<features-root>/layer_<n>/search/<clip_id>.npy` on the `[frames, 3584]`
   grid. `integration/step_audio2.py` can export layers
   `0,1,2,3,4,8` in one forward pass and supports `--quantization 4bit`.
3. Build enrollment rows in `reference/templates.jsonl`. Use multiple speakers
   per keyword, but never fit the centroid on the search split.
4. For every layer, either run `cli.py locate --window 8 --whitening center
   --pool-top 20 --include-gold` followed by `cli.py evaluate --pool-top 20`, or run
   `cli.py compare-layers` once over the feature root.
5. Import a static character-reading timestamp file on its original 40 ms grid.
   Do not smooth, resegment, or convert timestamps to the acoustic grid. This
   repository consumes the JSONL result only and ships no recognizer.
6. Implement an offline `AudioLMScorer`. Teacher-force every candidate after the
   instruction, audio, and its arm-specific prefix. Keep fp32 logits when the
   candidate token count is small.
7. Score A/P/X/S/C for deployable results; add O/T only behind an explicit
   evaluation-only flag.
8. Fuse with `z_pool(acoustic) + 4.0 z_pool(LM)` and report both occurrence and
   per-word Top-1. The validated plateau is 4.0 to 4.75; `1.5` was an inherited
   value, not a tuned optimum.
9. Run `cli.py audit` and publish the prefix-pollution table next to accuracy.
10. For the shape ablation, export aligned candidate columns (`lm_C`, `d1`,
    `res`, `vel`) and run `cli.py analyze-shape`; do not enable shape for
    production ranking without beating the published fixed-weight negative
    result.
11. For Stage-1 extensions, run baseline and each flag family (`adaptive`,
    `prototypes`, `whitening`, `layer fusion`) on the same manifest. Report the
    pool-recall ceiling separately from within-pool Top-1. Then run the
    nested CV with `scripts/stage1_cv.py --group speaker`; select the window
    and calibration only on the training groups.
12. For Stage-2, run `--no-kv-cache` and cached scoring on a held-out subset.
    Full-sequence candidates must be grouped by both prefix-token length and
    candidate-token length; otherwise left padding changes RoPE positions.
    Require high rank agreement before using cache results. Then compare `C`,
    `C + phonetic`, `C + alternate starts`, and PMI calibration. PMI needs the
    text-only prior, so PMI arms are omitted under `--no-kv-cache` instead of
    being silently reported as equal to plain `C`.
13. For shape, report fixed-weight and entropy-gated results side by side,
    including gate rate and mean LM entropy; never tune the gate on test labels.
    Use `scripts/shape_gate_cv.py --group speaker` on both the LM-only and the
    deployable `acoustic + 4.0 LM` baselines. A gain on the former does not
    establish an end-to-end gain.
14. For KV-cache claims, report warm-run medians, peak extra memory, max score
    difference and Top-1 agreement together. Do not quote speed-ups from a
    different candidate count, prefix length or dtype.

Report conventions:

- Pool recall is measured at Top 20, not with the number of dumped candidates.
- Per-word Top-1 averages per keyword first, then across keywords.
- O and T are ceilings, never headline system scores.
- Synthetic CPU fixtures verify code paths only; they do not reproduce corpus
  metrics.
- `reports/stage1_cv_speaker.json` is the primary Stage-1 generalization result;
  same-corpus ablations remain exploratory.
- `reports/stage2_cache_parity.json` covers the cache/full-sequence audit;
  `reports/stage2_real_ablation.json` covers the four scored Stage-2 branches.
