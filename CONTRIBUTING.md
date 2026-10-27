## Contributing

1. Run `python -m compileall .` and `pytest` before submitting a patch.
2. Keep locator, reranker, and analysis responsibilities separate.
3. Preserve the 80 ms and 40 ms grid distinction in names, comments, and tests.
4. Do not add fitted thresholds or evaluation-set-dependent constants without a
   pre-registered protocol and a held-out plan.
5. Every production ranking path must have a non-oracle test.
6. Analysis-only negative results are welcome; label them as audit operators.
