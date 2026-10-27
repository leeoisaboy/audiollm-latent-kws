import numpy as np

from scripts.shape_gate_real import baseline_scores, zscore


def test_fused_baseline_matches_the_deployable_pool_zscore():
    rows = [
        {"ac": [0.8, 0.4, 0.1], "lm_C": [-0.2, -3.0, -7.0]},
        {"ac": [0.2, 0.9, 0.5], "lm_C": [-5.0, -0.1, -2.0]},
    ]
    fused = baseline_scores(rows, "fused", lambda_lm=1.5)
    for row, value in zip(rows, fused):
        expected = zscore(np.asarray(row["ac"])) + 1.5 * zscore(
            np.asarray(row["lm_C"])
        )
        assert np.allclose(value, expected)


def test_lm_baseline_preserves_raw_lm_scores():
    rows = [{"ac": [0.8, 0.4, 0.1], "lm_C": [-0.2, -3.0, -7.0]}]
    values = baseline_scores(rows, "lm", lambda_lm=1.5)
    assert np.allclose(values[0], rows[0]["lm_C"])
