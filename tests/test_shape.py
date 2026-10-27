import numpy as np

from analysis.shape import (
    dtw_distance,
    first_difference,
    resample,
    sequence_entropy,
    shape_fusion_delta,
    shape_similarity,
    shape_template,
    shift_invariant_similarity,
    subsequence_dtw_distance,
)


def test_resample_preserves_endpoint_interpolation():
    values = np.array([[0.0, 0.0], [2.0, 2.0]])
    assert np.allclose(resample(values, 3), [[0, 0], [1, 1], [2, 2]])


def test_shape_operator_and_negative_fusion_helper():
    sequence = np.array([[0.0], [1.0], [3.0]])
    assert np.allclose(first_difference(sequence), [[1], [2]])
    template = shape_template([sequence], 1)
    assert shape_similarity(sequence, template, 1) > 0.99
    result = shape_fusion_delta(
        acoustic=[[0.0, 0.1]],
        language_model=[[0.0, 0.2]],
        shape=[[5.0, np.nan]],
        gold=[["candidate-0", "candidate-1"]],
        candidates=[["candidate-0", "candidate-1"]],
        beta=1.0,
    )
    assert result["n"] == 2
    assert result["finite_shape_ratio"] == 0.5


def test_dtw_and_shift_invariant_shape_operators_tolerate_boundary_jitter():
    sequence = np.array([[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]])
    shifted = np.array([[0.0, 0.0], [1.0, 0.0], [0.8, 0.2], [0.2, 0.8]])
    assert dtw_distance(sequence, sequence) < 1e-12
    assert dtw_distance(sequence, shifted) < 0.25
    assert subsequence_dtw_distance(sequence, shifted) < 0.25
    assert shift_invariant_similarity(sequence, shifted, max_shift=2) > 0.9


def test_entropy_gating_only_activates_for_uncertain_lm_pools():
    assert sequence_entropy(np.array([10.0, -10.0])) < 0.1
    assert sequence_entropy(np.array([0.0, 0.0])) > 0.99
    result = shape_fusion_delta(
        acoustic=[[0.0, 0.0], [0.0, 0.0]],
        language_model=[[0.0, 0.0], [10.0, -10.0]],
        shape=[[2.0, 0.0], [2.0, 0.0]],
        gold=[["candidate-0"], ["candidate-0"]],
        candidates=[["candidate-0", "candidate-1"], ["candidate-0", "candidate-1"]],
        beta=1.0,
        gating=True,
        entropy_threshold=0.5,
    )
    assert result["gate_rate"] == 0.5
    assert result["gated_top1"] == 1.0
