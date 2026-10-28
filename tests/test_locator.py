import numpy as np

from locator import (
    DirectoryLatentStore,
    build_pool,
    build_templates,
    fit_latent_calibrator,
)
from locator.search import fixed_span, window_means


def test_mean_template_localizes_a_synthetic_word(tmp_path):
    dimension = 8
    center = np.zeros(dimension)
    center[0] = 5.0
    enrollment = np.random.default_rng(7).normal(scale=0.05, size=(6, dimension))
    enrollment[1:4] += center
    np.save(tmp_path / "clip-0.npy", enrollment)
    template_rows = [
        {"clip_id": "clip-0", "word": "开灯", "kw_t0": 0.08, "kw_t1": 0.40}
    ]
    bank = build_templates(
        template_rows, DirectoryLatentStore(tmp_path, expected_width=dimension)
    )

    search = np.random.default_rng(11).normal(scale=0.05, size=(12, dimension))
    search[5:10] += center
    pool = build_pool("utt", search, bank, pool_top=1, window=5)
    assert pool.words == ["开灯"]
    assert 3 <= int(pool.starts[0]) <= 7


def test_mean_centering_removes_the_enrollment_offset():
    rng = np.random.default_rng(31)
    base = rng.normal(scale=0.05, size=(80, 6))
    offset = np.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    features = base + offset
    centered = fit_latent_calibrator([features], mode="center")
    assert np.allclose(centered.transform(features).mean(axis=0), 0.0, atol=1e-8)


def test_window_means_and_fixed_span_have_the_expected_shapes():
    features = np.arange(20, dtype=np.float64).reshape(10, 2)
    means = window_means(features, 4)
    assert means is not None and means.shape == (7, 2)
    assert np.allclose(means[0], features[:4].mean(axis=0))
    assert fixed_span(8, 4, 10) == (8, 10)
