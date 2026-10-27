import numpy as np

from locator import (
    DirectoryLatentStore,
    build_pool,
    build_templates,
    fit_latent_calibrator,
    grow_span,
)


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
    pool = build_pool("utt", search, bank, pool_top=1)
    assert pool.words == ["开灯"]
    assert 3 <= int(pool.starts[0]) <= 7


def test_duration_adaptive_windows_multi_prototypes_and_nms(tmp_path):
    dimension = 8
    rng = np.random.default_rng(23)
    rows = []
    for word_index, (word, duration) in enumerate((("开灯", 0.24), ("打开电视", 0.64))):
        center = np.zeros(dimension)
        center[word_index] = 5.0
        for take in range(4):
            clip_id = f"{word_index}-{take}"
            frames = rng.normal(scale=0.03, size=(14, dimension))
            frames[3:8] += center + take * 0.05
            np.save(tmp_path / f"{clip_id}.npy", frames)
            rows.append(
                {"clip_id": clip_id, "word": word, "kw_t0": 0.16, "kw_t1": 0.16 + duration}
            )
    bank = build_templates(
        rows,
        DirectoryLatentStore(tmp_path, expected_width=dimension),
        prototypes=2,
    )
    assert bank.windows["打开电视"] > bank.windows["开灯"]
    assert len(bank.prototypes_for("打开电视")) == 2
    search = rng.normal(scale=0.03, size=(20, dimension))
    search[5:10] += np.array([5.0] + [0.0] * (dimension - 1))
    pool = build_pool(
        "utt",
        search,
        bank,
        pool_top=2,
        adaptive_window=True,
        multi_scale=1,
        nms_candidates=2,
        nms_radius=2,
    )
    assert pool.words[0] == "开灯"
    assert len(pool.alt_starts[0]) == 2
    assert abs(pool.alt_starts[0][0] - pool.alt_starts[0][1]) > 2


def test_mean_and_abtt_calibrators_remove_anisotropic_offsets():
    rng = np.random.default_rng(31)
    base = rng.normal(scale=0.05, size=(80, 6))
    offset = np.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    features = base + offset
    centered = fit_latent_calibrator([features], mode="center")
    assert np.allclose(centered.transform(features).mean(axis=0), 0.0, atol=1e-8)
    abtt = fit_latent_calibrator([features], mode="abtt", rank=1)
    transformed = abtt.transform(features)
    first_component = abtt.components[0]
    assert abs(float(transformed.mean(axis=0) @ first_component)) < 1e-7


def test_grow_span_extends_the_peak_plateau_but_stops_at_the_valley():
    column = np.array([0.1, 0.2, 0.8, 0.9, 0.9, 0.85, 0.3, 0.1])
    assert grow_span(column, 3, 2, len(column), delta=0.1) == (2, 6)
