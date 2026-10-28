from reranker.align import (
    build_prefixes,
    phoneme_distance,
    phonetic_key,
    snap_index,
    snapped_cut_text,
)


READ = [(0, "请"), (2, "帮"), (4, "我"), (6, "开"), (8, "灯"), (10, "好吗")]


def test_grid_snapping_removes_the_candidate_first_character():
    prefix = snapped_cut_text(READ, "开灯", 0.48, 0.04)
    assert prefix == "请帮我"
    assert "开" not in prefix


def test_snap_index_prefers_the_nearest_matching_character():
    assert snap_index(READ, "开", 0.48, 0.04) == 3
    assert snap_index(READ, "好", 0.48, 0.04) == 5


def test_build_prefixes_returns_one_prefix_per_candidate():
    prefixes = build_prefixes(["开灯", "关窗"], [6, 6], READ, 0.04, frame_s=0.04)
    assert prefixes[0] == "请帮我"
    assert len(prefixes) == 2


def test_phonetic_snap_recovers_a_homophone_first_character():
    reading = [(0, "请"), (2, "官"), (4, "窗"), (6, "好")]
    assert phoneme_distance(phonetic_key("关"), phonetic_key("官")) <= 1
    assert snap_index(
        reading, "关", 0.08, 0.04, tolerance=0, phonetic=True
    ) == 1
    prefix = snapped_cut_text(
        reading,
        "关窗",
        0.08,
        0.04,
        tolerance=0,
        phonetic=True,
    )
    assert prefix == "请"
    assert "关" not in prefix
