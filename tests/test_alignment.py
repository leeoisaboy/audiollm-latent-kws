from reranker.align import (
    build_contexts,
    cut_text,
    phoneme_distance,
    phonetic_key,
    snap_index,
    snapped_cut_text,
)


READ = [(0, "请"), (2, "帮"), (4, "我"), (6, "开"), (8, "灯"), (10, "好吗")]


def test_naive_cut_leaks_the_candidates_own_first_character():
    prefix = cut_text(READ, 0.48, 0.04)
    assert "开" in prefix


def test_grid_snapping_removes_the_leak():
    prefix = snapped_cut_text(READ, "开灯", 0.48, 0.04)
    assert "开" not in prefix
    assert prefix == "请帮我"


def test_snap_index_prefers_the_nearest_matching_character():
    assert snap_index(READ, "开", 0.48, 0.04) == 3
    assert snap_index(READ, "好", 0.48, 0.04) == 5


def test_seven_arms_are_available_and_oracle_is_explicit():
    contexts = build_contexts(
        ["开灯"],
        [6],
        READ,
        0.04,
        frame_s=0.04,
        variants=("A", "P", "X", "S", "T", "C", "O"),
        null_chars=[(0, "别"), (2, "的")],
        gold=[("开灯", 6.0, 8.0)],
        transcript="请帮我开灯好吗",
    )
    assert tuple(contexts) == ("A", "P", "X", "S", "T", "C", "O")
    assert contexts["A"] == [""]
    assert "开" not in contexts["C"][0]
    assert contexts["T"][0] == "请帮我"
    assert contexts["O"][0] == "请帮我"


def test_phonetic_snap_recovers_a_homophone_first_character():
    reading = [(0, "请"), (2, "官"), (4, "窗"), (6, "好")]
    assert phoneme_distance(phonetic_key("关"), phonetic_key("官")) <= 1
    assert snap_index(reading, "关", 0.08, 0.04, tolerance=0) == 1
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
