import numpy as np

from analysis.audit import audit_prefix_pollution
from analysis.evaluation import score_records
from common.scoring import fuse, rank_of, zpool
from scripts.stage2_real_ablation import published_scores


def test_pool_zscore_prevents_raw_scale_domination():
    acoustic = np.array([0.80, 0.81, 0.79])
    language_model = np.array([-2.0, -0.1, -9.0])
    fused = fuse(acoustic, language_model, lambda_lm=1.5)
    assert int(np.argmax(fused)) == 1
    assert zpool(acoustic).std() > 0.9


def test_rank_counts_ties_as_half():
    assert rank_of([3, 3, 2], 0) == 1.5
    assert rank_of([3, 3, 2], 2) == 3.0


def test_audit_reports_pollution_for_naive_and_snapped_cuts():
    reading = [(0, "请"), (2, "帮"), (4, "我"), (6, "开"), (8, "灯")]
    pool = [
        {
            "utt": "u",
            "words": ["开灯"],
            "ps": [6],
            "score": [0.9],
            "gold": [{"word": "开灯", "ts": 6, "te": 8}],
        }
    ]
    report = audit_prefix_pollution(pool, {"u": {}}, {"u": reading}, 0.04, null_shift=0)
    assert report["P"]["contains_own_first_char"] == 1.0
    assert report["C"]["contains_own_first_char"] == 0.0


def test_pmi_calibration_removes_a_text_only_frequency_prior():
    pool = [
        {
            "utt": "u",
            "words": ["common", "target"],
            "score": [0.50, 0.49],
            "ps": [1, 2],
            "ntop": 2,
            "gold": [{"word": "target"}],
        }
    ]
    lm = {
        "u": {
            "scores": {"common": {"C": -0.2}, "target": {"C": -0.3}},
            "text_scores": {"common": {"C": -0.1}, "target": {"C": -2.0}},
        }
    }
    without_pmi = score_records(pool, lm, "C", lambda_lm=1.5)
    with_pmi = score_records(pool, lm, "C", lambda_lm=1.5, calibration="pmi", gamma=1.0)
    assert without_pmi[0]["winner"] == "common"
    assert with_pmi[0]["winner"] == "target"


def test_published_scores_transposes_the_word_and_arm_axes():
    rows = [
        {
            "utt": "u1",
            "scores": {
                "开灯": {"P": -2.0, "C": -0.5},
                "关窗": {"P": -3.0, "C": -4.0},
            },
        }
    ]
    scores = published_scores(rows)["u1"]["scores"]
    assert scores["C"] == {"开灯": -0.5, "关窗": -4.0}
    assert scores["P"]["关窗"] == -3.0
