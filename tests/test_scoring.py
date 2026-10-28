import numpy as np

from analysis.audit import audit_prefix_pollution, naive_cut_text
from analysis.evaluation import score_records
from common.config import is_cjk
from common.scoring import fuse, rank_of, zpool


def test_pool_zscore_prevents_raw_scale_domination():
    acoustic = np.array([0.80, 0.81, 0.79])
    language_model = np.array([-2.0, -0.1, -9.0])
    fused = fuse(acoustic, language_model, lambda_lm=1.5)
    assert int(np.argmax(fused)) == 1
    assert zpool(acoustic).std() > 0.9


def test_rank_counts_ties_as_half():
    assert rank_of([3, 3, 2], 0) == 1.5
    assert rank_of([3, 3, 2], 2) == 3.0


def test_empty_symbol_is_not_cjk():
    assert not is_cjk("")


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
    assert naive_cut_text(reading, 0.24, 0.04) == "请帮我开"


def test_fusion_uses_the_single_production_arm():
    pool = [
        {
            "utt": "u",
            "words": ["开灯", "关窗"],
            "score": [0.50, 0.49],
            "ps": [1, 2],
            "ntop": 2,
            "gold": [{"word": "关窗"}],
        }
    ]
    lm = {"u": {"scores": {"开灯": {"C": -0.2}, "关窗": {"C": -3.0}}}}
    records = score_records(pool, lm, "C", lambda_lm=4.0)
    assert records[0]["winner"] == "开灯"
    assert records[0]["rank"] == 2.0
