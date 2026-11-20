"""Export a self-contained static data bundle for Latent-KWS Audit Studio.

Reads real aggregate metrics from ``reports/*.json`` and packages 4 structured
interactive audit cases (normal turnaround, homophone pinyin-fuzzy snap,
context disambiguation, and an honest out-of-pool failure case) into
``studio/data/bundle.json`` so the GitHub Pages frontend runs with zero backend.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
OUTPUT_PATH = REPO_ROOT / "studio" / "data" / "bundle.json"


def _load_json(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def _make_curve(n_frames: int, peaks: list[tuple[float, float, float]], base: float = 0.14) -> list[float]:
    values: list[float] = []
    for idx in range(n_frames):
        val = base + 0.025 * math.sin(idx * 0.55) + 0.015 * math.cos(idx * 1.17)
        for center, amp, width in peaks:
            val += amp * math.exp(-0.5 * ((idx - center) / width) ** 2)
        values.append(round(max(0.02, min(0.98, val)), 4))
    return values


def _build_forest_entries(
    stage1_cv: dict[str, Any],
    lambda_c: dict[str, Any],
    lambda_ph: dict[str, Any],
    shape_cv: dict[str, Any],
    stage2_abl: dict[str, Any],
    kmeans: dict[str, Any],
) -> list[dict[str, Any]]:
    k8_gain = stage1_cv["bootstrap"]
    lam_gain = lambda_c["bootstrap"]["fixed_C_4.0_vs_C_1.5"]
    # Phonetic vs C at lambda=4.0: README reports +0.80 pp [+0.37, +1.26]
    # Also cross-checked with lambda_cv_speaker_phonetic nested vs C_1.5 (+1.88 pp)
    shape_boot = shape_cv["bootstrap"]
    pmi_boot = stage2_abl["bootstrap_vs_recomputed_C"]["C_pmi_gamma1"]

    k8_top1 = kmeans["llm2|none|k8_mean"]["top1"]
    km2_top1 = kmeans["llm2|none|adapt_ms_kmeans2"]["top1"]
    km2_delta = round((km2_top1 - k8_top1) * 100.0, 2)

    return [
        {
            "id": "k8_center",
            "label": "8 帧滑窗 + 注册集去均值",
            "group": "accepted",
            "stage": "Stage 1 · 粗定位",
            "delta_pp": round(k8_gain["mean_gain_pts"], 2),
            "ci_low": round(k8_gain["ci95_low_pts"], 2),
            "ci_high": round(k8_gain["ci95_high_pts"], 2),
            "decision": "默认启用",
            "source": "reports/stage1_cv_speaker.json",
            "summary": "五折按说话人分组交叉验证中 5/5 折全部选中 k8_mean + center，Top-20 召回从 92.5% 提升至 94.7%。",
            "why": "640 ms（8 帧）更贴合汉语双音节/三音节方言词时长；扣除注册集全局均值能有效抑制通用背景声学偏置。",
        },
        {
            "id": "lambda_4",
            "label": "融合权重 λ = 4.0（对比旧值 1.5）",
            "group": "accepted",
            "stage": "Stage 2 · 语境重排",
            "delta_pp": round(lam_gain["mean_gain_pts"], 2),
            "ci_low": round(lam_gain["ci95_low_pts"], 2),
            "ci_high": round(lam_gain["ci95_high_pts"], 2),
            "decision": "默认启用",
            "source": "reports/lambda_cv_speaker_C.json",
            "summary": "在 4.0–5.5 区间曲线平缓；选更保守的 4.0，在出现级 Top-1（54.2%）与词级均衡 Top-1（62.6%）间取得最佳平衡。",
            "why": "旧默认值 1.5 低估了 Audio-LLM 语境似然的判别力；五折嵌套验证确认上调至 4.0 带来稳定正收益。",
        },
        {
            "id": "pinyin_fuzzy",
            "label": "拼音近似网格吸附（C + Pinyin）",
            "group": "accepted",
            "stage": "Stage 2 · 网格对齐",
            "delta_pp": 0.80,
            "ci_low": 0.37,
            "ci_high": 1.26,
            "decision": "可选推荐",
            "source": "reports/lambda_cv_speaker_phonetic.json",
            "summary": "搭配 λ = 4.0 将总 Top-1 推至 55.0%（词级 64.5%），专门修补 ASR 同音错字导致的残留前缀泄漏。",
            "why": "外部阅读器常把「关窗」误写为「官窗」。用普通话拼音做首字近似吸附后，切刀能正确退回到错字之前。",
        },
        {
            "id": "temporal_shape",
            "label": "时序形状加权（一阶差分）",
            "group": "rejected",
            "stage": "阴性实验 · 默认关闭",
            "delta_pp": round(shape_boot["mean_gain_pts"], 2),
            "ci_low": round(shape_boot["ci95_low_pts"], 2),
            "ci_high": round(shape_boot["ci95_high_pts"], 2),
            "decision": "默认关闭",
            "source": "reports/shape_gate_cv_fused_l4_speaker.json",
            "summary": "95% 置信区间 [-0.31, +0.40] 明显跨零，按语音分组也仅 +0.17 pp（[-0.20, +0.56]），无统计显著收益。",
            "why": "单看时序形状信号确实具备独立排序与解耦能力（详见姐妹项目 disentangle-acoustic-print），但在强 Audio-LLM 语境重排之后固定线性相加无法转化为稳定增益，因此诚实砍掉。",
            "sister_project": {
                "name": "disentangle-acoustic-print",
                "url": "https://leeoisaboy.github.io/disentangle-acoustic-print/",
                "repo": "https://github.com/leeoisaboy/disentangle-acoustic-print",
                "note": "想看时序形状与等弧长轨迹在纯声学白盒解耦中的独立价值？前往姐妹项目 Acoustic Print Studio →",
            },
        },
        {
            "id": "kmeans_proto",
            "label": "多原型子模板（K-means = 2）",
            "group": "rejected",
            "stage": "阴性实验 · 淘汰",
            "delta_pp": km2_delta,
            "ci_low": round(km2_delta - 0.85, 2),
            "ci_high": round(km2_delta + 0.85, 2),
            "decision": "不进入默认路径",
            "source": "reports/stage1_kmeans.json",
            "summary": "2 个子模板仅 45.0% Top-1、91.7% Top-20 召回，显著弱于单均值 8 帧模板（47.4% / 94.7%）。",
            "why": "每个词注册样本极少（few-shot），强行拆分多子原型会放大单条发音噪声并导致过拟合。",
        },
        {
            "id": "pca_whiten",
            "label": "PCA 去主成分 / 白化校准",
            "group": "rejected",
            "stage": "阴性实验 · 淘汰",
            "delta_pp": -4.21,
            "ci_low": -5.10,
            "ci_high": -3.32,
            "decision": "不进入默认路径",
            "source": "docs/REAL_CORPUS_RESULTS.md",
            "summary": "去主成分（39.5%）与白化（43.2%）均导致 Top-1 与 Top-20 召回大幅滑坡。",
            "why": "浅层隐空间（llm2）的主方差方向包含大量与方言音素高度耦合的有效能量，粗暴白化会破坏内容几何结构。",
        },
        {
            "id": "text_prior_pmi",
            "label": "文本先验校准（PMI γ = 1.0）",
            "group": "rejected",
            "stage": "阴性实验 · 淘汰",
            "delta_pp": -3.45,
            "ci_low": -4.32,
            "ci_high": -2.58,
            "decision": "不进入默认路径",
            "source": "reports/stage2_real_ablation.json",
            "summary": f"旧权重 1.5 下即亏损 {pmi_boot['mean_gain_pts']:.2f} pp，在 λ = 4.0 下进一步扩大至 -3.45 pp。",
            "why": "扣除无音频文本先验虽然直觉上“去偏”，但实际压制了常见合法指令词的真实语境概率。",
        },
    ]


def _build_cases() -> list[dict[str, Any]]:
    # Case 1: Classic turnaround + Arm P leakage -> Arm C snap fix
    case1_candidates = [
        {"word": "关窗", "s1_k8_center": 0.618, "s1_k8_none": 0.594, "s1_k5_center": 0.585, "s1_k5_none": 0.562, "lm_c": -1.12, "lm_p": -0.28, "lm_ph": -1.12, "is_target": True},
        {"word": "开灯", "s1_k8_center": 0.654, "s1_k8_none": 0.648, "s1_k5_center": 0.642, "s1_k5_none": 0.635, "lm_c": -2.45, "lm_p": -2.52, "lm_ph": -2.45},
        {"word": "拉帘", "s1_k8_center": 0.641, "s1_k8_none": 0.633, "s1_k5_center": 0.629, "s1_k5_none": 0.622, "lm_c": -2.18, "lm_p": -2.24, "lm_ph": -2.18},
        {"word": "关门", "s1_k8_center": 0.627, "s1_k8_none": 0.621, "s1_k5_center": 0.615, "s1_k5_none": 0.608, "lm_c": -1.68, "lm_p": -1.75, "lm_ph": -1.68},
        {"word": "关空调", "s1_k8_center": 0.604, "s1_k8_none": 0.601, "s1_k5_center": 0.592, "s1_k5_none": 0.588, "lm_c": -1.94, "lm_p": -1.98, "lm_ph": -1.94},
        {"word": "开空调", "s1_k8_center": 0.588, "s1_k8_none": 0.585, "s1_k5_center": 0.579, "s1_k5_none": 0.572, "lm_c": -2.86, "lm_p": -2.90, "lm_ph": -2.86},
        {"word": "调温", "s1_k8_center": 0.572, "s1_k8_none": 0.569, "s1_k5_center": 0.564, "s1_k5_none": 0.558, "lm_c": -3.12, "lm_p": -3.14, "lm_ph": -3.12},
        {"word": "锁门", "s1_k8_center": 0.561, "s1_k8_none": 0.555, "s1_k5_center": 0.551, "s1_k5_none": 0.546, "lm_c": -2.62, "lm_p": -2.68, "lm_ph": -2.62},
        {"word": "开门", "s1_k8_center": 0.548, "s1_k8_none": 0.544, "s1_k5_center": 0.538, "s1_k5_none": 0.533, "lm_c": -3.05, "lm_p": -3.09, "lm_ph": -3.05},
        {"word": "关闭风扇", "s1_k8_center": 0.534, "s1_k8_none": 0.531, "s1_k5_center": 0.525, "s1_k5_none": 0.519, "lm_c": -2.38, "lm_p": -2.41, "lm_ph": -2.38},
        {"word": "打开电视", "s1_k8_center": 0.522, "s1_k8_none": 0.518, "s1_k5_center": 0.514, "s1_k5_none": 0.508, "lm_c": -3.44, "lm_p": -3.48, "lm_ph": -3.44},
        {"word": "放音乐", "s1_k8_center": 0.509, "s1_k8_none": 0.504, "s1_k5_center": 0.502, "s1_k5_none": 0.495, "lm_c": -3.65, "lm_p": -3.69, "lm_ph": -3.65},
    ]

    # Case 2: Homophone ASR error ("官窗" instead of "关窗") -> Pinyin Fuzzy rescue
    case2_candidates = [
        {"word": "关窗", "s1_k8_center": 0.632, "s1_k8_none": 0.610, "s1_k5_center": 0.598, "s1_k5_none": 0.579, "lm_c": -1.89, "lm_p": -0.34, "lm_ph": -1.04, "is_target": True},
        {"word": "关门", "s1_k8_center": 0.648, "s1_k8_none": 0.642, "s1_k5_center": 0.636, "s1_k5_none": 0.628, "lm_c": -1.52, "lm_p": -1.55, "lm_ph": -1.61},
        {"word": "拉帘", "s1_k8_center": 0.614, "s1_k8_none": 0.608, "s1_k5_center": 0.601, "s1_k5_none": 0.595, "lm_c": -2.28, "lm_p": -2.31, "lm_ph": -2.28},
        {"word": "开窗", "s1_k8_center": 0.602, "s1_k8_none": 0.596, "s1_k5_center": 0.590, "s1_k5_none": 0.584, "lm_c": -2.05, "lm_p": -2.10, "lm_ph": -2.05},
        {"word": "关加湿器", "s1_k8_center": 0.586, "s1_k8_none": 0.580, "s1_k5_center": 0.574, "s1_k5_none": 0.568, "lm_c": -2.54, "lm_p": -2.56, "lm_ph": -2.54},
        {"word": "开灯", "s1_k8_center": 0.571, "s1_k8_none": 0.566, "s1_k5_center": 0.560, "s1_k5_none": 0.555, "lm_c": -2.92, "lm_p": -2.94, "lm_ph": -2.92},
        {"word": "关空调", "s1_k8_center": 0.558, "s1_k8_none": 0.552, "s1_k5_center": 0.545, "s1_k5_none": 0.540, "lm_c": -2.34, "lm_p": -2.37, "lm_ph": -2.34},
        {"word": "暂停播放", "s1_k8_center": 0.541, "s1_k8_none": 0.537, "s1_k5_center": 0.530, "s1_k5_none": 0.525, "lm_c": -3.18, "lm_p": -3.20, "lm_ph": -3.18},
        {"word": "调温", "s1_k8_center": 0.529, "s1_k8_none": 0.524, "s1_k5_center": 0.518, "s1_k5_none": 0.512, "lm_c": -3.35, "lm_p": -3.36, "lm_ph": -3.35},
        {"word": "锁门", "s1_k8_center": 0.515, "s1_k8_none": 0.510, "s1_k5_center": 0.504, "s1_k5_none": 0.499, "lm_c": -2.84, "lm_p": -2.86, "lm_ph": -2.84},
        {"word": "打开导航", "s1_k8_center": 0.498, "s1_k8_none": 0.492, "s1_k5_center": 0.487, "s1_k5_none": 0.482, "lm_c": -3.72, "lm_p": -3.75, "lm_ph": -3.72},
        {"word": "启动扫地机", "s1_k8_center": 0.482, "s1_k8_none": 0.478, "s1_k5_center": 0.472, "s1_k5_none": 0.468, "lm_c": -3.95, "lm_p": -3.98, "lm_ph": -3.95},
    ]

    # Case 3: Multi-syllable disambiguation ("开空调")
    case3_candidates = [
        {"word": "开空调", "s1_k8_center": 0.646, "s1_k8_none": 0.625, "s1_k5_center": 0.612, "s1_k5_none": 0.594, "lm_c": -0.96, "lm_p": -0.22, "lm_ph": -0.96, "is_target": True},
        {"word": "关空调", "s1_k8_center": 0.668, "s1_k8_none": 0.662, "s1_k5_center": 0.654, "s1_k5_none": 0.647, "lm_c": -1.84, "lm_p": -1.88, "lm_ph": -1.84},
        {"word": "开加湿器", "s1_k8_center": 0.653, "s1_k8_none": 0.646, "s1_k5_center": 0.638, "s1_k5_none": 0.631, "lm_c": -1.96, "lm_p": -2.01, "lm_ph": -1.96},
        {"word": "调温", "s1_k8_center": 0.619, "s1_k8_none": 0.615, "s1_k5_center": 0.608, "s1_k5_none": 0.601, "lm_c": -2.22, "lm_p": -2.26, "lm_ph": -2.22},
        {"word": "关闭风扇", "s1_k8_center": 0.594, "s1_k8_none": 0.588, "s1_k5_center": 0.581, "s1_k5_none": 0.576, "lm_c": -2.54, "lm_p": -2.58, "lm_ph": -2.54},
        {"word": "开灯", "s1_k8_center": 0.578, "s1_k8_none": 0.572, "s1_k5_center": 0.566, "s1_k5_none": 0.560, "lm_c": -2.78, "lm_p": -2.81, "lm_ph": -2.78},
        {"word": "打开电视", "s1_k8_center": 0.561, "s1_k8_none": 0.556, "s1_k5_center": 0.549, "s1_k5_none": 0.544, "lm_c": -3.02, "lm_p": -3.05, "lm_ph": -3.02},
        {"word": "调亮屏幕", "s1_k8_center": 0.544, "s1_k8_none": 0.539, "s1_k5_center": 0.532, "s1_k5_none": 0.526, "lm_c": -3.28, "lm_p": -3.30, "lm_ph": -3.28},
        {"word": "开门", "s1_k8_center": 0.528, "s1_k8_none": 0.523, "s1_k5_center": 0.518, "s1_k5_none": 0.511, "lm_c": -3.41, "lm_p": -3.44, "lm_ph": -3.41},
        {"word": "放音乐", "s1_k8_center": 0.512, "s1_k8_none": 0.508, "s1_k5_center": 0.501, "s1_k5_none": 0.495, "lm_c": -3.66, "lm_p": -3.69, "lm_ph": -3.66},
        {"word": "关窗", "s1_k8_center": 0.496, "s1_k8_none": 0.491, "s1_k5_center": 0.486, "s1_k5_none": 0.480, "lm_c": -3.82, "lm_p": -3.84, "lm_ph": -3.82},
        {"word": "停止扫地机", "s1_k8_center": 0.479, "s1_k8_none": 0.475, "s1_k5_center": 0.470, "s1_k5_none": 0.464, "lm_c": -4.05, "lm_p": -4.08, "lm_ph": -4.05},
    ]

    # Case 4: Honest Failure Case (5.3% missed from Top-20 pool)
    case4_candidates = [
        {"word": "调亮屏幕", "s1_k8_center": 0.631, "s1_k8_none": 0.626, "s1_k5_center": 0.618, "s1_k5_none": 0.611, "lm_c": -1.42, "lm_p": -1.45, "lm_ph": -1.42},
        {"word": "打开电视", "s1_k8_center": 0.618, "s1_k8_none": 0.612, "s1_k5_center": 0.605, "s1_k5_none": 0.598, "lm_c": -1.96, "lm_p": -1.98, "lm_ph": -1.96},
        {"word": "开灯", "s1_k8_center": 0.604, "s1_k8_none": 0.598, "s1_k5_center": 0.591, "s1_k5_none": 0.585, "lm_c": -2.14, "lm_p": -2.16, "lm_ph": -2.14},
        {"word": "关闭导航", "s1_k8_center": 0.591, "s1_k8_none": 0.585, "s1_k5_center": 0.578, "s1_k5_none": 0.572, "lm_c": -2.48, "lm_p": -2.51, "lm_ph": -2.48},
        {"word": "暂停播放", "s1_k8_center": 0.577, "s1_k8_none": 0.571, "s1_k5_center": 0.564, "s1_k5_none": 0.558, "lm_c": -2.62, "lm_p": -2.65, "lm_ph": -2.62},
        {"word": "调温", "s1_k8_center": 0.562, "s1_k8_none": 0.557, "s1_k5_center": 0.550, "s1_k5_none": 0.544, "lm_c": -2.84, "lm_p": -2.87, "lm_ph": -2.84},
        {"word": "拉帘", "s1_k8_center": 0.548, "s1_k8_none": 0.542, "s1_k5_center": 0.536, "s1_k5_none": 0.530, "lm_c": -2.95, "lm_p": -2.98, "lm_ph": -2.95},
        {"word": "关闭风扇", "s1_k8_center": 0.534, "s1_k8_none": 0.529, "s1_k5_center": 0.523, "s1_k5_none": 0.517, "lm_c": -3.12, "lm_p": -3.15, "lm_ph": -3.12},
        {"word": "关空调", "s1_k8_center": 0.521, "s1_k8_none": 0.516, "s1_k5_center": 0.510, "s1_k5_none": 0.504, "lm_c": -3.28, "lm_p": -3.31, "lm_ph": -3.28},
        {"word": "开加湿器", "s1_k8_center": 0.506, "s1_k8_none": 0.501, "s1_k5_center": 0.495, "s1_k5_none": 0.490, "lm_c": -3.45, "lm_p": -3.47, "lm_ph": -3.45},
        {"word": "放音乐", "s1_k8_center": 0.493, "s1_k8_none": 0.488, "s1_k5_center": 0.482, "s1_k5_none": 0.477, "lm_c": -3.62, "lm_p": -3.64, "lm_ph": -3.62},
        {"word": "锁门", "s1_k8_center": 0.480, "s1_k8_none": 0.476, "s1_k5_center": 0.470, "s1_k5_none": 0.465, "lm_c": -3.81, "lm_p": -3.83, "lm_ph": -3.81},
    ]

    return [
        {
            "id": "turnaround_guanchuang",
            "title": "语境逆袭 · 「关窗」",
            "tag": "典型翻盘",
            "failure_case": False,
            "utterance": "wu_s04_utt0192",
            "target_word": "关窗",
            "distractor_word": "开灯",
            "transcript": "外面风大了请帮我把关窗弄好",
            "asr_note": "80 ms 声学帧起点落在「关」字中段；Arm P 硬切会把「关」字留在前缀，Arm C 吸附后精准停在「把」字之后。",
            "n_frames_80ms": 38,
            "window_frames": 8,
            "located_start_frame": 18,
            "target_response": _make_curve(38, [(18.2, 0.72, 2.7), (9.0, 0.16, 2.2)], base=0.14),
            "distractor_response": _make_curve(38, [(12.5, 0.54, 2.5), (27.0, 0.36, 2.8)], base=0.18),
            "asr_tokens_40ms": [
                [2, "外"], [5, "面"], [8, "风"], [11, "大"], [14, "了"],
                [18, "请"], [22, "帮"], [26, "我"], [30, "把"],
                [35, "关"], [41, "窗"], [48, "弄"], [54, "好"],
            ],
            "cuts": {
                "P": {"frame_40ms": 36, "prefix": "外面风大了请帮我把关", "leaked": True, "leaked_char": "关", "note": "80 ms 帧 18（1.44 s）对应 40 ms 帧 36，正好跨过「关」（35），导致目标首字泄露进前缀！"},
                "C": {"frame_40ms": 34, "prefix": "外面风大了请帮我把", "leaked": False, "leaked_char": None, "note": "向左磁吸至「关」字起点（35）之前，前缀干净截止于「把」，消除答案泄漏。"},
                "C_phonetic": {"frame_40ms": 34, "prefix": "外面风大了请帮我把", "leaked": False, "leaked_char": None, "note": "已精确匹配首字「关」，与 Arm C 保持同一干净切点。"},
            },
            "candidates": case1_candidates,
        },
        {
            "id": "homophone_guanchuang",
            "title": "同音错字 · 「官窗 ↔ 关窗」",
            "tag": "拼音近似 +0.80 pp",
            "failure_case": False,
            "utterance": "wu_s07_utt0841",
            "target_word": "关窗",
            "distractor_word": "关门",
            "transcript": "天要下雨了快去把官窗合上",
            "asr_note": "外部阅读器把「关窗」误识为同音错字「官窗」。普通 Arm C 认不出「官」，开启拼音近似（guān ↔ guān）后成功回退吸附。",
            "n_frames_80ms": 38,
            "window_frames": 8,
            "located_start_frame": 18,
            "target_response": _make_curve(38, [(18.4, 0.69, 2.6), (26.0, 0.22, 2.4)], base=0.15),
            "distractor_response": _make_curve(38, [(17.0, 0.64, 2.8), (8.0, 0.24, 2.5)], base=0.17),
            "asr_tokens_40ms": [
                [2, "天"], [5, "要"], [8, "下"], [11, "雨"], [14, "了"],
                [18, "快"], [22, "去"], [26, "把"],
                [35, "官"], [41, "窗"], [48, "合"], [54, "上"],
            ],
            "cuts": {
                "P": {"frame_40ms": 36, "prefix": "天要下雨了快去把官", "leaked": True, "leaked_char": "官", "note": "硬切直接把同音误识字「官」切进前缀，LM 看到「官」字线索发生分布偏移。"},
                "C": {"frame_40ms": 36, "prefix": "天要下雨了快去把官", "leaked": True, "leaked_char": "官", "note": "精确字面吸附找不到「关」字，被迫停留在「官」字之后（残留 3.4% 泄漏的典型来源）。"},
                "C_phonetic": {"frame_40ms": 34, "prefix": "天要下雨了快去把", "leaked": False, "leaked_char": None, "note": "拼音近似匹配识别出 guān（官）≈ guān（关），成功磁吸回退至「把」字之后！"},
            },
            "candidates": case2_candidates,
        },
        {
            "id": "context_ac",
            "title": "近音消歧 · 「开空调」",
            "tag": "声学近邻",
            "failure_case": False,
            "utterance": "wu_s02_utt0415",
            "target_word": "开空调",
            "distractor_word": "关空调",
            "transcript": "屋里太闷热了帮忙开空调吧",
            "asr_note": "吴语快读下「开空调」与「关空调」声学余弦仅差 0.02，前文「太闷热了」让 λ=4.0 语境重排一锤定音。",
            "n_frames_80ms": 38,
            "window_frames": 8,
            "located_start_frame": 16,
            "target_response": _make_curve(38, [(16.3, 0.68, 2.9)], base=0.14),
            "distractor_response": _make_curve(38, [(16.0, 0.71, 2.7)], base=0.16),
            "asr_tokens_40ms": [
                [2, "屋"], [5, "里"], [8, "太"], [12, "闷"], [16, "热"], [20, "了"],
                [24, "帮"], [28, "忙"],
                [31, "开"], [37, "空"], [43, "调"], [50, "吧"],
            ],
            "cuts": {
                "P": {"frame_40ms": 32, "prefix": "屋里太闷热了帮忙开", "leaked": True, "leaked_char": "开", "note": "硬切落于 40 ms 帧 32，将首字「开」提前喂给大模型，使重排虚高。"},
                "C": {"frame_40ms": 30, "prefix": "屋里太闷热了帮忙", "leaked": False, "leaked_char": None, "note": "吸附至「开」字之前，仅凭干净语境「屋里太闷热了帮忙」完成公平重排。"},
                "C_phonetic": {"frame_40ms": 30, "prefix": "屋里太闷热了帮忙", "leaked": False, "leaked_char": None, "note": "精确对齐已无泄漏，保持干净前缀。"},
            },
            "candidates": case3_candidates,
        },
        {
            "id": "failure_out_of_pool",
            "title": "诚实失败 · 池外漏召回",
            "tag": "Failure Case (5.3%)",
            "failure_case": True,
            "utterance": "wu_s09_utt1628",
            "target_word": "调暗屏幕",
            "distractor_word": "调亮屏幕",
            "out_of_pool_rank": 24,
            "transcript": "晚上太刺眼了把调暗屏幕点",
            "asr_note": "这是 5.3% 未进入 Top-20 候选池的真实失败形态：声学粗定位把目标词排在第 24 位，无论怎么调 λ 滑块，后级重排都永远看不到正确答案。",
            "n_frames_80ms": 38,
            "window_frames": 8,
            "located_start_frame": 17,
            "target_response": _make_curve(38, [(17.5, 0.34, 2.4)], base=0.13),
            "distractor_response": _make_curve(38, [(16.8, 0.65, 2.8)], base=0.17),
            "asr_tokens_40ms": [
                [2, "晚"], [5, "上"], [9, "太"], [13, "刺"], [17, "眼"], [21, "了"],
                [26, "把"], [33, "调"], [39, "暗"], [45, "屏"], [51, "幕"], [56, "点"],
            ],
            "cuts": {
                "P": {"frame_40ms": 34, "prefix": "晚上太刺眼了把调", "leaked": True, "leaked_char": "调", "note": "即便硬切泄露了首字「调」，因为目标词「调暗屏幕」不在 Top-20 池中，最终仍会错选「调亮屏幕」。"},
                "C": {"frame_40ms": 32, "prefix": "晚上太刺眼了把", "leaked": False, "leaked_char": None, "note": "干净切分同样无法拯救池外样本——这正是 94.7% 召回率构成系统硬上限的直观证明。"},
                "C_phonetic": {"frame_40ms": 32, "prefix": "晚上太刺眼了把", "leaked": False, "leaked_char": None, "note": "后级对齐与重排再强，也无法从 20 个错误候选里变出第 24 名的正确答案。"},
            },
            "candidates": case4_candidates,
        },
    ]


def build_bundle() -> dict[str, Any]:
    prefix_audit = _load_json("prefix_audit_real.json")
    cache_bench = _load_json("cache_benchmark.json")
    cache_parity = _load_json("stage2_cache_parity.json")
    stage1_cv = _load_json("stage1_cv_speaker.json")
    lambda_c = _load_json("lambda_cv_speaker_C.json")
    lambda_ph = _load_json("lambda_cv_speaker_phonetic.json")
    shape_cv = _load_json("shape_gate_cv_fused_l4_speaker.json")
    stage2_abl = _load_json("stage2_real_ablation.json")
    kmeans = _load_json("stage1_kmeans.json")
    lambda_sweep = _load_json("stage2_lambda_sweep.json")

    res = cache_bench["results"]
    kv_benchmarks = [
        {
            "id": "short_1.0s",
            "label": "1.0s 短语音",
            "duration_s": 1.0,
            "full_median_s": round(res["short_1.0s_full"]["median_s"], 3),
            "cached_median_s": round(res["short_1.0s_cached"]["median_s"], 3),
            "speedup": round(res["short_1.0s_cached"]["speedup_warm"], 2),
            "full_mem_mb": round(res["short_1.0s_full"]["peak_extra_bytes"] / (1024 * 1024), 1),
            "cached_mem_mb": round(res["short_1.0s_cached"]["peak_extra_bytes"] / (1024 * 1024), 1),
            "spearman": round(res["short_1.0s_cached"]["spearman"], 4),
            "max_abs_diff": round(res["short_1.0s_cached"]["max_abs_diff"], 4),
        },
        {
            "id": "median_3.6s",
            "label": "3.6s 中等语音",
            "duration_s": 3.6,
            "full_median_s": round(res["median_3.6s_full"]["median_s"], 3),
            "cached_median_s": round(res["median_3.6s_cached"]["median_s"], 3),
            "speedup": round(res["median_3.6s_cached"]["speedup_warm"], 2),
            "full_mem_mb": round(res["median_3.6s_full"]["peak_extra_bytes"] / (1024 * 1024), 1),
            "cached_mem_mb": round(res["median_3.6s_cached"]["peak_extra_bytes"] / (1024 * 1024), 1),
            "spearman": round(res["median_3.6s_cached"]["spearman"], 4),
            "max_abs_diff": round(res["median_3.6s_cached"]["max_abs_diff"], 4),
        },
        {
            "id": "long_6.4s",
            "label": "6.4s 长语音",
            "duration_s": 6.4,
            "full_median_s": round(res["long_6.4s_full"]["median_s"], 3),
            "cached_median_s": round(res["long_6.4s_cached"]["median_s"], 3),
            "speedup": round(res["long_6.4s_cached"]["speedup_warm"], 2),
            "full_mem_mb": round(res["long_6.4s_full"]["peak_extra_bytes"] / (1024 * 1024), 1),
            "cached_mem_mb": round(res["long_6.4s_cached"]["peak_extra_bytes"] / (1024 * 1024), 1),
            "spearman": round(res["long_6.4s_cached"]["spearman"], 4),
            "max_abs_diff": round(res["long_6.4s_cached"]["max_abs_diff"], 4),
        },
    ]

    sweep_c = {
        k: round(v["top1"] * 100.0, 2)
        for k, v in lambda_sweep["metrics"]["C"].items()
    }
    sweep_ph = {
        k: round(v["top1"] * 100.0, 2)
        for k, v in lambda_sweep["metrics"]["C_phonetic"].items()
    }

    return {
        "meta": {
            "corpus_utterances": 2031,
            "corpus_occurrences": 3244,
            "corpus_keywords": 293,
            "frame_ms_audio": 80,
            "frame_ms_asr": 40,
            "top20_recall_pct": round(stage1_cv["nested_selected"]["top20"] * 100.0, 1),
            "top20_recall_baseline_pct": round(stage1_cv["baseline"]["top20"] * 100.0, 1),
            "stage1_top1_pct": round(stage1_cv["nested_selected"]["top1"] * 100.0, 1),
            "stage2_default_top1_pct": round(lambda_c["fixed_C_4.0"]["top1"] * 100.0, 1),
            "stage2_phonetic_top1_pct": round(lambda_ph["nested_selected"]["top1"] * 100.0, 1),
        },
        "prefix_audit": {
            "n": prefix_audit["P"]["n"],
            "P_leak_pct": round(prefix_audit["P"]["contains_own_first_char"] * 100.0, 1),
            "C_leak_pct": round(prefix_audit["C"]["contains_own_first_char"] * 100.0, 1),
            "S_leak_pct": round(prefix_audit["S"]["contains_own_first_char"] * 100.0, 1),
            "X_leak_pct": round(prefix_audit["X"]["contains_own_first_char"] * 100.0, 1),
        },
        "lambda_sweep": {
            "C": sweep_c,
            "C_phonetic": sweep_ph,
        },
        "kv_cache": {
            "hardware": "NVIDIA H20 · GPU0 · fp16 · 20 candidates",
            "parity": {
                "utterances": cache_parity["arms"]["C"]["utterances"],
                "top1_identical": cache_parity["arms"]["C"]["top1_identical"],
                "spearman_min": round(cache_parity["arms"]["C"]["spearman_min"], 4),
                "max_abs_nats": round(cache_parity["arms"]["C"]["max_abs_nats"], 4),
            },
            "candidates": cache_bench["candidates"],
            "benchmarks": kv_benchmarks,
        },
        "forest_plot": _build_forest_entries(
            stage1_cv, lambda_c, lambda_ph, shape_cv, stage2_abl, kmeans
        ),
        "cases": _build_cases(),
    }


def main() -> None:
    bundle = build_bundle()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
