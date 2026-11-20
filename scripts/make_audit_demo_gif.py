"""Render a crisp, borderless animated demo GIF for Latent-KWS Audit Studio.

Reads real data from ``studio/data/bundle.json`` and produces
``docs/assets/audit-studio-demo.gif`` showcasing all four live modules:
1. Dual-grid alignment (Arm P 35.2% leak -> Arm C 3.4% magnetic snap)
2. Two-stage Slopegraph (lambda 0.0 -> 4.0 turnaround #4 -> #1)
3. Prefix KV-Cache reuse race (0.301s vs 0.141s, 2.14x speedup)
4. 95% CI ablation forest plot (3 accepted vs 4 rejected)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = REPO_ROOT / "studio" / "data" / "bundle.json"
OUT_GIF = REPO_ROOT / "docs" / "assets" / "audit-studio-demo.gif"

W, H = 960, 540
FPS = 16

BG = (251, 251, 253)
INK = (29, 29, 31)
INK_SOFT = (66, 66, 69)
MUTED = (110, 110, 115)
FAINT = (134, 134, 139)
BLUE = (0, 113, 227)
BLUE_SOFT = (228, 240, 255)
TEAL = (43, 156, 148)
TEAL_SOFT = (228, 246, 243)
CORAL = (230, 106, 78)
CORAL_SOFT = (253, 236, 231)
VIOLET = (123, 107, 176)
HAIRLINE = (232, 232, 236)

FONT_CN_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT_CN = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_MONO = Path(r"C:\Windows\Fonts\consola.ttf")
FONT_MONO_BOLD = Path(r"C:\Windows\Fonts\consolab.ttf")


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


F_EYEBROW = load_font(FONT_MONO_BOLD, 12)
F_TITLE = load_font(FONT_CN_BOLD, 30)
F_SUB = load_font(FONT_CN, 14)
F_MONO_SM = load_font(FONT_MONO, 12)
F_MONO_BD = load_font(FONT_MONO_BOLD, 12)
F_CN_SM = load_font(FONT_CN, 13)
F_CN_BD = load_font(FONT_CN_BOLD, 14)
F_CHAR = load_font(FONT_CN_BOLD, 16)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def ease_in_out(t: float) -> float:
    x = clamp(t, 0.0, 1.0)
    return -(math.cos(math.pi * x) - 1.0) * 0.5


def ease_out(t: float) -> float:
    x = clamp(t, 0.0, 1.0)
    return 1.0 - (1.0 - x) ** 3


def blend(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    x = clamp(t, 0.0, 1.0)
    return tuple(int(c1[i] * (1 - x) + c2[i] * x) for i in range(3))


def make_base_canvas() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    # Minimal topbar header
    draw.rounded_rectangle((36, 18, 56, 38), radius=6, fill=BLUE_SOFT)
    draw.line((41, 24, 51, 24), fill=BLUE, width=2)
    draw.line((41, 28, 48, 28), fill=BLUE, width=2)
    draw.line((41, 32, 51, 32), fill=BLUE, width=2)
    draw.text((64, 20), "Latent-KWS", font=F_CN_BD, fill=INK)
    draw.text((152, 21), "Audit Studio", font=F_MONO_SM, fill=MUTED)
    # Right status pill
    draw.rounded_rectangle((W - 216, 16, W - 36, 40), radius=12, fill=(242, 242, 245))
    draw.ellipse((W - 204, 24, W - 196, 32), fill=TEAL)
    draw.text((W - 188, 21), "3,244 实测样本 · 无边框", font=F_CN_SM, fill=INK_SOFT)
    return img


def draw_pill(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    text: str,
    bg: tuple[int, int, int],
    fg: tuple[int, int, int],
    font_obj: ImageFont.FreeTypeFont = F_MONO_BD,
) -> None:
    l, t, r, b = draw.textbbox((0, 0), text, font=font_obj)
    tw, th = r - l, b - t
    pw, ph = tw + 20, th + 10
    draw.rounded_rectangle((cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2), radius=ph / 2, fill=bg)
    draw.text((cx - tw / 2, cy - th / 2 - 1), text, font=font_obj, fill=fg)


def draw_header(draw: ImageDraw.ImageDraw, eyebrow: str, title_a: str, title_b: str, sub: str) -> None:
    draw.text((40, 58), eyebrow, font=F_EYEBROW, fill=BLUE)
    draw.text((40, 78), title_a, font=F_TITLE, fill=INK)
    l, t, r, b = draw.textbbox((40, 78), title_a, font=F_TITLE)
    draw.text((r + 8, 78), title_b, font=F_TITLE, fill=FAINT)
    draw.text((40, 120), sub, font=F_SUB, fill=MUTED)


def render_scene_grid(bundle: dict[str, Any], t: float) -> Image.Image:
    """Scene 1: Dual-Grid Alignment (Arm P 35.2% leak -> Arm C 3.4% magnetic snap)."""
    img = make_base_canvas()
    draw = ImageDraw.Draw(img)
    c = bundle["cases"][0]

    draw_header(
        draw,
        "01 / DUAL-GRID ALIGNMENT",
        "两套网格，",
        "差半步即泄漏。",
        "80 ms 声学帧遇上 40 ms 字符轨：硬切把首字「关」切进前缀，磁吸对齐压降泄漏 35.2% → 3.4%。",
    )

    # Transition from Arm P (t < 0.44) -> magnetic snap (0.44..0.58) -> Arm C (t >= 0.58)
    snap_p = ease_in_out((t - 0.44) / 0.14)
    is_leaking = snap_p < 0.5
    leak_pct = 35.2 * (1 - snap_p) + 3.4 * snap_p
    blade_f40 = 36.0 * (1 - snap_p) + 34.0 * snap_p

    # Status strip (borderless)
    badge_bg = CORAL if is_leaking else TEAL_SOFT
    badge_fg = (255, 255, 255) if is_leaking else TEAL
    badge_txt = f"PREFIX LEAKAGE · {leak_pct:.1f}%" if is_leaking else f"CLEAN ALIGNMENT · {leak_pct:.1f}%"
    draw_pill(draw, 145, 166, badge_txt, badge_bg, badge_fg)
    prefix_str = "左前缀：外面风大了请帮我把" + ("关" if is_leaking else "")
    draw.text((262, 158), prefix_str, font=F_CN_BD, fill=INK)
    if is_leaking:
        # Highlight leaked char
        l, _, r, _ = draw.textbbox((262, 158), "左前缀：外面风大了请帮我把", font=F_CN_BD)
        draw.rounded_rectangle((r - 1, 155, r + 18, 178), radius=5, fill=CORAL_SOFT)
        draw.text((r + 2, 158), "关", font=F_CN_BD, fill=(194, 51, 24))

    # Leakage mini bar on right
    draw.text((W - 220, 159), "泄漏率", font=F_CN_SM, fill=FAINT)
    draw.rounded_rectangle((W - 172, 164, W - 88, 170), radius=3, fill=HAIRLINE)
    bar_w = max(4, int(84 * (leak_pct / 40.0)))
    draw.rounded_rectangle((W - 172, 164, W - 172 + bar_w, 170), radius=3, fill=CORAL if is_leaking else TEAL)
    draw.text((W - 80, 158), f"{leak_pct:.1f}%", font=F_MONO_BD, fill=CORAL if is_leaking else TEAL)

    # Dual-grid geometry
    pad_l, pad_r = 96, 40
    usable_w = W - pad_l - pad_r
    n_audio = c["n_frames_80ms"]
    n_asr = n_audio * 2
    x_audio = lambda f: pad_l + usable_w * f / (n_audio - 1)
    x_asr = lambda f: pad_l + usable_w * f / (n_asr - 2)

    upper_top, upper_bot = 205, 345
    lower_y = 432
    blade_x = x_asr(blade_f40)

    # Lane labels
    draw.text((40, 215), "80 ms", font=F_MONO_BD, fill=BLUE)
    draw.text((40, 232), "声学帧", font=F_CN_SM, fill=FAINT)
    draw.text((40, 418), "40 ms", font=F_MONO_BD, fill=TEAL)
    draw.text((40, 435), "字符轨", font=F_CN_SM, fill=FAINT)

    # Vertical ticks
    for f in range(0, n_audio, 2):
        xx = x_audio(f)
        draw.line((xx, upper_top, xx, upper_bot), fill=(240, 240, 244), width=1)

    # Located 8-frame window highlight
    wx1, wx2 = x_audio(c["located_start_frame"]), x_audio(c["located_start_frame"] + 8)
    draw.rounded_rectangle((wx1, upper_top, wx2, upper_bot), radius=10, fill=BLUE_SOFT)

    # Baseline
    draw.line((pad_l, upper_bot, W - pad_r, upper_bot), fill=HAIRLINE, width=1)

    # Response curves
    y_map = lambda v: upper_bot - v * (upper_bot - upper_top - 16)
    dist_pts = [(x_audio(i), y_map(v)) for i, v in enumerate(c["distractor_response"])]
    tgt_pts = [(x_audio(i), y_map(v)) for i, v in enumerate(c["target_response"])]
    draw.line(dist_pts, fill=(175, 165, 206), width=2)
    draw.line(tgt_pts, fill=BLUE, width=3)

    # Live scan dot riding curve
    scan_i = int((t * 1.4 % 1.0) * (len(tgt_pts) - 1))
    sx, sy = tgt_pts[scan_i]
    draw.ellipse((sx - 8, sy - 8, sx + 8, sy + 8), fill=BLUE_SOFT)
    draw.ellipse((sx - 3.5, sy - 3.5, sx + 3.5, sy + 3.5), fill=BLUE)

    # Lower 40ms ASR character rail
    draw.line((pad_l, lower_y, W - pad_r, lower_y), fill=(195, 230, 226), width=2)
    for f40, ch in c["asr_tokens_40ms"]:
        tx = x_asr(f40)
        is_target_ch = ch in c["target_word"]
        is_leaked_ch = is_leaking and ch == "关"
        if is_leaked_ch:
            draw.ellipse((tx - 17, lower_y - 17, tx + 17, lower_y + 17), fill=CORAL_SOFT)
            fg = (194, 51, 24)
        elif is_target_ch:
            draw.ellipse((tx - 16, lower_y - 16, tx + 16, lower_y + 16), fill=BLUE_SOFT)
            fg = BLUE
        else:
            draw.ellipse((tx - 13, lower_y - 13, tx + 13, lower_y + 13), fill=BG)
            fg = INK_SOFT if f40 < blade_f40 else FAINT
        l, t0, r, b0 = draw.textbbox((0, 0), ch, font=F_CHAR)
        draw.text((tx - (r - l) / 2, lower_y - (b0 - t0) / 2 - 2), ch, font=F_CHAR, fill=fg)

    # Cut blade
    b_col = CORAL if is_leaking else TEAL
    draw.line((blade_x, upper_top - 10, blade_x, lower_y + 24), fill=b_col, width=3)
    draw.ellipse((blade_x - 5, upper_top - 15, blade_x + 5, upper_top - 5), fill=b_col)
    draw_pill(
        draw,
        blade_x,
        (upper_bot + lower_y) / 2 - 4,
        "LEAKAGE · 1.44s" if is_leaking else "GRID SNAP ✓ 1.36s",
        b_col,
        (255, 255, 255),
    )

    draw.text((40, H - 34), "Arm P 硬切 vs Arm C 网格吸附 · 双网格对齐白盒显微镜", font=F_MONO_SM, fill=FAINT)
    return img


def render_scene_slope(bundle: dict[str, Any], t: float) -> Image.Image:
    """Scene 2: Two-stage Slopegraph (lambda 0.0 -> 4.0 turnaround #4 -> #1)."""
    img = make_base_canvas()
    draw = ImageDraw.Draw(img)
    c = bundle["cases"][0]

    draw_header(
        draw,
        "02 / TWO-STAGE RERANKING",
        "声学撒网，",
        "语境定音。",
        "94.7% 先进入 Top-20 候选池；随着语境权重 λ 从 0.0 滑向 4.0，方言目标词「关窗」从 #4 一路逆袭至 #1。",
    )

    # Lambda smoothly glides 0.0 -> 4.0 (t: 0.15..0.68) and holds at 4.0
    lam_p = ease_in_out((t - 0.14) / 0.52)
    lam = 4.0 * lam_p
    top1_pct = 47.4 + (55.0 - 47.4) * lam_p

    # Slider visualization
    draw.text((40, 158), f"融合权重 λ = {lam:.1f}", font=F_MONO_BD, fill=BLUE)
    draw.rounded_rectangle((176, 164, 520, 169), radius=2, fill=HAIRLINE)
    kx = 176 + int((lam / 6.0) * (520 - 176))
    draw.rounded_rectangle((176, 164, kx, 169), radius=2, fill=BLUE)
    draw.ellipse((kx - 7, 159, kx + 7, 173), fill=BLUE)
    draw.text((545, 158), f"全语料 Top-1：{top1_pct:.1f}%  ·  Top-20 召回 94.7%", font=F_CN_SM, fill=MUTED)

    cands = c["candidates"][:8]
    left_list = sorted(cands, key=lambda x: -x["s1_k8_center"])
    # Interpolated ranks for silky motion
    right_0 = sorted(cands, key=lambda x: -x["s1_k8_center"])
    right_4 = sorted(cands, key=lambda x: -(x["s1_k8_center"] + 4.0 * x["lm_c"] * 0.1))
    r0_idx = {x["word"]: i for i, x in enumerate(right_0)}
    r4_idx = {x["word"]: i for i, x in enumerate(right_4)}

    col_l, col_r = 250, W - 250
    pad_t, row_step = 216, 36

    draw.text((40, 192), "STAGE-1 声学候选池 (8帧+去均值)", font=F_MONO_SM, fill=FAINT)
    draw.text((W - 265, 192), f"STAGE-2 语境重排 (λ = {lam:.1f})", font=F_MONO_BD, fill=BLUE)

    for left_rank, item in enumerate(left_list):
        w_str = item["word"]
        is_tgt = bool(item.get("is_target"))
        r_float = r0_idx[w_str] * (1 - lam_p) + r4_idx[w_str] * lam_p
        yl = pad_t + left_rank * row_step
        yr = pad_t + r_float * row_step

        # Smooth cubic bezier sampled into polyline
        mid_x = (col_l + col_r) * 0.5
        pts = []
        for s in range(25):
            u = s / 24.0
            mt = 1 - u
            px = mt**3 * (col_l + 8) + 3 * mt**2 * u * mid_x + 3 * mt * u**2 * mid_x + u**3 * (col_r - 8)
            py = mt**3 * yl + 3 * mt**2 * u * yl + 3 * mt * u**2 * yr + u**3 * yr
            pts.append((px, py))

        if is_tgt:
            draw.line(pts, fill=BLUE_SOFT, width=9)
            draw.line(pts, fill=BLUE, width=3)
        else:
            col = (180, 218, 214) if r4_idx[w_str] < left_rank else (225, 225, 230)
            draw.line(pts, fill=col, width=2)

        # Left label
        fg = BLUE if is_tgt else INK_SOFT
        draw.text((40, yl - 8), f"#{left_rank + 1}  {w_str}", font=F_CN_BD if is_tgt else F_CN_SM, fill=fg)
        draw.text((180, yl - 7), f"{item['s1_k8_center']:.3f}", font=F_MONO_SM, fill=FAINT)
        draw.ellipse((col_l - 4, yl - 4, col_l + 4, yl + 4), fill=BLUE if is_tgt else FAINT)

        # Right label
        cur_rank = int(round(r_float)) + 1
        draw.ellipse((col_r - 4, yr - 4, col_r + 4, yr + 4), fill=BLUE if is_tgt else FAINT)
        draw.text((col_r + 14, yr - 8), f"#{cur_rank}  {w_str}", font=F_CN_BD if is_tgt else F_CN_SM, fill=fg)
        if is_tgt and lam_p > 0.7:
            draw.text((W - 86, yr - 7), "▲ 3", font=F_MONO_BD, fill=TEAL)

    draw.text((40, H - 34), "Stage-1 (llm2) → Stage-2 (Step-Audio-2-mini) · 动态排名跃迁", font=F_MONO_SM, fill=FAINT)
    return img


def render_scene_kv(bundle: dict[str, Any], t: float) -> Image.Image:
    """Scene 3: Prefix KV-Cache reuse race (0.301s vs 0.141s, 2.14x speedup)."""
    img = make_base_canvas()
    draw = ImageDraw.Draw(img)
    bench = bundle["kv_cache"]["benchmarks"][2]  # 6.4s

    draw_header(
        draw,
        "03 / PREFIX KV-CACHE",
        "算一次前缀，",
        "跑二十个候选。",
        "NVIDIA H20 fp16 实测：共享音频前缀 KV 后耗时从 0.301 s 降至 0.141 s（2.14× 加速），r ≥ 0.997。",
    )

    race_p = clamp(t / 0.72, 0.0, 1.0)
    sim_s = race_p * bench["full_median_s"]
    pad_l, pad_r = 165, 48
    track_w = W - pad_l - pad_r
    max_s = bench["full_median_s"] * 1.04

    top_y, lane_h = 186, 128
    bot_y = top_y + lane_h + 36
    draw.line((40, top_y + lane_h + 18, W - 40, top_y + lane_h + 18), fill=HAIRLINE, width=1)

    # Top: Full Sequence
    draw.text((40, top_y + 18), "Full Sequence", font=F_CN_BD, fill=INK_SOFT)
    draw.text((40, top_y + 38), "20× 重算完整前缀", font=F_CN_SM, fill=FAINT)
    draw.text((40, top_y + 58), f"{min(sim_s, bench['full_median_s']):.3f} s", font=F_MONO_BD, fill=MUTED)

    full_w = track_w * (bench["full_median_s"] / max_s)
    full_ratio = clamp(sim_s / bench["full_median_s"], 0.0, 1.0)
    for i in range(20):
        y = top_y + i * 6.2
        lp = clamp(full_ratio * 1.15 - (i / 20.0) * 0.15, 0.0, 1.0)
        cw = full_w * lp
        draw.rectangle((pad_l, y, pad_l + full_w, y + 3.8), fill=(241, 241, 244))
        draw.rectangle((pad_l, y, pad_l + cw * 0.84, y + 3.8), fill=(176, 178, 185))
        if cw > 2:
            draw.rectangle((pad_l + cw * 0.84, y, pad_l + cw, y + 3.8), fill=BLUE)

    # Bottom: Prefix-KV Reuse
    draw.text((40, bot_y + 18), "Prefix-KV Reuse", font=F_CN_BD, fill=TEAL)
    draw.text((40, bot_y + 38), "1× 共享 + 20 路", font=F_CN_SM, fill=FAINT)
    draw.text((40, bot_y + 58), f"{min(sim_s, bench['cached_median_s']):.3f} s", font=F_MONO_BD, fill=TEAL)

    cached_w = track_w * (bench["cached_median_s"] / max_s)
    cached_ratio = clamp(sim_s / bench["cached_median_s"], 0.0, 1.0)
    trunk_w = cached_w * 0.66
    fork_w = cached_w - trunk_w
    trunk_fill = clamp(cached_ratio / 0.66, 0.0, 1.0) * trunk_w
    fork_fill = clamp((cached_ratio - 0.66) / 0.34, 0.0, 1.0) * fork_w

    ty1, ty2 = bot_y + 38, bot_y + 88
    draw.rounded_rectangle((pad_l, ty1, pad_l + trunk_w, ty2), radius=8, fill=BLUE_SOFT)
    if trunk_fill > 2:
        draw.rounded_rectangle((pad_l, ty1, pad_l + trunk_fill, ty2), radius=8, fill=BLUE)
        if trunk_fill > 110:
            draw.text((pad_l + 14, ty1 + 17), "Shared KV Prefix", font=F_MONO_BD, fill=(255, 255, 255))

    fx = pad_l + trunk_w
    for i in range(20):
        y = bot_y + i * 6.2
        draw.line((fx, (ty1 + ty2) / 2, fx + 12, y + 2), fill=(185, 224, 220) if fork_fill > 0 else HAIRLINE, width=1)
        draw.rectangle((fx + 12, y, fx + fork_w, y + 3.8), fill=TEAL_SOFT)
        if fork_fill > 1:
            draw.rectangle((fx + 12, y, fx + 12 + max(0, fork_w - 12) * (fork_fill / max(fork_w, 1)), y + 3.8), fill=TEAL)

    if cached_ratio >= 0.99:
        draw_pill(draw, pad_l + cached_w + 86, bot_y + 62, "✓ 0.141s · 2.14× 加速", TEAL, (255, 255, 255))

    draw.text((40, H - 34), "reports/cache_benchmark.json · 199/200 答案一致 · 额外显存 183 → 156 MB", font=F_MONO_SM, fill=FAINT)
    return img


def render_scene_forest(bundle: dict[str, Any], t: float) -> Image.Image:
    """Scene 4: 95% CI Ablation Forest Plot."""
    img = make_base_canvas()
    draw = ImageDraw.Draw(img)
    entries = bundle["forest_plot"]

    draw_header(
        draw,
        "04 / EMPIRICAL TRIBUNAL",
        "直觉归直觉，",
        "验证说了算。",
        "按说话人分组五折交叉验证（95% 置信区间）：绿色全部过零线保留，跨零或负收益的 7 项实验诚实淘汰。",
    )

    reveal = ease_out(t / 0.45)
    pad_l, pad_r = 250, 72
    plot_w = W - pad_l - pad_r
    x_of = lambda val: pad_l + ((val - (-5.5)) / (5.2 - (-5.5))) * plot_w
    zero_x = x_of(0.0)

    top_y, bot_y = 186, 474
    # Zero line & ticks
    for tick in (-4, -2, 0, 2, 4):
        tx = x_of(tick)
        col = (140, 140, 146) if tick == 0 else (236, 236, 240)
        draw.line((tx, top_y - 10, tx, bot_y), fill=col, width=2 if tick == 0 else 1)
        draw.text((tx - 14, bot_y + 6), f"{tick:+d} pp" if tick != 0 else "0 pp", font=F_MONO_SM, fill=INK_SOFT if tick == 0 else FAINT)

    draw.text((pad_l + 6, top_y - 24), "淘汰区（跨零或负收益）", font=F_MONO_BD, fill=CORAL)
    draw.text((W - pad_r - 130, top_y - 24), "通过区 (95% CI > 0)", font=F_MONO_BD, fill=TEAL)

    row_h = (bot_y - top_y) / len(entries)
    for idx, item in enumerate(entries):
        cy = top_y + row_h * (idx + 0.5)
        acc = item["group"] == "accepted"
        col = TEAL if acc else CORAL
        is_focus = idx == 3  # Highlight Temporal Shape (sister-project bridge)
        if is_focus:
            draw.rounded_rectangle((32, cy - 17, W - 32, cy + 17), radius=10, fill=CORAL_SOFT)

        label = item["label"]
        if len(label) > 14:
            label = label[:14] + "…"
        draw.text((42, cy - 8), label, font=F_CN_BD if is_focus else F_CN_SM, fill=INK if is_focus else INK_SOFT)

        x_lo = zero_x + (x_of(item["ci_low"]) - zero_x) * reveal
        x_hi = zero_x + (x_of(item["ci_high"]) - zero_x) * reveal
        x_mid = zero_x + (x_of(item["delta_pp"]) - zero_x) * reveal

        draw.line((x_lo, cy, x_hi, cy), fill=col, width=3)
        draw.line((x_lo, cy - 5, x_lo, cy + 5), fill=col, width=2)
        draw.line((x_hi, cy - 5, x_hi, cy + 5), fill=col, width=2)
        draw.ellipse((x_mid - 6, cy - 6, x_mid + 6, cy + 6), fill=(255, 255, 255), outline=col, width=3)
        draw.text((W - 64, cy - 7), f"{item['delta_pp']:+.2f}", font=F_MONO_BD, fill=col)

    draw.text((40, H - 34), "时序形状加权 (+0.03 pp，跨零关闭) — 纯声学时序解耦详见姐妹项目 disentangle-acoustic-print", font=F_CN_SM, fill=MUTED)
    return img


def main() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    frames: list[Image.Image] = []

    scenes = [
        (render_scene_grid, 38),
        (render_scene_slope, 38),
        (render_scene_kv, 34),
        (render_scene_forest, 30),
    ]

    for render_fn, count in scenes:
        for i in range(count):
            t = i / max(count - 1, 1)
            frames.append(render_fn(bundle, t))

    # Quantize with a shared clean palette for crisp edges and compact size
    quantized = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        OUT_GIF,
        save_all=True,
        append_images=quantized[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    size_kb = OUT_GIF.stat().st_size / 1024.0
    print(f"Wrote {OUT_GIF} ({len(quantized)} frames, {size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
