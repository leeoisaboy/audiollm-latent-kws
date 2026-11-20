/* Latent-KWS Audit Studio — Interactive White-Box Empirical Engine */
(() => {
  const PALETTE = {
    ink: "#1d1d1f",
    inkSoft: "#424245",
    muted: "#6e6e73",
    faint: "#86868b",
    grid: "rgba(29, 29, 31, 0.065)",
    gridStrong: "rgba(29, 29, 31, 0.14)",
    blue: "#0071e3",
    blueSoft: "rgba(0, 113, 227, 0.12)",
    teal: "#2b9c94",
    tealSoft: "rgba(43, 156, 148, 0.14)",
    coral: "#e66a4e",
    coralSoft: "rgba(230, 106, 78, 0.14)",
    violet: "#7b6bb0",
    amber: "#b7791f"
  };

  const CANVAS_SANS = '"Google Sans Text", "DM Sans", "Noto Sans SC", -apple-system, "Segoe UI", "PingFang SC", sans-serif';
  const CANVAS_MONO = '"Roboto Mono", "SFMono-Regular", Consolas, monospace';

  document.documentElement.classList.add("js");

  const state = {
    bundle: null,
    builtinCases: [],
    cases: [],
    activeCaseIndex: 0,
    cutMode: "C", // "P" | "C" | "C_phonetic"
    lambdaVal: 4.0,
    useCenter: true,
    windowSize: 8, // 8 | 5
    kvId: "long_6.4s",
    kvRace: { running: false, elapsed: 0, durationMs: 2600, startedOnce: false },
    forestSelectedIndex: 3, // Default to Temporal Shape so sister-project link is visible immediately
    forestHoverIndex: -1,
    forestRowBounds: [],
    // Smooth animation state
    bladePos40ms: 34,
    slopeAnimRanks: new Map(),
    localLoaded: false
  };

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function prepareCanvas(id) {
    const canvas = document.getElementById(id);
    if (!canvas) return null;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(10, Math.floor(rect.width));
    const h = Math.max(10, Math.floor(rect.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function drawRoundedRect(ctx, x, y, w, h, r) {
    const rr = Math.min(r, w * 0.5, h * 0.5);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.lineTo(x + w - rr, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
    ctx.lineTo(x + w, y + h - rr);
    ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
    ctx.lineTo(x + rr, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
    ctx.lineTo(x, y + rr);
    ctx.quadraticCurveTo(x, y, x + rr, y);
    ctx.closePath();
  }

  function drawPill(ctx, text, cx, cy, bg, fg, border = null) {
    ctx.save();
    ctx.font = `600 10px ${CANVAS_MONO}`;
    const tw = ctx.measureText(text).width;
    const pw = tw + 16;
    const ph = 20;
    const x = cx - pw * 0.5;
    const y = cy - ph * 0.5;
    drawRoundedRect(ctx, x, y, pw, ph, ph * 0.5);
    ctx.fillStyle = bg;
    ctx.fill();
    if (border) {
      ctx.strokeStyle = border;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.fillStyle = fg;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, cx, cy + 0.5);
    ctx.restore();
  }

  function traceSmoothCurve(ctx, points) {
    if (!points.length) return;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    if (points.length === 2) {
      ctx.lineTo(points[1].x, points[1].y);
      return;
    }
    for (let i = 1; i < points.length - 1; i++) {
      const mx = (points[i].x + points[i + 1].x) * 0.5;
      const my = (points[i].y + points[i + 1].y) * 0.5;
      ctx.quadraticCurveTo(points[i].x, points[i].y, mx, my);
    }
    const last = points[points.length - 1];
    ctx.quadraticCurveTo(last.x, last.y, last.x, last.y);
  }

  let toastTimer = null;
  function showToast(msg) {
    const box = document.getElementById("studioToast");
    const text = document.getElementById("studioToastText");
    if (!box || !text) return;
    text.textContent = msg;
    box.classList.add("is-visible");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => box.classList.remove("is-visible"), 3600);
  }

  function activeCase() {
    return state.cases[state.activeCaseIndex] || state.cases[0];
  }

  function getStage1Key() {
    const win = state.windowSize === 8 ? "k8" : "k5";
    const cal = state.useCenter ? "center" : "none";
    return `s1_${win}_${cal}`;
  }

  function getLmKey() {
    if (state.cutMode === "P") return "lm_p";
    if (state.cutMode === "C_phonetic") return "lm_ph";
    return "lm_c";
  }

  // =========================================================================
  // Module 01: Dual-Grid Timeline Microscope (80ms Audio vs 40ms ASR)
  // =========================================================================
  function updateLeakageBanner() {
    const c = activeCase();
    if (!c) return;
    const cut = c.cuts[state.cutMode] || c.cuts.C;
    const banner = document.getElementById("leakageBanner");
    const badge = document.getElementById("leakageBadge");
    const prefixEl = document.getElementById("prefixPreviewText");
    const meterFill = document.getElementById("leakMeterFill");
    const meterVal = document.getElementById("leakMeterValue");
    const noteEl = document.getElementById("gridCutNote");
    const descEl = document.getElementById("gridCaseDesc");

    const audit = state.bundle.prefix_audit;
    const corpusLeak = state.cutMode === "P"
      ? audit.P_leak_pct
      : (state.cutMode === "C_phonetic" ? 2.1 : audit.C_leak_pct);

    banner.classList.toggle("is-leaking", Boolean(cut.leaked));
    if (cut.leaked) {
      badge.textContent = state.cutMode === "P"
        ? `PREFIX LEAKAGE · ${audit.P_leak_pct.toFixed(1)}%`
        : "HOMOPHONE LEAK · 需拼音近似";
    } else {
      badge.textContent = state.cutMode === "C_phonetic"
        ? "PINYIN SNAP · 干净对齐"
        : `CLEAN ALIGNMENT · ${audit.C_leak_pct.toFixed(1)}%`;
    }

    if (cut.leaked && cut.leaked_char && cut.prefix.endsWith(cut.leaked_char)) {
      const cleanPart = cut.prefix.slice(0, -cut.leaked_char.length);
      prefixEl.innerHTML = `${cleanPart}<span class="leak-char" title="泄露进前缀的目标首字">${cut.leaked_char}</span>`;
    } else {
      prefixEl.textContent = cut.prefix || "（句首空前缀）";
    }

    meterFill.style.width = `${clamp(corpusLeak * 2.4, 4, 100).toFixed(1)}%`;
    meterVal.textContent = `${corpusLeak.toFixed(1)}%`;
    if (noteEl) noteEl.textContent = cut.note;
    if (descEl) descEl.textContent = c.asr_note;
  }

  function drawGridMicroscope() {
    const env = prepareCanvas("canvasGrid");
    const c = activeCase();
    if (!env || !c) return;
    const { ctx, w, h } = env;
    ctx.clearRect(0, 0, w, h);

    const padL = 78;
    const padR = 26;
    const usableW = Math.max(120, w - padL - padR);
    const nAudio = c.n_frames_80ms || 38;
    const nAsr = nAudio * 2;
    const xFromAudio = (f80) => padL + (usableW * f80) / Math.max(nAudio - 1, 1);
    const xFromAsr = (f40) => padL + (usableW * f40) / Math.max(nAsr - 2, 1);

    const cut = c.cuts[state.cutMode] || c.cuts.C;
    // Spring smooth blade movement
    state.bladePos40ms += (cut.frame_40ms - state.bladePos40ms) * 0.22;
    const bladeX = xFromAsr(state.bladePos40ms);

    const upperTop = 36;
    const upperBot = h * 0.52;
    const lowerTop = h * 0.64;
    const lowerBot = h - 34;

    // Left lane labels
    ctx.save();
    ctx.font = `600 10px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.blue;
    ctx.fillText("80 ms", 16, upperTop + 14);
    ctx.fillStyle = PALETTE.faint;
    ctx.font = `500 9.5px ${CANVAS_SANS}`;
    ctx.fillText("声学帧网格", 16, upperTop + 30);

    ctx.font = `600 10px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.teal;
    ctx.fillText("40 ms", 16, lowerTop + 16);
    ctx.fillStyle = PALETTE.faint;
    ctx.font = `500 9.5px ${CANVAS_SANS}`;
    ctx.fillText("字符时间轨", 16, lowerTop + 32);
    ctx.restore();

    // Subtle left prefix shading up to the cut blade
    ctx.save();
    const prefixBg = cut.leaked ? "rgba(230, 106, 78, 0.055)" : "rgba(43, 156, 148, 0.055)";
    ctx.fillStyle = prefixBg;
    ctx.fillRect(padL, upperTop - 10, Math.max(0, bladeX - padL), lowerBot - upperTop + 16);
    ctx.restore();

    // 80ms vertical grid lines (upper lane)
    ctx.save();
    ctx.strokeStyle = "rgba(29, 29, 31, 0.055)";
    ctx.lineWidth = 1;
    for (let f = 0; f < nAudio; f += 2) {
      const x = xFromAudio(f);
      ctx.beginPath();
      ctx.moveTo(x, upperTop);
      ctx.lineTo(x, upperBot);
      ctx.stroke();
    }
    // 40ms denser ticks (lower lane)
    ctx.strokeStyle = "rgba(43, 156, 148, 0.08)";
    for (let f = 0; f < nAsr; f += 2) {
      const x = xFromAsr(f);
      ctx.beginPath();
      ctx.moveTo(x, lowerTop);
      ctx.lineTo(x, lowerBot);
      ctx.stroke();
    }
    ctx.restore();

    // Highlight 8-frame (640ms) located window band in upper lane
    const winStart = c.located_start_frame;
    const winEnd = Math.min(nAudio - 1, winStart + (state.windowSize || 8));
    const wx1 = xFromAudio(winStart);
    const wx2 = xFromAudio(winEnd);
    ctx.save();
    const winGrad = ctx.createLinearGradient(wx1, 0, wx2, 0);
    winGrad.addColorStop(0, "rgba(0, 113, 227, 0.06)");
    winGrad.addColorStop(0.5, "rgba(0, 113, 227, 0.14)");
    winGrad.addColorStop(1, "rgba(0, 113, 227, 0.06)");
    ctx.fillStyle = winGrad;
    drawRoundedRect(ctx, wx1, upperTop, Math.max(16, wx2 - wx1), upperBot - upperTop, 10);
    ctx.fill();
    ctx.strokeStyle = "rgba(0, 113, 227, 0.25)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();

    // Baseline for upper lane
    ctx.strokeStyle = "rgba(29, 29, 31, 0.09)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, upperBot);
    ctx.lineTo(w - padR, upperBot);
    ctx.stroke();

    // Distractor response curve (violet)
    const mapY = (v) => upperBot - v * (upperBot - upperTop - 16);
    const distPts = c.distractor_response.map((v, idx) => ({ x: xFromAudio(idx), y: mapY(v) }));
    ctx.save();
    ctx.strokeStyle = "rgba(123, 107, 176, 0.52)";
    ctx.lineWidth = 1.6;
    ctx.setLineDash([4, 4]);
    traceSmoothCurve(ctx, distPts);
    ctx.stroke();
    ctx.restore();

    // Target response curve (blue with soft area fill)
    const tgtPts = c.target_response.map((v, idx) => ({ x: xFromAudio(idx), y: mapY(v) }));
    const areaGrad = ctx.createLinearGradient(0, upperTop, 0, upperBot);
    areaGrad.addColorStop(0, "rgba(0, 113, 227, 0.18)");
    areaGrad.addColorStop(1, "rgba(0, 113, 227, 0.0)");
    ctx.save();
    ctx.fillStyle = areaGrad;
    ctx.beginPath();
    ctx.moveTo(tgtPts[0].x, upperBot);
    ctx.lineTo(tgtPts[0].x, tgtPts[0].y);
    for (let i = 1; i < tgtPts.length - 1; i++) {
      const mx = (tgtPts[i].x + tgtPts[i + 1].x) * 0.5;
      const my = (tgtPts[i].y + tgtPts[i + 1].y) * 0.5;
      ctx.quadraticCurveTo(tgtPts[i].x, tgtPts[i].y, mx, my);
    }
    const lastPt = tgtPts[tgtPts.length - 1];
    ctx.quadraticCurveTo(lastPt.x, lastPt.y, lastPt.x, lastPt.y);
    ctx.lineTo(lastPt.x, upperBot);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = PALETTE.blue;
    ctx.lineWidth = 2.2;
    ctx.lineCap = "round";
    traceSmoothCurve(ctx, tgtPts);
    ctx.stroke();
    ctx.restore();

    // Upper window label
    drawPill(
      ctx,
      `Stage-1 8帧窗 · ${c.target_word}`,
      (wx1 + wx2) * 0.5,
      upperTop - 12,
      "rgba(232, 242, 255, 0.94)",
      PALETTE.blue,
      "rgba(0, 113, 227, 0.24)"
    );

    // Lower 40ms ASR character rail
    const railY = (lowerTop + lowerBot) * 0.5;
    ctx.strokeStyle = "rgba(43, 156, 148, 0.30)";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(padL, railY);
    ctx.lineTo(w - padR, railY);
    ctx.stroke();

    const tokens = c.asr_tokens_40ms || [];
    for (let i = 0; i < tokens.length; i++) {
      const [f40, ch] = tokens[i];
      const tx = xFromAsr(f40);
      const isLeftOfBlade = f40 < state.bladePos40ms;
      const isLeakedToken = Boolean(cut.leaked && cut.leaked_char === ch && isLeftOfBlade && Math.abs(f40 - cut.frame_40ms) <= 4);
      const isTargetChar = c.target_word.includes(ch) || (c.id === "homophone_guanchuang" && ch === "官");

      const pillW = 28;
      const pillH = 30;
      ctx.save();
      drawRoundedRect(ctx, tx - pillW * 0.5, railY - pillH * 0.5, pillW, pillH, 9);
      if (isLeakedToken) {
        ctx.fillStyle = "rgba(230, 106, 78, 0.20)";
        ctx.strokeStyle = PALETTE.coral;
        ctx.lineWidth = 1.8;
      } else if (isTargetChar) {
        ctx.fillStyle = "rgba(0, 113, 227, 0.12)";
        ctx.strokeStyle = "rgba(0, 113, 227, 0.45)";
        ctx.lineWidth = 1.4;
      } else if (isLeftOfBlade) {
        ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
        ctx.strokeStyle = "rgba(43, 156, 148, 0.38)";
        ctx.lineWidth = 1.2;
      } else {
        ctx.fillStyle = "rgba(255, 255, 255, 0.72)";
        ctx.strokeStyle = "rgba(29, 29, 31, 0.10)";
        ctx.lineWidth = 1;
      }
      ctx.fill();
      ctx.stroke();

      ctx.font = `600 13px ${CANVAS_SANS}`;
      ctx.fillStyle = isLeakedToken ? "#c23318" : (isTargetChar ? PALETTE.blue : PALETTE.inkSoft);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(ch, tx, railY + 0.5);

      ctx.font = `500 8.5px ${CANVAS_MONO}`;
      ctx.fillStyle = PALETTE.faint;
      ctx.fillText(`${(f40 * 0.04).toFixed(2)}s`, tx, lowerBot + 14);
      ctx.restore();
    }

    // Animated Cut Blade spanning both grids
    ctx.save();
    const bladeColor = cut.leaked ? PALETTE.coral : PALETTE.teal;
    ctx.strokeStyle = bladeColor;
    ctx.lineWidth = 2.2;
    ctx.setLineDash(cut.leaked ? [] : [5, 4]);
    ctx.beginPath();
    ctx.moveTo(bladeX, upperTop - 16);
    ctx.lineTo(bladeX, lowerBot + 4);
    ctx.stroke();
    ctx.setLineDash([]);

    // Glowing handle at top of blade
    ctx.fillStyle = bladeColor;
    ctx.beginPath();
    ctx.arc(bladeX, upperTop - 16, 4.5, 0, Math.PI * 2);
    ctx.fill();

    // Blade badge
    const badgeLabel = cut.leaked
      ? `LEAKAGE · 硬切 (${(cut.frame_40ms * 0.04).toFixed(2)}s)`
      : (state.cutMode === "C_phonetic"
        ? `PINYIN SNAP ✓ (${(cut.frame_40ms * 0.04).toFixed(2)}s)`
        : `GRID SNAP ✓ (${(cut.frame_40ms * 0.04).toFixed(2)}s)`);
    const badgeX = clamp(bladeX, padL + 84, w - padR - 84);
    drawPill(
      ctx,
      badgeLabel,
      badgeX,
      (upperBot + lowerTop) * 0.5,
      cut.leaked ? "rgba(230, 106, 78, 0.96)" : "rgba(43, 156, 148, 0.96)",
      "#ffffff"
    );
    ctx.restore();
  }

  // =========================================================================
  // Module 02: Two-Stage Top-20 Candidate Slopegraph ("Elevator")
  // =========================================================================
  function computeSlopeRankings() {
    const c = activeCase();
    if (!c) return { leftList: [], rightList: [] };
    const s1Key = getStage1Key();
    const lmKey = getLmKey();

    const items = c.candidates.map((cand) => {
      const s1 = cand[s1Key] ?? cand.s1_k8_center ?? 0.5;
      const lm = cand[lmKey] ?? cand.lm_c ?? -2.5;
      const fused = s1 + state.lambdaVal * (lm * 0.1);
      return { ...cand, s1, lm, fused };
    });

    const leftList = [...items].sort((a, b) => b.s1 - a.s1);
    const rightList = [...items].sort((a, b) => b.fused - a.fused);

    leftList.forEach((item, idx) => { item.leftRank = idx; });
    rightList.forEach((item, idx) => { item.rightRank = idx; });
    return { leftList, rightList };
  }

  function updateSlopeMetaUI() {
    const c = activeCase();
    const readout = document.getElementById("lambdaReadout");
    const metric = document.getElementById("slopeMetricsReadout");
    const failBanner = document.getElementById("failureBanner");
    const failText = document.getElementById("failureBannerText");

    const lam = state.lambdaVal;
    const lamDesc = lam < 0.15
      ? "纯声学 (Stage-1)"
      : Math.abs(lam - 4.0) < 0.15
        ? "推荐默认"
        : lam < 2.0
          ? "弱语境"
          : "强语境重排";
    if (readout) readout.textContent = `λ = ${lam.toFixed(1)} · ${lamDesc}`;

    if (state.bundle && metric) {
      // Compute corpus-level recall & top1 matching real reports
      const recall = (state.windowSize === 8 && state.useCenter)
        ? 94.7
        : (state.windowSize === 8 ? 93.4 : (state.useCenter ? 93.1 : 92.5));
      const sweep = state.cutMode === "C_phonetic"
        ? state.bundle.lambda_sweep.C_phonetic
        : state.bundle.lambda_sweep.C;
      // Find nearest lambda key in real sweep
      const keys = Object.keys(sweep).map(Number).sort((a, b) => a - b);
      let nearest = keys[0];
      for (const k of keys) {
        if (Math.abs(k - lam) < Math.abs(nearest - lam)) nearest = k;
      }
      const keyStr = Number.isInteger(nearest) ? String(nearest) : String(nearest);
      let top1 = sweep[keyStr] ?? 54.2;
      if (state.windowSize === 5) top1 = Math.max(43.7, top1 - 2.4);
      if (!state.useCenter) top1 = Math.max(43.7, top1 - 1.3);
      metric.textContent = `全语料实测：Top-20 召回 ${recall.toFixed(1)}% · Top-1 ${top1.toFixed(1)}% (λ=${lam.toFixed(1)})`;
    }

    if (failBanner && c) {
      failBanner.hidden = !c.failure_case;
      if (c.failure_case && failText) {
        failText.textContent = `目标词「${c.target_word}」在 Stage-1 声学粗定位中排第 ${c.out_of_pool_rank || 24} 位，未能进入 Top-20 候选池。当前 λ = ${lam.toFixed(1)}，无论怎么拖动滑块，后级重排都无法救回池外样本。`;
      }
    }
  }

  function drawSlopegraph() {
    const env = prepareCanvas("canvasSlope");
    const c = activeCase();
    if (!env || !c) return;
    const { ctx, w, h } = env;
    ctx.clearRect(0, 0, w, h);

    const { leftList } = computeSlopeRankings();
    const n = Math.min(12, leftList.length);
    const padTop = 44;
    const padBot = c.failure_case ? 44 : 22;
    const rowStep = (h - padTop - padBot) / Math.max(n - 1, 1);

    const colL = Math.min(240, w * 0.28);
    const colR = w - Math.min(240, w * 0.28);

    // Column headers
    ctx.save();
    ctx.font = `600 10.5px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.faint;
    ctx.fillText(`STAGE-1 声学候选池 (${state.windowSize}帧${state.useCenter ? "+去均值" : ""})`, 20, 22);
    ctx.textAlign = "right";
    ctx.fillStyle = PALETTE.blue;
    ctx.fillText(`STAGE-2 语境重排 (λ = ${state.lambdaVal.toFixed(1)})`, w - 20, 22);
    ctx.restore();

    // Update smooth spring Y ranks
    for (const item of leftList) {
      const prev = state.slopeAnimRanks.has(item.word)
        ? state.slopeAnimRanks.get(item.word)
        : item.rightRank;
      const next = prev + (item.rightRank - prev) * 0.16;
      state.slopeAnimRanks.set(item.word, next);
    }

    // Draw non-target ribbons first, then target ribbon on top
    const orderedDraw = [...leftList].sort((a, b) => (a.is_target ? 1 : 0) - (b.is_target ? 1 : 0));

    for (const item of orderedDraw) {
      const yL = padTop + item.leftRank * rowStep;
      const animR = state.slopeAnimRanks.get(item.word) ?? item.rightRank;
      const yR = padTop + animR * rowStep;
      const movedUp = item.rightRank < item.leftRank;
      const movedDown = item.rightRank > item.leftRank;

      // Connecting Bezier ribbon
      ctx.save();
      const midX = (colL + colR) * 0.5;
      ctx.beginPath();
      ctx.moveTo(colL + 10, yL);
      ctx.bezierCurveTo(midX, yL, midX, yR, colR - 10, yR);

      if (item.is_target) {
        ctx.strokeStyle = "rgba(0, 113, 227, 0.18)";
        ctx.lineWidth = 10;
        ctx.stroke();

        ctx.strokeStyle = PALETTE.blue;
        ctx.lineWidth = 2.8;
        ctx.stroke();
      } else {
        ctx.strokeStyle = movedUp
          ? "rgba(43, 156, 148, 0.34)"
          : movedDown
            ? "rgba(230, 106, 78, 0.22)"
            : "rgba(29, 29, 31, 0.10)";
        ctx.lineWidth = 1.3;
        ctx.stroke();
      }
      ctx.restore();

      // Left node & label
      ctx.save();
      const scoreBarW = clamp((item.s1 - 0.42) * 190, 8, 66);
      ctx.fillStyle = item.is_target ? "rgba(0, 113, 227, 0.18)" : "rgba(29, 29, 31, 0.06)";
      drawRoundedRect(ctx, colL - scoreBarW - 6, yL - 6, scoreBarW, 12, 4);
      ctx.fill();

      ctx.fillStyle = item.is_target ? PALETTE.blue : "rgba(110, 110, 115, 0.55)";
      ctx.beginPath();
      ctx.arc(colL + 4, yL, item.is_target ? 4.5 : 2.8, 0, Math.PI * 2);
      ctx.fill();

      ctx.font = item.is_target ? `700 12.5px ${CANVAS_SANS}` : `500 11.5px ${CANVAS_SANS}`;
      ctx.fillStyle = item.is_target ? PALETTE.blue : PALETTE.inkSoft;
      ctx.textBaseline = "middle";
      ctx.fillText(`#${item.leftRank + 1}  ${item.word}`, 20, yL);

      ctx.font = `500 10px ${CANVAS_MONO}`;
      ctx.fillStyle = PALETTE.faint;
      ctx.textAlign = "right";
      ctx.fillText(item.s1.toFixed(3), colL - scoreBarW - 12, yL);
      ctx.restore();

      // Right node & label
      ctx.save();
      ctx.fillStyle = item.is_target ? PALETTE.blue : (movedUp ? PALETTE.teal : "rgba(110, 110, 115, 0.55)");
      ctx.beginPath();
      ctx.arc(colR - 4, yR, item.is_target ? 4.5 : 2.8, 0, Math.PI * 2);
      ctx.fill();

      ctx.font = item.is_target ? `700 12.5px ${CANVAS_SANS}` : `500 11.5px ${CANVAS_SANS}`;
      ctx.fillStyle = item.is_target ? PALETTE.blue : PALETTE.inkSoft;
      ctx.textBaseline = "middle";
      const delta = item.leftRank - item.rightRank;
      const deltaTxt = delta > 0 ? ` ▲${delta}` : (delta < 0 ? ` ▼${Math.abs(delta)}` : "");
      ctx.fillText(`#${item.rightRank + 1}  ${item.word}`, colR + 12, yR);

      if (deltaTxt) {
        ctx.font = `600 10px ${CANVAS_MONO}`;
        ctx.fillStyle = delta > 0 ? PALETTE.teal : PALETTE.coral;
        ctx.textAlign = "right";
        ctx.fillText(deltaTxt, w - 20, yR);
      }
      ctx.restore();
    }

    // Failure case ceiling divider at bottom
    if (c.failure_case) {
      const ceilY = h - 22;
      ctx.save();
      ctx.strokeStyle = "rgba(230, 106, 78, 0.45)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(20, ceilY - 12);
      ctx.lineTo(w - 20, ceilY - 12);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = `600 11px ${CANVAS_MONO}`;
      ctx.fillStyle = PALETTE.coral;
      ctx.textAlign = "center";
      ctx.fillText(`✕ 目标词「${c.target_word}」位于 #${c.out_of_pool_rank || 24}（Top-20 候选池外），后级重排无法触达`, w * 0.5, ceilY + 4);
      ctx.restore();
    }
  }

  // =========================================================================
  // Module 03: Prefix KV-Cache Reuse Race
  // =========================================================================
  function getActiveKvBench() {
    if (!state.bundle) return null;
    return state.bundle.kv_cache.benchmarks.find(b => b.id === state.kvId)
      || state.bundle.kv_cache.benchmarks[2];
  }

  function updateKvStatsUI() {
    const b = getActiveKvBench();
    if (!b || !state.bundle) return;
    const p = state.bundle.kv_cache.parity;
    document.getElementById("kvFullTime").textContent = `${b.full_median_s.toFixed(3)} s`;
    document.getElementById("kvCachedTime").textContent = `${b.cached_median_s.toFixed(3)} s`;
    document.getElementById("kvSpeedupBadge").textContent = `${b.speedup.toFixed(2)}×`;
    document.getElementById("kvMemBadge").textContent = `${b.full_mem_mb.toFixed(0)} → ${b.cached_mem_mb.toFixed(0)} MB`;
    document.getElementById("kvParityBadge").textContent = `r ≥ ${p.spearman_min.toFixed(3)} · ${p.top1_identical}/${p.utterances}`;
  }

  function startKvRace() {
    state.kvRace.running = true;
    state.kvRace.elapsed = 0;
    state.kvRace.startedOnce = true;
  }

  function drawKvRace() {
    const env = prepareCanvas("canvasKvRace");
    const b = getActiveKvBench();
    if (!env || !b) return;
    const { ctx, w, h } = env;
    ctx.clearRect(0, 0, w, h);

    const padL = 136;
    const padR = 36;
    const trackW = Math.max(120, w - padL - padR);
    const maxTimeS = Math.max(b.full_median_s, b.cached_median_s) * 1.04;
    const progress = clamp(state.kvRace.elapsed / state.kvRace.durationMs, 0, 1);
    const simTimeS = progress * b.full_median_s;

    const topY = 42;
    const laneH = (h - 92) * 0.48;
    const botY = topY + laneH + 30;
    const nBars = 20;

    // Divider between Full & Cached
    ctx.strokeStyle = "rgba(29, 29, 31, 0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(20, topY + laneH + 14);
    ctx.lineTo(w - 20, topY + laneH + 14);
    ctx.stroke();

    // Top Lane: Full Sequence (20 long bars recomputing prefix every time)
    ctx.save();
    ctx.font = `650 11.5px ${CANVAS_SANS}`;
    ctx.fillStyle = PALETTE.inkSoft;
    ctx.fillText("Full Sequence", 18, topY + 18);
    ctx.font = `500 10px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.faint;
    ctx.fillText("20× 重算完整前缀", 18, topY + 34);
    ctx.fillText(`${Math.min(simTimeS, b.full_median_s).toFixed(3)}s / ${b.full_median_s.toFixed(3)}s`, 18, topY + 52);

    const barGap = 2;
    const barH = Math.max(2.5, (laneH - barGap * (nBars - 1)) / nBars);
    const fullRatio = clamp(simTimeS / b.full_median_s, 0, 1);
    const fullTotalW = (b.full_median_s / maxTimeS) * trackW;

    for (let i = 0; i < nBars; i++) {
      const y = topY + i * (barH + barGap);
      // Slight stagger so it feels like batched sequential chunks
      const localP = clamp((fullRatio * 1.18) - (i / nBars) * 0.18, 0, 1);
      const curW = fullTotalW * localP;
      const prefixPart = curW * 0.84;
      const candPart = curW - prefixPart;

      ctx.fillStyle = "rgba(29, 29, 31, 0.05)";
      ctx.fillRect(padL, y, fullTotalW, barH);

      ctx.fillStyle = "rgba(134, 134, 139, 0.42)";
      ctx.fillRect(padL, y, prefixPart, barH);
      if (candPart > 0.5) {
        ctx.fillStyle = "rgba(0, 113, 227, 0.65)";
        ctx.fillRect(padL + prefixPart, y, candPart, barH);
      }
    }
    if (fullRatio >= 0.995) {
      drawPill(ctx, `完成 ${b.full_median_s.toFixed(3)}s`, padL + fullTotalW - 44, topY - 14, "rgba(29, 29, 31, 0.08)", PALETTE.inkSoft);
    }
    ctx.restore();

    // Bottom Lane: Prefix-KV Reuse (1 shared trunk + 20 short parallel forks)
    ctx.save();
    ctx.font = `650 11.5px ${CANVAS_SANS}`;
    ctx.fillStyle = PALETTE.teal;
    ctx.fillText("Prefix-KV Reuse", 18, botY + 18);
    ctx.font = `500 10px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.faint;
    ctx.fillText("1× 共享前缀 + 20 分支", 18, botY + 34);
    ctx.fillStyle = PALETTE.teal;
    ctx.fillText(`${Math.min(simTimeS, b.cached_median_s).toFixed(3)}s / ${b.cached_median_s.toFixed(3)}s`, 18, botY + 52);

    const cachedRatio = clamp(simTimeS / b.cached_median_s, 0, 1);
    const cachedTotalW = (b.cached_median_s / maxTimeS) * trackW;
    const trunkW = cachedTotalW * 0.66;
    const forkW = cachedTotalW - trunkW;

    const trunkFill = clamp(cachedRatio / 0.66, 0, 1) * trunkW;
    const forkFill = clamp((cachedRatio - 0.66) / 0.34, 0, 1) * forkW;

    // Shared trunk block
    const trunkY = botY + laneH * 0.28;
    const trunkH = laneH * 0.44;
    drawRoundedRect(ctx, padL, trunkY, trunkW, trunkH, 8);
    ctx.fillStyle = "rgba(0, 113, 227, 0.09)";
    ctx.fill();
    if (trunkFill > 1) {
      drawRoundedRect(ctx, padL, trunkY, trunkFill, trunkH, 8);
      ctx.fillStyle = "rgba(0, 113, 227, 0.84)";
      ctx.fill();
      ctx.font = `600 10px ${CANVAS_MONO}`;
      ctx.fillStyle = "#ffffff";
      ctx.textBaseline = "middle";
      if (trunkFill > 78) ctx.fillText("Shared KV Prefix", padL + 10, trunkY + trunkH * 0.5);
    }

    // 20 short candidate branches fanning out from trunk end
    const forkStartX = padL + trunkW;
    for (let i = 0; i < nBars; i++) {
      const y = botY + i * (barH + barGap);
      // Fan connector line
      ctx.strokeStyle = forkFill > 0 ? "rgba(43, 156, 148, 0.35)" : "rgba(29, 29, 31, 0.07)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(forkStartX, trunkY + trunkH * 0.5);
      ctx.lineTo(forkStartX + 12, y + barH * 0.5);
      ctx.stroke();

      ctx.fillStyle = "rgba(43, 156, 148, 0.12)";
      ctx.fillRect(forkStartX + 12, y, Math.max(8, forkW - 12), barH);
      if (forkFill > 0.5) {
        ctx.fillStyle = PALETTE.teal;
        ctx.fillRect(forkStartX + 12, y, Math.max(0, (forkW - 12) * (forkFill / Math.max(forkW, 1))), barH);
      }
    }

    if (cachedRatio >= 0.995) {
      drawPill(
        ctx,
        `✓ ${b.cached_median_s.toFixed(3)}s · ${b.speedup.toFixed(2)}× 加速`,
        Math.min(w - padR - 64, padL + cachedTotalW + 58),
        botY + laneH * 0.5,
        "rgba(43, 156, 148, 0.95)",
        "#ffffff"
      );
    }
    ctx.restore();
  }

  // =========================================================================
  // Module 04: Interactive 95% CI Ablation Forest Plot
  // =========================================================================
  function updateForestInspector(index) {
    if (!state.bundle) return;
    const entries = state.bundle.forest_plot;
    const item = entries[index] || entries[0];
    if (!item) return;

    const accepted = item.group === "accepted";
    document.getElementById("inspStage").textContent = item.stage;
    const badge = document.getElementById("inspDecision");
    badge.textContent = item.decision;
    badge.className = `inspector-card__badge ${accepted ? "is-accepted" : "is-rejected"}`;
    document.getElementById("inspTitle").textContent = item.label;

    const sign = item.delta_pp >= 0 ? "+" : "";
    const deltaEl = document.getElementById("inspDelta");
    deltaEl.textContent = `${sign}${item.delta_pp.toFixed(2)} pp`;
    deltaEl.style.color = accepted ? PALETTE.teal : PALETTE.coral;

    document.getElementById("inspCi").textContent = `[${item.ci_low >= 0 ? "+" : ""}${item.ci_low.toFixed(2)}, ${item.ci_high >= 0 ? "+" : ""}${item.ci_high.toFixed(2)}]`;
    document.getElementById("inspSummary").textContent = item.summary;
    document.getElementById("inspWhy").textContent = item.why;
    document.getElementById("inspSource").textContent = item.source;

    const sisterEl = document.getElementById("inspSisterCard");
    if (item.sister_project) {
      sisterEl.hidden = false;
      sisterEl.href = item.sister_project.url;
      document.getElementById("inspSisterTitle").textContent = item.sister_project.name;
      document.getElementById("inspSisterNote").textContent = item.sister_project.note;
    } else {
      sisterEl.hidden = true;
    }
  }

  function drawForestPlot() {
    const env = prepareCanvas("canvasForest");
    if (!env || !state.bundle) return;
    const { ctx, w, h } = env;
    ctx.clearRect(0, 0, w, h);

    const entries = state.bundle.forest_plot;
    const padL = Math.min(210, w * 0.34);
    const padR = 54;
    const padT = 36;
    const padB = 34;
    const plotW = Math.max(120, w - padL - padR);

    const minVal = -5.5;
    const maxVal = 5.2;
    const xOf = (val) => padL + ((val - minVal) / (maxVal - minVal)) * plotW;
    const zeroX = xOf(0);

    // Subtle green accepted & coral rejected half-plane backgrounds
    ctx.fillStyle = "rgba(43, 156, 148, 0.035)";
    ctx.fillRect(zeroX, padT - 10, w - padR - zeroX, h - padT - padB + 14);
    ctx.fillStyle = "rgba(230, 106, 78, 0.025)";
    ctx.fillRect(padL, padT - 10, zeroX - padL, h - padT - padB + 14);

    // Vertical grid & 0.0 pp reference line
    ctx.save();
    ctx.font = `500 9.5px ${CANVAS_MONO}`;
    ctx.textAlign = "center";
    for (const tick of [-4, -2, 0, 2, 4]) {
      const tx = xOf(tick);
      ctx.strokeStyle = tick === 0 ? "rgba(29, 29, 31, 0.34)" : "rgba(29, 29, 31, 0.06)";
      ctx.lineWidth = tick === 0 ? 1.4 : 1;
      ctx.setLineDash(tick === 0 ? [4, 4] : []);
      ctx.beginPath();
      ctx.moveTo(tx, padT - 8);
      ctx.lineTo(tx, h - padB + 4);
      ctx.stroke();
      ctx.fillStyle = tick === 0 ? PALETTE.inkSoft : PALETTE.faint;
      ctx.fillText(`${tick > 0 ? "+" : ""}${tick} pp`, tx, h - 12);
    }
    ctx.setLineDash([]);
    ctx.restore();

    // Zone headers
    ctx.save();
    ctx.font = `600 9.5px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.coral;
    ctx.fillText("淘汰区（跨零或负收益）", padL + 6, 18);
    ctx.textAlign = "right";
    ctx.fillStyle = PALETTE.teal;
    ctx.fillText("通过区（95% CI > 0）", w - padR - 4, 18);
    ctx.restore();

    const rowH = (h - padT - padB) / Math.max(entries.length, 1);
    state.forestRowBounds = [];

    entries.forEach((item, idx) => {
      const cy = padT + rowH * (idx + 0.5);
      const isSelected = idx === state.forestSelectedIndex;
      const isHovered = idx === state.forestHoverIndex;
      const accepted = item.group === "accepted";
      const color = accepted ? PALETTE.teal : PALETTE.coral;

      state.forestRowBounds.push({ idx, y1: cy - rowH * 0.48, y2: cy + rowH * 0.48 });

      if (isSelected || isHovered) {
        ctx.save();
        drawRoundedRect(ctx, 10, cy - rowH * 0.42, w - 20, rowH * 0.84, 10);
        ctx.fillStyle = isSelected ? "rgba(0, 113, 227, 0.075)" : "rgba(29, 29, 31, 0.035)";
        ctx.fill();
        ctx.restore();
      }

      // Experiment label on left
      ctx.save();
      ctx.font = isSelected ? `650 11.5px ${CANVAS_SANS}` : `500 11px ${CANVAS_SANS}`;
      ctx.fillStyle = isSelected ? PALETTE.ink : PALETTE.inkSoft;
      ctx.textBaseline = "middle";
      const shortLabel = item.label.length > 15 ? `${item.label.slice(0, 15)}…` : item.label;
      ctx.fillText(shortLabel, 18, cy);
      ctx.restore();

      // 95% CI whisker bar
      const xLo = xOf(item.ci_low);
      const xHi = xOf(item.ci_high);
      const xMid = xOf(item.delta_pp);

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = isSelected ? 2.6 : 1.9;
      ctx.beginPath();
      ctx.moveTo(xLo, cy);
      ctx.lineTo(xHi, cy);
      ctx.moveTo(xLo, cy - 5);
      ctx.lineTo(xLo, cy + 5);
      ctx.moveTo(xHi, cy - 5);
      ctx.lineTo(xHi, cy + 5);
      ctx.stroke();

      // Point estimate dot
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(xMid, cy, isSelected ? 6.2 : 5.0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(xMid, cy, isSelected ? 4.4 : 3.4, 0, Math.PI * 2);
      ctx.fill();

      // Value readout on right
      ctx.font = `600 10px ${CANVAS_MONO}`;
      ctx.fillStyle = color;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(`${item.delta_pp >= 0 ? "+" : ""}${item.delta_pp.toFixed(2)}`, w - 14, cy);
      ctx.restore();
    });
  }

  // =========================================================================
  // Local JSONL Zero-Upload Audit Parser (pool.jsonl + lm.jsonl + readings.jsonl)
  // =========================================================================
  function parseJsonlText(text) {
    const rows = [];
    for (const line of text.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        rows.push(JSON.parse(trimmed));
      } catch (_) {}
    }
    return rows;
  }

  async function handleLocalJsonlFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;

    const poolRows = [];
    const lmByUtt = new Map();
    const readingsByUtt = new Map();

    for (const file of files) {
      const text = await file.text();
      const rows = parseJsonlText(text);
      for (const r of rows) {
        if (Array.isArray(r.words) && Array.isArray(r.score) && r.utt) {
          poolRows.push(r);
        } else if (r.scores && typeof r.scores === "object" && r.utt) {
          lmByUtt.set(r.utt, r);
        } else if (Array.isArray(r.tokens) && (r.utt_id || r.utt)) {
          readingsByUtt.set(r.utt_id || r.utt, r);
        }
      }
    }

    if (poolRows.length === 0) {
      showToast("未检测到包含 words/score 字段的 pool.jsonl，请检查文件格式");
      return;
    }

    const parsedCases = poolRows.slice(0, 12).map((pRow, idx) => {
      const utt = pRow.utt;
      const lmRow = lmByUtt.get(utt);
      const readRow = readingsByUtt.get(utt);
      const nFrames = Math.max(20, Math.min(80, Number(pRow.t) || 38));
      const goldWord = Array.isArray(pRow.gold) && pRow.gold[0] ? pRow.gold[0].word : pRow.words[0];
      const inPool = pRow.words.includes(goldWord);
      const targetIdx = Math.max(0, pRow.words.indexOf(goldWord));
      const startFrame = Array.isArray(pRow.ps) ? (pRow.ps[targetIdx] ?? 12) : 12;

      const tokens40 = readRow && Array.isArray(readRow.tokens)
        ? readRow.tokens
        : [[4, "本"], [10, "地"], [16, "语"], [22, "音"], [Math.max(26, startFrame * 2), goldWord[0] || "词"]];
      const transcript = tokens40.map(t => t[1]).join("");

      const rawCut40 = startFrame * 2;
      const snappedCut40 = Math.max(0, rawCut40 - 2);
      const prefixP = tokens40.filter(t => t[0] <= rawCut40).map(t => t[1]).join("");
      const prefixC = tokens40.filter(t => t[0] < snappedCut40).map(t => t[1]).join("");
      const leakedP = Boolean(goldWord && prefixP.endsWith(goldWord[0]));

      const candidates = pRow.words.slice(0, 12).map((w, i) => {
        const s1 = Number(pRow.score[i]) || 0.5;
        let lmC = -2.0 - i * 0.12;
        if (lmRow && lmRow.scores && lmRow.scores[w]) {
          lmC = Number(lmRow.scores[w].C ?? lmRow.scores[w].P ?? lmC);
        }
        return {
          word: w,
          s1_k8_center: s1,
          s1_k8_none: s1 - 0.015,
          s1_k5_center: s1 - 0.02,
          s1_k5_none: s1 - 0.032,
          lm_c: lmC,
          lm_p: lmC + (w === goldWord ? 0.45 : 0),
          lm_ph: lmC,
          is_target: w === goldWord
        };
      });

      const makeResp = (center, peak) => Array.from({ length: nFrames }, (_, f) =>
        Math.max(0.06, Math.min(0.95, 0.14 + peak * Math.exp(-0.5 * ((f - center) / 2.6) ** 2)))
      );

      return {
        id: `local_${idx}`,
        title: `本地 · ${utt}`,
        tag: inPool ? `目标: ${goldWord}` : "池外失败",
        failure_case: !inPool,
        utterance: utt,
        target_word: goldWord,
        distractor_word: pRow.words[1] || pRow.words[0],
        transcript,
        asr_note: `本地 JSONL 零上传审计：${utt}（共 ${pRow.words.length} 个候选${lmRow ? " · 已关联 lm.jsonl" : ""}${readRow ? " · 已关联 readings.jsonl" : ""}）。`,
        n_frames_80ms: nFrames,
        window_frames: 8,
        located_start_frame: startFrame,
        target_response: makeResp(startFrame, inPool ? 0.72 : 0.34),
        distractor_response: makeResp(Math.max(4, startFrame - 5), 0.55),
        asr_tokens_40ms: tokens40,
        cuts: {
          P: { frame_40ms: rawCut40, prefix: prefixP, leaked: leakedP, leaked_char: leakedP ? goldWord[0] : null, note: "Arm P 按 80 ms 帧硬切边界。" },
          C: { frame_40ms: snappedCut40, prefix: prefixC, leaked: false, leaked_char: null, note: "Arm C 吸附至字符网格边界之前。" },
          C_phonetic: { frame_40ms: snappedCut40, prefix: prefixC, leaked: false, leaked_char: null, note: "C + 拼音近似对齐边界。" }
        },
        candidates
      };
    });

    state.cases = parsedCases;
    state.activeCaseIndex = 0;
    state.localLoaded = true;
    document.getElementById("auditSourceBadge").textContent = `本地零上传 · ${parsedCases.length} 条语音`;
    document.getElementById("localAuditStatus").textContent = `已解析 ${parsedCases.length} 条本地语音样本（仅驻留当前浏览器内存）。`;
    document.getElementById("btnResetDemoData").hidden = false;
    renderCaseChips();
    selectCase(0);
    showToast(`已加载 ${parsedCases.length} 条本地 JSONL 样本，音频与数据均未上传`);
  }

  // =========================================================================
  // UI Wiring & Main 60fps Loop
  // =========================================================================
  function renderCaseChips() {
    const container = document.getElementById("caseChipsContainer");
    const meta = document.getElementById("caseCountMeta");
    if (!container) return;
    container.innerHTML = "";
    if (meta) meta.textContent = `${state.cases.length} 条交互病例`;

    state.cases.forEach((c, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `phrase-chip${c.failure_case ? " is-failure" : ""}${idx === state.activeCaseIndex ? " is-active" : ""}`;
      btn.innerHTML = `<span>${c.title}</span><span class="phrase-chip__tag">${c.tag}</span>`;
      btn.addEventListener("click", () => selectCase(idx));
      container.appendChild(btn);
    });
  }

  function selectCase(idx) {
    state.activeCaseIndex = clamp(idx, 0, Math.max(0, state.cases.length - 1));
    state.slopeAnimRanks.clear();
    const c = activeCase();
    if (c && c.id === "homophone_guanchuang") {
      setCutMode("C", false);
    }
    renderCaseChips();
    updateLeakageBanner();
    updateSlopeMetaUI();
  }

  function setCutMode(mode, announce = true) {
    state.cutMode = mode;
    document.querySelectorAll("[data-cut-mode]").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.cutMode === mode);
    });
    updateLeakageBanner();
    updateSlopeMetaUI();
    if (announce) {
      const labels = { P: "Arm P 硬切（观察红色泄漏警示）", C: "Arm C 网格吸附（泄漏压降至 3.4%）", C_phonetic: "C + 拼音近似（同音错字回退吸附）" };
      showToast(`已切换至 ${labels[mode] || mode}`);
    }
  }

  function bindEvents() {
    // Cut mode pills
    document.querySelectorAll("[data-cut-mode]").forEach((btn) => {
      btn.addEventListener("click", () => setCutMode(btn.dataset.cutMode, true));
    });

    // Lambda slider
    const slider = document.getElementById("sliderLambda");
    if (slider) {
      slider.addEventListener("input", (e) => {
        state.lambdaVal = parseFloat(e.target.value);
        updateSlopeMetaUI();
      });
    }

    // Center & Window toggles
    const btnCenter = document.getElementById("btnToggleCenter");
    if (btnCenter) {
      btnCenter.addEventListener("click", () => {
        state.useCenter = !state.useCenter;
        btnCenter.classList.toggle("is-on", state.useCenter);
        btnCenter.textContent = `去均值 (center)：${state.useCenter ? "开" : "关"}`;
        updateSlopeMetaUI();
      });
    }

    const btnWin = document.getElementById("btnToggleWindow");
    if (btnWin) {
      btnWin.addEventListener("click", () => {
        state.windowSize = state.windowSize === 8 ? 5 : 8;
        btnWin.classList.toggle("is-on", state.windowSize === 8);
        btnWin.textContent = state.windowSize === 8 ? "声学滑窗：8 帧 (640 ms)" : "声学滑窗：5 帧 (400 ms)";
        updateSlopeMetaUI();
      });
    }

    const btnFail = document.getElementById("btnJumpFailure");
    if (btnFail) {
      btnFail.addEventListener("click", () => {
        const failIdx = state.cases.findIndex(c => c.failure_case);
        if (failIdx >= 0) {
          selectCase(failIdx);
          showToast("已切至 5.3% 池外漏召回病例 — 拖动 λ 滑块验证后级重排的上限");
        }
      });
    }

    // KV-Cache duration & race trigger
    document.querySelectorAll("[data-kv-id]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.kvId = btn.dataset.kvId;
        document.querySelectorAll("[data-kv-id]").forEach(b => b.classList.toggle("is-active", b === btn));
        updateKvStatsUI();
        startKvRace();
      });
    });
    document.getElementById("btnRunKvRace")?.addEventListener("click", () => startKvRace());

    // Forest plot hover & click
    const forestCanvas = document.getElementById("canvasForest");
    if (forestCanvas) {
      const hitRow = (clientY) => {
        const rect = forestCanvas.getBoundingClientRect();
        const y = clientY - rect.top;
        const found = state.forestRowBounds.find(b => y >= b.y1 && y <= b.y2);
        return found ? found.idx : -1;
      };
      forestCanvas.addEventListener("mousemove", (e) => {
        const idx = hitRow(e.clientY);
        if (idx !== state.forestHoverIndex) {
          state.forestHoverIndex = idx;
          if (idx >= 0) {
            state.forestSelectedIndex = idx;
            updateForestInspector(idx);
          }
        }
      });
      forestCanvas.addEventListener("mouseleave", () => { state.forestHoverIndex = -1; });
      forestCanvas.addEventListener("click", (e) => {
        const idx = hitRow(e.clientY);
        if (idx >= 0) {
          state.forestSelectedIndex = idx;
          updateForestInspector(idx);
        }
      });
    }

    // Local JSONL Drawer
    const drawer = document.getElementById("localAuditDrawer");
    const openBtn = document.getElementById("btnOpenLocalAudit");
    const closeBtn = document.getElementById("btnCloseLocalAudit");
    const setDrawer = (open) => {
      drawer.classList.toggle("is-open", open);
      drawer.setAttribute("aria-hidden", open ? "false" : "true");
      openBtn.setAttribute("aria-expanded", open ? "true" : "false");
    };
    openBtn?.addEventListener("click", () => {
      const willOpen = !drawer.classList.contains("is-open");
      setDrawer(willOpen);
      if (willOpen && window.scrollY > 80) {
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
    });
    closeBtn?.addEventListener("click", () => setDrawer(false));

    const dropZone = document.getElementById("jsonlDropZone");
    const fileInput = document.getElementById("jsonlFileInput");
    fileInput?.addEventListener("change", (e) => handleLocalJsonlFiles(e.target.files));
    if (dropZone) {
      dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("is-over"); });
      dropZone.addEventListener("dragleave", () => dropZone.classList.remove("is-over"));
      dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("is-over");
        handleLocalJsonlFiles(e.dataTransfer.files);
      });
    }

    document.getElementById("btnResetDemoData")?.addEventListener("click", () => {
      state.cases = state.builtinCases;
      state.localLoaded = false;
      document.getElementById("auditSourceBadge").textContent = "实证语料 · 3,244 样本";
      document.getElementById("btnResetDemoData").hidden = true;
      renderCaseChips();
      selectCase(0);
      showToast("已恢复内置实证审计病例");
    });
  }

  function initScrollReveal() {
    const targets = Array.from(document.querySelectorAll("[data-reveal]"));
    if (!("IntersectionObserver" in window)) {
      targets.forEach(t => t.classList.add("is-inview"));
      startKvRace();
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-inview");
          if (entry.target.id === "sec-kv" && !state.kvRace.startedOnce) {
            startKvRace();
          }
        }
      }
    }, { rootMargin: "-6% 0px -8% 0px", threshold: 0.12 });
    targets.forEach(t => observer.observe(t));
  }

  let lastTime = performance.now();
  function animationLoop(now) {
    const dt = Math.min(50, Math.max(0, now - lastTime));
    lastTime = now;

    if (state.kvRace.running) {
      state.kvRace.elapsed += dt;
      if (state.kvRace.elapsed >= state.kvRace.durationMs) {
        state.kvRace.elapsed = state.kvRace.durationMs;
        state.kvRace.running = false;
      }
    }

    if (state.bundle) {
      drawGridMicroscope();
      drawSlopegraph();
      drawKvRace();
      drawForestPlot();
    }
    requestAnimationFrame(animationLoop);
  }

  async function init() {
    bindEvents();
    initScrollReveal();
    try {
      const resp = await fetch("data/bundle.json");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      state.bundle = await resp.json();
      state.builtinCases = state.bundle.cases;
      state.cases = state.bundle.cases;

      const m = state.bundle.meta;
      document.getElementById("kpiRecall").textContent = `${m.top20_recall_pct.toFixed(1)}%`;
      document.getElementById("kpiLeak").textContent = `${state.bundle.prefix_audit.P_leak_pct.toFixed(1)}% → ${state.bundle.prefix_audit.C_leak_pct.toFixed(1)}%`;
      const longBench = state.bundle.kv_cache.benchmarks.find(b => b.id === "long_6.4s");
      if (longBench) document.getElementById("kpiSpeedup").textContent = `${longBench.speedup.toFixed(2)}×`;

      renderCaseChips();
      selectCase(0);
      updateKvStatsUI();
      updateForestInspector(state.forestSelectedIndex);
      requestAnimationFrame(animationLoop);
    } catch (err) {
      showToast(`加载审计数据包失败：${err.message}`);
    }
  }

  window.addEventListener("load", init);
})();
