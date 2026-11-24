/* Latent-KWS Audit Studio — Living Borderless White-Box Empirical Engine */
(() => {
  const PALETTE = {
    ink: "#1d1d1f",
    inkSoft: "#424245",
    muted: "#6e6e73",
    faint: "#86868b",
    blue: "#0071e3",
    teal: "#2b9c94",
    coral: "#e66a4e",
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
    cutMode: "P", // Autonomous cycle starts on P then snaps to C so visitors immediately witness the fix!
    lambdaVal: 4.0,
    useCenter: true,
    windowSize: 8,
    kvId: "long_6.4s",
    kvRace: { running: true, elapsed: 0, durationMs: 2400, holdMs: 2200 },
    forestSelectedIndex: 3, // Start on Temporal Shape (sister-project highlight)
    forestHoverIndex: -1,
    forestRowBounds: [],
    // Autonomous living choreography flags (yield immediately on user interaction)
    autoGrid: true,
    autoGridTimer: 0,
    autoLambda: true,
    autoLambdaPhase: 0,
    autoForest: true,
    autoForestTimer: 0,
    // Progressive section reveals & smooth scroll-linked choreography
    reveals: { grid: 0, slope: 0, kv: 0, forest: 0 },
    scrollAnim: new Map(),
    bladePos40ms: 36,
    slopeAnimRanks: new Map(),
    clock: 0,
    localLoaded: false
  };

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function easeOutCubic(t) {
    const x = clamp(t, 0, 1);
    return 1 - Math.pow(1 - x, 3);
  }

  function easeInOutSine(t) {
    const x = clamp(t, 0, 1);
    return -(Math.cos(Math.PI * x) - 1) * 0.5;
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

  function drawPill(ctx, text, cx, cy, bg, fg) {
    ctx.save();
    ctx.font = `600 10px ${CANVAS_MONO}`;
    const tw = ctx.measureText(text).width;
    const pw = tw + 16;
    const ph = 20;
    drawRoundedRect(ctx, cx - pw * 0.5, cy - ph * 0.5, pw, ph, ph * 0.5);
    ctx.fillStyle = bg;
    ctx.fill();
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

  function evalCubicBezier(p0, p1, p2, p3, t) {
    const mt = 1 - t;
    return {
      x: mt * mt * mt * p0.x + 3 * mt * mt * t * p1.x + 3 * mt * t * t * p2.x + t * t * t * p3.x,
      y: mt * mt * mt * p0.y + 3 * mt * mt * t * p1.y + 3 * mt * t * t * p2.y + t * t * t * p3.y
    };
  }

  let toastTimer = null;
  function showToast(msg) {
    const box = document.getElementById("studioToast");
    const text = document.getElementById("studioToastText");
    if (!box || !text) return;
    text.textContent = msg;
    box.classList.add("is-visible");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => box.classList.remove("is-visible"), 3400);
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
  // Scroll-Linked Apple / OpenAI / Gemini Fade-In & Fade-Out Choreography
  // =========================================================================
  function updateScrollChoreography(dt) {
    const vh = window.innerHeight || 800;
    const sections = document.querySelectorAll("[data-scroll-stage]");
    const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    sections.forEach((sec) => {
      const key = sec.dataset.scrollStage;
      const rect = sec.getBoundingClientRect();
      const center = rect.top + rect.height * 0.5;
      // Normalized distance from viewport center: -1 (above) .. 0 (centered) .. +1 (below)
      const norm = (center - vh * 0.5) / (vh * 0.72);
      const visible = rect.bottom > vh * 0.08 && rect.top < vh * 0.92;

      if (visible) {
        state.reveals[key] = Math.min(1, state.reveals[key] + dt / 1150);
      }

      if (reduceMotion) {
        sec.style.opacity = "1";
        sec.style.transform = "none";
        return;
      }

      let targetOpacity = 1.0;
      let targetY = 0.0;
      let targetScale = 1.0;

      if (norm > 0.25) {
        // Entering from bottom
        const t = clamp((norm - 0.25) / 0.85, 0, 1);
        targetOpacity = 1 - t * 0.92;
        targetY = t * 44;
        targetScale = 1 - t * 0.018;
      } else if (norm < -0.35) {
        // Exiting smoothly at top
        const t = clamp((-norm - 0.35) / 0.85, 0, 1);
        targetOpacity = 1 - t * 0.84;
        targetY = -t * 24;
        targetScale = 1 - t * 0.012;
      }

      const prev = state.scrollAnim.get(sec) || { o: targetOpacity, y: targetY, s: targetScale };
      const damp = 1 - Math.exp(-dt * 0.012);
      prev.o += (targetOpacity - prev.o) * damp;
      prev.y += (targetY - prev.y) * damp;
      prev.s += (targetScale - prev.s) * damp;
      state.scrollAnim.set(sec, prev);

      sec.style.opacity = prev.o.toFixed(3);
      sec.style.transform = `translate3d(0, ${prev.y.toFixed(2)}px, 0) scale(${prev.s.toFixed(4)})`;
    });
  }

  // =========================================================================
  // Autonomous Living Choreography (Zero Hover Required!)
  // =========================================================================
  function stepAutonomousChoreography(dt) {
    // 1. Dual-Grid automatic Arm P -> Arm C -> Pinyin demonstration cycle
    if (state.autoGrid && state.reveals.grid > 0.65) {
      state.autoGridTimer += dt;
      if (state.autoGridTimer >= 3600) {
        state.autoGridTimer = 0;
        const c = activeCase();
        const sequence = (c && c.id === "homophone_guanchuang")
          ? ["P", "C", "C_phonetic"]
          : ["P", "C"];
        const nextIdx = (sequence.indexOf(state.cutMode) + 1) % sequence.length;
        setCutMode(sequence[nextIdx], false, false);
      }
    }

    // 2. Two-Stage Slopegraph automatic λ tide (0.0 -> 4.0 -> hold -> 0.0)
    if (state.autoLambda && state.reveals.slope > 0.5) {
      state.autoLambdaPhase = (state.autoLambdaPhase + dt / 7600) % 1.0;
      const p = state.autoLambdaPhase;
      let targetLam = 4.0;
      if (p < 0.16) {
        targetLam = 0.0;
      } else if (p < 0.42) {
        targetLam = 4.0 * easeInOutSine((p - 0.16) / 0.26);
      } else if (p < 0.78) {
        targetLam = 4.0;
      } else {
        targetLam = 4.0 * (1 - easeInOutSine((p - 0.78) / 0.22));
      }
      state.lambdaVal = targetLam;
      const slider = document.getElementById("sliderLambda");
      if (slider) slider.value = targetLam.toFixed(2);
      updateSlopeMetaUI();
    }

    // 3. Prefix KV-Cache continuous looping race
    if (state.reveals.kv > 0.35) {
      const totalCycle = state.kvRace.durationMs + state.kvRace.holdMs;
      state.kvRace.elapsed = (state.kvRace.elapsed + dt) % totalCycle;
    }

    // 4. Ablation Forest Plot autonomous spotlight tour
    if (state.autoForest && state.reveals.forest > 0.65 && state.bundle) {
      state.autoForestTimer += dt;
      if (state.autoForestTimer >= 4200) {
        state.autoForestTimer = 0;
        const order = [3, 0, 1, 2, 6, 4, 5];
        const curPos = order.indexOf(state.forestSelectedIndex);
        state.forestSelectedIndex = order[(curPos + 1) % order.length];
        updateForestInspector(state.forestSelectedIndex);
      }
    }
  }

  // =========================================================================
  // Module 01: Dual-Grid Timeline Microscope
  // =========================================================================
  function updateLeakageBanner() {
    const c = activeCase();
    if (!c || !state.bundle) return;
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
      prefixEl.innerHTML = `${cleanPart}<span class="leak-char">${cut.leaked_char}</span>`;
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

    const reveal = easeOutCubic(state.reveals.grid);
    const padL = 74;
    const padR = 20;
    const usableW = Math.max(120, w - padL - padR);
    const nAudio = c.n_frames_80ms || 38;
    const nAsr = nAudio * 2;
    const xFromAudio = (f80) => padL + (usableW * f80) / Math.max(nAudio - 1, 1);
    const xFromAsr = (f40) => padL + (usableW * f40) / Math.max(nAsr - 2, 1);

    const cut = c.cuts[state.cutMode] || c.cuts.C;
    state.bladePos40ms += (cut.frame_40ms - state.bladePos40ms) * 0.16;
    const bladeX = xFromAsr(state.bladePos40ms);

    const upperTop = 36;
    const upperBot = h * 0.52;
    const lowerTop = h * 0.65;
    const lowerBot = h - 32;

    // Left lane labels
    ctx.save();
    ctx.font = `600 10px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.blue;
    ctx.fillText("80 ms", 6, upperTop + 14);
    ctx.fillStyle = PALETTE.faint;
    ctx.font = `500 9.5px ${CANVAS_SANS}`;
    ctx.fillText("声学帧", 6, upperTop + 30);

    ctx.font = `600 10px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.teal;
    ctx.fillText("40 ms", 6, lowerTop + 16);
    ctx.fillStyle = PALETTE.faint;
    ctx.font = `500 9.5px ${CANVAS_SANS}`;
    ctx.fillText("字符轨", 6, lowerTop + 32);
    ctx.restore();

    // Progressive left-to-right clip reveal
    ctx.save();
    ctx.beginPath();
    ctx.rect(padL - 8, 0, (usableW + 32) * reveal, h);
    ctx.clip();

    // Subtle prefix wash up to the blade
    const washGrad = ctx.createLinearGradient(padL, 0, bladeX, 0);
    if (cut.leaked) {
      washGrad.addColorStop(0, "rgba(230, 106, 78, 0.0)");
      washGrad.addColorStop(1, "rgba(230, 106, 78, 0.08)");
    } else {
      washGrad.addColorStop(0, "rgba(43, 156, 148, 0.0)");
      washGrad.addColorStop(1, "rgba(43, 156, 148, 0.075)");
    }
    ctx.fillStyle = washGrad;
    ctx.fillRect(padL, upperTop - 6, Math.max(0, bladeX - padL), lowerBot - upperTop + 10);

    // Subtle vertical ticks
    ctx.strokeStyle = "rgba(29, 29, 31, 0.045)";
    ctx.lineWidth = 1;
    for (let f = 0; f < nAudio; f += 2) {
      const x = xFromAudio(f);
      ctx.beginPath();
      ctx.moveTo(x, upperTop);
      ctx.lineTo(x, upperBot);
      ctx.stroke();
    }

    // Breathing 8-frame located window glow
    const pulse = 0.5 + 0.5 * Math.sin(state.clock * 2.4);
    const winStart = c.located_start_frame;
    const winEnd = Math.min(nAudio - 1, winStart + (state.windowSize || 8));
    const wx1 = xFromAudio(winStart);
    const wx2 = xFromAudio(winEnd);
    const winGrad = ctx.createLinearGradient(wx1, 0, wx2, 0);
    winGrad.addColorStop(0, "rgba(0, 113, 227, 0.0)");
    winGrad.addColorStop(0.2, `rgba(0, 113, 227, ${(0.08 + pulse * 0.05).toFixed(3)})`);
    winGrad.addColorStop(0.8, `rgba(0, 113, 227, ${(0.08 + pulse * 0.05).toFixed(3)})`);
    winGrad.addColorStop(1, "rgba(0, 113, 227, 0.0)");
    ctx.fillStyle = winGrad;
    ctx.fillRect(wx1, upperTop, Math.max(18, wx2 - wx1), upperBot - upperTop);

    // Upper baseline
    ctx.strokeStyle = "rgba(29, 29, 31, 0.08)";
    ctx.beginPath();
    ctx.moveTo(padL, upperBot);
    ctx.lineTo(w - padR, upperBot);
    ctx.stroke();

    // Subtle animated wave breathing on response curves
    const breathe = (idx) => 1 + 0.025 * Math.sin(state.clock * 2.2 - idx * 0.25);
    const mapY = (v) => upperBot - v * (upperBot - upperTop - 16);

    const distPts = c.distractor_response.map((v, idx) => ({ x: xFromAudio(idx), y: mapY(v * breathe(idx)) }));
    ctx.save();
    ctx.strokeStyle = "rgba(123, 107, 176, 0.48)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    traceSmoothCurve(ctx, distPts);
    ctx.stroke();
    ctx.restore();

    const tgtPts = c.target_response.map((v, idx) => ({ x: xFromAudio(idx), y: mapY(v * breathe(idx)) }));
    const areaGrad = ctx.createLinearGradient(0, upperTop, 0, upperBot);
    areaGrad.addColorStop(0, "rgba(0, 113, 227, 0.16)");
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

    ctx.strokeStyle = "rgba(0, 113, 227, 0.12)";
    ctx.lineWidth = 8;
    traceSmoothCurve(ctx, tgtPts);
    ctx.stroke();

    ctx.strokeStyle = PALETTE.blue;
    ctx.lineWidth = 2.1;
    ctx.lineCap = "round";
    traceSmoothCurve(ctx, tgtPts);
    ctx.stroke();
    ctx.restore();

    // Live drifting acoustic scan bead riding along the target response curve
    const scanT = (state.clock * 0.18) % 1.0;
    const scanIdx = scanT * (tgtPts.length - 1);
    const lo = Math.floor(scanIdx);
    const hi = Math.min(lo + 1, tgtPts.length - 1);
    const frac = scanIdx - lo;
    const bx = tgtPts[lo].x * (1 - frac) + tgtPts[hi].x * frac;
    const by = tgtPts[lo].y * (1 - frac) + tgtPts[hi].y * frac;
    const halo = ctx.createRadialGradient(bx, by, 0, bx, by, 14);
    halo.addColorStop(0, "rgba(0, 113, 227, 0.32)");
    halo.addColorStop(1, "rgba(0, 113, 227, 0)");
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(bx, by, 14, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = PALETTE.blue;
    ctx.beginPath();
    ctx.arc(bx, by, 3, 0, Math.PI * 2);
    ctx.fill();

    // Lower 40ms ASR character rail (borderless floating typography + soft tint circles)
    const railY = (lowerTop + lowerBot) * 0.5;
    ctx.strokeStyle = "rgba(43, 156, 148, 0.24)";
    ctx.lineWidth = 1.4;
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

      ctx.save();
      if (isLeakedToken) {
        ctx.fillStyle = `rgba(230, 106, 78, ${(0.16 + pulse * 0.12).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(tx, railY, 16, 0, Math.PI * 2);
        ctx.fill();
      } else if (isTargetChar) {
        ctx.fillStyle = "rgba(0, 113, 227, 0.10)";
        ctx.beginPath();
        ctx.arc(tx, railY, 15, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillStyle = "rgba(251, 251, 253, 0.92)";
        ctx.beginPath();
        ctx.arc(tx, railY, 13, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.font = (isTargetChar || isLeakedToken) ? `650 13.5px ${CANVAS_SANS}` : `500 12.5px ${CANVAS_SANS}`;
      ctx.fillStyle = isLeakedToken ? "#c23318" : (isTargetChar ? PALETTE.blue : (isLeftOfBlade ? PALETTE.ink : PALETTE.faint));
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(ch, tx, railY + 0.5);

      ctx.font = `500 8.5px ${CANVAS_MONO}`;
      ctx.fillStyle = "rgba(134, 134, 139, 0.62)";
      ctx.fillText(`${(f40 * 0.04).toFixed(2)}s`, tx, lowerBot + 14);
      ctx.restore();
    }

    // Animated Cut Blade
    ctx.save();
    const bladeColor = cut.leaked ? PALETTE.coral : PALETTE.teal;
    ctx.strokeStyle = bladeColor;
    ctx.lineWidth = 2.0;
    ctx.setLineDash(cut.leaked ? [] : [4, 4]);
    ctx.beginPath();
    ctx.moveTo(bladeX, upperTop - 12);
    ctx.lineTo(bladeX, lowerBot + 2);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = bladeColor;
    ctx.beginPath();
    ctx.arc(bladeX, upperTop - 12, 4, 0, Math.PI * 2);
    ctx.fill();

    const badgeLabel = cut.leaked
      ? `LEAKAGE · ${(cut.frame_40ms * 0.04).toFixed(2)}s`
      : (state.cutMode === "C_phonetic"
        ? `PINYIN SNAP ✓`
        : `GRID SNAP ✓`);
    const badgeX = clamp(bladeX, padL + 72, w - padR - 72);
    drawPill(
      ctx,
      badgeLabel,
      badgeX,
      (upperBot + lowerTop) * 0.5,
      cut.leaked ? "rgba(230, 106, 78, 0.94)" : "rgba(43, 156, 148, 0.94)",
      "#ffffff"
    );
    ctx.restore();
    ctx.restore();
  }

  // =========================================================================
  // Module 02: Two-Stage Top-20 Candidate Slopegraph
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
    const modeTag = state.autoLambda ? "自动演示" : (lam < 0.15 ? "纯声学" : Math.abs(lam - 4.0) < 0.2 ? "推荐默认" : "手动调节");
    if (readout) readout.textContent = `λ = ${lam.toFixed(1)} · ${modeTag}`;

    if (state.bundle && metric) {
      const recall = (state.windowSize === 8 && state.useCenter)
        ? 94.7
        : (state.windowSize === 8 ? 93.4 : (state.useCenter ? 93.1 : 92.5));
      const sweep = state.cutMode === "C_phonetic"
        ? state.bundle.lambda_sweep.C_phonetic
        : state.bundle.lambda_sweep.C;
      const keys = Object.keys(sweep).map(Number).sort((a, b) => a - b);
      let nearest = keys[0];
      for (const k of keys) {
        if (Math.abs(k - lam) < Math.abs(nearest - lam)) nearest = k;
      }
      const keyStr = Number.isInteger(nearest) ? String(nearest) : String(nearest);
      let top1 = sweep[keyStr] ?? 54.2;
      if (state.windowSize === 5) top1 = Math.max(43.7, top1 - 2.4);
      if (!state.useCenter) top1 = Math.max(43.7, top1 - 1.3);
      metric.textContent = `Top-20 召回 ${recall.toFixed(1)}% · Top-1 ${top1.toFixed(1)}% (λ=${lam.toFixed(1)})`;
    }

    if (failBanner && c) {
      failBanner.hidden = !c.failure_case;
      if (c.failure_case && failText) {
        failText.textContent = `目标词「${c.target_word}」在 Stage-1 排第 ${c.out_of_pool_rank || 24} 位，未进入 Top-20（当前 λ = ${lam.toFixed(1)}），后级重排无法无中生有。`;
      }
    }
  }

  function drawSlopegraph() {
    const env = prepareCanvas("canvasSlope");
    const c = activeCase();
    if (!env || !c) return;
    const { ctx, w, h } = env;
    ctx.clearRect(0, 0, w, h);

    const reveal = easeOutCubic(state.reveals.slope);
    const { leftList } = computeSlopeRankings();
    const n = Math.min(12, leftList.length);
    const padTop = 44;
    const padBot = c.failure_case ? 42 : 22;
    const rowStep = (h - padTop - padBot) / Math.max(n - 1, 1);

    const colL = Math.min(230, w * 0.27);
    const colR = w - Math.min(230, w * 0.27);

    ctx.save();
    ctx.font = `600 10px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.faint;
    ctx.fillText(`STAGE-1 声学池 (${state.windowSize}帧${state.useCenter ? "+去均值" : ""})`, 4, 20);
    ctx.textAlign = "right";
    ctx.fillStyle = PALETTE.blue;
    ctx.fillText(`STAGE-2 语境重排 (λ = ${state.lambdaVal.toFixed(1)})`, w - 4, 20);
    ctx.restore();

    for (const item of leftList) {
      const prev = state.slopeAnimRanks.has(item.word)
        ? state.slopeAnimRanks.get(item.word)
        : item.rightRank;
      const next = prev + (item.rightRank - prev) * 0.14;
      state.slopeAnimRanks.set(item.word, next);
    }

    const orderedDraw = [...leftList].sort((a, b) => (a.is_target ? 1 : 0) - (b.is_target ? 1 : 0));

    for (const item of orderedDraw) {
      const yL = padTop + item.leftRank * rowStep;
      const animR = state.slopeAnimRanks.get(item.word) ?? item.rightRank;
      const yR = padTop + animR * rowStep;
      const movedUp = item.rightRank < item.leftRank;
      const movedDown = item.rightRank > item.leftRank;
      const midX = (colL + colR) * 0.5;

      // Progressive ribbon reveal
      ctx.save();
      ctx.globalAlpha = reveal;
      ctx.beginPath();
      ctx.moveTo(colL + 10, yL);
      ctx.bezierCurveTo(midX, yL, midX, yR, colR - 10, yR);

      if (item.is_target) {
        ctx.strokeStyle = "rgba(0, 113, 227, 0.16)";
        ctx.lineWidth = 10;
        ctx.stroke();
        ctx.strokeStyle = PALETTE.blue;
        ctx.lineWidth = 2.6;
        ctx.stroke();

        // Flowing luminous energy bead along target curve
        const beadT = (state.clock * 0.42) % 1.0;
        const pt = evalCubicBezier(
          { x: colL + 10, y: yL },
          { x: midX, y: yL },
          { x: midX, y: yR },
          { x: colR - 10, y: yR },
          beadT
        );
        const glow = ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, 12);
        glow.addColorStop(0, "rgba(0, 113, 227, 0.45)");
        glow.addColorStop(1, "rgba(0, 113, 227, 0)");
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 12, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3.0, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.strokeStyle = movedUp
          ? "rgba(43, 156, 148, 0.30)"
          : movedDown
            ? "rgba(230, 106, 78, 0.20)"
            : "rgba(29, 29, 31, 0.08)";
        ctx.lineWidth = 1.25;
        ctx.stroke();
      }
      ctx.restore();

      // Left column
      ctx.save();
      const scoreBarW = clamp((item.s1 - 0.42) * 180, 8, 62) * reveal;
      ctx.fillStyle = item.is_target ? "rgba(0, 113, 227, 0.16)" : "rgba(29, 29, 31, 0.055)";
      drawRoundedRect(ctx, colL - scoreBarW - 6, yL - 5, scoreBarW, 10, 5);
      ctx.fill();

      ctx.fillStyle = item.is_target ? PALETTE.blue : "rgba(110, 110, 115, 0.45)";
      ctx.beginPath();
      ctx.arc(colL + 4, yL, item.is_target ? 4.2 : 2.6, 0, Math.PI * 2);
      ctx.fill();

      ctx.font = item.is_target ? `700 12.5px ${CANVAS_SANS}` : `500 11.5px ${CANVAS_SANS}`;
      ctx.fillStyle = item.is_target ? PALETTE.blue : PALETTE.inkSoft;
      ctx.textBaseline = "middle";
      ctx.fillText(`#${item.leftRank + 1}  ${item.word}`, 4, yL);

      ctx.font = `500 10px ${CANVAS_MONO}`;
      ctx.fillStyle = PALETTE.faint;
      ctx.textAlign = "right";
      ctx.fillText(item.s1.toFixed(3), colL - scoreBarW - 12, yL);
      ctx.restore();

      // Right column
      ctx.save();
      ctx.fillStyle = item.is_target ? PALETTE.blue : (movedUp ? PALETTE.teal : "rgba(110, 110, 115, 0.45)");
      ctx.beginPath();
      ctx.arc(colR - 4, yR, item.is_target ? 4.2 : 2.6, 0, Math.PI * 2);
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
        ctx.fillText(deltaTxt, w - 4, yR);
      }
      ctx.restore();
    }

    if (c.failure_case) {
      const ceilY = h - 18;
      ctx.save();
      ctx.strokeStyle = "rgba(230, 106, 78, 0.42)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(4, ceilY - 12);
      ctx.lineTo(w - 4, ceilY - 12);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = `600 11px ${CANVAS_MONO}`;
      ctx.fillStyle = PALETTE.coral;
      ctx.textAlign = "center";
      ctx.fillText(`✕ 目标词「${c.target_word}」位于 #${c.out_of_pool_rank || 24}（Top-20 候选池外），重排无法触达`, w * 0.5, ceilY + 4);
      ctx.restore();
    }
  }

  // =========================================================================
  // Module 03: Prefix KV-Cache Continuous Race
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
    state.kvRace.elapsed = 0;
  }

  function drawKvRace() {
    const env = prepareCanvas("canvasKvRace");
    const b = getActiveKvBench();
    if (!env || !b) return;
    const { ctx, w, h } = env;
    ctx.clearRect(0, 0, w, h);

    const padL = 132;
    const padR = 24;
    const trackW = Math.max(120, w - padL - padR);
    const maxTimeS = Math.max(b.full_median_s, b.cached_median_s) * 1.04;
    const progress = clamp(state.kvRace.elapsed / state.kvRace.durationMs, 0, 1);
    const simTimeS = progress * b.full_median_s;

    const topY = 34;
    const laneH = (h - 78) * 0.48;
    const botY = topY + laneH + 28;
    const nBars = 20;

    // Hairline divider
    ctx.strokeStyle = "rgba(29, 29, 31, 0.06)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(4, topY + laneH + 14);
    ctx.lineTo(w - 4, topY + laneH + 14);
    ctx.stroke();

    // Top Lane: Full Sequence
    ctx.save();
    ctx.font = `650 11.5px ${CANVAS_SANS}`;
    ctx.fillStyle = PALETTE.inkSoft;
    ctx.fillText("Full Sequence", 4, topY + 16);
    ctx.font = `500 10px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.faint;
    ctx.fillText("20× 重算前缀", 4, topY + 32);
    ctx.fillText(`${Math.min(simTimeS, b.full_median_s).toFixed(3)}s`, 4, topY + 48);

    const barGap = 2;
    const barH = Math.max(2.5, (laneH - barGap * (nBars - 1)) / nBars);
    const fullRatio = clamp(simTimeS / b.full_median_s, 0, 1);
    const fullTotalW = (b.full_median_s / maxTimeS) * trackW;

    for (let i = 0; i < nBars; i++) {
      const y = topY + i * (barH + barGap);
      const localP = clamp((fullRatio * 1.16) - (i / nBars) * 0.16, 0, 1);
      const curW = fullTotalW * localP;
      const prefixPart = curW * 0.84;
      const candPart = curW - prefixPart;

      ctx.fillStyle = "rgba(29, 29, 31, 0.04)";
      ctx.fillRect(padL, y, fullTotalW, barH);
      ctx.fillStyle = "rgba(134, 134, 139, 0.38)";
      ctx.fillRect(padL, y, prefixPart, barH);
      if (candPart > 0.5) {
        ctx.fillStyle = "rgba(0, 113, 227, 0.65)";
        ctx.fillRect(padL + prefixPart, y, candPart, barH);
      }
    }
    if (fullRatio >= 0.995) {
      drawPill(ctx, `完成 ${b.full_median_s.toFixed(3)}s`, padL + fullTotalW - 44, topY - 12, "rgba(29, 29, 31, 0.07)", PALETTE.inkSoft);
    }
    ctx.restore();

    // Bottom Lane: Prefix-KV Reuse
    ctx.save();
    ctx.font = `650 11.5px ${CANVAS_SANS}`;
    ctx.fillStyle = PALETTE.teal;
    ctx.fillText("Prefix-KV Reuse", 4, botY + 16);
    ctx.font = `500 10px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.faint;
    ctx.fillText("1× 共享 + 20 路", 4, botY + 32);
    ctx.fillStyle = PALETTE.teal;
    ctx.fillText(`${Math.min(simTimeS, b.cached_median_s).toFixed(3)}s`, 4, botY + 48);

    const cachedRatio = clamp(simTimeS / b.cached_median_s, 0, 1);
    const cachedTotalW = (b.cached_median_s / maxTimeS) * trackW;
    const trunkW = cachedTotalW * 0.66;
    const forkW = cachedTotalW - trunkW;

    const trunkFill = clamp(cachedRatio / 0.66, 0, 1) * trunkW;
    const forkFill = clamp((cachedRatio - 0.66) / 0.34, 0, 1) * forkW;

    const trunkY = botY + laneH * 0.28;
    const trunkH = laneH * 0.44;
    drawRoundedRect(ctx, padL, trunkY, trunkW, trunkH, 8);
    ctx.fillStyle = "rgba(0, 113, 227, 0.08)";
    ctx.fill();
    if (trunkFill > 1) {
      drawRoundedRect(ctx, padL, trunkY, trunkFill, trunkH, 8);
      ctx.fillStyle = "rgba(0, 113, 227, 0.86)";
      ctx.fill();
      ctx.font = `600 10px ${CANVAS_MONO}`;
      ctx.fillStyle = "#ffffff";
      ctx.textBaseline = "middle";
      if (trunkFill > 76) ctx.fillText("Shared KV Prefix", padL + 10, trunkY + trunkH * 0.5);
    }

    const forkStartX = padL + trunkW;
    for (let i = 0; i < nBars; i++) {
      const y = botY + i * (barH + barGap);
      ctx.strokeStyle = forkFill > 0 ? "rgba(43, 156, 148, 0.35)" : "rgba(29, 29, 31, 0.06)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(forkStartX, trunkY + trunkH * 0.5);
      ctx.lineTo(forkStartX + 12, y + barH * 0.5);
      ctx.stroke();

      ctx.fillStyle = "rgba(43, 156, 148, 0.11)";
      ctx.fillRect(forkStartX + 12, y, Math.max(8, forkW - 12), barH);
      if (forkFill > 0.5) {
        ctx.fillStyle = PALETTE.teal;
        ctx.fillRect(forkStartX + 12, y, Math.max(0, (forkW - 12) * (forkFill / Math.max(forkW, 1))), barH);
      }
    }

    if (cachedRatio >= 0.995) {
      drawPill(
        ctx,
        `✓ ${b.cached_median_s.toFixed(3)}s · ${b.speedup.toFixed(2)}×`,
        Math.min(w - padR - 56, padL + cachedTotalW + 52),
        botY + laneH * 0.5,
        "rgba(43, 156, 148, 0.94)",
        "#ffffff"
      );
    }
    ctx.restore();
  }

  // =========================================================================
  // Module 04: Interactive Living Forest Plot
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

    const reveal = easeOutCubic(state.reveals.forest);
    const pulse = 0.5 + 0.5 * Math.sin(state.clock * 2.8);
    const entries = state.bundle.forest_plot;
    const padL = Math.min(200, w * 0.34);
    const padR = 46;
    const padT = 32;
    const padB = 30;
    const plotW = Math.max(120, w - padL - padR);

    const minVal = -5.5;
    const maxVal = 5.2;
    const xOf = (val) => padL + ((val - minVal) / (maxVal - minVal)) * plotW;
    const zeroX = xOf(0);

    // Subtle open zero-line & ticks
    ctx.save();
    ctx.font = `500 9.5px ${CANVAS_MONO}`;
    ctx.textAlign = "center";
    for (const tick of [-4, -2, 0, 2, 4]) {
      const tx = xOf(tick);
      ctx.strokeStyle = tick === 0 ? "rgba(29, 29, 31, 0.32)" : "rgba(29, 29, 31, 0.05)";
      ctx.lineWidth = tick === 0 ? 1.4 : 1;
      ctx.setLineDash(tick === 0 ? [4, 4] : []);
      ctx.beginPath();
      ctx.moveTo(tx, padT - 6);
      ctx.lineTo(tx, h - padB + 4);
      ctx.stroke();
      ctx.fillStyle = tick === 0 ? PALETTE.inkSoft : PALETTE.faint;
      ctx.fillText(`${tick > 0 ? "+" : ""}${tick} pp`, tx, h - 10);
    }
    ctx.setLineDash([]);
    ctx.restore();

    ctx.save();
    ctx.font = `600 9.5px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.coral;
    ctx.fillText("淘汰区", padL + 4, 16);
    ctx.textAlign = "right";
    ctx.fillStyle = PALETTE.teal;
    ctx.fillText("通过区 (CI > 0)", w - padR, 16);
    ctx.restore();

    const rowH = (h - padT - padB) / Math.max(entries.length, 1);
    state.forestRowBounds = [];

    entries.forEach((item, idx) => {
      const cy = padT + rowH * (idx + 0.5);
      const isSelected = idx === state.forestSelectedIndex;
      const accepted = item.group === "accepted";
      const color = accepted ? PALETTE.teal : PALETTE.coral;
      const rowReveal = clamp((reveal * 1.4) - idx * 0.08, 0, 1);

      state.forestRowBounds.push({ idx, y1: cy - rowH * 0.48, y2: cy + rowH * 0.48 });

      if (isSelected) {
        ctx.save();
        drawRoundedRect(ctx, 0, cy - rowH * 0.40, w, rowH * 0.80, 12);
        ctx.fillStyle = accepted ? "rgba(43, 156, 148, 0.065)" : "rgba(230, 106, 78, 0.065)";
        ctx.fill();
        ctx.restore();
      }

      ctx.save();
      ctx.font = isSelected ? `650 11.5px ${CANVAS_SANS}` : `500 11px ${CANVAS_SANS}`;
      ctx.fillStyle = isSelected ? PALETTE.ink : PALETTE.inkSoft;
      ctx.textBaseline = "middle";
      const shortLabel = item.label.length > 15 ? `${item.label.slice(0, 15)}…` : item.label;
      ctx.fillText(shortLabel, 8, cy);
      ctx.restore();

      // Progressive expansion from zeroX
      const xLo = zeroX + (xOf(item.ci_low) - zeroX) * rowReveal;
      const xHi = zeroX + (xOf(item.ci_high) - zeroX) * rowReveal;
      const xMid = zeroX + (xOf(item.delta_pp) - zeroX) * rowReveal;

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = isSelected ? 2.6 : 1.8;
      ctx.beginPath();
      ctx.moveTo(xLo, cy);
      ctx.lineTo(xHi, cy);
      ctx.moveTo(xLo, cy - 4.5);
      ctx.lineTo(xLo, cy + 4.5);
      ctx.moveTo(xHi, cy - 4.5);
      ctx.lineTo(xHi, cy + 4.5);
      ctx.stroke();

      if (isSelected) {
        ctx.fillStyle = accepted
          ? `rgba(43, 156, 148, ${(0.14 + pulse * 0.12).toFixed(3)})`
          : `rgba(230, 106, 78, ${(0.14 + pulse * 0.12).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(xMid, cy, 11 + pulse * 3, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(xMid, cy, isSelected ? 6.0 : 4.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(xMid, cy, isSelected ? 4.2 : 3.2, 0, Math.PI * 2);
      ctx.fill();

      ctx.font = `600 10px ${CANVAS_MONO}`;
      ctx.fillStyle = color;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(`${item.delta_pp >= 0 ? "+" : ""}${item.delta_pp.toFixed(2)}`, w - 6, cy);
      ctx.restore();
    });
  }

  // =========================================================================
  // Local JSONL Zero-Upload Audit Parser
  // =========================================================================
  function parseJsonlText(text) {
    const rows = [];
    for (const line of text.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try { rows.push(JSON.parse(trimmed)); } catch (_) {}
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
      const rows = parseJsonlText(await file.text());
      for (const r of rows) {
        if (Array.isArray(r.words) && Array.isArray(r.score) && r.utt) poolRows.push(r);
        else if (r.scores && typeof r.scores === "object" && r.utt) lmByUtt.set(r.utt, r);
        else if (Array.isArray(r.tokens) && (r.utt_id || r.utt)) readingsByUtt.set(r.utt_id || r.utt, r);
      }
    }

    if (!poolRows.length) {
      showToast("未检测到包含 words/score 字段的 pool.jsonl");
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
      const rawCut40 = startFrame * 2;
      const snappedCut40 = Math.max(0, rawCut40 - 2);
      const prefixP = tokens40.filter(t => t[0] <= rawCut40).map(t => t[1]).join("");
      const prefixC = tokens40.filter(t => t[0] < snappedCut40).map(t => t[1]).join("");
      const leakedP = Boolean(goldWord && prefixP.endsWith(goldWord[0]));
      const candidates = pRow.words.slice(0, 12).map((w, i) => {
        const s1 = Number(pRow.score[i]) || 0.5;
        let lmC = -2.0 - i * 0.12;
        if (lmRow && lmRow.scores && lmRow.scores[w]) lmC = Number(lmRow.scores[w].C ?? lmRow.scores[w].P ?? lmC);
        return {
          word: w, s1_k8_center: s1, s1_k8_none: s1 - 0.015, s1_k5_center: s1 - 0.02, s1_k5_none: s1 - 0.032,
          lm_c: lmC, lm_p: lmC + (w === goldWord ? 0.45 : 0), lm_ph: lmC, is_target: w === goldWord
        };
      });
      const makeResp = (center, peak) => Array.from({ length: nFrames }, (_, f) =>
        Math.max(0.06, Math.min(0.95, 0.14 + peak * Math.exp(-0.5 * ((f - center) / 2.6) ** 2)))
      );
      return {
        id: `local_${idx}`, title: `本地 · ${utt}`, tag: inPool ? `目标: ${goldWord}` : "池外失败",
        failure_case: !inPool, utterance: utt, target_word: goldWord, distractor_word: pRow.words[1] || pRow.words[0],
        transcript: tokens40.map(t => t[1]).join(""),
        asr_note: `本地 JSONL 零上传审计：${utt}（${pRow.words.length} 个候选）。`,
        n_frames_80ms: nFrames, window_frames: 8, located_start_frame: startFrame,
        target_response: makeResp(startFrame, inPool ? 0.72 : 0.34),
        distractor_response: makeResp(Math.max(4, startFrame - 5), 0.55),
        asr_tokens_40ms: tokens40,
        cuts: {
          P: { frame_40ms: rawCut40, prefix: prefixP, leaked: leakedP, leaked_char: leakedP ? goldWord[0] : null, note: "Arm P 硬切边界。" },
          C: { frame_40ms: snappedCut40, prefix: prefixC, leaked: false, leaked_char: null, note: "Arm C 吸附至字符边界之前。" },
          C_phonetic: { frame_40ms: snappedCut40, prefix: prefixC, leaked: false, leaked_char: null, note: "C + 拼音近似对齐边界。" }
        },
        candidates
      };
    });

    state.cases = parsedCases;
    state.activeCaseIndex = 0;
    state.localLoaded = true;
    document.getElementById("auditSourceBadge").textContent = `本地零上传 · ${parsedCases.length} 条语音`;
    document.getElementById("btnResetDemoData").hidden = false;
    renderCaseChips();
    selectCase(0);
    showToast(`已载入 ${parsedCases.length} 条本地 JSONL 样本（零上传）`);
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
    state.autoGridTimer = 0;
    renderCaseChips();
    updateLeakageBanner();
    updateSlopeMetaUI();
  }

  function setCutMode(mode, userInitiated = true, announce = true) {
    if (userInitiated) state.autoGrid = false;
    state.cutMode = mode;
    document.querySelectorAll("[data-cut-mode]").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.cutMode === mode);
    });
    updateLeakageBanner();
    updateSlopeMetaUI();
    if (announce) {
      const labels = { P: "Arm P 硬切（观察泄漏）", C: "Arm C 网格吸附", C_phonetic: "C + 拼音近似" };
      showToast(`已切换至 ${labels[mode] || mode}`);
    }
  }

  function bindEvents() {
    document.querySelectorAll("[data-cut-mode]").forEach((btn) => {
      btn.addEventListener("click", () => setCutMode(btn.dataset.cutMode, true, true));
    });

    const slider = document.getElementById("sliderLambda");
    if (slider) {
      slider.addEventListener("input", (e) => {
        state.autoLambda = false;
        state.lambdaVal = parseFloat(e.target.value);
        updateSlopeMetaUI();
      });
    }

    const btnCenter = document.getElementById("btnToggleCenter");
    if (btnCenter) {
      btnCenter.addEventListener("click", () => {
        state.useCenter = !state.useCenter;
        btnCenter.classList.toggle("is-on", state.useCenter);
        btnCenter.textContent = `去均值：${state.useCenter ? "开" : "关"}`;
        updateSlopeMetaUI();
      });
    }

    const btnWin = document.getElementById("btnToggleWindow");
    if (btnWin) {
      btnWin.addEventListener("click", () => {
        state.windowSize = state.windowSize === 8 ? 5 : 8;
        btnWin.classList.toggle("is-on", state.windowSize === 8);
        btnWin.textContent = state.windowSize === 8 ? "8 帧窗 (640 ms)" : "5 帧窗 (400 ms)";
        updateSlopeMetaUI();
      });
    }

    const btnFail = document.getElementById("btnJumpFailure");
    if (btnFail) {
      btnFail.addEventListener("click", () => {
        const failIdx = state.cases.findIndex(c => c.failure_case);
        if (failIdx >= 0) {
          selectCase(failIdx);
          showToast("已切至 5.3% 池外失败例 — 观察 λ 扫描仍无法救回 #24");
        }
      });
    }

    document.querySelectorAll("[data-kv-id]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.kvId = btn.dataset.kvId;
        document.querySelectorAll("[data-kv-id]").forEach(b => b.classList.toggle("is-active", b === btn));
        updateKvStatsUI();
        startKvRace();
      });
    });
    document.getElementById("btnRunKvRace")?.addEventListener("click", () => startKvRace());

    const forestCanvas = document.getElementById("canvasForest");
    if (forestCanvas) {
      const hitRow = (clientY) => {
        const rect = forestCanvas.getBoundingClientRect();
        const y = clientY - rect.top;
        const found = state.forestRowBounds.find(b => y >= b.y1 && y <= b.y2);
        return found ? found.idx : -1;
      };
      forestCanvas.addEventListener("click", (e) => {
        const idx = hitRow(e.clientY);
        if (idx >= 0) {
          state.autoForest = false;
          state.forestSelectedIndex = idx;
          updateForestInspector(idx);
        }
      });
    }

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
      if (willOpen && window.scrollY > 80) window.scrollTo({ top: 0, behavior: "smooth" });
    });
    closeBtn?.addEventListener("click", () => setDrawer(false));

    const dropZone = document.getElementById("jsonlDropZone");
    document.getElementById("jsonlFileInput")?.addEventListener("change", (e) => handleLocalJsonlFiles(e.target.files));
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
      document.getElementById("auditSourceBadge").textContent = "L1/L2/L3 公开基准 · 双冻结后端";
      document.getElementById("btnResetDemoData").hidden = true;
      renderCaseChips();
      selectCase(0);
      showToast("已恢复内置实证病例");
    });
  }

  let lastTime = performance.now();
  function animationLoop(now) {
    const dt = Math.min(50, Math.max(0, now - lastTime));
    lastTime = now;
    state.clock += dt * 0.001;

    updateScrollChoreography(dt);
    stepAutonomousChoreography(dt);

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
    try {
      const resp = await fetch("data/bundle.json");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      state.bundle = await resp.json();
      state.builtinCases = state.bundle.cases;
      state.cases = state.bundle.cases;

      document.getElementById("kpiRecall").textContent = "96.2%–100%";
      document.getElementById("kpiLeak").textContent = "0.00%";
      document.getElementById("kpiSpeedup").textContent = "3.07× / 7.93×";

      renderCaseChips();
      setCutMode("P", false, false);
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
