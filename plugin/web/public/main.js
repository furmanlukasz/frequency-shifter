// Holy Shifter WebView UI — responsive layout. Sections stack and the whole UI scrolls,
// so it fits any size (iPhone / iPad / desktop / AUv3 host view). No CSS zoom: pointer
// coordinates map 1:1 to layout, so slider/knob drags are accurate on touch.
import * as Juce from "./juce.js";

// Surface JS errors + status to native stderr (so the UI can be debugged headlessly).
const nlog = (...a) => { try { Juce.getNativeFunction("jsLog")(a.map(String).join(" ")); } catch (e) {} };
window.addEventListener("error", (e) => nlog("JS ERROR:", e.message, "@", (e.filename || "") + ":" + e.lineno));
window.addEventListener("unhandledrejection", (e) => nlog("PROMISE REJECT:", (e && e.reason && e.reason.message) || (e && e.reason)));
nlog("main.js start; HOLY_PARAMS =", (window.HOLY_PARAMS || []).length);

const NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

const stage = document.createElement("div"); stage.id = "stage";
const wrap = document.createElement("div"); wrap.className = "wrap"; stage.append(wrap);
document.getElementById("app").replaceWith(stage);

// ---- helpers -------------------------------------------------------------
function mkEl(tag, cls, parent, txt) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  if (parent) parent.append(e);
  return e;
}

// ---- generic controls ----------------------------------------------------
// Per-control value formatting + tempo-sync, matching Visage HolySlider exactly.
// opts: { decimals=1, suffix="", adaptive=false, secondsAbove=null,
//         sync:{ toggleId, divisionId, reversed } }  — when the sync toggle is ON the row
// shows the division label (e.g. "1/4"), snaps to the division choices, and (if reversed)
// maps slow→left / fast→right; otherwise it's a normal continuous slider.
function sliderRow(parent, id, label, opts = {}) {
  const row = mkEl("div", "row", parent);
  mkEl("span", "lbl", row, label);
  const trk = mkEl("div", "track", row);
  mkEl("div", "sbg", trk);
  const fill = mkEl("div", "sfill", trk);
  const dot = mkEl("div", "sdot", trk);
  const val = mkEl("span", "sval", row);
  const st = Juce.getSliderState(id);
  const sync = opts.sync || null;
  const syncSt = sync ? Juce.getToggleState(sync.toggleId) : null;
  const divSt = sync ? Juce.getComboBoxState(sync.divisionId) : null;
  const synced = () => !!(syncSt && syncSt.getValue());

  // Free-mode readout — mirrors HolySlider::formatValue (seconds-above, 0dp=truncate, adaptive).
  const fmtFree = () => {
    let v = st.getScaledValue(); if (!isFinite(v)) v = 0;
    if (opts.secondsAbove != null && Math.abs(v) >= opts.secondsAbove) return (v / 1000).toFixed(2) + " s";
    if (opts.decimals === 0) return String(Math.trunc(v)) + (opts.suffix || "");
    let dec = opts.decimals != null ? opts.decimals : 1;
    if (opts.adaptive) { const a = Math.abs(v); if (a > 0 && a < 0.01) dec = Math.max(dec, 3); else if (a < 0.1) dec = Math.max(dec, 2); }
    return v.toFixed(dec) + (opts.suffix || "");
  };
  const divData = () => { const choices = (divSt.properties && divSt.properties.choices) || []; return { choices, idx: divSt.getChoiceIndex(), n: choices.length }; };

  const render = () => {
    if (synced()) {
      const { choices, idx, n } = divData();
      const norm = n > 1 ? idx / (n - 1) : 0;
      const vis = sync.reversed ? 1 - norm : norm;
      fill.style.width = (vis * 100) + "%"; dot.style.left = (vis * 100) + "%";
      val.textContent = choices[idx] != null ? choices[idx] : "";
    } else {
      const nv = st.getNormalisedValue();
      fill.style.width = (nv * 100) + "%"; dot.style.left = (nv * 100) + "%";
      val.textContent = fmtFree();
    }
  };
  const setFromX = (clientX) => {
    const r = trk.getBoundingClientRect();
    const vis = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
    if (synced()) {
      const { n } = divData();
      const norm = sync.reversed ? 1 - vis : vis;
      divSt.setChoiceIndex(Math.max(0, Math.min(n - 1, Math.round(norm * (n - 1)))));
      render();
    } else {
      st.setNormalisedValue(vis);
      fill.style.width = (vis * 100) + "%"; dot.style.left = (vis * 100) + "%"; val.textContent = fmtFree();
    }
  };
  let dragging = false;
  trk.addEventListener("pointerdown", (e) => { dragging = true; trk.setPointerCapture(e.pointerId); if (!synced()) st.sliderDragStarted(); setFromX(e.clientX); });
  trk.addEventListener("pointermove", (e) => { if (dragging) setFromX(e.clientX); });
  const end = (e) => { if (!dragging) return; dragging = false; try { trk.releasePointerCapture(e.pointerId); } catch {} if (!synced()) st.sliderDragEnded(); render(); };
  trk.addEventListener("pointerup", end);
  trk.addEventListener("pointercancel", end);
  st.valueChangedEvent.addListener(render); st.propertiesChangedEvent.addListener(render);
  if (syncSt) syncSt.valueChangedEvent.addListener(render);
  if (divSt) { divSt.valueChangedEvent.addListener(render); divSt.propertiesChangedEvent.addListener(render); }
  render();
  return row;
}

// Dimming helpers (mirror Visage setDimmed): grey + disable a set of controls.
function dimEls(els, dim) { els.forEach((e) => e && e.classList.toggle("dimmed", dim)); }
function wireEnableDim(toggleId, els) {
  const st = Juce.getToggleState(toggleId);
  const apply = () => dimEls(els, !st.getValue());
  st.valueChangedEvent.addListener(apply); apply();
}

function switchEl(parent, id, label) {
  const el = mkEl("div", "switch", parent);
  const trk = mkEl("div", "track", el); mkEl("div", "dot", trk);
  if (label) mkEl("span", "swlabel", el, label);
  const st = Juce.getToggleState(id);
  const refresh = () => el.classList.toggle("on", st.getValue());
  el.addEventListener("click", () => st.setValue(!st.getValue()));
  st.valueChangedEvent.addListener(refresh); refresh();
  return el;
}

function comboEl(parent, id) {
  const sel = mkEl("select", "combo", parent);
  const st = Juce.getComboBoxState(id);
  const populate = () => {
    const ch = (st.properties && st.properties.choices) || []; sel.innerHTML = "";
    ch.forEach((c, i) => { const o = document.createElement("option"); o.value = i; o.textContent = c; sel.append(o); });
    sel.value = String(st.getChoiceIndex());
  };
  sel.addEventListener("change", () => st.setChoiceIndex(parseInt(sel.value, 10)));
  st.propertiesChangedEvent.addListener(populate);
  st.valueChangedEvent.addListener(() => { sel.value = String(st.getChoiceIndex()); }); populate();
  return sel;
}

function segmentedEl(parent, id, segs, onChange) {
  const el = mkEl("div", "seg", parent);
  const st = Juce.getComboBoxState(id);
  const opts = segs.map((s, i) => { const o = mkEl("div", "segopt", el, s); o.addEventListener("click", () => st.setChoiceIndex(i)); return o; });
  const refresh = () => { const idx = st.getChoiceIndex(); opts.forEach((o, i) => o.classList.toggle("on", i === idx)); if (onChange) onChange(idx); };
  st.valueChangedEvent.addListener(refresh); st.propertiesChangedEvent.addListener(refresh); refresh();
  return el;
}

function pianoEl(parent) {
  const el = mkEl("div", "piano", parent);
  const whiteIdx = [0, 2, 4, 5, 7, 9, 11];
  const wpct = 100 / 7;
  whiteIdx.forEach((pc, i) => makeKey(pc, i * wpct, wpct, 100, "white"));
  const blackAfter = { 1: 0, 3: 1, 6: 3, 8: 4, 10: 5 };
  const bwpct = wpct * 0.62;
  for (const pc of [1, 3, 6, 8, 10]) makeKey(pc, (blackAfter[pc] + 1) * wpct - bwpct / 2, bwpct, 62, "black");
  function makeKey(pc, leftPct, widthPct, heightPct, kind) {
    const k = mkEl("div", "pk " + kind, el, NOTE[pc]);
    k.style.left = leftPct + "%"; k.style.width = widthPct + "%"; k.style.height = heightPct + "%";
    const st = Juce.getToggleState("scaleNote" + pc);
    const refresh = () => k.classList.toggle("on", st.getValue());
    k.addEventListener("click", () => st.setValue(!st.getValue()));
    st.valueChangedEvent.addListener(refresh); refresh();
  }
  return el;
}

// ---- rotary shift knob (bipolar, symmetric-log display) -------------------
function shiftKnob(parent) {
  const el = mkEl("div", "knob", parent);
  const cv = mkEl("canvas", null, el);
  const ctx = cv.getContext("2d");
  let W = 210, H = 210;
  function sizeCanvas() {
    const r = cv.getBoundingClientRect();
    W = Math.round(r.width) || 210; H = Math.round(r.height) || W;
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    cv.width = Math.max(1, Math.round(W * dpr)); cv.height = Math.max(1, Math.round(H * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  const st = Juce.getSliderState("shiftHz");
  const A0 = -2.35619, A1 = 2.35619, LS = 10, LMAX = Math.log(1 + 5000 / LS);
  const dispFromKnob = (kn) => { const s = kn * 2 - 1, sg = s >= 0 ? 1 : -1, a = Math.abs(s); return sg * LS * (Math.exp(a * LMAX) - 1); };
  const pnFromKnob = (kn) => (dispFromKnob(kn) + 20000) / 40000;
  const knobFromPn = (pn) => { const pv = pn * 40000 - 20000, sg = pv >= 0 ? 1 : -1, a = Math.min(Math.abs(pv), 5000); return (sg * Math.log(1 + a / LS) / LMAX + 1) / 2; };

  function draw() {
    const kn = dragging ? dragKn : knobFromPn(st.getNormalisedValue());
    const ang = A0 + kn * (A1 - A0);
    const cx = W / 2, cy = H / 2, r = Math.min(W, H) * 0.36;
    ctx.clearRect(0, 0, W, H);
    const arc = (a0, a1, col, lw) => { ctx.beginPath(); ctx.lineWidth = lw; ctx.strokeStyle = col; ctx.lineCap = "round"; ctx.arc(cx, cy, r, a0 - Math.PI / 2, a1 - Math.PI / 2); ctx.stroke(); };
    arc(A0, A1, "#252320", 2.5);
    const mid = A0 + 0.5 * (A1 - A0);
    if (Math.abs(ang - mid) > 0.01) arc(Math.min(mid, ang), Math.max(mid, ang), "#c9a96e", 2.5);
    for (let i = 0; i < 5; i++) { const t = A0 + (i / 4) * (A1 - A0); const ir = r + 7, or = r + 12;
      ctx.beginPath(); ctx.lineWidth = 1; ctx.strokeStyle = (i === 2) ? "#8a857d" : "#3e3a34";
      ctx.moveTo(cx + ir * Math.sin(t), cy - ir * Math.cos(t)); ctx.lineTo(cx + or * Math.sin(t), cy - or * Math.cos(t)); ctx.stroke(); }
    ctx.beginPath(); ctx.fillStyle = "#c9a96e"; ctx.arc(cx + r * Math.sin(ang), cy - r * Math.cos(ang), 6, 0, 2 * Math.PI); ctx.fill();
    const v = dispFromKnob(kn);
    ctx.fillStyle = "#e8e4db"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.font = "300 " + Math.round(W * 0.16) + "px 'Inter'";
    ctx.fillText(Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1), cx, cy - W * 0.02);
    ctx.fillStyle = "#3e3a34"; ctx.font = "500 " + Math.round(W * 0.052) + "px 'Inter'"; ctx.fillText("HZ", cx, cy + W * 0.11);
  }
  let dragging = false, dragKn = 0, startY = 0, startKn = 0, fine = false;
  cv.addEventListener("pointerdown", (e) => { dragging = true; fine = e.shiftKey; cv.setPointerCapture(e.pointerId);
    startY = e.clientY; startKn = knobFromPn(st.getNormalisedValue()); dragKn = startKn; st.sliderDragStarted(); });
  cv.addEventListener("pointermove", (e) => { if (!dragging) return;
    const sens = fine ? 3600 : 360;   // base 360 px (you preferred this over Visage's 300); Shift = 10x finer
    dragKn = Math.min(1, Math.max(0, startKn + (startY - e.clientY) / sens));
    st.setNormalisedValue(pnFromKnob(dragKn)); draw(); });
  const end = (e) => { if (!dragging) return; dragging = false; try { cv.releasePointerCapture(e.pointerId); } catch {} st.sliderDragEnded(); draw(); };
  cv.addEventListener("pointerup", end); cv.addEventListener("pointercancel", end);
  cv.addEventListener("dblclick", () => { st.setNormalisedValue(0.5); draw(); });
  st.valueChangedEvent.addListener(draw); st.propertiesChangedEvent.addListener(draw);
  new ResizeObserver(() => { sizeCanvas(); draw(); }).observe(cv);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { sizeCanvas(); draw(); });
  sizeCanvas(); draw();
  return el;
}

// ---- preset bar ----------------------------------------------------------
function presetBar(parent) {
  const row = mkEl("div", "presetrow", parent);
  const prev = mkEl("button", "presetbtn", row, "‹");
  const nameEl = mkEl("div", "preset-name", row);
  const dd = mkEl("div", "preset-dd", nameEl);
  const next = mkEl("button", "presetbtn", row, "›");
  const save = mkEl("button", "txtbtn primary", row, "SAVE");
  const del = mkEl("button", "txtbtn outline", row, "DEL");

  const fn = (n) => Juce.getNativeFunction(n);
  const list = fn("presetList"), cur = fn("presetCurrent"), load = fn("presetLoad"), saveP = fn("presetSave"), delP = fn("presetDelete");
  let names = [], current = "";
  const nameText = mkEl("span", null, nameEl); nameText.style.cssText = "flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
  const setName = (n) => { current = n || ""; nameText.textContent = current || "—"; };

  async function refresh() {
    try { names = JSON.parse(await list()); } catch { names = []; }
    try { current = await cur(); } catch {}
    setName(current);
  }
  const closeDD = () => dd.classList.remove("open");
  function openDD() {
    dd.innerHTML = "";
    if (!names.length) mkEl("div", "preset-dd-item", dd, "(no presets)");
    names.forEach((n) => {
      const it = mkEl("div", "preset-dd-item" + (n === current ? " current" : ""), dd, n);
      it.addEventListener("click", async (e) => { e.stopPropagation(); await load(n); setName(n); closeDD(); });
    });
    dd.classList.add("open");
  }
  nameEl.addEventListener("click", (e) => { e.stopPropagation(); dd.classList.contains("open") ? closeDD() : openDD(); });
  document.addEventListener("click", closeDD);

  async function step(d) { if (!names.length) return; let i = names.indexOf(current); if (i < 0) i = 0; i = (i + d + names.length) % names.length; await load(names[i]); setName(names[i]); }
  prev.addEventListener("click", (e) => { e.stopPropagation(); step(-1); });
  next.addEventListener("click", (e) => { e.stopPropagation(); step(1); });

  save.addEventListener("click", (e) => {
    e.stopPropagation(); closeDD();
    nameEl.classList.add("editing"); nameText.textContent = "";
    const inp = mkEl("input", "preset-edit", nameText); inp.value = current || "Untitled"; inp.focus(); inp.select();
    let done = false;
    const finish = async (ok) => {
      if (done) return; done = true; nameEl.classList.remove("editing");
      const v = inp.value.trim(); setName(current);
      if (ok && v) { await saveP(v); await refresh(); setName(v); }
    };
    inp.addEventListener("click", (ev) => ev.stopPropagation());
    inp.addEventListener("keydown", (ev) => { if (ev.key === "Enter") { ev.preventDefault(); finish(true); } else if (ev.key === "Escape") finish(false); });
    inp.addEventListener("blur", () => finish(false));
  });
  del.addEventListener("click", async (e) => { e.stopPropagation(); if (current) { await delP(current); await refresh(); } });
  refresh();
}

// ---- L/R Decorrelate (native flag, not a relay) --------------------------
function decorrelateToggle(parent) {
  const el = mkEl("div", "switch", parent);
  const trk = mkEl("div", "track", el); mkEl("div", "dot", trk);
  mkEl("span", "swlabel", el, "L/R Decorr");
  const get = Juce.getNativeFunction("decorrelateGet"), set = Juce.getNativeFunction("decorrelateSet");
  let on = false;
  const refresh = () => el.classList.toggle("on", on);
  get().then((v) => { on = !!v; refresh(); }).catch(() => {});
  el.addEventListener("click", async () => { on = !on; refresh(); await set(on); });
  return el;
}

// ---- section helper ------------------------------------------------------
function section(label, enableId) {
  const sec = mkEl("div", "sec", wrap);
  const hdr = mkEl("div", "sec-hdr", sec);
  if (enableId) switchEl(hdr, enableId, "");
  if (label) mkEl("span", "h", hdr, label);
  return sec;
}

// ---- assemble ============================================================
// header
const hdrRow = mkEl("div", "hdr-row", wrap);
const titleBox = mkEl("div", null, hdrRow);
mkEl("div", "title", titleBox, "HOLY SHIFTER");
mkEl("div", "subtitle", titleBox, "Frequency Shifter with Harmonic Quantisation");
const logo = mkEl("img", "logo", hdrRow); logo.src = "heathen-machines-logo.png";

// presets + mode (the mode segmented control is built LAST into this slot so its initial
// dimming pass can see every section; the slot keeps it in the right visual position)
presetBar(wrap);
const modeSlot = mkEl("div", null, wrap);

// shift knob
const knobSec = mkEl("div", "sec", wrap);
const knobWrap = mkEl("div", "knobwrap", knobSec);
shiftKnob(knobWrap);
const warmRow = mkEl("div", "subrow", knobWrap); warmRow.style.justifyContent = "center";
switchEl(warmRow, "warm", "WARM");

// spectral  (Visage: Quantize/Envelope/Transients 1dp no suffix, Sens/Texture/Density int, Smear 1dp ms)
const specSec = section("SPECTRAL CONTROLS");
pianoEl(specSec);
sliderRow(specSec, "quantizeStrength", "Quantize", { decimals: 1 });
sliderRow(specSec, "preserve", "Envelope", { decimals: 1 });
sliderRow(specSec, "transients", "Transients", { decimals: 1 });
sliderRow(specSec, "sensitivity", "Sens", { decimals: 0 });
switchEl(mkEl("div", "subrow", specSec), "peakSnap", "Tones Only");
const texRow = sliderRow(specSec, "noiseMix", "Texture", { decimals: 0 });
const denRow = sliderRow(specSec, "peakSens", "Density", { decimals: 0 });
sliderRow(specSec, "smear", "Smear", { decimals: 1, suffix: " ms" });
mkEl("div", "tip", specSec, "Adjust with care while playing");
wireEnableDim("peakSnap", [texRow, denRow]);   // Texture/Density active only when Tones Only is on

// freq modulation  (lfoRate: tempo divisions when Sync on)
const fmSec = section("FREQ MODULATION", "lfoEnabled");
const fmDepth = sliderRow(fmSec, "lfoDepth", "Depth", { adaptive: true });
const fmRate = sliderRow(fmSec, "lfoRate", "Rate", { adaptive: true, suffix: " Hz",
  sync: { toggleId: "lfoSync", divisionId: "lfoDivision", reversed: true } });
const fmSub = mkEl("div", "subrow", fmSec);
switchEl(fmSub, "lfoSync", "Sync"); comboEl(fmSub, "lfoShape");
wireEnableDim("lfoEnabled", [fmDepth, fmRate, fmSub]);

// delay  (delayTime: ms, seconds above 1000; tempo divisions when Sync on)
const dlySec = section("DELAY", "delayEnabled");
sliderRow(dlySec, "delayTime", "Time", { decimals: 1, suffix: " ms", secondsAbove: 1000,
  sync: { toggleId: "delaySync", divisionId: "delayDivision", reversed: true } });
sliderRow(dlySec, "delayFeedback", "Feedback", { decimals: 1 });
sliderRow(dlySec, "delayDamping", "Damping", { decimals: 1 });
const slopeRow = sliderRow(dlySec, "delaySlope", "Slope", { decimals: 1 });
const dlySub = mkEl("div", "subrow", dlySec);
switchEl(dlySub, "delaySync", "Sync"); decorrelateToggle(dlySub);

// delay modulation  (dlyLfoRate: tempo divisions when Sync on)
const dmSec = section("DELAY MODULATION", "dlyLfoEnabled");
const dmDepth = sliderRow(dmSec, "dlyLfoDepth", "Depth", { adaptive: true, suffix: " ms" });
const dmRate = sliderRow(dmSec, "dlyLfoRate", "Rate", { adaptive: true, suffix: " Hz",
  sync: { toggleId: "dlyLfoSync", divisionId: "dlyLfoDivision", reversed: true } });
const dmSub = mkEl("div", "subrow", dmSec);
switchEl(dmSub, "dlyLfoSync", "Sync"); comboEl(dmSub, "dlyLfoShape");
wireEnableDim("dlyLfoEnabled", [dmDepth, dmRate, dmSub]);

// mask  (Transition 2dp oct, Low/High integer Hz)
const maskSec = section("MASK", "maskEnabled");
const maskSub = mkEl("div", "subrow", maskSec); comboEl(maskSub, "maskMode").classList.add("tan");
sliderRow(maskSec, "maskTransition", "Transition", { decimals: 2, suffix: " oct" });
sliderRow(maskSec, "maskLowFreq", "Low", { decimals: 0 });
sliderRow(maskSec, "maskHighFreq", "High", { decimals: 0 });

// mix
const mixSec = section("MIX");
sliderRow(mixSec, "dryWet", "Dry / Wet", { decimals: 1 });

// ---- mode dimming (spectral-only controls greyed in Classic) -------------
function updateMode(idx) {
  const classic = (idx === 0);
  specSec.classList.toggle("dimmed", classic);
  maskSec.classList.toggle("dimmed", classic);
  slopeRow.classList.toggle("dimmed", classic);
}

// built last so the initial dimming pass sees every section defined above
segmentedEl(modeSlot, "processingMode", ["CLASSIC", "SPECTRAL"], updateMode);

nlog("UI assembled OK; sections =", wrap.querySelectorAll(".sec").length, "rows =", wrap.querySelectorAll(".row").length);
