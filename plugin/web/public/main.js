// Holy Shifter WebView UI — data-driven from window.HOLY_PARAMS, bound to the
// native APVTS via JUCE's juce.js relays. Responsive: the panel grid (style.css)
// reflows on resize; controls read their range/choices/label live from the relay
// state's `properties`, so no values are hard-coded here.
import * as Juce from "./juce.js";

const NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
const BLACK = new Set([1, 3, 6, 8, 10]);
const params = window.HOLY_PARAMS || [];

const $ = (tag, cls, txt) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
};

// Note: getScaledValue() maps through start/end/skew only. Most Holy Shifter curved
// params use lambda NormalisableRanges (skew left at 1), so this readout is exact for
// linear params and approximate for curved ones — the value WRITTEN is always correct
// (setNormalisedValue passes the normalised value through the param's real curve in C++).
function fmtValue(state) {
  const p = state.properties || {};
  let v = state.getScaledValue();
  if (!isFinite(v)) v = 0;
  const a = Math.abs(v);
  let s;
  if (a >= 100) s = v.toFixed(0);
  else if (a >= 1) s = v.toFixed(1);
  else if (a > 0 && a < 0.1) s = v.toFixed(3);
  else s = v.toFixed(2);
  return s + (p.label ? " " + p.label : "");
}

function makeSlider(p) {
  const wrap = $("div", "ctl ctl-slider");
  const head = $("div", "ctl-head");
  const valEl = $("span", "ctl-val", "");
  head.append($("span", "ctl-label", p.name || p.id), valEl);
  const input = $("input", "slider");
  input.type = "range"; input.min = "0"; input.max = "1"; input.step = "0.0005";
  wrap.append(head, input);

  const state = Juce.getSliderState(p.id);
  let dragging = false;
  const refresh = () => { if (!dragging) input.value = String(state.getNormalisedValue()); valEl.textContent = fmtValue(state); };
  input.addEventListener("input", () => { state.setNormalisedValue(parseFloat(input.value)); valEl.textContent = fmtValue(state); });
  input.addEventListener("pointerdown", () => { dragging = true; state.sliderDragStarted(); });
  input.addEventListener("pointerup", () => { dragging = false; state.sliderDragEnded(); refresh(); });
  input.addEventListener("pointercancel", () => { dragging = false; state.sliderDragEnded(); refresh(); });
  state.valueChangedEvent.addListener(refresh);
  state.propertiesChangedEvent.addListener(refresh);
  refresh();
  return wrap;
}

function makeCombo(p) {
  const wrap = $("div", "ctl ctl-combo");
  wrap.append($("span", "ctl-label", p.name || p.id));
  const sel = $("select", "select");
  wrap.append(sel);
  const state = Juce.getComboBoxState(p.id);
  const populate = () => {
    const ch = (state.properties && state.properties.choices) || [];
    sel.innerHTML = "";
    ch.forEach((c, i) => { const o = $("option", null, c); o.value = String(i); sel.append(o); });
    sel.value = String(state.getChoiceIndex());
  };
  sel.addEventListener("change", () => state.setChoiceIndex(parseInt(sel.value, 10)));
  state.propertiesChangedEvent.addListener(populate);
  state.valueChangedEvent.addListener(() => { sel.value = String(state.getChoiceIndex()); });
  populate();
  return wrap;
}

function makeToggle(p) {
  const btn = $("button", "ctl ctl-toggle", p.name || p.id);
  const state = Juce.getToggleState(p.id);
  const refresh = () => btn.classList.toggle("on", state.getValue());
  btn.addEventListener("click", () => state.setValue(!state.getValue()));
  state.valueChangedEvent.addListener(refresh);
  refresh();
  return btn;
}

function makePiano(items) {
  const wrap = $("div", "piano");
  const sorted = items.slice().sort((a, b) => parseInt(a.id.slice(9), 10) - parseInt(b.id.slice(9), 10));
  for (const p of sorted) {
    const idx = parseInt(p.id.slice(9), 10); // strip "scaleNote"
    const key = $("button", "key " + (BLACK.has(idx) ? "black" : "white"), NOTE_NAMES[idx]);
    const state = Juce.getToggleState(p.id);
    const refresh = () => key.classList.toggle("on", state.getValue());
    key.addEventListener("click", () => state.setValue(!state.getValue()));
    state.valueChangedEvent.addListener(refresh);
    refresh();
    wrap.append(key);
  }
  return wrap;
}

async function buildPresetBar(host) {
  const bar = $("div", "presetbar");
  const sel = $("select", "preset-select");
  const name = $("input", "preset-name"); name.type = "text"; name.placeholder = "preset name…";
  const saveBtn = $("button", "pbtn", "Save");
  const delBtn = $("button", "pbtn", "Delete");
  bar.append($("span", "preset-lbl", "Preset"), sel, name, saveBtn, delBtn);
  host.append(bar);

  const fn = (n) => Juce.getNativeFunction(n);
  const list = fn("presetList"), cur = fn("presetCurrent"), load = fn("presetLoad"),
        save = fn("presetSave"), del = fn("presetDelete");

  async function refresh() {
    let names = [];
    try { names = JSON.parse(await list()); } catch (e) { names = []; }
    let current = ""; try { current = await cur(); } catch (e) {}
    sel.innerHTML = "";
    names.forEach((n) => { const o = $("option", null, n); o.value = n; sel.append(o); });
    if (current) sel.value = current;
    name.value = current || "";
  }
  sel.addEventListener("change", async () => { await load(sel.value); name.value = sel.value; });
  saveBtn.addEventListener("click", async () => {
    const n = (name.value || sel.value || "Untitled").trim();
    if (!n) return; await save(n); await refresh(); sel.value = n; name.value = n;
  });
  delBtn.addEventListener("click", async () => { if (sel.value) { await del(sel.value); await refresh(); } });
  await refresh();
}

function build() {
  const app = document.getElementById("app");
  app.innerHTML = "";

  const header = $("header", "app-header");
  header.append($("div", "brand", "HOLY SHIFTER"), $("div", "vendor", "Heathen Machines"));
  app.append(header);

  buildPresetBar(app).catch(() => {});

  const order = [], bySec = {};
  for (const p of params) {
    if (!bySec[p.section]) { bySec[p.section] = []; order.push(p.section); }
    bySec[p.section].push(p);
  }

  const grid = $("div", "panels");
  for (const sec of order) {
    const panel = $("section", "panel");
    panel.append($("h2", "panel-title", sec));
    const body = $("div", "panel-body");
    if (sec === "Scale") {
      body.append(makePiano(bySec[sec]));
    } else {
      for (const p of bySec[sec]) {
        body.append(p.type === "float" ? makeSlider(p) : p.type === "choice" ? makeCombo(p) : makeToggle(p));
      }
    }
    panel.append(body);
    grid.append(panel);
  }
  app.append(grid);
}

// Module scripts are deferred, so the DOM is parsed and params.js has run by now.
build();
