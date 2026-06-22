# Engineering Plan: Port "Holy Shifter" to a Resizable, Responsive JUCE WebView UI

> Modeled on the Parallax plugin's JUCE 8 `WebBrowserComponent` architecture. The primary goal is a **cool, responsive UI the user can freely resize** — replacing today's fixed 700×928 non-resizable Visage GPU UI.

---

## 1. Executive Summary + Verdict

**Verdict: Feasible and recommended.** JUCE 8's `WebBrowserComponent` is a production path for a single responsive HTML/CSS/JS UI that runs natively on macOS (WKWebView), iOS (WKWebView), Windows (WebView2), and Linux (WebKitGTK). Parallax — a sibling plugin in this same workspace — implements most of the pattern end to end (native relays + resource provider + embedded vanilla web bundle, no bundler), so we have a **working, copyable macOS/WKWebView reference in-tree** rather than a greenfield design. **Two important caveats up front, both confirmed by inspecting Parallax:** (a) Parallax has **zero Windows/WebView2 handling** — no `NEEDS_WEBVIEW2`, no `Backend::webview2`, no `JUCE_USE_WIN_WEBVIEW2` anywhere — so the *entire Windows story is net-new and unproven in this workspace*; and (b) Parallax's CSS is **fixed-px throughout** (no `clamp()`, no `vmin`, no `@container`, no `transform: scale`, no `aspect-ratio`), so its UI "resizes" only by reflow/compression at fixed metrics. The genuinely responsive layout we want is **net-new development**, not reference reuse. We call these out explicitly wherever they touch a task estimate.

- **Resize is the headline win and it is directly solvable.** Today `VisageHostEditor` hardcodes `kBaseWidth = 700, kBaseHeight = 928` and the editor is non-resizable. The webview path makes the editor resizable (`setResizable(true, …)` + `setResizeLimits(...)`), stretches the webview to fill bounds in `resized()`, and lets CSS reflow/scale the layout. Parallax does the native half of this (`setResizable(true, true)`, `setResizeLimits(720, 480, 2400, 1600)`, `webView.setBounds(getLocalBounds())`); the responsive *CSS* half is ours to build.
- **Rough effort: ~5–7 focused weeks** for full desktop parity (50 params + 12 piano bools + presets + all the non-trivial control behaviours), including cross-platform/host QA and a Windows bring-up that has **no in-tree precedent**. A de-risking spike (Section 10) gets a resizable webview talking to ~3 real params in ~3–4 days **and must actually build-and-run on Windows**, not assume it from Parallax.
- **Biggest single risk: faithfully reproducing the non-trivial control behaviours** — especially the **curved sliders' write path** (see below), the shift knob's bespoke symmetric-log mapping (visual ±5000 Hz over a ±20000 Hz param), the three dual-purpose sync sliders with **reversed division labels**, adaptive decimals, the delay-time seconds switch, and the 12-key piano multi-toggle. These are where "looks done but subtly wrong" bugs live.
- **A specific, confirmed correctness trap: curved sliders.** Many params (`quantizeStrength` pow²; `lfoDepth`/`dlyLfoDepth` pow^5.3; `lfoRate`/`dlyLfoRate`, `maskLowFreq`/`maskHighFreq`, `delayTime` log) define their curve via **lambda `NormalisableRange` converters with skew left at 1** (verified in `PluginProcessor.cpp`: e.g. `quantizeRange` uses `std::pow(normalised, 2.0f)`, `lfoDepthRange` uses `std::pow(normalised, lfoDepthExp)`). The JUCE JS bridge and parameter attachments only transport **skew**, not the lambdas, so they treat these as **linear** (skew=1). The naïve "invert the skew in JS" approach is wrong twice over and will silently corrupt values on write. Section 5.3 specifies the correct write path. `shiftHz` is genuinely **linear** (`NormalisableRange(-20000, 20000, 0.1f)`, no lambda), so the knob's value transport is fine; only its *visual* mapping is custom.
- **Secondary risks:** **Windows WebView2 build SDK** (NuGet acquisition — the configure step hard-fails without it), the WebView2 **end-user runtime**, keyboard focus in webviews (preset-name field; `EDITOR_WANTS_KEYBOARD_FOCUS` is currently `FALSE`), CSP correctness under the custom `juce://` scheme (no in-tree precedent), and **editor-construction timeouts in validation/CI** (this repo's CI already records pluginval/pedalboard hangs at plugin load with the *current* Visage editor).
- **Strategic note: this is plausibly also the iOS UI — but unproven.** Per the iOS migration notes, that effort is blocked because Visage has no iOS windowing, and WKWebView *is* the OS-native view on iOS (touch for free). That makes a webview UI a strong candidate to sidestep the blocker. However, the relay bridge + ResourceProvider running inside an **AUv3 app-extension sandbox on iOS** is not demonstrated anywhere in this workspace, and resize inside an AUv3 host view is unverified. Treat iOS as **plausible, deferred, and requiring its own spike** (Section 8.3) — not a guaranteed near-free win.

**Recommendation:** Build behind a `HOLY_SHIFTER_USE_WEBVIEW` CMake flag (mirroring Parallax's `PARALLAX_USE_WEBVIEW`) so Visage and WebView coexist during the port, then flip the default and retire Visage once parity + QA (including a pluginval re-run) pass.

---

## 2. Reference Architecture — How Parallax Does It (the pattern to replicate)

Parallax is the template **for the macOS/WKWebView core**. Replicate these concrete mechanisms; do **not** assume it covers Windows, responsive CSS, dev-server live-reload, CSP, or self-hosted fonts (it covers none of those — see the explicit corrections below).

### 2.1 Compile gating
- The entire WebView editor TU is wrapped in `#if PARALLAX_USE_WEBVIEW` (`PluginEditor.h:6`, `PluginEditor.cpp:3`). When off, `createEditor()` returns a `GenericAudioProcessorEditor`. CMake drives **both** `JUCE_WEB_BROWSER` and `PARALLAX_USE_WEBVIEW` from one `PARALLAX_WEBVIEW_VALUE` (`CMakeLists.txt:35-46`), and links `juce::juce_gui_extra` only in the WebView branch (`CMakeLists.txt:60`).
- **Holy Shifter addition (not in Parallax):** also guard the editor with `#if JUCE_WEB_BROWSER_RESOURCE_PROVIDER_AVAILABLE` so that a *misconfigured* Windows build (missing `JUCE_USE_WIN_WEBVIEW2`) fails loudly at compile time instead of silently shipping a blank UI (see Section 4 and 7.2).

### 2.2 WebBrowserComponent construction
- Built inline in the member-init list (`PluginEditor.cpp:167`) with `Options{}`:
  - `.withNativeIntegrationEnabled()` — injects the `window.__JUCE__` bridge.
  - `.withOptionsFrom(<relay>)` — one call per APVTS parameter relay.
  - `.withNativeFunction("name", lambda)` — for non-parameter state.
  - `.withResourceProvider([this](auto& url){ return getResource(url); })` — last in the chain.
- Parallax uses the **default backend** (WKWebView on macOS); no Windows options, no page-load-error handling. **This is a gap, not a pattern to copy.** For Holy Shifter we must add `.withBackend(Backend::webview2)` and `.withWinWebView2Options(...)` on Windows. The earlier framing that "Parallax leans on JUCE's auto-acquisition" of WebView2 is **incorrect** — there is no auto-acquisition of the build SDK, and Parallax simply has no Windows path at all.

### 2.3 Serving the UI (ResourceProvider, no localhost in release)
- `addAndMakeVisible(webView)` then `webView.goToURL(WebBrowserComponent::getResourceProviderRoot())` (`PluginEditor.cpp:440-441`) — navigates the in-memory `juce://` scheme, not http/file. The custom scheme registered on macOS is literally `juce` (`juce_WebBrowserComponent_mac.mm`: `setURLSchemeHandler:…forURLScheme:@"juce"`); content loads from a `juce://…` origin.
- `getResource(url)` (`PluginEditor.cpp:459-473`): strips query, maps `/`→`index.html`, mangles the filename (`.` and `-` → `_`) to match `juce_add_binary_data`, looks up `BinaryData::getNamedResource`, returns `{bytes, mime}` or `std::nullopt`.
- **Load-bearing gotcha to carry over:** MIME must be derived from the *resolved filename*, not the raw URL — `/` has no extension and would be served as octet-stream, aborting the WKWebView frame load (`PluginEditor.cpp:12-16`).
- **Windows note:** on Windows, `getResource()`/`getResourceProviderRoot()` only exist when `JUCE_WEB_BROWSER_RESOURCE_PROVIDER_AVAILABLE` is 1, which on Windows requires `JUCE_USE_WIN_WEBVIEW2` (verified `juce_WebBrowserComponent.h:39-43`). Without it the ResourceProvider compiles out and the editor loads nothing — hence the compile-time guard in 2.1.

### 2.4 Parameter bridge
- Strict **1 relay + 1 attachment per APVTS param**, relay name == param id. `WebSliderRelay` (continuous), `WebComboBoxRelay` (choice), `WebToggleButtonRelay` (bool), each constructed **name-only** (`WebControlRelays.h:73 WebSliderRelay(StringRef)` — so any stale `{browser, name}` two-arg snippets are wrong). Bound via `Web*ParameterAttachment(*apvts.getParameter("id"), relay, nullptr)`.
- **Declaration/destruction order matters:** relays first, WebBrowserComponent second, attachments last → on teardown attachments die first, browser second, relays last. Getting this wrong crashes on editor close.

### 2.5 Non-parameter state & push
- `withNativeFunction` for presets (`presetList/presetSave/presetDelete`), host/transport polling (`hostState` ~20 Hz), utility (`panic/resetAll`), and a UI preset name (`uiPresetGet/Set`). Each lambda **must call `complete(...)` exactly once**; long ops run on a background thread and marshal back via `MessageManager::callAsync` guarded by `Component::SafePointer`.
- One native→JS push channel via `emitEventIfBrowserIsVisible("agentStream", payload)` (Holy Shifter only needs the equivalent if we later add streaming/meters; it currently has none).

### 2.6 Web bundle (no build step)
- Vanilla ES-module SPA in `web/public/`: `index.html`, `main.js`, `style.css`, `juce.js` (official bridge), `check_native_interop.js` (mock `window.__JUCE__` for standalone browser testing). **No bundler/minify/transpile.**
- CMake: `file(GLOB ... CONFIGURE_DEPENDS "web/public/*")` → `juce_add_binary_data(ParallaxWebAssets ...)` → link. The same files run standalone via `python3 -m http.server` for Playwright tests.
- **Do NOT mirror Parallax's fonts.** Parallax's `index.html:7-9` fetches fonts from `fonts.googleapis.com`/`fonts.gstatic.com` via `<link>`. That is a remote network fetch from inside a plugin — an ATS/CSP/hardened-runtime, offline-failure, and privacy/telemetry surface. We self-host instead (Section 5.1/7.3).

### 2.7 Resize plumbing — and what Parallax does *not* do
- Native half (copyable): `setResizable(true, true)`, `setResizeLimits(720, 480, 2400, 1600)`, `setSize(1140, 720)`; `resized()` does `webView.setBounds(getLocalBounds())`. **No native→JS size message.**
- **Correction:** Parallax's CSS is **not** "fully responsive to window size." Its `style.css` uses ~77 fixed-px gap/padding values, fixed grid tracks (`150px 1fr 280px`), fixed font sizes (`8.5px`/`13px`), and contains **zero** `clamp()`, `vmin/vmax`, `@container`, `transform: scale`, or `aspect-ratio`. It resizes purely by reflow/compression at fixed metrics. Parallax also does **not** use `setFixedAspectRatio`. So both our aspect-lock approach and our genuinely-fluid CSS are net-new (Section 6), and must be prototyped in the spike rather than assumed.

---

## 3. Target Surface — Holy Shifter's Full Parameter + Control Inventory

The web UI must reproduce every working-tree control. **Bind everything by string id** (not index) — the v0.2.3 PeakSnap branch inserts params before `processingMode`, shifting indices. APVTS accessor is `processor.getValueTreeState()` (confirmed `PluginProcessor.h:57`). Preset/decorrelate calls go through `processor.getPresetManager()` and `processor.setStereoDecorrelate(...)` / `processor.getStereoDecorrelate()`.

**Confirmed parameter total: 50** (verified against `createParameterLayout`): **19 `AudioParameterFloat`, 21 `AudioParameterBool` (9 standalone + the 12 `scaleNote0..11` piano bools built in a loop), 10 `AudioParameterChoice`.** Earlier "~42" counts were wrong; use **50** throughout.

### 3.1 Parameters → relay type mapping

| Relay type | APVTS param ids |
|---|---|
| **WebSliderRelay** (continuous) | `shiftHz`, `quantizeStrength`, `dryWet`, `smear`, `lfoDepth`, `lfoRate`, `dlyLfoDepth`, `dlyLfoRate`, `maskLowFreq`, `maskHighFreq`, `maskTransition`, `delayTime`, `delaySlope`, `delayFeedback`, `delayDamping`, `delayGain`*, `preserve`, `transients`, `sensitivity` |
| **WebToggleButtonRelay** (bool) | `scaleNote0..11` (×12, piano), `lfoSync`, `lfoEnabled`, `dlyLfoSync`, `dlyLfoEnabled`, `maskEnabled`, `delayEnabled`, `delaySync`, `warm`, `logScale`* |
| **WebComboBoxRelay** (choice) | `lfoDepthMode`* (hidden, zero-sized in Visage), `lfoDivision`, `lfoShape`, `dlyLfoDivision`, `dlyLfoShape`, `maskMode`, `delayDivision`, `processingMode`, `rootNote`* (deprecated), `scaleType`* (deprecated) |

`*` = registered for state compatibility but **not user-visible** (`delayGain`, `logScale`, `lfoDepthMode`, deprecated `rootNote`/`scaleType`). **Binding decision (corrected):** APVTS persists *every* registered parameter via `get/setStateInformation` **regardless of any editor binding** — the editor is transient, so a relay+attachment is **not required for state round-trip**. Therefore **do not** add relays/attachments for purely-hidden/deprecated params the web UI never reads or writes (`logScale`, `delayGain`, `rootNote`, `scaleType`). The one exception is **`lfoDepthMode`**, which in Visage is *added then zero-sized* (`setBounds(0,0,0,0)`, `HolyShifterUI.cpp:719`) rather than absent; if the web UI ever needs to read it (e.g. to choose Hz vs Deg readout), bind it and render it **present-but-hidden** (`display:none` + a `data-testid` mirror). Dropping the four truly-unused bindings removes pure lifetime-ordering surface for zero state benefit.

### 3.2 Control types the web UI must implement

1. **Rotary knob** (`shiftHz`) — vertical-delta drag, Shift = 10× fine, double-click reset, bipolar arc from top, custom symmetric-log *visual* mapping (visual ±5000 Hz / param ±20000), big `%d`≥100 else `%.1f` readout + "HZ". The param itself is **linear**, so value transport is the plain normalised round-trip; only the visual mapping is bespoke.
2. **Continuous sliders** — `quantizeStrength` (pow² curve), `preserve` ("Envelope"), `transients`, `sensitivity` ("Sens", 0 dp), `smear` (" ms"), `delayFeedback`, `delayDamping`, `delaySlope` (dimmed in Classic), `maskLowFreq`/`maskHighFreq` (log, 0 dp), `maskTransition` (" oct", 2 dp), `dryWet` (mix footer), plus the depth sliders `lfoDepth`/`dlyLfoDepth` (pow^5.3 curve, adaptive decimals). **All curve shapes live in lambda converters with skew=1 — see the write-path rule in 5.3.**
3. **Dual-purpose sync sliders** (×3: `lfoRate`/`lfoDivision`, `dlyLfoRate`/`dlyLfoDivision`, `delayTime`/`delayDivision`) — Sync toggle swaps the active param from float→Choice; synced = discrete-snapped to 14 positions with tick marks; **reversed labels** (16/1 left, 1/32 right) where fill = `1 - paramNorm` but the readout shows the true value; `delayTime` switches to "x.xx s" above 1000 ms.
4. **Toggles** — sync toggles (`lfoSync`/`dlyLfoSync`/`delaySync`); the LFO enable toggles (`lfoEnabled`/`dlyLfoEnabled`) which dim their *own* sub-controls (see dimming note below); `maskEnabled`/`delayEnabled` (state only — they do **not** drive dimming in the current UI); `warm`; and **`L/R Decorr`** which is NOT an APVTS param (calls `setStereoDecorrelate` directly via native function, not saved in state).
5. **Combos** — `lfoShape`/`dlyLfoShape` (6 shapes), `maskMode` (3), `lfoDepthMode` (hidden Hz/Deg).
6. **Segmented control** — `processingMode` (CLASSIC/SPECTRAL); drives mode dimming + crossfade.
7. **Piano keyboard** — 12 white/black keys bound to `scaleNote0..11` bools; **must reproduce the exact pitch-class mapping and geometry** from `HolyPianoKeyboard.cpp` (`whiteKeyPC_`/`blackKeyPC_`/`blackAfterWhite_`, black-key-first hit-test) so visual key *i* toggles the correct `scaleNote{pc}` — off-by-one is a real risk; multi-select; dimmed/non-interactive in Classic mode.
8. **Preset strip + modal** — prev/next, click-to-open dropdown (current + hover highlight), Save modal (text field, maxLen 40, full keyboard handling, factory→" Custom" suffix), Delete confirm modal (disabled for factory/empty).
9. **Mode/enable-driven dimming (corrected to match live UI):**
   - **Classic mode** (`updateControlsForMode`, `HolyShifterUI.cpp:456-475`) dims the spectral panel (piano, `quantizeStrength`, `preserve`, `transients`, `sensitivity`, `smear`, `lfoDepthMode`), **the entire Mask section** (`maskEnabled`, `maskMode`, `maskTransition`, `maskLowFreq`, `maskHighFreq`), and `delaySlope`.
   - **LFO enables** (`updateLfoEnableUI`, `HolyShifterUI.cpp:420-434`) dim only *their own* depth/rate/shape/sync sub-controls — `lfoEnabled` → {`lfoDepth`,`lfoRate`,`lfoShape`,`lfoSync`}; `dlyLfoEnabled` → {`dlyLfoDepth`,`dlyLfoRate`,`dlyLfoShape`,`dlyLfoSync`}.
   - **`maskEnabled` and `delayEnabled` do NOT drive any dimming** in the current UI (mask dims via Classic mode only; `delayEnabled` dims nothing). Decision to surface to the user (Section 11): match this exactly, **or** add delay/mask enable-dimming as a deliberate, documented improvement.
10. **State polling** — equivalent of `pollState()` (`HolyShifterUI.cpp`): re-read preset name, mode, sync/enable states on automation/preset load (handled by relay `valueChangedEvent` + a host→UI poll for non-param state).
11. **Artwork** — `heathen-machines-logo.png` (99 KB) top-right, `pagan-background.png` (3.5 MB!) full-window backdrop + scrim, title "H O L Y SHIFTER", gold/amber accent theme.

> **Note on the 3.5 MB background:** Parallax already bakes a 706 KB JPG into every plugin copy and warns about it. At 3.5 MB, re-export `pagan-background.png` to a compressed/resized WebP/JPG (or CSS gradient + smaller texture overlay) before embedding — it lands in every plugin instance.

---

## 4. Native Architecture for Holy Shifter's WebView Editor

New file pair `src/WebViewEditor.{h,cpp}` (gated `#if HOLY_SHIFTER_USE_WEBVIEW`, and additionally `#if JUCE_WEB_BROWSER_RESOURCE_PROVIDER_AVAILABLE` around the ResourceProvider-dependent body so Windows misconfiguration fails at compile time), replacing `VisageHostEditor` as the `createEditor()` return when the flag is on.

### 4.1 createEditor wiring
`PluginProcessor.cpp:2173-2175` currently `return new VisageHostEditor(*this);`. Change to:
```cpp
juce::AudioProcessorEditor* FrequencyShifterProcessor::createEditor() {
#if HOLY_SHIFTER_USE_WEBVIEW
    return new WebViewEditor(*this);
#else
    return new VisageHostEditor(*this);
#endif
}
```

### 4.2 WebBrowserComponent setup (member-init list)
```cpp
webView(juce::WebBrowserComponent::Options{}
    .withNativeIntegrationEnabled()
#if JUCE_WINDOWS
    .withBackend(juce::WebBrowserComponent::Backend::webview2)
    .withWinWebView2Options(juce::WebBrowserComponent::Options::WinWebView2{}
        // Stable, per-plugin folder — NOT tempDirectory (clearable/locked/shared).
        .withUserDataFolder(juce::File::getSpecialLocation(
            juce::File::userApplicationDataDirectory)
            .getChildFile("HolyShifter").getChildFile("WebView2")))
#endif
    .withOptionsFrom(shiftHzRelay)
    // ... one .withOptionsFrom per *visible/bound* relay (~46 calls; see 4.3) ...
    .withNativeFunction("presetList",     [this](auto& a, auto c){ /* ... */ })
    .withNativeFunction("presetSave",     /* ... */)
    .withNativeFunction("presetDelete",   /* ... */)
    .withNativeFunction("presetLoad",     /* ... */)
    .withNativeFunction("presetPrev",     /* ... */)
    .withNativeFunction("presetNext",     /* ... */)
    .withNativeFunction("presetName",     /* ... */)   // current name + isFactory
    .withNativeFunction("getDecorrelate", /* ... */)   // hydrate L/R Decorr on init
    .withNativeFunction("setDecorrelate", /* ... */)   // non-APVTS L/R Decorr write
    .withNativeFunction("resizeTo",       /* ... */)   // JS resize handle → setSize (Section 6)
    .withResourceProvider([this](auto& url){ return getResource(url); }))
```

### 4.3 Relays + attachments (declaration order)
Declare **relays first, webView second, attachments last** (per the destruction-order rule). With ~46 bound params, this ordering is real and load-bearing — getting it wrong crashes on close.

**Generation correctness (corrected):** relays and attachments are **non-movable JUCE members tied to construction order**, so you cannot loop over a `const char*[]` table to populate a member-init list, and a struct table cannot declare differently-typed members in a fixed order. The cleanest correct approach is the **explicit named-member pattern Parallax uses** (verbose, but the order guarantee is automatic from declaration order). A `static` table may *only* drive the repetitive `.withOptionsFrom(...)` registration and attachment-construction *call sites* — it does **not** remove the requirement to hand-declare each relay/webView/attachment member in the header in strict order. If a table-driven design is truly wanted, hold relays/attachments in `std::vector<std::unique_ptr<...>>` built in the ctor body, but then you must ensure the webView still sees the relays before its own construction (`withOptionsFrom` needs them first) — which is awkward; prefer the explicit pattern.

Pattern per param: `WebSliderRelay shiftHzRelay{"shiftHz"};` … `WebSliderParameterAttachment shiftHzAtt{*apvts.getParameter("shiftHz"), shiftHzRelay, nullptr};` with `apvts = processor.getValueTreeState()`.

**Do not** create relays/attachments for the purely-hidden/deprecated params `logScale`, `delayGain`, `rootNote`, `scaleType` — APVTS persists them with no editor binding (Section 3.1). Bind `lfoDepthMode` only if the web UI reads it.

### 4.4 Piano keyboard (12 bool relays)
No special native machinery — it's just 12 `WebToggleButtonRelay`s (`scaleNote0..11`). The "keyboard" is purely a JS DOM/SVG widget that toggles those 12 toggle states; multi-select and Classic-mode dimming are JS concerns. The **pitch-class-to-key geometry must be ported verbatim** from `HolyPianoKeyboard.cpp:19-64` (see 3.2 §7 and 5.3).

### 4.5 Presets (native functions, mirror Parallax's preset bridge)
Holy Shifter already has a `PresetManager` (confirmed API: `getAllPresetNames`, `getCurrentPresetName`, `isFactoryPreset`, `loadPreset`, `savePreset`, `deletePreset`, `loadNextPreset`, `loadPreviousPreset`). Expose via native functions returning JSON / acks, each calling `complete(...)` once. Preset *recall* changes params on the C++ side, which the relays push back to JS automatically via `valueChangedEvent` — no manual re-push of each param needed. **Window size must NOT be reapplied on preset recall** (Section 6.5).

### 4.6 L/R Decorrelate (special-case, with getter)
Not an APVTS param and not saved in state (`PluginProcessor.cpp:2178`). The setter `setStereoDecorrelate(bool)` and **getter `getStereoDecorrelate()` both exist** (`PluginProcessor.h:133-134`). Bridge via `.withNativeFunction("setDecorrelate", …)` and **`.withNativeFunction("getDecorrelate", …)`**; on init the JS toggle must **hydrate from `getDecorrelate`** so it doesn't desync on editor reopen. The JS toggle is local-only (no relay).

### 4.7 ResourceProvider
Copy Parallax's `getResource` verbatim, including the filename-mangling and **MIME-from-resolved-filename** rule. **Extend `mimeForExtension` for every embedded asset type**, including the chosen **self-hosted font extension** (`.woff2` or `.ttf`) and `.webp` — otherwise fonts/images load as `application/octet-stream` and silently fail.

---

## 5. Web UI Architecture

Vanilla ES-module SPA, no bundler (mirror Parallax's *build approach*). Single page (Holy Shifter is one screen, unlike Parallax's 8 pages) — simpler than the reference.

### 5.1 File/module structure (`plugin/web/public/`)
- `index.html` — static markup: header (logo, title), mode segmented control, the spectral panel (knob + piano + sliders), the LFO/Delay/Mask sections, mix footer, preset strip, hidden preset modal. Mount `<div id="...">`s filled by JS. Include hidden mirror nodes (`data-testid`) for Playwright assertions, like Parallax. **No external `<link>` to Google Fonts.**
- `style.css` — theme tokens from `HolyTheme.h` as CSS custom properties (see 5.7), responsive layout (Section 6).
- `juce.js` — official JUCE 8 bridge (copy unmodified).
- `check_native_interop.js` — mock `window.__JUCE__` for standalone browser dev/test.
- `main.js` — param metadata table (incl. the **per-param curve table**, 5.3), widget factories, presets, piano, dimming logic, init + a `window.__hs` test command API.
- `assets/` — compressed `pagan-background` + `heathen-machines-logo`, **self-hosted fonts** (Inter + IBM Plex Mono — same families as Visage) embedded as `@font-face` from the existing `plugin/fonts/*.ttf` (TTF works directly; converting to `.woff2` is an optional size win). **Never fetch remote fonts** — offline/hardened-runtime hosts (and any CSP) would block them and fall back to system fonts.

### 5.2 Binding to native (juce.js)
- Sliders/knobs: `Juce.getSliderState(id)` → `getScaledValue/getNormalisedValue/setNormalisedValue`, `sliderDragStarted/sliderDragEnded`, `properties{start,end,skew,interval}`, `valueChangedEvent`/`propertiesChangedEvent`. **Improvement over Parallax:** *do* call `sliderDragStarted/Ended` (Parallax omits them, losing host gesture boundaries) so DAW automation records clean gestures — matching the Visage `beginGesture()/endGesture()` behaviour (confirmed in `HolyRotaryKnob::mouseDown/mouseUp`).
- Toggles: `Juce.getToggleState(id)` → `getValue/setValue`.
- Combos: `Juce.getComboBoxState(id)` → `getChoiceIndex/setChoiceIndex`, populate options from `properties.choices` (authoritative from C++) with JS lists only as fallback.
- Native fns: `Juce.getNativeFunction(name)` (Promise), guarded by presence in `__JUCE__.initialisationData.__juce__functions`.

### 5.3 Control implementations

- **Curved sliders — the correct read/write contract (critical).** The bridge transports only **skew** (which is 1 for every curved param), so it treats these params as **linear**. Do **not** apply the param's curve when *writing* and do **not** use the (dead) "if skew≠1 pow(n,skew)" branch — that double-applies or no-ops. Instead:
  - **Write:** convert the real-unit value to the **linear-inverse** normalised position `n = (V − start) / (end − start)` and pass that to `setNormalisedValue`. The C++ side re-applies the real curve via its `convertTo0to1` lambda, storing the correct value. (Writing the *curved* normalised value would make C++ re-curve it and corrupt the stored value.)
  - **Read / thumb + label placement:** read `getScaledValue()` (already the correctly de-curved real value) and place the visual thumb/label by replicating **each param's exact C++ curve in JS** (a per-param curve table in `main.js`): `quantizeStrength` → `pow(n, 2)` (inverse `sqrt`); `lfoDepth`/`dlyLfoDepth` → `pow(n, 5.3)`; `lfoRate`/`dlyLfoRate`, `maskLowFreq`/`maskHighFreq`, `delayTime` → their log/exp mappings. This keeps the *visual* curve faithful while the *transport* stays linear-normalised.
  - **Validate with a Playwright round-trip test** per curved param: set a real value in JS → read back `getScaledValue()` → assert equality within the param's snap resolution.
  - **`shiftHz` is linear** (no lambda), so its slider/knob value transport is the plain normalised round-trip; only its *visual* ±5000/±20000 symmetric-log mapping is custom (below).
- **Rotary knob (`shiftHz`)** — `<div>` + inline SVG (background arc, bipolar value arc via `stroke-dasharray` on `pathLength=100`, indicator dot, 5 ticks) + visually-hidden `<input type=range>`. Interaction = **vertical-delta pointer drag** (`deltaNorm = (startY - y) / sensitivity`), Shift→×10 fine, double-click→default — exactly the Visage logic. Apply the **3-lambda symmetric-log visual mapping** (knob-norm↔param-norm) in JS so visual span is ±5000 Hz while the param is ±20000; readout `%d`≥100 else `%.1f` + "HZ".
- **Plain sliders** — `.slider-bg/.slider-fill/.slider-knob` over a hidden range input. **Adaptive decimals**: <0.1→2dp, <0.01→3dp. Double-click→default; Shift→fine (no snapping).
- **Dual-purpose sync sliders** — a JS `synced` flag (driven by the sync toggle) switches the bound state from the float relay to the division combo relay. Synced: snap to 14 discrete positions, draw ticks, label = division. **Reversed mapping:** `fillNorm = 1 - paramNorm` (16/1 left, 1/32 right) while the label always shows the true value — get *both* right (subtle bug surface). `delayTime`: `value≥1000` renders `value/1000` + " s". **Crucially, each sync slider must subscribe to its sync-toggle relay's `valueChangedEvent`** (`lfoSync`/`dlyLfoSync`/`delaySync`) so that preset/automation-driven sync changes swap the active float↔division binding, re-snap to divisions, and redraw ticks/label — not just user clicks. Add a Playwright test for the automation-driven swap.
- **Toggles** — checkbox + CSS switch. The two LFO enable toggles add/remove a `.dimmed` class on **their own sub-controls** (not whole sections). `maskEnabled`/`delayEnabled` set state but drive no dimming (match current behaviour unless Decision 11 says otherwise). `L/R Decorr` calls `setDecorrelate`/hydrates from `getDecorrelate` (no relay).
- **Combos** — styled native `<select>` (SVG chevron), options from `properties.choices`.
- **Segmented control (`processingMode`)** — two buttons writing `setChoiceIndex`; `onChange` toggles a `body[data-mode="classic|spectral"]` attribute that CSS uses for global Classic-mode dimming (spectral panel + entire mask section + `delaySlope`) + crossfade feel.
- **Piano keyboard** — 7 white + 5 black keys (SVG or absolutely-positioned divs), **black-key-first hit-test**, each toggling one of `scaleNote0..11` via the **exact `whiteKeyPC_`/`blackKeyPC_`/`blackAfterWhite_` arrays + geometry from `HolyPianoKeyboard.cpp:19-64`**. Disabled (CSS `pointer-events:none` + dim) when `data-mode="classic"`. Playwright-assert that each visual key toggles the expected `scaleNoteN`.
- **Preset strip + modal** — prev/next call `presetPrev/Next`; name area opens a dropdown built from `presetList`; Save opens a modal with a real `<input>` (maxLen 40, Enter/Escape) → `presetSave`; Delete → confirm modal → `presetDelete`, disabled when `presetName.isFactory`. The webview keyboard-focus caveat (Section 9) most affects this field — requires `EDITOR_WANTS_KEYBOARD_FOCUS TRUE` for the WebView build (Section 7.3) and per-host testing.

### 5.4 Artwork + gradients
- Background: compressed `pagan-background` as a `background-image` with a CSS scrim overlay (`linear-gradient(rgba(10,10,12,.7), …)`), or a CSS radial/linear gradient base + a smaller texture PNG. Panel gradients (`HolyPanelGradTop/Bot`, `HolyMixGradTop`) become CSS `linear-gradient`s. Accent glow (`HolyAccentGlow 0x26C9A96E`) → `box-shadow`/`filter: drop-shadow`.
- Logo: `<img>` top-right.

### 5.5 State hydration + two-way sync
- juce.js fires `requestInitialUpdate` per state on load → backend replays current values + properties. Widgets `refresh()` once at build.
- Two-way: every widget listens to `valueChangedEvent` + `propertiesChangedEvent` so automation/preset recall push back into the DOM. Sync sliders additionally subscribe to their sync-toggle relays (5.3).
- Async init getters: `presetList` → fill dropdown; `presetName` → current name + factory flag (label only; params already restored via relays); **`getDecorrelate` → hydrate the L/R Decorr toggle**. A light `setInterval` poll (or one-shot on focus) for non-param UI state if needed (Holy Shifter has no live meters, so polling needs are minimal vs Parallax's 20 Hz `hostState`).
- `body[data-ready]` flips on init completion (harness signal).

### 5.6 Attribution nuance
Preset/automation writes go through `setNormalisedValue/setChoiceIndex` directly (no DOM event), so only human gestures hit DOM listeners — clean separation, same as Parallax.

### 5.7 Theme tokens (reuse Visage palette)
Lift `HolyTheme.h` colors verbatim into CSS variables, e.g.:
```css
:root{
  --bg:#0A0A0C; --surface:#111113; --strip:#0E0E10; --raised:#161618;
  --border:#1E1E22; --text:#E8E4DB; --text-sec:#8A857D; --text-muted:#3E3A34;
  --accent:#C9A96E; --accent-dim:#6B5D3D; --accent-glow:#26C9A96E; --track:#252320;
  --panel-grad-top:#19191D; --panel-grad-bot:#101013;
}
```
Fonts: Inter (Thin/Light/Regular/Medium/SemiBold) + IBM Plex Mono Light for numeric readouts — same as Visage; the same TTFs in `plugin/fonts/` are **self-hosted** via `@font-face` (TTF or converted `.woff2`).

---

## 6. RESIZABLE / RESPONSIVE DESIGN (the centerpiece)

This is the whole point — Visage is locked at 700×928. **Note up front:** neither the aspect-lock path nor a genuinely-fluid CSS layout exists in Parallax to copy; both are net-new and must be prototyped in the spike (Section 10).

### 6.1 Native: make the editor resizable
Call order is load-bearing. `setResizeLimits` ends by calling `defaultConstrainer.setSizeLimits` + `setBoundsConstrained`, and `getConstrainer()` returns that default constrainer, so `getConstrainer()->setFixedAspectRatio(...)` does compile and apply. **But if you ever install a custom constrainer via `setConstrainer()`, `setResizeLimits` `jassertfalse`s and no-ops** (`AudioProcessorEditor.cpp:92`). So use exactly this order and **no custom constrainer**:
```cpp
setResizable(true, /*useBottomRightCornerComp*/ false);  // no JUCE corner over the webview (6.2)
setResizeLimits(560, 742, 1680, 2227);                   // strictly on the 700:928 ratio (see below)
getConstrainer()->setFixedAspectRatio(700.0 / 928.0);    // lock portrait ratio
setSize(700, 928);                                        // preserve current default
```
`resized()`: `webView.setBounds(getLocalBounds());` — guarded against re-entrancy (6.5).

**Limits must be exactly on the locked ratio.** `700/928 = 0.75431`. The earlier `560×740 … 1680×2220` corners are *not* on that ratio (`560/740 = 0.7568`), so a fixed-aspect constrainer would fight/round the corners. Use min `560×742` and max `1680×2227` (or any pair where `w/h == 700/928` exactly).

**Aspect ratio decision:** locking the ratio gives predictable uniform scaling and is the safest *visual* outcome, but it is the **less host-tested path** (Parallax does free resize within limits, no aspect lock) and `setFixedAspectRatio` combined with host-driven *edge* drags can constrain on one axis only or produce snap-back/jitter loops depending on which edge the host drives. Therefore: **prototype aspect-lock against a real host-embedded webview in the spike** (add it to the spike's exit criteria), and keep a **fallback to the proven free-resize-within-limits + CSS reflow** if aspect-lock fights hosts. (Persist the chosen size — see 6.5.)

### 6.2 The resize-grip pitfall (confirmed)
Native webviews render **on top of** sibling JUCE components — on macOS the webview is an `NSViewComponent`. A JUCE `ResizableCornerComponent` overlaid on the webview ends up behind/non-interactive. Therefore:
- Pass `useBottomRightCornerComp = false` (no JUCE corner over the webview).
- Rely on **host-driven edge resize** (works once `setResizable`/limits are set), **and/or** implement a CSS resize handle in the web UI that calls `.withNativeFunction("resizeTo", …)` → `editor.setSize(w,h)` (clamped to limits, re-entrancy guarded). Recommended: do both — host edges for DAW-native feel, plus an in-UI grip for hosts with weak/corner-only edge support.

### 6.3 Web: responsive layout strategy (net-new; choose ONE and prototype it)
Parallax provides **no** responsive primitives to copy. Pick one concrete strategy in the spike and budget it as new development:

**Option A — true fluid CSS (recommended for crispness):**
- Make **every** metric `clamp()`/`%`/`fr`/`vmin`; drive font-size off a container/root unit; `display:grid` sections with `repeat(N, minmax(0,1fr))`; `aspect-ratio` on the knob; `@container` queries on section wrappers so the LFO/Delay/Mask strips **collapse to single-column at narrow widths** instead of compressing (an improvement over Parallax, which just compresses at fixed px). More work, but **crisp at every size and DPR**, with no transform softening. This is the only approach where the "SVG/CSS is crisp at any DPR" claim is automatically true.

**Option B — transform-scale wrapper (predictable with locked aspect):**
- `transform: scale` does **not** change the layout box, so a naïve `transform: scale(var(--ui-scale))` will not fill bounds. The correct machinery (none of which exists to copy): a wrapper `position:absolute; left:50%; top:50%; width:700px; height:928px; transform: translate(-50%,-50%) scale(s)` where `s` is computed by a `ResizeObserver` as `min(clientW/700, clientH/928)`. With locked aspect ratio (6.1) this is viable and predictable. **Caveat:** non-integer `s` softens text, 1px borders, and SVG strokes on the webview raster — so either accept slight softening or **snap `s` to device-pixel-aligned steps**. The fluid-CSS crispness guarantee does **not** apply here.

Ship **one** of these from the spike; do not present either as a Parallax freebie.

### 6.4 Canvas DPR (only if a visualizer is added)
The SVG approach avoids most DPR issues *under fluid CSS* (Option A). If any widget ever uses `<canvas>` (e.g. a future spectrum), copy Parallax's `fitCanvas` verbatim (`main.js:524`): read `getBoundingClientRect()`, `dpr = min(2, devicePixelRatio)`, clamp to `[1,4096]`, reassign `canvas.width/height` only when changed, position canvases `absolute; inset:0`. **Note:** under transform-scale (Option B), SVG strokes and 1px borders soften at non-integer scale exactly like canvas — crispness is only automatic under Option A.

### 6.5 Host resize negotiation + size persistence (corrected)
`setResizable(true,…)` + `setResizeLimits(...)` makes JUCE report resize support (VST3 `checkSizeConstraint`/`IPlugViewContentScaleSupport`; AU/AUv3 preferred sizes). Caveats: some hosts (older Logic AU, Pro Tools) clamp/ignore programmatic `setSize` — always honor host-initiated resizes, use the (default) constrainer, and **fixed-aspect edge resize is the most host-divergent path** (test Logic, Live, Reaper, Pro Tools, Bitwig, Cubase).

**Persisting editor size — do NOT put it in a parameter.** Window size is UI chrome, not DSP state. Storing it as a parameter (or anywhere preset/automation recall reads) makes **every preset recall resize the window** and fights hosts that restore editor bounds themselves. Instead, store size in a **dedicated non-parameter child node of the APVTS state ValueTree** (e.g. an `<editor>` child written/read in `get/setStateInformation`), read **once on editor construction** to seed `setSize`, and **never reapply on preset load**. Guard `resized()` and the `resizeTo` native function against **re-entrancy** so a host-initiated resize → `resized()` → JS → `resizeTo` → `setSize` loop cannot form (ignore programmatic `setSize` echoes). Verify against hosts that already restore editor bounds to avoid double-restore fights.

---

## 7. Build Pipeline

Mirror Parallax's **no-bundler** approach for the web assets; the Windows half is net-new (Parallax has none).

### 7.1 Embedding (CMake), behind a flag
Add to `plugin/CMakeLists.txt`:
```cmake
option(HOLY_SHIFTER_USE_WEBVIEW "Build the WebView UI instead of Visage" OFF)
set(HS_WEBVIEW_VALUE 0)
if(HOLY_SHIFTER_USE_WEBVIEW)
  set(HS_WEBVIEW_VALUE 1)
  target_link_libraries(FrequencyShifter PRIVATE juce::juce_gui_extra)
  set(HS_WEB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/web/public")
  file(GLOB HS_WEB_ASSETS CONFIGURE_DEPENDS "${HS_WEB_DIR}/*")
  if(HS_WEB_ASSETS)
    juce_add_binary_data(HolyShifterWebAssets SOURCES ${HS_WEB_ASSETS})
    target_link_libraries(FrequencyShifter PRIVATE HolyShifterWebAssets)
  endif()
endif()
```
Flip the compile defs (currently hardcoded `JUCE_WEB_BROWSER=0` at `CMakeLists.txt:165`). On Windows the ResourceProvider **only exists when `JUCE_USE_WIN_WEBVIEW2` is set**, so define it (the static-linking variant implies it):
```cmake
target_compile_definitions(FrequencyShifter PUBLIC
    JUCE_WEB_BROWSER=${HS_WEBVIEW_VALUE}
    JUCE_USE_CURL=0
    HOLY_SHIFTER_USE_WEBVIEW=${HS_WEBVIEW_VALUE})
if(HOLY_SHIFTER_USE_WEBVIEW AND WIN32)
  target_compile_definitions(FrequencyShifter PUBLIC
      JUCE_USE_WIN_WEBVIEW2_WITH_STATIC_LINKING=1)  # implies JUCE_USE_WIN_WEBVIEW2=1
endif()
```
- `CONFIGURE_DEPENDS` re-globs when files change; a *new* file triggers reconfigure on next build. **Don't glob junk** (`.DS_Store`, source maps) into the binary — list explicitly or `.gitignore` strays.
- Re-compress `pagan-background` before embedding (3.5 MB × every instance).

### 7.2 Windows WebView2 — net-new, and the build will NOT configure as-is
**This is the single biggest reproducibility hole.** Parallax has zero Windows WebView2 handling, so there is no in-tree reference; treat all of this as new work and prove it by actually building and running on Windows.

1. **Pass `NEEDS_WEBVIEW2 TRUE` to `juce_add_plugin(...)`** (WebView branch only). This triggers `find_package(WebView2 REQUIRED)` (`JUCEUtils.cmake:300-301`). For a direct `juce_add_plugin` you must pass it explicitly — `NEEDS_WEBVIEW2` is auto-derived only from the `_WITH_STATIC_LINKING` *pip* flag, which does not apply here.
2. **Install the Microsoft.Web.WebView2 NuGet package before configure.** `FindWebView2.cmake` searches `$ENV{USERPROFILE}/AppData/Local/PackageManagement/NuGet/Packages/*Microsoft.Web.WebView2*` (or `JUCE_WEBVIEW2_PACKAGE_LOCATION`). **Nothing currently installs this**, and the existing Windows CI (`.github/workflows/build.yml:161-175`) runs a bare `cmake -B build -DCMAKE_BUILD_TYPE=Release` with **no nuget step**, so configure will hard-fail with *"Could not find a package configuration file provided by WebView2"* the instant `NEEDS_WEBVIEW2` is on. Fix: add an explicit acquisition step to CI before configure —
   ```
   nuget install Microsoft.Web.WebView2 -OutputDirectory <dir>
   cmake -B build -DJUCE_WEBVIEW2_PACKAGE_LOCATION=<dir> ...
   ```
   — or vendor the SDK headers/loader and point `JUCE_WEBVIEW2_PACKAGE_LOCATION` at it. Prefer **static linking** (`JUCE_USE_WIN_WEBVIEW2_WITH_STATIC_LINKING=1`, set in 7.1) so the loader is static; the WebView2 **runtime** is still required at run time.
3. **Guard the editor with `#if JUCE_WEB_BROWSER_RESOURCE_PROVIDER_AVAILABLE`** (Section 4) so a Windows build missing `JUCE_USE_WIN_WEBVIEW2` fails at compile time rather than shipping a blank UI.
4. **Separate the two failure modes** the plan previously conflated: (a) the **build-time NuGet SDK** absence is a developer/CI problem (above); (b) the **end-user Evergreen runtime** absence shows a blank/error view at run time. For (b): set a stable per-plugin `withUserDataFolder` under `userApplicationDataDirectory` (NOT `tempDirectory`); detect a missing runtime via `GetAvailableCoreWebView2BrowserVersionString` and show a **native JUCE fallback label with a download link** instead of a blank webview; ship/detect the Evergreen bootstrapper. Add a **page-load-error path** (host it in the `SinglePageBrowser` subclass mentioned in 7.4).
5. **Add a Windows smoke test to CI that actually loads the editor** — a clean configure does not prove the WebView renders.

### 7.3 macOS/iOS
WKWebView is a system framework (no extra package); `CMAKE_OSX_DEPLOYMENT_TARGET 11.0` and OBJC/OBJCXX are already enabled. Serve from the in-memory ResourceProvider (avoid `file://` sandbox issues). **Self-host fonts (don't fetch Google Fonts — this is the Parallax anti-pattern; offline/hardened-runtime/CSP would break it).** **Set `EDITOR_WANTS_KEYBOARD_FOCUS TRUE` for the WebView build** (currently `FALSE` at `CMakeLists.txt:138`, mapped to `setWantsKeyboardFocus` in the AU client) so the preset-name `<input>` receives typing — gate it so the Visage build keeps its current value. The existing Visage macOS patches (`patches/`) are Visage-only and become irrelevant once Visage is removed; keep them gated `if(APPLE)` while both UIs coexist.

### 7.4 Dev (live reload) vs Release (embedded) — dev path is net-new
- **Embedded (default plugin):** ResourceProvider path; only `CMAKE_BUILD_TYPE`/arch differ between debug/release.
- **Dev hot-reload (NEW — Parallax does not do this in the plugin):** the 2-arg `withResourceProvider(provider, allowedOrigin)` signature is real (`juce_WebBrowserComponent.h:385`), but Parallax implements no localhost/`pageAboutToLoad`/`SinglePageBrowser`/`allowedOrigin` in the plugin. So this is new design: pass a dev-server origin to `withResourceProvider(provider, "http://localhost:4321")` and `goToURL("http://localhost:4321")`, subclass `WebBrowserComponent`'s `pageAboutToLoad` to allow **only** the resource root + dev origin, and **gate the whole thing behind a debug-only define — never ship it.** A release build pointing `goToURL` at `http://localhost` is a remote-content + mixed-scheme security regression.
- **Standalone browser + Playwright (genuinely copyable from Parallax):** serve the same `web/public/` via `python3 -m http.server 4321` with `check_native_interop.js`'s mock backend; Playwright asserts the hidden mirror nodes — Parallax's actual Tier-5 setup (`tests/ui/playwright.config.js`). Keep this distinction clear: the standalone harness is reuse; the in-plugin dev server is not.

---

## 8. Migration Strategy

### 8.1 Coexist behind a flag, then flip
- Keep Visage as the default (`HOLY_SHIFTER_USE_WEBVIEW=OFF`) while building the webview UI. `createEditor()` switches on the flag (Section 4.1). This de-risks the port — Visage stays shippable until parity + QA pass.
- Per memory: `PluginEditor.cpp` is **dead legacy** and the live UI is Visage (`HolyShifterUI` via `VisageHostEditor`). The webview editor replaces `VisageHostEditor` as the active editor; once parity holds **and the pluginval/pedalboard validation re-run passes** (Section 9), retire Visage (`HolyShifterUI`, `controls/`, the Visage FetchContent + macOS patches, font/image `add_embedded_resources` if not reused), shrinking the build.

### 8.2 What to reuse (no rewrite)
- **DSP + parameters:** untouched — APVTS, `createParameterLayout`, `parameterChanged`, the % scaling, `STFT`/`MusicalQuantizer`/`Scales`. The UI swap is purely the editor layer.
- **PresetManager:** reuse as-is via native functions (Section 4.5).
- **Theme:** lift `HolyTheme.h` colors → CSS variables (Section 5.7).
- **Fonts:** reuse `plugin/fonts/` TTFs as **self-hosted** `@font-face` (TTF or `.woff2`).
- **Artwork:** reuse `assets/heathen-machines-logo.png`; re-compress `pagan-background.png`.
- **Behaviour spec:** the Visage control sources (`HolyRotaryKnob.cpp`, `HolySlider.cpp/.h`, `HolyPianoKeyboard.cpp`) are the **authoritative behaviour reference** for the JS reimplementation (drag math, fine mode, reset, sync/reverse/seconds, adaptive decimals, **piano pitch-class arrays/geometry**, the per-param curves). Confirmed APIs: `setSynced`, `setSyncReversed`, `setSecondsAbove`, `setAdaptiveDecimals`. Confirmed dimming sources: `updateControlsForMode` and `updateLfoEnableUI` (Section 3.2 §9).

### 8.3 iOS synergy — plausible but unproven; treat as a separate spike
A responsive webview UI is the most credible way to unblock iOS (Visage has no iOS windowing), and several enablers are verified:
- `JUCE_WEB_BROWSER_RESOURCE_PROVIDER_AVAILABLE` **does include `JUCE_IOS`** (`juce_WebBrowserComponent.h:40`).
- **WKWebView is the OS-native view on iOS** and handles touch/gestures/scrolling/on-screen keyboard.

**But the load-bearing unknowns are not demonstrated anywhere in this workspace** (Parallax has never been built for iOS):
- The relay/`WebSliderRelay` bridge + ResourceProvider running inside an **AUv3 app-extension sandbox** (tighter memory limits, different view-attachment lifecycle) is unverified — *prove it on a single param before quoting any reuse percentage.*
- Editor **resize inside an AUv3 host view** is unverified (Section 8.3's own "to verify" item).
- The "~80–90% shared" / "resize just works" figures were unsubstantiated estimates; **drop them** until a one-param iOS AUv3 spike confirms the bridge works on-device.

Remaining iOS-specific work (unavoidable for any iOS port): AUv3 extension target + Info.plist + provisioning/entitlements, App Sandbox file access (use the in-memory ResourceProvider, not `file://`), on-screen-keyboard focus, and host-view resize verification. The `AUv3` format is already declared in `juce_add_plugin`.

**Recommendation:** Design the responsive layout (Section 6) with touch + tablet breakpoints from day one (cheap insurance), but keep iOS strictly **Phase 6/deferred and gated behind its own spike**; do not use it as a justification multiplier elsewhere.

---

## 9. Risks & Cross-Platform Caveats

1. **Curved-slider write path (highest correctness risk):** curves are lambda `NormalisableRange` converters with skew=1, which the JS bridge treats as linear; writing the curved normalised value double-curves and corrupts state. Mitigate with the linear-inverse write + JS-replicated curve for placement + per-param Playwright round-trip (Section 5.3). `shiftHz` is linear and safe.
2. **Other non-trivial control fidelity:** shift-knob symmetric-log *visual* mapping, reversed sync sliders (fill=`1-norm` but true label) **and their sync-toggle subscription**, adaptive decimals, delay-seconds switch, piano black-key-first hit-test with exact pitch-class geometry, mode/enable dimming **matching the live rules** (Classic dims spectral+mask+slope; LFO enables dim only their own sub-controls; mask/delay enables dim nothing). Port math directly from the Visage `.cpp` sources; assert via hidden mirror nodes in Playwright.
3. **Windows build SDK (critical, net-new):** `NEEDS_WEBVIEW2` triggers `find_package(WebView2 REQUIRED)`; without a NuGet acquisition step the configure **hard-fails**. Add the nuget step to CI + `JUCE_WEBVIEW2_PACKAGE_LOCATION`, set `JUCE_USE_WIN_WEBVIEW2_WITH_STATIC_LINKING=1`, guard the editor with `JUCE_WEB_BROWSER_RESOURCE_PROVIDER_AVAILABLE` (Section 7.2).
4. **Windows end-user runtime:** Win10 may lack Evergreen → blank/error view; set a stable per-plugin `withUserDataFolder` (not tempDirectory), detect missing runtime and show a native fallback with a download link, add a page-load-error path, ship/detect the bootstrapper (Section 7.2 §4).
5. **Keyboard focus in webview:** `EDITOR_WANTS_KEYBOARD_FOCUS` is `FALSE` (`CMakeLists.txt:138`); the preset-name `<input>` will not reliably receive typing until it's flipped `TRUE` for the WebView build. A `visage-plugin-keyfocus.patch` already exists in `patches/`, confirming this first-responder problem bit the project before. Enabling editor focus can swallow DAW transport shortcuts — test space/play passthrough across AU (Logic), VST3 (Live/Cubase), and Standalone (this is Decision 9's trade-off).
6. **Responsive layout is net-new:** Parallax's CSS is fixed-px with no fluid primitives; neither fluid CSS (Option A) nor transform-scale (Option B) nor aspect-lock is copyable. Prototype the chosen approach in the spike; transform-scale softens non-integer scales (snap or accept).
7. **Aspect-lock + host resize:** `setFixedAspectRatio` is unproven against a host-embedded `NSViewComponent`; limits must be exactly on the 700:928 ratio; keep call order `setResizable → setResizeLimits → getConstrainer()->setFixedAspectRatio → setSize` with **no custom constrainer**; fixed-aspect edge resize is the most host-divergent path; have the free-resize fallback ready (Section 6.1).
8. **Editor size persistence:** store in a dedicated non-parameter state child, seed `setSize` once on construction, never reapply on preset load, guard `resized()`/`resizeTo` against re-entrancy (Section 6.5).
9. **CSP is unproven under the `juce://` scheme:** the earlier example `default-src 'self' juce: https://juce.backend; script-src 'self'` is partly fabricated — there is **no** `https://juce.backend` origin, and under a custom non-http scheme `'self'` may not resolve to the `juce://` origin, so a naïve `script-src 'self'` can **block your own bundled `main.js`/CSS and white-screen the UI**. Parallax ships no CSP, so there is no in-tree precedent. Budget a **spike task**: empirically log `document.location` / CSP violation reports on both WKWebView and WebView2, write the policy against the **real** origin, and treat CSP as defense-in-depth on top of `withNativeIntegrationEnabled` + local-only content (the load-bearing trust boundary).
10. **Performance/memory + load-reliability vs Visage:** each open editor adds a browser engine + helper/GPU processes (tens of MB RSS) and a JS↔native bridge; multi-instance multiplies it. Visage is materially leaner. **The harder risk is editor-construction timeouts in validation/CI:** this repo's CI already records pluginval/pedalboard *hangs/timeouts at plugin load* with the current Visage editor (`.github/workflows/build.yml:72,105`), and the macOS pluginval gate (`build.yml:114-132`) could break if WebView2/WKWebView process startup pushes editor-create over the timeout. **Re-run pluginval + the headless pedalboard tier against the webview build before flipping the default**, consider lazy/deferred webView construction, confirm editor open under pluginval's strictness levels, and quantify per-instance RSS with a quick multi-instance test rather than asserting "Visage is leaner."
11. **Embedded asset bloat & MIME:** `file(GLOB)` bakes every `web/public/*` into the binary; keep it clean and re-compress the 3.5 MB background. **Extend `mimeForExtension` for the chosen font extension (`.woff2`/`.ttf`) and `.webp`** or they load as octet-stream and fail.
12. **`withNativeIntegrationEnabled` security:** only safe with fully-controlled local content; never point at remote origins in shipping builds (and never ship the dev-server path).
13. **API drift:** relays take a **name only** and are wired via `.withOptionsFrom` (`WebControlRelays.h:73`) — ignore stale `{browser, name}` blog snippets.

---

## 10. Phased Roadmap + De-Risk Spike

**Spike (3–4 days) — prove the resizable bridge on BOTH platforms.** Add the `HOLY_SHIFTER_USE_WEBVIEW` flag + CMake glob/embed + the Windows NuGet/WebView2 wiring (7.2); a minimal `WebViewEditor` with `setResizable` + aspect-locked limits (exact-ratio corners) **and** the free-resize fallback ready; a tiny `index.html` binding **3 real params** (`shiftHz` slider/knob, `warm` toggle, `processingMode` combo) + `presetList`; prototype **one** responsive strategy (Option A fluid CSS or Option B transform-scale, 6.3); and a first pass at CSP origin logging (Risk 9). **Exit criteria:** (1) editor resizes in a DAW with the web UI reflowing/scaling and **aspect-lock validated against a real host-embedded webview** (or fallback chosen); (2) params round-trip both ways including a **curved param** to validate the write path; (3) presets list loads; (4) the build **actually configures, links, and renders on Windows** (not assumed from Parallax) with WebView2 static-linked and a stable user-data folder. This validates the riskiest unknowns (curved write path + resize/aspect + Windows bring-up + CSP) cheaply.

**Phase 1 — Native scaffold (3–4 days).** Full relay+attachment set for all bound params (~46 of the 50 — exclude the 4 truly-unused hidden/deprecated ones, 3.1), all preset native functions, `setDecorrelate`/`getDecorrelate`, `resizeTo` (re-entrancy guarded). ResourceProvider copied from Parallax + extended MIME map. Standalone-browser harness + Playwright config wired. (~1 wk cumulative.)

**Phase 2 — Core controls + layout (1–1.5 wks).** Knob (symmetric-log visual mapping), all plain sliders with the **correct curved write path + JS curve table + adaptive decimals + double-click/fine**, toggles, combos, segmented control, theme CSS + artwork, the chosen responsive scaffold. Visual parity pass against Visage/Figma + curved-param round-trip Playwright tests.

**Phase 3 — Hard controls + sections (1–1.5 wks).** Three dual-purpose sync sliders (reversed labels + seconds switch + **sync-toggle relay subscription**), piano keyboard (exact pitch-class geometry, Playwright per-key assertion), mode/enable dimming **matching live rules**, preset strip + Save/Delete modals, two-way hydration + polling, L/R Decorr hydrate-on-init.

**Phase 4 — Responsive polish + resize (3–5 days).** Finalize exact-ratio min/max limits, optional Option-A `@container` reflow breakpoints, in-UI resize grip, persist editor size in the dedicated state child, tune `clamp()`/`vmin` typography, snap/accept transform-scale softening if Option B. Verify fixed-aspect edge-drag across hosts.

**Phase 5 — Cross-platform + QA (1–1.5 wks).** Windows: NuGet-in-CI, static link, stable user-data folder, missing-runtime fallback, editor-render smoke test. macOS: empirically-derived CSP, self-hosted fonts, `EDITOR_WANTS_KEYBOARD_FOCUS TRUE` + per-host keyboard/transport-shortcut testing. **Re-run pluginval + headless pedalboard against the webview build** (load-timeout fragility), multi-instance RSS check. Then flip the default flag and retire Visage.

**Phase 6 (optional, deferred) — iOS AUv3** behind its own spike: prove the relay bridge + ResourceProvider + resize on a single param in an actual AUv3 build before committing; then extension target, entitlements, sandbox/keyboard verification, touch tuning.

**Total to desktop parity + QA: ~5–7 weeks** (Windows bring-up is net-new, not a one-liner; excludes iOS).

---

## 11. Decisions Needed From the User

1. **Aspect ratio on resize:** lock 700:928 (uniform scale — visually safest but the *less host-tested* path; spike-validated with a free-resize fallback) **or** allow free reflow with breakpoints (proven Parallax-style, more responsive CSS)? (Affects Phase 4 scope.)
2. **Min/max editor sizes:** confirm bounds **exactly on the 700:928 ratio** (proposed `560×742 … 1680×2227`, default `700×928`).
3. **Coexistence vs hard cutover:** keep Visage behind the flag during the port (recommended) or remove it immediately?
4. **Target the v0.2.3 PeakSnap params** (`peakSnap`/`Tones Only`, `noiseMix`/`Texture`, `peakSens`/`Density`) in the new UI now, or match the current working tree (50 params) and add them later? (They exist in DSP on `benji/main` but have no Visage control yet — the webview is a clean place to introduce them.)
5. **Dimming behaviour:** match the current UI exactly (only `lfoEnabled`/`dlyLfoEnabled` dim their own sub-controls; Classic dims spectral+mask+slope; `maskEnabled`/`delayEnabled` dim nothing) **or** add delay/mask enable-dimming as a deliberate improvement?
6. **Responsive strategy:** Option A fluid CSS (crisp at all sizes, more work) **or** Option B transform-scale wrapper (predictable with aspect-lock, slight non-integer softening)?
7. **iOS priority:** design touch/tablet breakpoints from day one (cheap) but keep the iOS build itself deferred behind its own spike — confirm this framing, vs investing in iOS earlier (higher risk, unproven bridge in AUv3).
8. **Framework:** vanilla ES modules like Parallax (no build step, fastest to match the reference) — confirm, vs introducing a bundler/React (more tooling, not needed for one screen).
9. **Background asset:** approve re-compressing/replacing the 3.5 MB `pagan-background.png` (WebP/JPG or CSS gradient + small texture) before embedding.
10. **Windows distribution:** confirm we'll add the WebView2 NuGet step to CI **and** ship/detect the Evergreen runtime (accepting the Win10-without-runtime edge case with a native fallback prompt).
11. **Keyboard focus:** approve flipping `EDITOR_WANTS_KEYBOARD_FOCUS TRUE` for the WebView build, accepting that DAW global/transport shortcuts may be swallowed while the webview has focus (text fields then work), pending per-host testing.