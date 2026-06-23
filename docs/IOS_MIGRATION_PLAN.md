# Holy Shifter → iOS (iPhone + iPad) App Store — Migration Plan

_Status: **in progress** — build-system spike underway on branch `feat/ios-port`. Target: ship
Holy Shifter on the iOS App Store as an **AUv3 plug‑in + standalone container app**, universal
(iPhone + iPad)._

> **Update (the WebView changes everything).** Since this plan was first written we shipped a
> **WebView UI** (`juce::WebBrowserComponent`, branch merged to `main`). On iOS that component is
> backed by **WKWebView**, which is fully supported — so the WebView UI *is* the iOS UI. This
> **eliminates the project's central risk** (porting Visage's windowing layer to iOS) and the
> longest phase (~2–4 wk + an ongoing Visage fork). The new path is **Path C** (§5). Paths A/B
> are retained below only for context.

**Decisions locked in:** universal **iPhone + iPad**; **min iOS 16**; **paid up‑front** (no
StoreKit/IAP); **Benji's "Heathen Machines" team** (`DU92Z6L82F`) owns signing + the listing.

---

## 1. Executive summary

Shipping Holy Shifter on the iOS App Store is **feasible and now low‑risk**. The audio engine and
the JUCE plug‑in plumbing port with little change, and the **WebView UI runs on iOS as‑is** via
WKWebView — so we do **not** need to port Visage to iOS.

- ✅ **DSP core** — pure portable C++ (`src/dsp/*`), no platform headers. Cross‑compiles to
  iOS arm64 as‑is. (`juce_dsp` FFT uses Apple Accelerate/vDSP — fine on iOS.)
- ✅ **JUCE wrapper** — JUCE 8 supports **AUv3** and **Standalone** on iOS. We already declare
  `AUv3` in `FORMATS` and set `PLUGIN_AU_EXPORT_PREFIX` / `AU_MAIN_TYPE`.
- ✅ **UI** — the **WebView UI** (HTML/CSS/JS + the `juce.js` relay bridge) renders in **WKWebView**
  on iOS, with touch handled natively by the web view. Same UI codebase as desktop WebView.
- ✅ **Build system** — Visage is now fully gated behind `HOLY_BUILD_VISAGE` (off on iOS); iOS
  forces `HOLY_SHIFTER_USE_WEBVIEW=ON` and never compiles/links Visage. (Done on `feat/ios-port`.)
- ⚠️ **Remaining work** — iOS build/sign/provisioning, touch‑ergonomics + iPhone‑size responsive
  layout, preset sharing via App Group, audio‑session/state, and App Store packaging. None are
  research‑risk; they're standard iOS integration.

**Recommendation:** **Path C** — ship the WebView UI on iOS. Visage stays desktop‑only. The
Phase 0 spike is now just "prove the WebView UI renders + takes touch in the iOS simulator,"
which is far cheaper than the old Visage‑port spike.

---

## 2. Current architecture (what we're porting)

- **JUCE 8.0.4** plug‑in via CMake/FetchContent. Formats: AU, VST3, AUv3, Standalone.
  `BUNDLE_ID com.harmonictools.frequencyshifter`, `PLUGIN_CODE Fshf`, manufacturer `Hrmt`,
  product **"Holy Shifter"**, company **"Heathen Machines"**, v0.2.2.
- **DSP** (`src/dsp/`): STFT / PhaseVocoder / HilbertShifter / MusicalQuantizer / SpectralDelay /
  SpectralMask / LFO / DriftModulator / FeedbackDelay / Scales. Pure C++.
- **UI**: Visage (GPU, bgfx/Metal). Custom `Holy*` controls, embedded fonts (Inter, IBM Plex
  Mono) and images (logo, pagan background), Figma‑matched. Embedded by `VisageHostEditor` via
  `visageWindow_->show(nativeHandle)` — i.e. Visage already attaches to a host‑provided **NSView**.
- **macOS‑specific code**: `visage_windowing/macos/windowing_macos.mm` + two local patches
  (`visage-au-hidpi.patch`, `visage-plugin-keyfocus.patch`), both guarded `if(APPLE)`.
- `src/PluginEditor.cpp` is **dead legacy JUCE editor** (not in the build).

---

## 3. Target shape on iOS

Apple distributes audio plug‑ins as **AUv3 app extensions** that ship **inside a container app**.
So we build two products from the same core:

1. **AUv3 extension** — the plug‑in, loaded by hosts (AUM, GarageBand, Cubasis, Drambo, Loopy
   Pro, etc.).
2. **Standalone container app** — required by Apple to distribute the extension, and doubles as a
   usable standalone effect. Apple rejects "thin"/empty container apps, so this must be a real,
   functional app (it already is — we have a Standalone format).

Architecture: arm64 device builds + simulator builds for dev. Shared C++ core; AUv3 and Standalone
are thin JUCE wrappers around it (as today).

---

## 4. Portability assessment (component by component)

| Component | iOS status | Work required |
|---|---|---|
| DSP core (`src/dsp/*`) | ✅ portable | Build for arm64; confirm no desktop assumptions; verify FFT path |
| JUCE plug‑in client (AUv3, Standalone) | ✅ supported | iOS toolchain + iOS plug‑in properties |
| Visage renderer (bgfx Metal) | ✅ works on iOS | none (already Metal) |
| **Visage windowing + touch** | ❌ **missing** | **Write `visage_windowing/ios/` (UIView/CAMetalLayer/UITouch)** — see §5 |
| `VisageHostEditor` (embeds Visage in JUCE editor) | ⚠️ macOS NSView path | Add iOS UIView embedding path |
| Preset manager (`~/Library/...` paths) | ⚠️ desktop paths | Use app container/Documents + an **App Group** to share presets between app & extension |
| Embedded assets/fonts | ✅ | Fine (note ~3.6 MB background image → app size) |
| macOS Visage patches | ⚠️ | Keep `if(APPLE AND NOT IOS)`; iOS uses the new `ios` windowing |
| Audio I/O / session | ⚠️ | AUv3: host‑driven. Standalone: `AVAudioSession` + Background Audio mode |

---

## 5. The UI decision — **resolved: Path C (WebView)**

### Path C — Ship the WebView UI on iOS **(chosen)**

The WebView UI we built for desktop (`juce::WebBrowserComponent` + the `juce.js` relay/native‑
function bridge, `src/WebViewEditor.{h,cpp}` + `web/public/*`) runs on iOS **unchanged**:

- On iOS, `WebBrowserComponent` is backed by **WKWebView** (first‑class on the platform). The same
  resource provider serves the embedded `web/public` bundle; the same relays drive the parameters.
- **Touch is free** — WKWebView delivers native touch to the HTML controls; our knob/slider drag
  already work on pointer deltas, which WebKit synthesises from touch. (Ergonomics still need a pass
  — see Phase 3.)
- **One UI codebase** for desktop‑WebView and iOS, and no Visage fork to maintain on iOS.
- Build wiring is done: iOS forces `HOLY_SHIFTER_USE_WEBVIEW=ON` and `HOLY_BUILD_VISAGE=OFF`, so
  Visage is never fetched, compiled, or linked (`plugin/CMakeLists.txt`).

**Cons / open items:** the UI is currently aspect‑locked to 700×928 (portrait‑ish) — fine on iPad,
but iPhone needs a responsive reflow (or letterboxing) for small/landscape sizes (Phase 3). Per‑host
AUv3 WKWebView embedding still needs on‑device testing (Phase 5).

> Paths A and B below are kept for historical context. With Path C working, **we do not port Visage
> to iOS** unless we later want the GPU UI specifically on iPad.

### Path A — Port Visage windowing to iOS (not pursued)

Write `visage_windowing/ios/windowing_ios.mm`, the iOS analog of `windowing_macos.mm`:

- A `UIView` subclass backed by a **CAMetalLayer** that bgfx renders into (bgfx already supports
  this on iOS).
- Attach to a host‑provided parent `UIView` (the JUCE AUv3 editor's view) — mirroring the current
  `WindowMac(... void* parent_handle)` path that takes the NSView.
- **Touch → input**: map `touchesBegan/Moved/Ended` → Visage `mouseDown/Drag/Up` (single‑touch
  first; add multi‑touch later for two knobs at once). The existing vertical‑drag knob and slider
  drag already work on deltas, so they translate cleanly to touch.
- Backing scale from `UIScreen.scale` (Retina) — analogous to the macOS `setDpiScale`/HiDPI patch
  we already maintain.
- Lifecycle: view appear/disappear, occlusion, app background/foreground.

**Pros:** reuse the entire UI (controls, theme, fonts, Figma artwork) → visual parity, one UI
codebase, smallest rework. **Cons:** we maintain an iOS windowing layer as a fork/patch of the
pinned Visage (upstream has no iOS today — verify before/at spike); Metal‑on‑iOS + AUv3 view
embedding has edge cases (we already hit the macOS HiDPI + first‑responder ones, so expect similar
on iOS).

### Path B — Rebuild the UI in JUCE for iOS (fallback)

Use JUCE's own GUI (native iOS + touch + AUv3 editor). Re‑implement the `Holy*` controls in JUCE.

**Pros:** fully supported path, no Visage porting, touch handled by JUCE. **Cons:** re‑implement
the whole custom UI (lose or redo the Visage/Figma look); two UI codebases to maintain (desktop
Visage + iOS JUCE); the legacy `PluginEditor.cpp` is abandoned and not a usable base. Larger UI
effort and visual divergence from desktop.

**Decision driver:** preserve the UI investment and parity → **Path A**, gated by a spike. If the
spike shows Metal/AUv3 embedding is too fragile, fall back to B.

---

## 6. Build‑system changes

- **iOS CMake toolchain**: `-DCMAKE_SYSTEM_NAME=iOS`, `CMAKE_OSX_ARCHITECTURES=arm64` (device) and
  a separate simulator config, `CMAKE_OSX_DEPLOYMENT_TARGET` (propose **iOS 15**), Xcode generator,
  code signing identities/profiles.
- **`juce_add_plugin`**: ensure `AUv3` + `Standalone` build for iOS; add iOS properties — app
  category, **icons**, **launch storyboard**, `ITSAppUsesNonExemptEncryption=false`, microphone
  usage string (if standalone input), **Background Audio** UIBackgroundMode (standalone),
  `REQUIRES_FULL_SCREEN`/orientation, and the **App Group** entitlement (shared presets).
- **Patch guard**: change the Visage patch gate from `if(APPLE)` to "macOS only" so iOS uses the new
  `ios` windowing and skips the AppKit‑specific patches.
- **Separate build dir** (`build-ios`) so desktop CI/builds are untouched.
- **CI**: add an iOS build job (build + archive; signed builds need the Apple cert in CI). Keep it
  separate from the desktop matrix.

---

## 7. iOS‑specific app concerns

- **Memory/CPU**: AUv3 runs in‑process in the host. Watch total footprint (the big background image
  + fonts + STFT buffers) and DSP CPU on iPad (profile thermals/battery).
- **Audio session**: AUv3 is host‑driven (sample rate/buffer from host). Standalone uses
  `AVAudioSession` + Background Audio so it keeps running.
- **Preset storage**: `PresetManager` currently writes `~/Library/Audio/Presets/...` — on iOS use
  the app's `Documents`/container and an **App Group** so the standalone app and the AUv3 extension
  share the same user presets.
- **State**: verify AUv3 `fullState` save/restore works embedded (we already exercise plug‑in state
  on desktop).
- **Input** (standalone): an effect needs input — mic or Inter‑App Audio; decide standalone UX.

---

## 8. App Store & business

- **Apple Developer Program** ($99/yr) — use Benji's team ("benjamin vaughan", team `DU92Z6L82F`;
  the same identity that signs the desktop release). Create App Store Connect app + **two bundle
  IDs** (container app + extension) + App Store distribution profiles.
- **Monetization** (decision needed): paid app, or free app + **StoreKit IAP** non‑consumable
  unlock (common for audio apps; allows a try‑before‑buy/limited demo). Affects engineering
  (StoreKit + feature gating).
- **Review**: AUv3 must function in a host; the container/standalone app must be genuinely useful
  (anti "thin app" rule). Needs iPad screenshots, privacy policy/nutrition label, metadata, app
  icon, launch screen, encryption declaration (standard crypto → exempt).
- **Branding**: ships as **Holy Shifter** by **Heathen Machines** (matches desktop).

---

## 9. Risks & unknowns (highest first)

1. **Visage iOS windowing port** — the central risk. Metal layer + touch + AUv3 view embedding +
   Retina scale + lifecycle. → **Spike to de‑risk (Phase 0).**
2. **AUv3 view embedding across hosts** — behaviour differs in AUM / GarageBand / Cubasis. Test
   early on real hardware.
3. **Performance on iPad** — STFT/FFT CPU, thermal throttling, battery. Profile and possibly add a
   lighter mode / smaller FFT option.
4. **Touch ergonomics** — desktop drag model needs finger‑friendly hit targets (44 pt), gesture
   tuning, and the piano keyboard sized for touch.
5. **App Store review** — container‑app usefulness, metadata completeness.
6. **Visage maintenance** — carrying an iOS windowing fork against the pinned SHA (like our current
   patches).

---

## 10. Phased roadmap (Path C)

- **Phase 0 — WebView‑on‑iOS spike (simulator). _In progress._** Build the Standalone (which embeds
  the AUv3) for the iOS simulator and confirm the WebView UI renders + takes touch. Cheap now that
  it's "does WKWebView show our HTML" rather than "port Visage." Build wiring already done on
  `feat/ios-port` (Visage gated off, WebView forced on).
- **Phase 1 — Build system, ~0.5 wk.** iOS CMake/Xcode config for **device** (signing, bundle IDs,
  provisioning via Benji's team); confirm AUv3 + Standalone archive. _(Simulator config done.)_
- **Phase 2 — Responsive layout, ~1–2 wk.** Make the UI reflow for **iPhone** sizes + landscape
  (today it's aspect‑locked 700×928 — fine on iPad, needs work on phone). Replaces the old
  "Visage iOS windowing" phase, which Path C removes entirely.
- **Phase 3 — Touch UX pass, ~1 wk.** 44 pt hit targets, gesture tuning for the knob/slider/piano,
  prevent unwanted scroll/zoom/selection in WKWebView, fine‑drag affordance.
- **Phase 4 — iOS integration, ~1–2 wk.** Presets via **App Group** (share between app + extension),
  audio session, `fullState` save/restore, app icon, launch screen, container/extension wiring.
- **Phase 5 — Host testing + performance, ~1–2 wk.** AUM / GarageBand / Cubasis on a real device;
  profile STFT/FFT CPU + thermals on iPad/iPhone.
- **Phase 6 — App Store, ~1 wk + review.** App Store Connect (Benji's team), signing, **paid‑app
  pricing** (no IAP), metadata, iPhone+iPad screenshots, encryption declaration, submit + iterate.

**Rough total (solo, experienced JUCE/C++/iOS dev): ~5–9 weeks** — Path C removes the ~2–4 wk
Visage‑port phase and its ongoing fork tax. Estimates are coarse and will tighten as the spike and
device builds land.

---

## 11. Decisions — resolved

1. **UI strategy** — ✅ **Path C (WebView on iOS).** Visage stays desktop‑only.
2. **Monetization** — ✅ **Paid up‑front.** No StoreKit/IAP code; single App Store price.
3. **Device scope** — ✅ **Universal (iPhone + iPad)**, **min iOS 16**.
4. **Apple account ownership** — ✅ **Benji's "Heathen Machines" team** (`DU92Z6L82F`) — same
   identity that signs the desktop release; owns the App Store Connect app + revenue.
5. **Who builds it** — engineering here; **Benji handles signing/notarization + the App Store
   submission** (as with desktop releases).
