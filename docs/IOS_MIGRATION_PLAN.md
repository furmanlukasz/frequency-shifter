# Holy Shifter → iPad (iOS) App Store — Migration Plan

_Status: research + plan (no code changes yet). Author: handoff doc. Target: ship Holy Shifter
on the iPad App Store as an **AUv3 plug‑in + standalone container app**._

---

## 1. Executive summary

Shipping Holy Shifter on the iPad App Store is **feasible**. The audio engine and the JUCE
plug‑in plumbing port to iOS with little change. There is exactly **one significant
engineering risk**: the **Visage GPU UI has no iOS support** today.

- ✅ **DSP core** — pure portable C++ (`src/dsp/*`), no platform headers. Cross‑compiles to
  iOS arm64 as‑is. (`juce_dsp` FFT uses Apple Accelerate/vDSP — fine on iOS.)
- ✅ **JUCE wrapper** — JUCE 8 supports **AUv3** and **Standalone** on iOS. We already declare
  `AUv3` in `FORMATS` and set `PLUGIN_AU_EXPORT_PREFIX` / `AU_MAIN_TYPE`.
- ✅ **Renderer** — Visage draws via **bgfx**, which has a working **Metal** backend on iOS
  (`renderer_mtl.mm`, `entry_ios.mm`).
- ❌ **Visage windowing + input** — `visage_windowing/` has `emscripten / win32 / macos /
  linux` only. **No `ios/`.** The macOS layer is **AppKit** (`NSWindow`/`NSView`, 22 refs) and
  is **mouse‑only** (no touch). On iOS, CMake's `APPLE` is also true, so it would try to compile
  the AppKit file → **won't build for iOS**. This is the crux of the project.

**Recommendation:** port Visage's windowing layer to iOS (UIView + CAMetalLayer + UITouch) so we
keep the entire existing UI and visual parity (Path A). De‑risk with a 1–2 week spike before
committing. Fallback is rebuilding the UI in JUCE (Path B) — larger and splits the UI codebase.

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

## 5. The UI decision (the crux)

### Path A — Port Visage windowing to iOS **(recommended)**

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

## 10. Phased roadmap

- **Phase 0 — Spike (de‑risk Visage on iOS), ~1–2 wk.** Minimal iOS app that renders the live Holy
  Shifter UI in a `UIView` with touch input on a real iPad. **Go/no‑go for Path A.**
- **Phase 1 — Build system, ~0.5–1 wk.** iOS CMake toolchain; AUv3 + Standalone iOS targets
  building (DSP/headless first, then linking the UI).
- **Phase 2 — Visage iOS windowing layer, ~2–4 wk** (or Path B UI rebuild if spike fails).
- **Phase 3 — Touch UX pass, ~1–2 wk.** Hit targets, gestures, knob/slider/keyboard tuning, multi‑
  touch.
- **Phase 4 — iOS integration, ~1–2 wk.** Presets via App Group, audio session, state, icons,
  launch screen, container/extension wiring.
- **Phase 5 — Host testing + performance, ~1–2 wk.** AUM / GarageBand / Cubasis on device; profile.
- **Phase 6 — App Store, ~1 wk + review.** App Store Connect, signing, (IAP), metadata, screenshots,
  submission + review iteration.

**Rough total (solo, experienced JUCE/C++/iOS dev): ~7–14 weeks** for Path A, gated by the Phase 0
spike. Path B replaces Phase 2 with a larger UI rebuild (~3–6 wk of UI work) plus an ongoing
two‑UI maintenance tax. Estimates are coarse and will be recalibrated after the spike.

---

## 11. Decisions needed before starting

1. **UI strategy** — Path A (port Visage to iOS, recommended) vs Path B (rebuild UI in JUCE)?
   (Default: run the Phase 0 spike, then decide.)
2. **Monetization** — paid app vs free + IAP unlock?
3. **Device scope** — iPad‑only, or universal (iPhone too)? Minimum iOS version (proposed: 15)?
4. **Apple account ownership** — use Benji's "benjamin vaughan / Heathen Machines" team for App
   Store Connect, or a separate Heathen Machines org account?
5. **Who builds it** — same dev(s), or is this a separate workstream/handoff?
