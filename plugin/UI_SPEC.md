# Holy Shifter — UI Spec & Build Reference

> Exact-value reference for matching the plugin's Visage UI to the current Figma design.
> Source design: **Holy Shifter UI** — `figma.com/design/827bxfu41L9O6P3sxhApOQ` (frame `27:2`, 700×928).
> Generated 2026-05-31 from Figma node values + the live source tree. Colors/geometry verified against `src/ui/HolyTheme.h`.

## TL;DR — how close are we already?

The active UI (`HolyShifterUI`) was built from an **earlier rev of this same design**, so most of it already matches to the pixel.

- ✅ **Colors** — every palette value already matches the Figma exactly (see token table).
- ✅ **Geometry constants** — slider 1.5px, thumb Ø7, toggle 30×15, dot Ø11 all match.
- ✅ **Parameters** — every design control maps to an existing param (or runtime flag); no gaps.
- ❗ **Fonts** — the one substantive gap: UI renders with **system Helvetica/Arial**; design is **Inter + IBM Plex Mono**.
- ✅ **Secondary text color** — set to design `#8a857d` (still WCAG AA, ~5.4:1 on bg).
- ✅ **Enhanced / Diffuse removed** — both deleted from the design (2026-05-31); nothing to build or stub.
- ➕ **Two new enable toggles** — Freq-Mod and Delay-Mod gained header on/off toggles in the latest design rev.

---

## 1. Framework & canvas

- **JUCE 8.0.4 + Visage** (pinned `b0b2ee8…`), C++20. Editor = `VisageHostEditor` → `HolyShifterUI : visage::Frame`.
- **Canvas: 700 × 928** logical, matches Figma 1:1. **Window resizable, default 80% (560 × 742)**, aspect locked 700:928, via a single Visage content scale — all control coords stay in 700×928 space; Figma remains the 100% reference.
- Render idiom: per-frame `draw(visage::Canvas&)` using `fill / roundedRectangle / roundedRectangleBorder / circle / roundedArc / segment / text`. 30 Hz timer drives `pollState()` → `redrawAll()`.
- Param binding: `VisageParamAttachment` (one APVTS param ↔ one control). `holy::dimColor(c, dimmed)` scales alpha to 25% for disabled sections.

## 2. Design tokens — colors (Figma → current code)

| Role | Figma hex | Code constant (`HolyTheme.h`) | Status |
|---|---|---|---|
| Header/toolbar row bg | `#0c0c0e` | `HolyModeSelectorBg 0xFF0C0C0E` | ✅ exact |
| Section (strip) bg | `#0e0e10` | `HolyStrip 0xFF0E0E10` | ✅ exact |
| Panel gradient top | `#19191d` | `HolyPanelGradTop 0xFF19191D` | ✅ exact |
| Panel gradient bottom | `#101013` | `HolyPanelGradBot 0xFF101013` | ✅ exact |
| Divider line | `#1a1a1d` | `HolyStripBorder 0xFF1A1A1D` | ✅ exact |
| Panel border | `#1c1c20` | `HolyPanelBorder 0xFF1C1C20` | ✅ exact |
| Control track / toggle-off | `#252320` | `HolyTrack 0xFF252320` | ✅ exact |
| **Gold accent** | `#c9a96e` | `HolyAccent 0xFFC9A96E` | ✅ exact |
| Accent glow | `~15% #c9a96e` | `HolyAccentGlow 0x26C9A96E` | ✅ exact |
| Primary text / values | `#e8e4db` | `HolyText 0xFFE8E4DB` | ✅ exact |
| Dim text (subtitle, unit, caret) | `#3e3a34` | `HolyTextMuted 0xFF3E3A34` | ✅ exact |
| Black-key fill | `#0a0a0c` | `colors::background 0xFF0A0A0C` | ✅ exact |
| Black-key border | `#151517` | `HolyBorderDim 0xFF151517` | ✅ exact |
| Segmented active fill | `rgba(107,93,61,.5)` | (drawn inline) | verify |
| Segmented active border | `rgba(201,169,110,.4)` | (drawn inline) | verify |
| Keyboard white-key tint | `rgba(201,169,110,.2)` | (drawn inline) | verify |
| Secondary label text | `#8a857d` | `HolyTextSec 0xFF8A857D` | ✅ matched (was `#b8b2a6`; AA ~5.4:1) |

## 3. Typography ❗ (the main gap)

Design uses **Inter** (Thin/Light/Regular/Medium/SemiBold) + **IBM Plex Mono Light**. Both are free/OFL → embeddable for an exact match. Code currently loads **system Helvetica/Arial** via `holy::makeFont()` — no embedded fonts, no BinaryData.

| Role | Font / weight | Size | Tracking | Color |
|---|---|---|---|---|
| Title `HOLY  SHIFTER` | Inter Thin | 26 | 9.1 | `#e8e4db` |
| Subtitle | Inter Regular | 10 | 0.5 | `#3e3a34` |
| Section label (`DELAY`…) | Inter Medium | 8 | 1.6 | `#c9a96e` |
| Body label (`Time`…) | Inter Regular | 10 | — | `#8a857d` |
| **Value readout** (`200.0 ms`) | **IBM Plex Mono Light** | 10 | — | `#e8e4db` |
| Toggle label (`WARM`/`Sync`/`L/R Decorr`) | Inter Medium | 9–10 | 0.5–1.35 | `#8a857d` / `#c9a96e` |
| Segment active / inactive | Inter SemiBold / Medium | 10 | 1.5 | `#c9a96e` / `#3e3a34` |
| Knob value `0` / unit `HZ` | Inter Light 34 / Inter Medium 9 | — | —/2.25 | `#e8e4db` / `#3e3a34` |
| Helper note (Smear) | Inter Light | 8 | 0.4 | `#8a857d` |

**Work:** bundle TTFs (`Inter-Thin/Light/Regular/Medium/SemiBold`, `IBMPlexMono-Light`), load via Visage/BinaryData, refactor `makeFont(size)` → `makeFont(size, weight, family)`.

## 4. Component patterns (theme values already match)

| Pattern | Spec | Code value | Status |
|---|---|---|---|
| Slider track | h 1.5px, r 0.75, `#252320` | `HolySliderHeight 1.5` | ✅ |
| Slider active fill | same h, `#c9a96e`, width = value% | — | ✅ |
| Slider thumb | Ø7 gold dot + soft glow | `HolyThumbSize 7.0` | ✅ |
| Bipolar slider (Sens) | fill from center | — | verify |
| Toggle | pill 30×15 r7.5 `#252320`; dot Ø11 | `HolyToggleWidth/Height/DotSize 30/15/11` | ✅ |
| Knob | 270° arc (2.5 width), radial glow, 5 ticks, Ø12 indicator | `HolyKnobArcWidth 2.5` | ✅ |
| Segmented control | 220×26 r4 `#252320`; active half translucent-gold | — | verify |
| Spectral panel | **vertical** gradient `#19191d`→`#101013`, border `#1c1c20`, r6; top hairline fades transparent→gold(12%)→transparent | colors defined, but **panel drawn FLAT** (top color only); hairline flat | ⚠️ |
| Keyboard | 400×42 r3; 7 white keys (gold-tint, 55.1px) + 5 black (`#0a0a0c`, 31.4px, h26) | — | verify |

## 5. Section layout (Figma Y-offsets, frame `27:2`)

| Section | x | y | w | h |
|---|---|---|---|---|
| Title / Subtitle | 28 | 14 / 46 | — | — |
| Preset bar (`◂ ▸ Default`) | 37–92 | 70 | — | — |
| Mode Selector (segmented + Warm) | 0 | 102 | 700 | 36 |
| Shift Knob | 28 | 169 | 210 | 218 |
| Spectral Panel | 245 | 158 | 430 | 240 |
| Freq Modulation | 0 | 406 | 700 | 110 |
| Delay | 0 | 518 | 700 | 134 |
| Delay Modulation | 0 | 652 | 700 | 92 |
| Mask | 0 | 762 | 700 | 90 |
| Mix Footer (Dry/Wet) | 0 | 852 | 700 | 72 |
| Logo | 608 | 18 | 53 | 53 |

## 6. Control → parameter mapping

All ✅ unless noted. Param IDs from `PluginProcessor.h`.

| Design control | Param ID | Type |
|---|---|---|
| Shift knob (Hz) | `shiftHz` | Float ±20000 (knob shows ±5000 log) |
| CLASSIC / SPECTRAL | `processingMode` | Choice |
| WARM | `warm` | Bool |
| Note keyboard (12) | `scaleNote0..11` | Bool ×12 |
| Quantize | `quantizeStrength` | Float |
| Envelope *(labeled)* | `preserve` | Float |
| Transients | `transients` | Float |
| Sens | `sensitivity` | Float |
| Smear (ms) | `smear` | Float |
| Freq-Mod enable / Depth / Rate / Sync / wave | `lfoEnabled` / `lfoDepth` / `lfoRate` / `lfoSync` / `lfoShape` | — |
| Delay enable / Time / Sync / Feedback / Damping / Slope | `delayEnabled` / `delayTime` / `delaySync` / `delayFeedback` / `delayDamping` / `delaySlope` | — |
| L/R Decorr | *(runtime flag — non-APVTS)* | `setStereoDecorrelate()` |
| Delay-Mod enable / Depth / Rate / Sync / wave | `dlyLfoEnabled` / `dlyLfoDepth` / `dlyLfoRate` / `dlyLfoSync` / `dlyLfoShape` | — |
| Mask enable / type / Transition / Low / High | `maskEnabled` / `maskMode` / `maskTransition` / `maskLowFreq` / `maskHighFreq` | — |
| Dry / Wet | `dryWet` | Float |
| Preset prev/next/name | `PresetManager` | non-APVTS |

## 7. Punch-list (current → target)

1. **Fonts** — embed Inter (Thin/Light/Regular/Medium/SemiBold) + IBM Plex Mono Light via Visage's `add_embedded_resources(...)` → `visage::EmbeddedFile`, loaded with `visage::Font(size, file)` (same path Visage uses for its built-in fonts; compiled into the binary, no runtime paths). Refactor `makeFont(size)` → `makeFont(size, weight)`; update every text draw with the per-role weight/size/tracking from §3. **— WEIGHTS DONE** (embedded via `add_embedded_resources`; `makeFont(size, FontWeight)` wired across `HolyShifterUI` + all controls: slider readouts → Mono, knob value → Light, section labels/toggles → Medium, active segment → SemiBold, title → Thin, tooltip → Light). Sizes/positions deferred to the verify pass. **✓ VERIFIED** in a Standalone build — Inter (labels/title) + IBM Plex Mono (numeric readouts) render correctly.
2. **Secondary text color** — ✅ DONE: `HolyTextSec` / `colors::textSec` set to `#8a857d` (all active draws reference the constant; verified no other hardcoded instances).
3. **Freq-Mod & Delay-Mod enable toggles** — add header on/off toggles (bind to `lfoEnabled` / `dlyLfoEnabled`; both already exist) per the latest design rev.
4. **Per-section geometry pass** — ✅ **DONE & verified.** Spectral Panel repositioned `245/158, 430×240` (code `px/py` + child bounds updated); Smear row reflowed up; **"MIX" footer label removed** (footer = just "DRY / WET"); keyboard key fill reconciled to Figma's subtle gold tint (white-key on `0x33C9A96E`, border `0x26C9A96E`; black keys SOLID opaque + border `#151517`. Selection = **color change** (not a tab): ON = opaque `#2d2922` (the white-selected look ≈`#3c362d` — 20% gold over the panel — taken ~25% darker), OFF = `#0a0a0c`. NB: the earlier *translucent* black-key fill (`0x4DC9A96E`) let the white-key seam show through as a **split down the middle** — fixed by keeping the fill solid/opaque). Enable toggles already existed. Remaining nicety (optional): preset arrows are ASCII `</>` in code vs `◂ ▸` in Figma — could swap to `triangleLeft/Right`.
5. **Spectral panel gradient** — ✅ **DONE & verified.** Panel now a true vertical `Brush::vertical(#19191d → #101013)`; **Mix footer** same. (Top hairline still a flat ~12% gold bar — fine; the transparent→gold→transparent fade is a minor optional polish.)
   - ✅ **Section divider bars DONE & verified** — Figma `32:15` (y516, Freq-Mod↔Delay) & `32:18` (y760, Delay-Mod↔Mask) are **solid 2px `#48402F` full-width** bars. Code had drawn them via `drawGradientAccentLine` (transparent→gold→transparent, fading at the edges) so they didn't read as dividers. Now `setColor(0xFF48402F); fill(0, 516/760, w, 2)`.
   - ✅ **Section toggles aligned (all four) DONE & verified** — moved off the shared x160 column to their Figma x, each tucked right after its label: FreqMod **119**, Delay **55**, DlyMod **120**, Mask **56**.
   - ✅ **Mask dropdown text DONE & verified** — muted tan **`#a59777` @ 9px** via new per-instance `HolyComboBox::setTextColor()/setTextSize()` (Mask only; the shared waveform "Sine" combos keep defaults).
6. **Window sizing** — make the editor **resizable, 80% default (560×742)**, aspect locked to 700:928, min/max ≈60–130%. ⚠️ **DEFERRED — first attempt reverted.** FAILED approach: `setNativeWindowDimensions` + overwriting the content frame's `dpiScale` as a zoom factor → blank render. Root cause: in Visage the window/canvas `dpiScale` IS the display backing factor (drives the native layer `contentsScale`, the framebuffer, AND native-px→points), so repurposing it corrupts the surface. CORRECT path (per Visage `examples/ClapPlugin`): on macOS call `visageWindow_->setWindowDimensions(logicalW, logicalH)` (NOT `setNativeWindowDimensions`), keep `dpiScale` = true backing, and make the `HolyShifterUI` child fill the window — but the design uses fixed 700×928 child coords, so the child needs a content-scale transform / scaling flex root. Pair with `ApplicationWindow::setFixedAspectRatio`+`setMinimumDimensions` and JUCE `getConstrainer()->setFixedAspectRatio/setSizeLimits`. Currently reverted to known-good `setResizable(false,false)` + `setSize(700,928)`; `applyContentScale()` is a no-op stub. ⚠️ When testing the Standalone: kill ALL stale instances first — multiple instances fight over the GPU surface and render blank (NOT a code bug).

## 8. Assets

- **Logo** — `Heathen Machines` lockup. ✅ **DONE & verified.** Cleaned RGBA master (`~/Downloads/heathen-machines-logo-trimmed.png`, 3862×3987) downscaled to **`plugin/assets/heathen-machines-logo.png`** (248×256, ~100 KB), embedded via `add_embedded_resources(HolyShifterImages "holy_images.h" "holy::images")` → `holy::images::heathen_machines_logo_png`, drawn in `HolyShifterUI::draw()` with `canvas.image(file, 609, 18, 51, 53)` (white brush = untinted; aspect-preserved in the Figma 53×53 slot, node `174:40`). NB: the original Downloads files were fake-transparent (baked checkerboard) — these are the keyed RGBA versions. Only bitmap asset in the design; everything else is vector primitives.

## 9. Open decisions (unresolved)

- **Fonts:** embed Inter + IBM Plex Mono (exact) vs keep system fonts (approx). *Planned: embed.*
- ~~Secondary text~~ — resolved: matched design `#8a857d`.
- ~~Enhanced / Diffuse~~ — resolved: deleted from design.
