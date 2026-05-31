# Holy Shifter — Revision Specs

Status: implemented in build v0.1.1 (2026-05-24). Source of truth for this round of changes.

## Scope & locations

All work is in the source repo: `plugin/src/`.

| Abbrev | File |
|---|---|
| **PP** | `plugin/src/PluginProcessor.cpp` |
| **PPH** | `plugin/src/PluginProcessor.h` |
| **SD** | `plugin/src/dsp/SpectralDelay.h` |
| **UI** | `plugin/src/ui/HolyShifterUI.cpp` |
| **UIH** | `plugin/src/ui/HolyShifterUI.h` |

Stable plugin identity — do **not** change: AU `aufx Fshf Hrmt`, bundle `com.harmonictools.frequencyshifter`. Build/validate with `auval -v aufx Fshf Hrmt` and the turnkey `plugin/scripts/build_and_notarize.sh`.

Line numbers below are anchors at time of writing; confirm by symbol if the file has shifted.

---

## R1 — Quantize: remap slider so 0 = 0.2

**Intent.** The Quantize slider keeps its 0–100 travel, but its range is remapped so **slider 0 → internal 0.2** (what 20% gives today) and **slider 100 → internal 1.0** (full). Default = slider at 0, so a fresh instance quantizes at 0.2. Linear across the travel — no dead zone. There is intentionally no setting below 0.2.

**Current behavior.**
- `PARAM_QUANTIZE_STRENGTH`, range 0–100 % with quadratic display skew, **default `100.0f`** (`PP:122-140`).
- Listener stores `newValue / 100` → internal 0–1 (`PP:510-512`).
- Consumed (smoothed) at `PP:1156-1157`, passed to `quantizer->quantizeSpectrum(…)` at `PP:~1703`. **Spectral-mode only.**

**Change.**
1. `PP:139` — default `100.0f` → **`0.0f`** (slider rests at 0).
2. Remap at point of use, after the existing block-rate smoother: `effectiveQuant = 0.2f + 0.8f * currentQuantizeStrength`, and pass `effectiveQuant` to `quantizeSpectrum` (`PP:~1700-1703`). Apply at every consumption site (currently one).
   - Rationale for consumer-side remap: the stored param value stays the raw 0–100 slider position (preset-safe) and it's robust to listener init order. Equivalent alternative: remap in the listener (`PP:510-512`) and initialise the atomic to `0.2f`.

**Consequences (intended).**
- Slider still displays 0–100 %, but that number is the **slider position**, not the literal quantize fraction (display 0 % = 0.2 applied; 100 % = full).
- Minimum effective quantize is 0.2; no "off" below it. A true off would need a separate enable (out of scope).
- The quadratic skew now gives fine resolution just above 0.2.

**Acceptance.** Fresh instance, slider at 0 → audible 0.2 quantize in Spectral; slider at 100 → full quantize; smooth sweep, no clicks; presets restore slider position.

---

## R2 — Delay "Diffuse": true diffusion, no added feedback

**Background — two feedback paths in Spectral mode.** (Established by reading `PP:1585-1793`.)
1. **Time-domain feedback loop** — feedback is read and added to the input *before* the FFT (`PP:1596-1652`, scaled by the Feedback knob `currentFeedbackAmount`), and the processed output is written back *after* the IFFT through a 150 Hz HPF + one-pole damping LPF (`PP:1744-1778`). **This is the feedback heard at Diffuse 0** and is independent of Diffuse.
2. **SpectralDelay per-bin delay + feedback** (`PP:1717-1722`). Its audible contribution is gated by the Diffuse/`mix` crossfade (`SD:189-199`); its internal regeneration is driven by the *same* Feedback knob (`PP:1127`).

**Why diffusion currently runs away.** Raising Diffuse blends in path-2's delayed, self-regenerating copy, and since the SpectralDelay output also feeds the time-domain write-back (`PP:1738`→`1754`), that energy recirculates through path 1 and **compounds** → "diffusion creates too much feedback."

**Intent.** Diffuse smears/disperses the sound (and the feedback tail) **without raising its level** — energy-preserving, adds zero feedback. At Diffuse 0, behavior is exactly as today (clean path-1 feedback). Slope keeps doing per-bin frequency-dependent delay; its delayed copy may recirculate and add *some* feedback — accepted.

**Change.**
1. **Remove the SpectralDelay's internal feedback.** At `PP:1127`, stop passing the knob into `setFeedback` — keep it at 0 (matches the original intent already noted at `PP:885`, "Disable spectral delay internal feedback"). This kills the runaway/compounding regeneration; the time-domain loop (path 1) is the sole feedback engine.
2. **Make Diffuse an energy-preserving smear** instead of a wet/dry crossfade (`SD:189-199`): drive a per-bin **phase decorrelation** from Diffuse (0 = no smear / identity, 100 = full decorrelation), optionally a small magnitude bin-spread. Because energy is preserved, what recirculates through path 1 is *diffused but not louder*. Rename `setMix`→`setDiffusion` (or keep the setter, change semantics); update the call at `PP:1129`.
3. **Keep Slope unchanged** (`setFrequencySlope` / `computeDelayTimes`, `SD:95-100, 233-255`). Per-bin delay still happens; its delayed signal remains the SpectralDelay's output contribution. The author accepts this can re-introduce some feedback via path-1 recirculation.

**Open implementation detail (finalize by ear).** Diffuse no longer controls the wet/dry blend of the per-bin-delayed signal, so the SpectralDelay needs a defined wet level for that delayed (Slope-shaped) signal. Recommended: keep the existing dry/wet crossfade structure but drive its wet amount from a fixed internal constant (start at the prior default ~0.5) decoupled from Diffuse, and apply the Diffuse phase-smear on top. Alternative: sum the delayed signal additively at the existing delay `Gain` level (`SD:134`). Pick by listening; this is the one spot not pinned down.

**Diffuse default → 0** (clean unless dialed in). Update the param default for `PARAM_DELAY_DIFFUSE` accordingly. Range stays 0–100 %.

**UI / scope.** Keep `delayDiffuseSlider_` and the "Diffuse" label (`UI:175`). Spectral-only — greyed in Classic (see R4).

**Acceptance.** With Feedback fixed, sweeping Diffuse 0→100 must not increase output level or lengthen the feedback tail — only smear. Diffuse 0 = clean repeats exactly as today. Slope may add some feedback (accepted, not a regression).

---

## R3 — On/off toggles for both LFOs

**Intent.** Each LFO (Freq Modulation and Delay Modulation) gets an enable toggle that behaves exactly like the Delay's, independent of depth.

**Reference pattern (Delay enable).** `PARAM_DELAY_ENABLED` constant (`PPH:94`) → `AudioParameterBool` default `false` (`PP:383-387`) → `std::atomic<bool> delayEnabled` (`PPH:219`) → registered/handled in `parameterChanged` (`PP:~628-630`) → `delayEnabledToggle_` (`UIH:94`, bound `UI:154`) → DSP gate `delayEnabled.load()` (`PP:1171`).

**Change — two new bool params mirroring that pattern.**

| New param / atomic | Drives | DSP gate |
|---|---|---|
| `PARAM_LFO_ENABLED` / `lfoEnabled` | Freq-shift LFO | wrap the Freq-LFO block `PP:1204-1298`; when off, force LFO output to 0 (no modulation) |
| `PARAM_DLY_LFO_ENABLED` / `dlyLfoEnabled` | Delay-time LFO | wrap the Delay-LFO block `PP:1303-1379`; when off, `modulatedDelayTimeMs` = base delay time (`PP:~1382`) |

Per param: add the constant in **PPH** (next to `PARAM_LFO_*` / `PARAM_DLY_LFO_*`), `AudioParameterBool` default **`false`** in `createParameterLayout`, atomic in **PPH**, register + handle in `parameterChanged`, read `…Enabled.load()` once per block beside the other LFO reads (`PP:~1163-1170`).

**UI.** Add `lfoEnabledToggle_` and `dlyLfoEnabledToggle_` (UIH), styled like `delayEnabledToggle_` (empty label = strip header), bound next to the existing LFO bindings (`UI:129-151`, `UI:186-202`), placed as each section's header toggle. When a toggle is off, dim that LFO's own controls (depth/rate/shape/sync) via `setDimmed()`, refreshed alongside `updateControlsForMode()` / `pollState()`.

**Default = OFF** (matches the Delay). ⚠️ Backward-compat: any saved preset/session with LFO depth > 0 will load with the LFO **off** (new param defaults off). If existing presets must keep modulating, default these **ON** instead.

**Acceptance.** Toggle off → LFO has zero effect even with depth up. Toggle on → modulates as before. Toggles persist in state and automate.

---

## R4 — Grey Slope + Diffuse in Classic mode

**Intent.** In Classic, grey only the delay controls that are Spectral-only. Keep Time, Feedback, Sync, Damping (newly wired in by R5) and Delay Modulation active.

**Current behavior.** `updateControlsForMode()` (`UI:356-372`) dims only spectral-*panel* controls. No delay control is greyed. Classic genuinely ignores Slope/Diffuse/Gain — the Classic path (`PP:1424-1567`) reads only enable, time, feedback.

**Change.** In `updateControlsForMode()` add:
```cpp
delaySlopeSlider_.setDimmed(isClassic);   // UI:172
delayDiffuseSlider_.setDimmed(isClassic); // UI:175
```
Leave `delayTimeSlider_`, `delayFeedbackSlider_`, `delaySyncToggle_`, `delayDampingSlider_`, and the Delay-Modulation controls active in both modes. (These two sliders live in the delay strip, not the spectral panel box, so dimming them is independent of the panel dimming.)

**Acceptance.** Switching to Classic dims Slope + Diffuse only; Time/Feedback/Sync/Damping/Delay-Mod stay bright and functional. Switching back to Spectral restores them.

---

## R5 — Damping on the Classic-mode delay feedback

**Intent.** The Damping knob should shape the Classic feedback tone the same way it does in Spectral (darker repeats as Damping rises).

**Current behavior.** Classic feedback filtering is HPF (150 Hz) → **fixed 12 kHz 4th-order Butterworth LPF** (`classicFbLpfCoeffs`, computed once at prepare `PP:1006-1036`, applied `PP:1514-1534`). Damping drives only `feedbackFilterCoeff` (`PP:1136-1138`), which is used **only in the Spectral** write path (`PP:1774`). So Damping does nothing in Classic today.

**Change.** Make the Classic feedback LPF track Damping with the same mapping Spectral uses: `cutoff = 12000 · (1000/12000)^(damping/100)` → 0 % = 12 kHz, 100 % = 1 kHz (`PP:903-904`, `PP:1136-1138`). Recommended (lowest risk): insert the already-damping-driven one-pole stage into the Classic feedback chain — after the 150 Hz HPF (`PP:1508`) and before the buffer write (`PP:1537`): `lpf = prev + feedbackFilterCoeff * (lpf - prev)`, exactly like `PP:1774`. `feedbackFilterCoeff` is already recomputed on Damping change (`PP:1138`), so no new coefficient plumbing. Add/reset a per-channel one-pole state beside the existing Classic feedback states (`PP:1039-1043`). (Alternative: recompute `classicFbLpfCoeffs` from the damping cutoff in the `delayNeedsUpdate` block — heavier, keeps the 4th-order slope.)

**Acceptance.** Classic with feedback up: Damping 0 % → bright repeats (~12 kHz), 100 % → progressively darker (~1 kHz), smooth and stable (no runaway, no zipper noise).

---

## Warm — no change (reference only)

Left as-is per decision. Warm is a 2-pole Butterworth low-pass at **4500 Hz**, Q = 0.707, applied to the **wet signal only** before the Dry/Wet mix, in all three branches (`PP:906-928`; applied `PP:1816` Classic, `:1920` Spectral, `:2010` switching). It is correctly wired — its gentleness comes from being a wet-only high-cut.
