# Envelope + Delay Feedback — Open Issue (handoff)

Status: **unresolved as of v0.1.7** (2026-05-30). Two implementations attempted, neither fixes the perceived problem. Recommend the next session **start with diagnostics** before another fix attempt — the mechanism is not yet pinned down.

---

## The problem

When **Envelope (Preserve)** is engaged AND **Delay** is enabled AND the **frequency shift is set above ~200 Hz**, the user perceives the feedback as building up / running away / becoming excessive.

**Reproduction steps:**
1. Spectral mode.
2. Enable Delay with some Feedback (e.g. 40–60%) and a moderate Delay Time (e.g. 200–400 ms).
3. Set the frequency Shift > ~200 Hz.
4. Push **Envelope past ~50%**.
5. Audible feedback becomes excessive.

**It does NOT happen when:**
- Delay is off (Envelope alone works correctly — captures source amplitude/articulation cleanly).
- Envelope is off (Delay+Feedback alone is stable at the same Shift/Feedback settings).

**Hard constraint from the user:** the fix must **not** change Envelope's behavior in the no-delay case. The Envelope is doing its job there and they like it.

---

## Code locations (paths are project-relative to `plugin/`)

| What | File:line |
|---|---|
| Envelope (Preserve) algorithm — spectral-only, runs in the per-sample mixing loop | `src/PluginProcessor.cpp:~1908–1955` |
| Spectral feedback **read** — adds `feedbackBuffers[ch][readPos] * feedback` to `inputSample` BEFORE the FFT (gated on `currentDelayEnabled && proc == 0`) | `src/PluginProcessor.cpp:~1596–1652` |
| Spectral feedback **write** — `outputSample` → 150 Hz HPF → damping LPF → `feedbackBuffers[ch]` (also gated on `currentDelayEnabled && proc == 0`) | `src/PluginProcessor.cpp:~1744–1778` |
| Envelope state arrays + coefficients | `src/PluginProcessor.h:~263–268` |
| Envelope coefficients computed in `prepareToPlay` | `src/PluginProcessor.cpp:~725–736` |
| SpectralDelay (per-bin) — currently `setFeedback(0)`, fixed internal wet `mix = 0.5` | `src/dsp/SpectralDelay.h` |
| Quantizer (where shifts > 200 Hz happen) | `src/dsp/MusicalQuantizer.*` |

Plugin identity (stable, do not change): AU `aufx Fshf Hrmt`, bundle `com.harmonictools.frequencyshifter`. Build target names: `FrequencyShifter_VST3`, `FrequencyShifter_AU`. Build script `plugin/scripts/build_and_notarize.sh` for shipping; for iteration just `cmake --build build --target FrequencyShifter_VST3 FrequencyShifter_AU --parallel 8` (auto-installs to `~/Library/Audio/Plug-Ins/`). Bump `PROJECT_VERSION` in `plugin/CMakeLists.txt` so Ableton re-scans; then `killall -9 AudioComponentRegistrar` and `/usr/bin/auval -v aufx Fshf Hrmt`. **JUCE codesign gotcha:** after the build, `xattr -cr` the AU bundle and `codesign --force --sign -` both the inner binary and the bundle.

---

## What's established about the signal flow

Spectral-mode per-sample flow (within `processBlock`):

1. **Proc loop** (per proc, per sample, only proc 0 touches feedback):
   - `inputSample = drySignal[i]`
   - If delay on: `inputSample += feedbackRead * Feedback` (the only feedback that recirculates).
   - Write to inputBuf; on FFT-frame boundary: FFT → quantize → mask → SpectralDelay → IFFT → overlap-add into outputBuf.
   - `outputSample = outputBuf[readPos]` (then zero, advance).
   - If delay on: `outputSample` → 150 Hz HPF → damping LPF → write to `feedbackBuffers[ch]`. **This write uses `outputSample` PRE-Preserve.**
   - `procOutput[i] = outputSample`.
2. **Mixing loop** (per channel, per sample, runs AFTER the entire proc loop for that channel):
   - `wetSample = delayCompBuffers[ch][readIdx]` (delay-compensated `procOutput`).
   - If `currentPreserve > 0.01`: compute `gainCorrection = inputEnv / outputEnv`, clamp, `wetSample *= blendedCorrection`.
   - Warm filter.
   - `channelData[i] = mix(delayedDry, wetSample, dryWetMix)`.

**Key established fact:** Envelope's gain is applied to `wetSample` in the mixing loop, AFTER `outputSample` has been written to `feedbackBuffers`. So Envelope **does not compound the loop's regenerative gain** — the loop gain is governed entirely by the Feedback knob, the 150 Hz HPF, and the damping LPF.

**Discrepancy worth noting:** that established fact and the user's report ("feedback runs away with Envelope") are in tension. The next investigator should not assume Envelope is multiplying loop gain — verify what's actually happening before treating it as a feedback-runaway problem. See "Diagnostics to run first" below.

---

## Attempts that did NOT work

### Attempt 1 — Source-locked Preserve (v0.1.6)

Modified Envelope itself:
- Added a per-channel **input peak follower** (instant attack, ~400 ms release): `inputPeak = max(|dry|, inputPeak * envPeakReleaseCoeff)`.
- Computed `sourceGate = clamp(inputEnvelope / inputPeak, 0, 1)` — ~1 when source near its peak, decays as source falls.
- Gated **only the upward (energy-restoring) part** of `gainCorrection` by `sourceGate`; downward correction left at full.
- Reduced boost cap from 4× to 2×.

Rationale: relax the boost as the source decays so the feedback tail isn't propped up; cap worst-case amplification.

User feedback: didn't fix the in-note feedback amplification, and changed Envelope's behavior (which the user explicitly didn't want). **Reverted in v0.1.7.**

### Attempt 2 — Feedback-loop compensation (v0.1.7, current)

Reverted Envelope to its original algorithm (clamp [0.25, 4.0], `pow(preserve, 0.7)` shape, no source gate). Then added:
- A per-channel `lastPreserveGain[ch]` state, updated in the mixing loop after each Preserve correction (`= blendedCorrection`); set to 1.0 when Preserve is off.
- In the **spectral feedback write** (`PP:~1744`), scaled `outputSample` by `1/lastPreserveGain` (only when `> 1.0`; identity otherwise): `float x0 = outputSample * fbCompensation;`.

Rationale (mathematical): if Envelope post-multiplies the wet by G, scale the loop input by 1/G. After the loop's own regeneration and Envelope's post-loop boost, the audible feedback level should be the same as Envelope-off. Direct signal still source-matched. Identity when Envelope is off or delay is off, so the no-delay case is byte-for-byte unchanged.

User feedback: this didn't work either. **The mechanism causing the perceived problem is not the simple "Envelope multiplies the audible feedback level" model that this fix assumed.**

---

## Diagnostics to run first (before another fix attempt)

The two attempts assumed the user's "feedback" perception was caused by Envelope amplifying the wet signal's feedback content. Both failed, which suggests the mechanism is something else. Before changing code, **characterize the actual behavior**:

1. **Bisect with the Feedback knob.** With Envelope > 50%, Delay on, Shift > 200 Hz, **set Feedback to 0** — does the "runaway feedback" perception remain? If yes, the issue is not the time-domain feedback loop at all.
2. **Bisect with Shift.** Hold Envelope and Delay constant; sweep Shift from 0 up. At what Shift value does the problem onset? Is it gradual or threshold-like at ~200 Hz?
3. **Bisect with SpectralDelay.** The SpectralDelay (per-bin, `SpectralDelay.h`) currently has `setFeedback(0)` but still blends a 50% delayed wet (`mix = 0.5` fixed). Temporarily set `mix = 0` and rebuild — does the issue go away?
4. **Instrument with DBG logs** in `processBlock` (the codebase already uses `DBG()` — see `PP:~1639–1651`). Log per-second: `inputEnvelope[0]`, `outputEnvelope[0]`, `blendedCorrection`, `|outputSample|` peak, `|wetSample|` peak post-Preserve, `feedbackBuffers[0][writePos]` peak. Run with the problematic settings and watch the numbers actually evolve.
5. **Controlled signal.** Feed a short tone burst (not music). Does the feedback behaviour follow expectation, or do you see runaway in the logs?
6. **Confirm the user's "feedback" perception literally.** Is it (a) progressively louder echoes, (b) a sustained drone/howl, or (c) a sudden level spike? Each implicates a different mechanism.

---

## Hypotheses to consider (ranked roughly by likelihood given current evidence)

1. **The "feedback" is the spectral delay's per-bin blend interacting with shift accumulation.** Even though `setFeedback(0)`, the `mix = 0.5` per-bin blend adds a 200 ms-ago shifted spectrum each frame; combined with the time-domain feedback loop's barber-pole cascade, the spectral content piles up at shifted frequencies. Envelope then amplifies whatever's there.
2. **At shift > 200 Hz, the shifter loses significant per-pass energy.** Envelope's `inputEnv/outputEnv` ratio is consistently high → near-max boost → audible level dominated by amplified content. The compensation fix reduces loop input but Envelope's measurement of `outputEnv` (which contains the partial result) may have lag/dynamics that cause incomplete cancellation.
3. **Per-block lag of `lastPreserveGain`.** It's updated in the mixing loop (runs after the proc loop for the block), so the feedback write uses the previous block's gain. For fast envelope changes, the compensation lags.
4. **Quantizer behavior at large shifts.** The MusicalQuantizer may snap shifted bins to scale notes in a way that concentrates energy, and Envelope amplifies that concentration. Worth instrumenting.
5. **It's actually not new feedback — it's the existing feedback content being made *audible* by Envelope** (which the user perceives as "getting feedback"). If true, no math compensation will satisfy them; the answer is to either change the *qualitative* sound (e.g. attenuate the feedback path more, add a limiter) or accept that Envelope + delay is loud.

---

## Constraints / acceptance criteria for any fix

- **Envelope behavior in the no-delay case must be identical** to the current (v0.1.7) behavior — the user has confirmed it works there.
- With delay on + Envelope on + shift > 200 Hz: no audible feedback buildup / runaway-feel.
- Source amplitude/articulation matching must continue to work when delay is on (i.e. the *direct* shifted signal still tracks the source's envelope).

---

## Useful prior context

- `REVISIONS.md` (same directory) — the spec for the earlier round of work (R1–R5), all of which shipped and were confirmed by the user.
- `CLAUDE.md` in the source repo root and in the `…/FrequencyShifter-All-Platforms 3/` packaging workspace — codebase orientation, build/sign workflow, plugin identity.
- The two failed attempts are in git history at v0.1.6 and v0.1.7. The compensation code added at v0.1.7 is still present in `PP:~1744` and `PP:~1908–1955` — decide whether to keep, remove, or evolve it before the next iteration.

---

## Suggested first move for the next session

Don't immediately code. Run diagnostic #1 (Feedback=0 with Envelope+Shift>200) and #4 (DBG instrumentation) to **observe what's actually happening** in the audio path. The fact that two principled fixes both failed is strong evidence the mental model is off — pin the mechanism down first.
