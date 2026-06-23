# Holy Shifter — Test Plan

Two layers:

1. **Objective harness** (`scripts/run_audio_tests.sh`) — loads the real VST3 via
   [pedalboard](https://github.com/spotify/pedalboard), runs synthetic reference signals,
   and prints a pass/fail table. This is the **regression net** — run it on every build.
2. **Manual Ableton suite** (below) — your ears judging "does it sound musical." The
   harness can't hear glassiness or groove; you can.

---

## 1. Objective harness

```bash
bash plugin/scripts/run_audio_tests.sh
# or test a specific build:
bash plugin/scripts/run_audio_tests.sh "/path/to/Some Build.vst3"
```

First run creates an isolated venv (`~/.cache/holyshifter-audiotest/`) and installs
pedalboard+numpy. All measurements are latency-robust (steady-state windows / peak-finding),
and pitch is read by **autocorrelation** (the audible pitch comes from the phase-vocoder's
phase advance, so a magnitude-FFT reading is misleading).

| Test | What it checks | Pass |
|------|----------------|------|
| T1 snap accuracy | off-scale sine lands on a scale note | <40¢ off-scale & moved |
| T2 stability (chatter) | boundary tone locks to ONE note (hysteresis) | <12¢ pitch wobble |
| T3 attack (transients) | percussion stays sharp (crest factor) | >3.5 |
| T4 level stability (pump) | steady chord holds a steady level | RMS CV <0.15 |
| T5 ringing/decay | output decays to silence after input stops | tail <−55 dB |
| T6 noise vs broadband | Noise knob doesn't reduce broadband | ratio ≥0.9 |

**v119 PeakSnap baseline: 6/6 pass.** Known characteristic (not a bug): peak-snap is
inherently tonal on *pure* broadband noise — even Noise=100 can't fully mask the snapped
peaks (T6 flatness stays low). Use lower Peak Sens / per-bin mode for very noisy sources.

Tune thresholds/signals in `scripts/audio_test_harness.py` as the algorithm evolves.

---

## 2. Manual Ableton suite (Spectral mode unless noted)

**Use the same fixed sample per source type every build** (save a test rack), and keep the
previous build (`v118 PhaseTex`) loaded for A/B. Shorthand: **Q**=Quantize, **N**=Noise,
**S**=Peak Sens, **Sm**=Smear, **Mix**=Dry/Wet.

### A — Peak-snap quality
| # | Source | Settings | Pass ✓ / Fail ✗ |
|---|--------|----------|------------------|
| A1 | Sustained vocal "ahh" / organ chord | PeakSnap **ON**, Q100, N30, S50, C major | ✓ steady clean in-tune tones ✗ fizzy/gritty shimmer |
| A2 | Held distorted-guitar chord | PeakSnap **ON**, Q100, N40, **S70** | ✓ pitches rock-steady ✗ bubbling/warble |
| A3 | Breathy flute / airy vocal | PeakSnap **ON**, Q100, sweep **N 0→100** | ✓ 0%=pure tones, 100%=air returns, smooth ✗ clicks |
| A4 | Rhodes / e-piano | PeakSnap **ON**, Q100, N50, sweep **S 0→100** | ✓ low=natural, high=in-key, smooth ✗ wobble at high S |

### B — Transients / attack
| # | Source | Settings | Pass ✓ / Fail ✗ |
|---|--------|----------|------------------|
| B1 | Congas / bongos (dry) | PeakSnap **ON**, Q100, N30, S50 | ✓ hits punch; ring snaps to scale ✗ soft/smeared |
| B2 | Snare / rimshot | PeakSnap **ON**, vary **Sm low→high** | ✓ low Sm snappier, high Sm softer but audible ✗ attack vanishes |
| B3 | Marimba / kalimba | PeakSnap **ON**, Q100, N40 | ✓ crisp ping **and** in-tune body ✗ mushy or detuned |

### C — Regression: per-bin must be unchanged
| # | Source | Settings | Pass ✓ / Fail ✗ |
|---|--------|----------|------------------|
| C1 | Pad | PeakSnap **OFF**, Q100, Preserve 50 | ✓ identical to v118 PhaseTex ✗ any difference |
| C2 | Sustained vocal | toggle PeakSnap **ON↔OFF** | ✓ OFF denser/glassier, ON cleaner+natural, both stable |

### D — Classic mode untouched
| # | Source | Settings | Pass ✓ / Fail ✗ |
|---|--------|----------|------------------|
| D1 | Any | **Classic**, Shift +200 Hz | ✓ same inharmonic Bode shift ✗ any change |

### E — Transport sync
| # | Source | Settings | Pass ✓ / Fail ✗ |
|---|--------|----------|------------------|
| E1 | Pad/chord | LFO **ON**, Sync **ON**, **1 bar**, Sine | ✓ bar-locked; identical sweep each replay; starts centered ✗ random phase |
| E2 | Pad | LFO **ON**, Sync **OFF** | ✓ starts at 0 every Play ✗ resumes mid-cycle |
| E3 | Rhythmic stab | Delay **ON**, Sync **ON**, 1/8 | ✓ taps on 1/8s; follows tempo |

### F — Edge cases
| # | Source | Settings | Pass ✓ / Fail ✗ |
|---|--------|----------|------------------|
| F1 | Chord that cuts off | PeakSnap **ON**, Q100 | ✓ clean decay ✗ lingering ring/tinnitus |
| F2 | Busy chord w/ movement | PeakSnap **ON**, N40 | ✓ steady level ✗ pumping/breathing (→ add anti-pump) |
| F3 | White noise / cymbal wash | PeakSnap **ON**, Q100, N30 | ✓ stays noise-like ✗ manufactures pitched tones |

---

## Workflow tips
- **Render each test to audio** so you can A/B builds by ear-comparing renders — even null-test
  (invert ON vs OFF) to confirm "per-bin unchanged."
- Keep **pluginval** in the loop for crash/automation/state regression:
  `pluginval --strictness-level 5 --skip-gui-tests --validate "<bundle>"`.
- The objective harness catches what ears miss between builds; the manual suite catches what
  numbers miss (musicality).
