# Holy Shifter v107 — Test Suite Documentation

> Auto-generated from test source files. All tests run against the **actual compiled VST3 binary** via [pedalboard](https://github.com/spotify/pedalboard).

## Summary

| Test File | Purpose | Classes | Tests | Tier |
|-----------|---------|---------|-------|------|
| test_plugin_basic.py | Plugin loading, bypass, safety | 4 | 15 | 1: Foundation |
| test_plugin_spectral.py | Spectral (FFT) mode processing | 7 | 15 | 2: Core |
| test_plugin_classic.py | Classic (Hilbert) mode | 3 | 8 | 2: Core |
| test_plugin_signals.py | Square waves, impulses, sweeps | 5 | 12 | 2: Signals |
| test_plugin_edge_cases.py | Extreme params, sample rates | 3 | 10+ | 3: Stress |
| test_plugin_lfo_depth.py | LFO depth streaming behavior | 4 | 12 | 3: Regression |
| test_plugin_quality_gates.py | Release criteria | 1 | 8 | 4: Release |
| **Total** | | **27** | **~80** | |

**Last run:** 89 passed, 1 known failure (22050 Hz), 1 xfailed, 1 xpassed

---

## Fixtures (conftest.py)

### Plugin Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `plugin` | session | Shared VST3 instance (fast, has state between tests) |
| `fresh_plugin` | function | Fresh VST3 instance per test (clean state) |

### Signal Fixtures

| Fixture | Signal | Use Case |
|---------|--------|----------|
| `sine_440` | 440 Hz sine, 1s | Pitch reference |
| `sine_1000` | 1000 Hz sine, 1s | Mid-range test |
| `square_440` | 440 Hz bandlimited square | Rich odd harmonics |
| `impulse` | Single-sample pulse | Impulse response |
| `pulse_train` | 10 Hz pulse train | Periodic transients |
| `silence` | 1s zeros | Null test |
| `dc_offset` | Constant 0.5 | DC handling |
| `white_noise` | Seeded random (seed=42) | Broadband |
| `two_tone` | 440 + 880 Hz | Harmonic pair |
| `sweep` | 20 Hz → 20 kHz chirp, 2s | Full range |
| `piano_like` | 440 Hz + decaying harmonics | Realistic polyphonic |

### Helpers

| Function | Returns | Purpose |
|----------|---------|---------|
| `peak_frequency(signal, sr)` | Hz | Dominant frequency via FFT |
| `peak_frequencies(signal, n, sr)` | [Hz] | Top N peaks |
| `rms_db(signal)` | dB | RMS level |
| `has_nan_or_inf(signal)` | bool | Safety check |
| `energy_ratio_db(out, in)` | dB | Energy change |
| `spectral_energy_above(signal, freq)` | dB | HF energy |
| `spectral_energy_below(signal, freq)` | dB | LF energy |
| `dc_component_db(signal)` | dB | DC magnitude |
| `process_with_params(plugin, signal, **p)` | ndarray | Process with params, handles mono/stereo |

---

## Tier 1: Foundation — test_plugin_basic.py

### TestPluginLoads

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_plugin_loads` | VST3 binary loads without crash | Plugin is not None |
| `test_plugin_has_shift_parameter` | Shift (Hz) knob exposed | `shift_hz_hz` in parameters |
| `test_plugin_has_dry_wet` | Dry/Wet knob exposed | `dry_wet` in parameters |
| `test_plugin_has_mode` | Mode selector exposed | `mode` in parameters |
| `test_plugin_has_quantize` | Quantize knob exposed | `quantize` in parameters |
| `test_plugin_has_feedback` | Feedback knob exposed | `feedback` in parameters |

### TestBypass

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_bypass_sine_440` | Zero shift preserves pitch | Peak at 440 Hz (±10 Hz) |
| `test_bypass_preserves_energy` | Zero shift preserves level | RMS change < 3 dB |
| `test_reset_produces_same_output` | Plugin reset clears state | Two consecutive runs produce identical output |

### TestSilence

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_silence_in_silence_out` | Silent input → silent output | Output < -90 dB |
| `test_noise_floor` | Noise floor with shift active | Output < -80 dB |

### TestSafety

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_no_nan_on_sine` | No NaN from sine | No NaN/Inf |
| `test_no_nan_on_impulse` | No NaN from impulse | No NaN/Inf |
| `test_no_nan_on_noise` | No NaN from white noise | No NaN/Inf |
| `test_output_bounded` | Peak amplitude safe | Peak < +12 dBFS |
| `test_dc_not_amplified` | DC input doesn't explode | Peak < +12 dBFS |

---

## Tier 2: Core — test_plugin_spectral.py

Spectral mode uses FFT (bin resolution ~10 Hz at 44.1kHz/4096). Tolerances are ±15 Hz for single tones, ±30 Hz for complex signals.

### TestFrequencyShifting

| Test | Input | Shift | Expected Output | Tolerance |
|------|-------|-------|-----------------|-----------|
| `test_shift_sine_up_100` | 440 Hz | +100 | 540 Hz | ±15 Hz |
| `test_shift_sine_down_200` | 1000 Hz | -200 | 800 Hz | ±15 Hz |
| `test_shift_two_tones` | 440+880 Hz | +50 | 490+930 Hz | ±30 Hz |
| `test_shift_zero_is_identity` | 440 Hz | 0 | 440 Hz | ±15 Hz |
| `test_shift_large_positive` | 440 Hz | +5000 | ~5440 Hz | ±30 Hz |
| `test_shift_large_negative` | 1000 Hz | -800 | ~200 Hz | ±15 Hz |
| `test_shift_preserves_energy` | 440 Hz | +100 | RMS ≈ input | < 6 dB change |

### TestQuantization

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_quantize_snaps_to_scale` | Quantize snaps detuned tone to C major | Snaps to nearest scale note |
| `test_quantize_strength_zero_no_effect` | Zero strength = no quantization | Unquantized frequency (±30 Hz) |
| `test_quantize_no_nan` | No NaN with quantize on noise | No NaN/Inf |

### TestSpectralMask (xfail)

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_mask_lowpass` | LP mask cuts above 3 kHz | HF energy drops ≥10 dB |
| `test_mask_highpass` | HP mask cuts below 1 kHz | LF energy drops ≥10 dB |

### TestSpectralDelay

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_delay_produces_output` | Delay+feedback creates repeats | Late energy > -80 dB |

### TestPhaseVocoder

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_phase_vocoder_on_still_shifts` | PV on: shift works | Peak ~540 Hz (±30 Hz) |
| `test_phase_vocoder_off_still_shifts` | PV off: shift works | Peak ~540 Hz (±30 Hz) |

### TestSmear

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_smear_small` | 5ms FFT (small window) works | No NaN, RMS > -30 dB |
| `test_smear_large` | 93ms FFT (fine resolution) works | Peak ~540 Hz (±30 Hz) |

### TestDryWet

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_dry_wet_50_has_both_frequencies` | 50% mix has original + shifted | Both ~440 Hz and ~640 Hz present |

---

## Tier 2: Core — test_plugin_classic.py

Classic mode uses Hilbert transform (8-stage allpass networks). Near-zero latency (~12 samples).

### TestClassicShifting

| Test | Input | Shift | Expected | Tolerance |
|------|-------|-------|----------|-----------|
| `test_classic_shift_up_100` | 440 Hz | +100 | 540 Hz | ±10 Hz |
| `test_classic_shift_down_200` | 1000 Hz | -200 | 800 Hz | ±10 Hz |
| `test_classic_zero_shift` | 440 Hz | 0 | 440 Hz | ±10 Hz |

### TestClassicSafety

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_classic_no_nan_impulse` | No NaN from impulse at +500 Hz | No NaN/Inf |
| `test_classic_bounded_square` | Square wave bounded | Peak < +12 dBFS |
| `test_classic_silence` | Silent input stays silent | Output < -80 dB |

### TestClassicFeatures

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_classic_warm_filter` | WARM filter reduces HF | Energy >5 kHz drops ≥6 dB |
| `test_classic_produces_impulse_response` | Impulse produces output | Peak > 0.001 |

---

## Tier 2: Signals — test_plugin_signals.py

### TestSquareWave

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_square_harmonics_shift` | Odd harmonics shift together | Fundamental ~540 Hz, 3rd ~1420 Hz |
| `test_square_energy_preserved` | Energy preserved | RMS change < 6 dB |
| `test_square_no_nan` | No NaN | No NaN/Inf |

### TestImpulse

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_impulse_spectral_mode` | Valid spectral impulse response | No NaN, RMS > -100 dB |
| `test_impulse_classic_mode` | Valid classic impulse response | No NaN, peak > 0.001 |
| `test_impulse_bounded` | Impulse doesn't explode | Peak < +12 dBFS |

### TestPulseTrain

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_pulse_train_no_crash` | Pulse train safe | No NaN/Inf |
| `test_pulse_train_has_energy` | Output has energy | RMS > -60 dB |
| `test_pulse_train_bypass` | Periodicity preserved | Peak freq > 5 Hz |

### TestSweep

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_sweep_no_clicks` | No clicks during chirp | Max sample jump < 1.0 |
| `test_sweep_continuous` | No dropouts | Mid-section RMS > -40 dB |

### TestDCHandling / TestPianoLike

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_dc_rejected_after_shift` | DC rejected by FFT | DC < -10 dB |
| `test_piano_shifted` | Harmonic signal shifts correctly | Peak ~540 Hz (±15 Hz) |
| `test_piano_energy_reasonable` | Harmonic energy preserved | RMS change < 12 dB |

---

## Tier 3: Stress — test_plugin_edge_cases.py

### TestExtremeParameters

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_extreme_shift_positive` | +20 kHz shift | No NaN |
| `test_extreme_shift_negative` | -20 kHz shift | No NaN |
| `test_max_feedback` | 95% feedback + delay | No NaN, peak < 10.0 |
| `test_all_params_max` | All params at max | No NaN |
| `test_all_params_min` | All params at min | No NaN |
| `test_rapid_parameter_change` | Mid-stream automation | No NaN |

### TestSampleRates (parametrized)

| Sample Rate | Pass Criteria |
|------------|---------------|
| 22050 Hz | No NaN (**known failure** — Hilbert coefficients not designed for sub-44.1kHz) |
| 44100 Hz | No NaN, shift accurate (±15 Hz) |
| 48000 Hz | No NaN, shift accurate (±15 Hz) |
| 88200 Hz | No NaN, shift accurate (±15 Hz) |
| 96000 Hz | No NaN, shift accurate (±15 Hz) |

### TestBlockSizes (parametrized)

| Block Size | Pass Criteria |
|-----------|---------------|
| 64, 128, 256, 512, 1024 | No NaN, RMS > -40 dB |

---

## Tier 3: Regression — test_plugin_lfo_depth.py

Tests for the LFO depth "stuck" bug (fixed in v107 patch). Uses **streaming mode** (`reset=False`) to simulate real DAW behavior.

### TestLfoDepthStreaming (bug reproduction tests)

| Test | Scenario | Pass Criteria |
|------|----------|---------------|
| `test_depth_increase_streaming_512` | Depth 0→300 at 512-sample blocks | Spectral width > 50 Hz within 1s |
| `test_depth_increase_streaming_2048` | Depth 0→300 at 2048-sample blocks | Spectral width > 50 Hz within 1s |
| `test_depth_decrease_streaming` | Depth 300→0 at 1024-sample blocks | Spectral width < 20 Hz within 1s |
| `test_depth_multiple_changes_streaming` | 0→300→0→500 rapid changes | Each change clearly distinguished (>20 Hz delta) |

### TestLfoDepthBlockSizeConsistency

| Test | What It Verifies | Pass Criteria |
|------|-----------------|---------------|
| `test_block_size_invariance` | Same depth at 128/512/2048 blocks = same result | Min width > 70% of max width |
| `test_convergence_rate_invariance` | Convergence speed independent of block size | 1024-block width > 50% of 128-block width |

---

## Tier 4: Release — test_plugin_quality_gates.py

**All must pass before shipping to beta testers.**

| Gate | Metric | Threshold |
|------|--------|-----------|
| `test_gate_bypass_snr` | Bypass signal fidelity | RMS change < 3 dB |
| `test_gate_shift_accuracy` | +100 Hz shift precision | Error < 5 Hz |
| `test_gate_no_dc_leakage` | DC after shifting | < -40 dB |
| `test_gate_no_nan_param_sweep` | NaN across 50 param combos | 0 failures |
| `test_gate_output_ceiling` | Peak output level | < +12 dBFS |
| `test_gate_silence_floor` | Silence → output level | < -90 dB |
| `test_gate_no_clicks_on_start` | First 100ms click-free | Max jump < 0.5 |
| `test_gate_stereo_balance` | L/R balance in bypass | < 0.5 dB difference |

---

## Running Tests

```bash
# Full suite
pytest tests/ -v

# Quick smoke test
pytest tests/test_plugin_basic.py -v

# Release gates only
pytest tests/test_plugin_quality_gates.py -v

# LFO regression tests
pytest tests/test_plugin_lfo_depth.py -v

# Skip known failures
pytest tests/ -v -k "not 22050"
```
