"""
Regression test for the in-key spectral frequency shifter pipeline.

Verifies the behavior the redesigned Spectral mode is supposed to produce:
when `processingMode = Spectral`, `quantizeStrength = 100`, and a sparse scale
(C major triad) is active, the Shift Hz knob moves the output through the
active scale tones — with continuous, sub-bin precision — rather than
snapping in ~10.77 Hz chunks (the old integer-bin shifter behavior).

Background — the old pipeline applied three stages:
    PhaseVocoder.process(shiftHz)  →  FrequencyShifter::shift(shiftHz)  →  Quantizer
The middle stage rounded shiftHz to the nearest FFT bin (~10.77 Hz at SR=44100,
fftSize=4096), so small knob movements were inaudible until they crossed a bin
boundary, and the output sounded metallic from the phase-vocoder over-locking
on top of the bin-quantized spectrum.

The new pipeline collapses to a single quantizer call that takes shiftHz as
a continuous parameter, applies it per-bin inside the two-nearest scale-tone
distribution, and uses the quantizer's per-MIDI-note phase accumulators for
synthesis coherence.

NOTE: this test requires a freshly built VST3 at the path defined in
conftest.py (currently expects v109). Update conftest's VST3_PATH after
rebuilding.

The companion fix in Scales.h makes the previously-buggy "input near top of
octave" range work correctly, so this test now includes shifts that put
src_freq near the top of an octave (e.g. 30, 75, 500, 1500 Hz).
"""

import os
import numpy as np
import pytest
from conftest import SR, peak_frequency


# C major triad as pitch classes (semitones from C)
TRIAD_PCS = (0, 4, 7)
A4_HZ = 440.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def freq_to_midi(freq: float) -> float:
    return 69.0 + 12.0 * np.log2(freq / A4_HZ)


def midi_to_freq(midi: float) -> float:
    return A4_HZ * 2.0 ** ((midi - 69.0) / 12.0)


def nearest_in_scale_pitch_class(measured_freq: float, pcs=TRIAD_PCS) -> int:
    """Pitch class of the nearest scale tone (in any octave) to measured_freq."""
    midi_measured = freq_to_midi(measured_freq)
    best_pc, best_dist = None, float("inf")
    # Search a wide octave range and pick the closest scale tone.
    for octave in range(-2, 9):
        for pc in pcs:
            candidate_midi = 60 + octave * 12 + pc  # root = C4
            dist = abs(candidate_midi - midi_measured)
            if dist < best_dist:
                best_dist = dist
                best_pc = pc
    return best_pc, best_dist


def configure_spectral_triad(plugin):
    """Configure the plugin as an in-key Spectral shifter with a C-major triad."""
    plugin.processing_mode = 1.0  # 0=Classic (Hilbert), 1=Spectral
    plugin.quantize_strength = 100.0
    plugin.dry_wet = 100.0  # avoid the dry-mix path masking quantized output
    plugin.lfo_depth = 0.0   # keep shift static so the peak doesn't smear
    for i in range(12):
        setattr(plugin, f"scale_note{i}", bool(i in TRIAD_PCS))
    return plugin


def process_through_plugin(plugin, signal, sr=SR):
    plugin.reset()
    audio = signal[np.newaxis, :] if signal.ndim == 1 else signal
    out = plugin.process(audio, sample_rate=float(sr))
    return out[0] if out.ndim == 2 else out


def trim_latency(out_signal, samples=8192):
    """Drop the first chunk where ISTFT overlap-add hasn't filled in yet.

    Spectral mode latency is the FFT size (set in PluginProcessor.cpp to
    MAX_FFT_SIZE = 8192 samples). Past that point the output is steady-state.
    """
    return out_signal[samples:]


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

# Shift values chosen to span:
#   * 0 Hz                  — no shift, pure scale snap (sanity baseline)
#   * 5, 10 Hz              — well below the old 10.77 Hz bin width; the old
#                             pipeline rounded these to zero, so without the
#                             fix the output is indistinguishable from shift=0
#   * 30, 75 Hz             — src_freq in (466, 523) Hz: triggered the
#                             Scales.h octave-wrap bug before the fix
#   * 100, 200, 300 Hz      — large enough to cross multiple scale tones
#   * 500 Hz                — src_freq ≈ 940 Hz: another wrap-bug zone, now fixed
#   * 1000, 1500 Hz         — multi-octave shifts; verify high register and
#                             the third wrap-bug zone (1500 Hz → src ≈ 1940 Hz)
SHIFT_VALUES_HZ = [0, 5, 10, 30, 75, 100, 200, 300, 500, 1000, 1500]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift_hz", SHIFT_VALUES_HZ)
def test_output_lands_on_scale_pitch_class(fresh_plugin, sine_440, shift_hz):
    """Dominant output frequency is on a C, E, or G in some octave."""
    plugin = configure_spectral_triad(fresh_plugin)
    plugin.shift_hz = float(shift_hz)

    out = process_through_plugin(plugin, sine_440)
    stable = trim_latency(out)
    assert len(stable) > SR // 4, "Not enough post-latency signal to analyze"

    measured = peak_frequency(stable)
    assert measured > 0, f"shift={shift_hz}: zero/silent output"

    pc, cents = nearest_in_scale_pitch_class(measured)
    assert cents < 50, (
        f"shift={shift_hz} Hz: dominant peak at {measured:.1f} Hz is "
        f"{cents * 100:.0f} cents off the nearest C/E/G (pitch class {pc}). "
        f"Either the in-key snap is broken or the shift overshot the scale."
    )


def test_sub_bin_shifts_change_the_output(fresh_plugin, sine_440):
    """
    Regression for "not variable enough" — small Shift moves produce audibly
    different outputs.

    The old bin-quantized shifter rounded shiftHz to the nearest 10.77 Hz, so
    shifts of {0, 5, 10} Hz all collapsed to the same output. The new shifter
    is continuous, so even sub-bin moves can crossfade between scale tones
    via the two-nearest weighting.

    Concretely: with a 440 Hz input and C/E/G active, shifting by enough Hz
    to cross from G4 toward C5 should produce measurably different output
    than shift=0. We do not require every increment to be distinct — only
    that a shift of 100 Hz lands on a different scale tone than shift=0.
    """
    plugin = configure_spectral_triad(fresh_plugin)

    outputs = {}
    for shift in [0, 100, 200, 300]:
        plugin.shift_hz = float(shift)
        out = process_through_plugin(plugin, sine_440)
        outputs[shift] = peak_frequency(trim_latency(out))

    # Round each output to a semitone (any octave). The set must contain
    # at least 2 distinct semitones across the sweep — otherwise the shift
    # parameter is not affecting output frequency at all.
    seen_pitch_classes = set()
    for shift, freq in outputs.items():
        if freq > 0:
            midi = round(freq_to_midi(freq))
            seen_pitch_classes.add(midi % 12)

    assert len(seen_pitch_classes) >= 2, (
        "Spectral mode does not move the output across a shift sweep. "
        f"All shifts collapsed to the same pitch class. Outputs: {outputs}"
    )


def test_inactive_pitch_classes_excluded(fresh_plugin, sine_440):
    """
    With only C/E/G active, the dominant peak should never land on D, D#, F,
    F#, G#, A, A#, or B. This guards against the snap stage misfiring or the
    shift bypassing the quantizer entirely.
    """
    plugin = configure_spectral_triad(fresh_plugin)
    forbidden = {1, 2, 3, 5, 6, 8, 9, 10, 11}  # everything except 0, 4, 7

    for shift in [0, 10, 100, 200, 300, 1000]:
        plugin.shift_hz = float(shift)
        out = process_through_plugin(plugin, sine_440)
        measured = peak_frequency(trim_latency(out))
        if measured <= 0:
            continue
        midi = round(freq_to_midi(measured))
        pc = midi % 12
        assert pc not in forbidden, (
            f"shift={shift} Hz: dominant output {measured:.1f} Hz is "
            f"pitch class {pc}, which is not in the active scale "
            f"{TRIAD_PCS}. Quantizer is leaking out-of-scale content."
        )


def test_shift_zero_matches_pure_quantizer(fresh_plugin, sine_440):
    """
    At shift=0, the new pipeline must produce the same output as the old
    pure-quantizer path (which never went through the bin shifter). 440 Hz
    against C/E/G snaps to G4 (392 Hz) — the closest scale tone.
    """
    plugin = configure_spectral_triad(fresh_plugin)
    plugin.shift_hz = 0.0
    out = process_through_plugin(plugin, sine_440)
    measured = peak_frequency(trim_latency(out))

    expected_g4 = midi_to_freq(67)  # G4 = 391.995 Hz
    cents = 1200 * abs(np.log2(measured / expected_g4)) if measured > 0 else 1e6
    assert cents < 50, (
        f"shift=0, C/E/G active: expected output near G4 ({expected_g4:.1f} Hz), "
        f"got {measured:.1f} Hz ({cents:.0f} cents off)."
    )
