"""
Tests for LFO depth parameter dynamic changes.

Reproduces the bug: "when you move the modulation amount for the Freq LFO,
it seems to get stuck at the first position you set it."

Root cause: The LFO depth smoothing coefficient is designed for per-sample
application (~20ms time constant) but is applied once per processBlock call.
With large block sizes (512-4096 samples), the effective convergence time
is ~10-40 seconds instead of 20ms.

IMPORTANT: pedalboard's plugin.process() has reset=True by default, which
resets smoothedLfoDepth to the current value on each call. To reproduce the
DAW behavior (continuous streaming without resets), we must use reset=False
after the initial call.
"""

import os
import numpy as np
import pytest
from conftest import SR, process_with_params, peak_frequency, rms_db, has_nan_or_inf

VST3_PATH = os.path.expanduser(
    "~/Library/Audio/Plug-Ins/VST3/Holy Shifter v107.vst3"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLOCK_DURATION = 0.5  # seconds per processing chunk
BLOCK_SAMPLES = int(SR * BLOCK_DURATION)


def generate_sine(freq_hz: float, n_samples: int, sr: int = SR) -> np.ndarray:
    """Generate a sine wave."""
    t = np.arange(n_samples) / sr
    return (np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def process_streaming(plugin, signal: np.ndarray, sr: int = SR,
                      buffer_size: int = 512) -> np.ndarray:
    """Process a signal in streaming mode (no reset between blocks).

    This accurately simulates how a DAW streams audio: processBlock is called
    repeatedly without resetting the plugin state between calls. The smoothing
    coefficient bug manifests here because smoothedLfoDepth only updates once
    per processBlock call.

    Args:
        plugin: The loaded VST3 plugin
        signal: 1D audio signal
        sr: Sample rate
        buffer_size: The DAW's buffer size — determines how often processBlock
                     is called and thus how often smoothing is applied.
    """
    if signal.ndim == 1:
        audio = signal[np.newaxis, :]
    else:
        audio = signal

    n_samples = audio.shape[1]
    output_chunks = []

    for i, start in enumerate(range(0, n_samples, buffer_size)):
        end = min(start + buffer_size, n_samples)
        chunk = audio[:, start:end]
        # First chunk: reset=True to initialize; subsequent: reset=False
        # This matches DAW behavior where processBlock streams continuously
        out_chunk = plugin.process(
            chunk, sample_rate=float(sr),
            buffer_size=buffer_size,
            reset=(i == 0)
        )
        output_chunks.append(out_chunk)

    output = np.concatenate(output_chunks, axis=1)
    return output[0] if output.ndim == 2 else output


def measure_spectral_width(output: np.ndarray, center_freq: float,
                           sr: int = SR) -> float:
    """Measure the spectral width around a center frequency.

    LFO modulation spreads energy across a range of frequencies.
    Returns the bandwidth in Hz containing 90% of energy near center_freq.
    """
    if len(output) < 256:
        return 0.0

    windowed = output * np.hanning(len(output))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(output), 1.0 / sr)

    # Focus on region around center frequency (±500 Hz)
    mask = (freqs >= center_freq - 500) & (freqs <= center_freq + 500)
    local_spectrum = spectrum.copy()
    local_spectrum[~mask] = 0

    total_energy = np.sum(local_spectrum ** 2)
    if total_energy < 1e-20:
        return 0.0

    cumulative = np.cumsum(local_spectrum[mask] ** 2) / total_energy
    local_freqs = freqs[mask]

    low_idx = np.searchsorted(cumulative, 0.05)
    high_idx = np.searchsorted(cumulative, 0.95)

    if high_idx >= len(local_freqs):
        high_idx = len(local_freqs) - 1

    return float(local_freqs[high_idx] - local_freqs[low_idx])


def load_fresh_plugin():
    """Load a fresh plugin instance."""
    if not os.path.exists(VST3_PATH):
        pytest.skip("VST3 not built")
    from pedalboard import load_plugin
    return load_plugin(VST3_PATH)


# ---------------------------------------------------------------------------
# Tests: Basic LFO depth behavior (with reset — should always pass)
# ---------------------------------------------------------------------------


class TestLfoDepthBasic:
    """Basic LFO depth tests using process_with_params (with reset).
    These verify the LFO works at all, independent of the smoothing bug."""

    def test_lfo_depth_zero_no_modulation(self, fresh_plugin, sine_440):
        """With LFO depth=0, output should be a clean shifted tone."""
        out = process_with_params(
            fresh_plugin, sine_440, SR,
            shift_hz=100.0,
            lfo_depth=0.0,
            lfo_rate=2.0,
            dry_wet=100.0,
            processing_mode=1.0,
        )
        width = measure_spectral_width(out, 540.0)
        assert width < 30.0, f"Expected narrow spectrum, got width={width:.1f} Hz"

    def test_lfo_depth_nonzero_has_modulation(self, fresh_plugin, sine_440):
        """With LFO depth > 0, output should show frequency spread."""
        out = process_with_params(
            fresh_plugin, sine_440, SR,
            shift_hz=100.0,
            lfo_depth=200.0,
            lfo_rate=2.0,
            dry_wet=100.0,
            processing_mode=1.0,
        )
        width = measure_spectral_width(out, 540.0)
        assert width > 30.0, f"Expected wide spectrum from LFO, got width={width:.1f} Hz"

    def test_no_nan_on_depth_change(self, fresh_plugin):
        """Changing LFO depth should never produce NaN."""
        signal = generate_sine(440.0, SR * 2)

        fresh_plugin.reset()
        fresh_plugin.shift_hz = 100.0
        fresh_plugin.lfo_rate = 5.0
        fresh_plugin.dry_wet = 100.0
        fresh_plugin.processing_mode = 1.0

        depths = [0.0, 5000.0, 0.0, 2500.0, 5000.0]
        for depth in depths:
            fresh_plugin.lfo_depth = depth
            audio = signal[np.newaxis, :SR // 2]
            out = fresh_plugin.process(audio, sample_rate=float(SR))
            assert not has_nan_or_inf(out), f"NaN/Inf at lfo_depth={depth}"

    def test_output_bounded_with_lfo(self, fresh_plugin, sine_440):
        """Output should stay bounded even with maximum LFO depth."""
        out = process_with_params(
            fresh_plugin, sine_440, SR,
            shift_hz=100.0,
            lfo_depth=5000.0,
            lfo_rate=10.0,
            dry_wet=100.0,
            processing_mode=1.0,
        )
        peak = float(np.max(np.abs(out)))
        assert peak < 4.0, f"Output unbounded with max LFO: peak={peak:.2f}"


# ---------------------------------------------------------------------------
# Tests: LFO depth parameter changes in streaming mode (reproduce bug)
# ---------------------------------------------------------------------------


class TestLfoDepthStreaming:
    """Tests that simulate DAW streaming behavior using reset=False.

    The bug: smoothedLfoDepth uses a per-sample coefficient (0.99887) but
    it's applied once per processBlock. With buffer_size=1024:
    - Time constant = 1024 / (44100 * -ln(0.99887)) ≈ 20.5 seconds
    - After 1s: only reaches ~4.8% of target
    - After 5s: only reaches ~21.7% of target

    These tests SHOULD FAIL with the current buggy code.
    """

    def test_depth_increase_streaming_512(self):
        """BUG: LFO depth 0→300 should converge within 1s at buffer_size=512."""
        plugin = load_fresh_plugin()

        signal = generate_sine(440.0, SR * 3)

        # Configure plugin
        plugin.shift_hz = 0.0  # No base shift, only LFO
        plugin.lfo_depth = 0.0
        plugin.lfo_rate = 3.0
        plugin.dry_wet = 100.0
        plugin.processing_mode = 0.0  # Classic mode (no FFT latency)

        # Stream 0.5s with depth=0 to initialize (uses reset=True on first block)
        init_out = process_streaming(plugin, signal[:BLOCK_SAMPLES], buffer_size=512)

        # NOW change depth to 300 Hz and stream 1s WITHOUT reset
        plugin.lfo_depth = 300.0
        audio_after = signal[np.newaxis, :SR]
        output_chunks = []
        bs = 512
        for start in range(0, SR, bs):
            end = min(start + bs, SR)
            chunk = audio_after[:, start:end]
            out = plugin.process(chunk, sample_rate=float(SR),
                                 buffer_size=bs, reset=False)
            output_chunks.append(out)

        combined = np.concatenate(output_chunks, axis=1)[0]

        # Measure spectral width of last 0.3s
        tail = combined[-int(SR * 0.3):]
        width = measure_spectral_width(tail, 0.0)

        # With 300 Hz depth and 3 Hz rate, we expect significant spectral spread
        assert width > 50.0, (
            f"LFO depth not converging at buffer_size=512: "
            f"width={width:.1f} Hz after 1s (expected >50 Hz for 300 Hz depth)"
        )

    def test_depth_increase_streaming_2048(self):
        """BUG: LFO depth 0→300 should converge within 1s at buffer_size=2048."""
        plugin = load_fresh_plugin()

        signal = generate_sine(440.0, SR * 3)

        plugin.shift_hz = 0.0
        plugin.lfo_depth = 0.0
        plugin.lfo_rate = 3.0
        plugin.dry_wet = 100.0
        plugin.processing_mode = 0.0

        # Initialize
        process_streaming(plugin, signal[:BLOCK_SAMPLES], buffer_size=2048)

        # Change depth and stream without reset
        plugin.lfo_depth = 300.0
        audio_after = signal[np.newaxis, :SR]
        output_chunks = []
        bs = 2048
        for start in range(0, SR, bs):
            end = min(start + bs, SR)
            chunk = audio_after[:, start:end]
            out = plugin.process(chunk, sample_rate=float(SR),
                                 buffer_size=bs, reset=False)
            output_chunks.append(out)

        combined = np.concatenate(output_chunks, axis=1)[0]

        tail = combined[-int(SR * 0.3):]
        width = measure_spectral_width(tail, 0.0)

        assert width > 50.0, (
            f"LFO depth not converging at buffer_size=2048: "
            f"width={width:.1f} Hz after 1s (expected >50 Hz for 300 Hz depth)"
        )

    def test_depth_decrease_streaming(self):
        """BUG: LFO depth 300→0 should converge within 1s."""
        plugin = load_fresh_plugin()

        signal = generate_sine(440.0, SR * 4)
        bs = 1024

        # Start with depth=300 and stream 1.5s to let it fully settle
        plugin.shift_hz = 0.0
        plugin.lfo_depth = 300.0
        plugin.lfo_rate = 3.0
        plugin.dry_wet = 100.0
        plugin.processing_mode = 0.0

        process_streaming(plugin, signal[:int(SR * 1.5)], buffer_size=bs)

        # NOW change depth to 0 and stream 1s without reset
        plugin.lfo_depth = 0.0
        audio = signal[np.newaxis, :SR]
        output_chunks = []
        for start in range(0, SR, bs):
            end = min(start + bs, SR)
            chunk = audio[:, start:end]
            out = plugin.process(chunk, sample_rate=float(SR),
                                 buffer_size=bs, reset=False)
            output_chunks.append(out)

        combined = np.concatenate(output_chunks, axis=1)[0]

        # After setting depth=0, the last 0.3s should show minimal modulation
        tail = combined[-int(SR * 0.3):]
        width = measure_spectral_width(tail, 0.0)

        assert width < 20.0, (
            f"LFO depth still active after setting to 0 (buffer_size={bs}): "
            f"width={width:.1f} Hz after 1s (expected <20 Hz)"
        )

    def test_depth_multiple_changes_streaming(self):
        """BUG: Multiple rapid depth changes should all take effect in streaming mode."""
        plugin = load_fresh_plugin()

        signal = generate_sine(440.0, SR * 5)
        bs = 1024

        plugin.shift_hz = 0.0
        plugin.lfo_rate = 3.0
        plugin.dry_wet = 100.0
        plugin.processing_mode = 0.0

        # Initialize
        plugin.lfo_depth = 0.0
        process_streaming(plugin, signal[:int(SR * 0.5)], buffer_size=bs)

        # Sequence of depth changes, 1s each, without reset
        depths = [0.0, 300.0, 0.0, 500.0]
        widths = []

        for depth in depths:
            plugin.lfo_depth = depth
            audio = signal[np.newaxis, :SR]
            output_chunks = []
            for start in range(0, SR, bs):
                end = min(start + bs, SR)
                chunk = audio[:, start:end]
                out = plugin.process(chunk, sample_rate=float(SR),
                                     buffer_size=bs, reset=False)
                output_chunks.append(out)

            combined = np.concatenate(output_chunks, axis=1)[0]
            # Measure last 0.3s
            tail = combined[-int(SR * 0.3):]
            widths.append(measure_spectral_width(tail, 0.0))

        # Each depth setting should be clearly distinguishable
        assert widths[1] > widths[0] + 20.0, (
            f"0→300 change not detected: {widths[0]:.1f} vs {widths[1]:.1f}"
        )
        assert widths[2] < widths[1] - 20.0, (
            f"300→0 change not detected: {widths[1]:.1f} vs {widths[2]:.1f}"
        )
        assert widths[3] > widths[2] + 20.0, (
            f"0→500 change not detected: {widths[2]:.1f} vs {widths[3]:.1f}"
        )


class TestLfoDepthBlockSizeConsistency:
    """The modulation depth should be the SAME regardless of buffer size.

    If the smoothing coefficient is correctly applied per-sample (or adjusted
    for block rate), the output should be identical whether the DAW uses
    128-sample or 2048-sample buffers.
    """

    def test_block_size_invariance(self):
        """Same depth should produce same modulation at different block sizes."""
        widths = {}

        for bs in [128, 512, 2048]:
            plugin = load_fresh_plugin()
            signal = generate_sine(440.0, SR * 2)

            plugin.shift_hz = 0.0
            plugin.lfo_depth = 300.0
            plugin.lfo_rate = 3.0
            plugin.dry_wet = 100.0
            plugin.processing_mode = 0.0

            out = process_streaming(plugin, signal, buffer_size=bs)
            # Measure last 0.5s (after any settling)
            tail = out[-BLOCK_SAMPLES:]
            widths[bs] = measure_spectral_width(tail, 0.0)

        # All block sizes should produce similar spectral width (within 30%)
        max_width = max(widths.values())
        min_width = min(widths.values())

        assert min_width > 0.7 * max_width, (
            f"Block size affects modulation depth: {widths} "
            f"(ratio={min_width/max(max_width, 0.1):.2f}, expected >0.7)"
        )

    def test_convergence_rate_invariance(self):
        """After a depth change, convergence speed should be block-size independent."""
        convergence_widths = {}

        for bs in [128, 1024]:
            plugin = load_fresh_plugin()
            signal = generate_sine(440.0, SR * 3)

            plugin.shift_hz = 0.0
            plugin.lfo_depth = 0.0
            plugin.lfo_rate = 3.0
            plugin.dry_wet = 100.0
            plugin.processing_mode = 0.0

            # Initialize at depth=0
            process_streaming(plugin, signal[:BLOCK_SAMPLES], buffer_size=bs)

            # Change to depth=300 and stream 0.5s without reset
            plugin.lfo_depth = 300.0
            audio = signal[np.newaxis, :BLOCK_SAMPLES]
            output_chunks = []
            for start in range(0, BLOCK_SAMPLES, bs):
                end = min(start + bs, BLOCK_SAMPLES)
                chunk = audio[:, start:end]
                out = plugin.process(chunk, sample_rate=float(SR),
                                     buffer_size=bs, reset=False)
                output_chunks.append(out)

            combined = np.concatenate(output_chunks, axis=1)[0]
            tail = combined[-int(SR * 0.2):]
            convergence_widths[bs] = measure_spectral_width(tail, 0.0)

        # Convergence at 128-sample blocks vs 1024 should be similar
        ratio = convergence_widths[1024] / max(convergence_widths[128], 0.1)
        assert ratio > 0.5, (
            f"Convergence rate depends on block size: "
            f"128-blocks={convergence_widths[128]:.1f} Hz, "
            f"1024-blocks={convergence_widths[1024]:.1f} Hz "
            f"(ratio={ratio:.2f}, expected >0.5)"
        )


class TestLfoDepthNoArtifacts:
    """Ensure LFO depth changes don't produce clicks or NaN in streaming mode."""

    def test_no_clicks_on_depth_change(self):
        """Rapid depth changes should not produce clicks."""
        plugin = load_fresh_plugin()
        signal = generate_sine(440.0, SR * 3)
        bs = 512

        plugin.shift_hz = 50.0
        plugin.lfo_rate = 1.0
        plugin.dry_wet = 100.0
        plugin.processing_mode = 0.0  # Classic
        plugin.lfo_depth = 0.0

        # Initialize
        process_streaming(plugin, signal[:int(SR * 0.2)], buffer_size=bs)

        all_output = []
        chunk_size = SR // 10  # 100ms chunks

        for i in range(20):
            plugin.lfo_depth = 500.0 if (i % 2 == 0) else 0.0

            audio = signal[np.newaxis, :chunk_size]
            output_chunks = []
            for start in range(0, chunk_size, bs):
                end = min(start + bs, chunk_size)
                chunk = audio[:, start:end]
                out = plugin.process(chunk, sample_rate=float(SR),
                                     buffer_size=bs, reset=False)
                output_chunks.append(out)

            combined = np.concatenate(output_chunks, axis=1)[0]
            all_output.append(combined)

        full_output = np.concatenate(all_output)
        diffs = np.abs(np.diff(full_output))
        max_diff = float(np.max(diffs))
        assert max_diff < 0.5, f"Click detected: max sample jump = {max_diff:.3f}"

    def test_no_nan_streaming(self):
        """No NaN in streaming mode with parameter changes."""
        plugin = load_fresh_plugin()
        signal = generate_sine(440.0, SR * 2)
        bs = 1024

        plugin.shift_hz = 100.0
        plugin.lfo_rate = 5.0
        plugin.dry_wet = 100.0
        plugin.processing_mode = 0.0
        plugin.lfo_depth = 0.0

        process_streaming(plugin, signal[:int(SR * 0.2)], buffer_size=bs)

        for depth in [0.0, 5000.0, 0.0, 2500.0]:
            plugin.lfo_depth = depth
            audio = signal[np.newaxis, :int(SR * 0.3)]
            output_chunks = []
            for start in range(0, int(SR * 0.3), bs):
                end = min(start + bs, int(SR * 0.3))
                chunk = audio[:, start:end]
                out = plugin.process(chunk, sample_rate=float(SR),
                                     buffer_size=bs, reset=False)
                output_chunks.append(out)
                assert not has_nan_or_inf(out), f"NaN at depth={depth}, block {start}"
