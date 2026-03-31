"""
Shared fixtures for Holy Shifter v103 VST3 plugin tests.

All signal fixtures are deterministic (exact math or seeded RNG).
The plugin fixture loads the actual built VST3 binary.
"""

import os
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SR = 44100  # Default sample rate
DURATION = 1.0  # Default signal duration in seconds
VST3_PATH = os.path.expanduser(
    "~/Library/Audio/Plug-Ins/VST3/Holy Shifter v109.vst3"
)


# ---------------------------------------------------------------------------
# Plugin fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def plugin():
    """Load the actual built Holy Shifter VST3 plugin."""
    if not os.path.exists(VST3_PATH):
        pytest.skip("VST3 not built — run: cmake --build plugin/build --config Release")
    from pedalboard import load_plugin

    return load_plugin(VST3_PATH)


@pytest.fixture()
def fresh_plugin():
    """Load a fresh plugin instance per test (no shared state)."""
    if not os.path.exists(VST3_PATH):
        pytest.skip("VST3 not built")
    from pedalboard import load_plugin

    return load_plugin(VST3_PATH)


# ---------------------------------------------------------------------------
# Signal fixtures (all deterministic)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sine_440():
    """440 Hz sine wave, 1 second, 44100 Hz."""
    t = np.arange(int(SR * DURATION)) / SR
    return (np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture()
def sine_1000():
    """1000 Hz sine wave, 1 second, 44100 Hz."""
    t = np.arange(int(SR * DURATION)) / SR
    return (np.sin(2 * np.pi * 1000 * t)).astype(np.float32)


@pytest.fixture()
def square_440():
    """440 Hz bandlimited square wave (sum of odd harmonics up to Nyquist)."""
    t = np.arange(int(SR * DURATION)) / SR
    signal = np.zeros_like(t)
    for k in range(1, 40, 2):  # odd harmonics: 1, 3, 5, ...
        freq = 440 * k
        if freq >= SR / 2:
            break
        signal += (1.0 / k) * np.sin(2 * np.pi * freq * t)
    signal *= 4 / np.pi  # normalize to ±1
    return signal.astype(np.float32)


@pytest.fixture()
def impulse():
    """Single-sample unit impulse at t=0, 1 second long."""
    sig = np.zeros(int(SR * DURATION), dtype=np.float32)
    sig[0] = 1.0
    return sig


@pytest.fixture()
def pulse_train():
    """10 Hz pulse train (impulse every 4410 samples), 1 second."""
    sig = np.zeros(int(SR * DURATION), dtype=np.float32)
    spacing = SR // 10  # 4410 samples
    for i in range(0, len(sig), spacing):
        sig[i] = 1.0
    return sig


@pytest.fixture()
def silence():
    """1 second of silence."""
    return np.zeros(int(SR * DURATION), dtype=np.float32)


@pytest.fixture()
def dc_offset():
    """Constant DC offset of 0.5 for 1 second."""
    return np.full(int(SR * DURATION), 0.5, dtype=np.float32)


@pytest.fixture()
def white_noise():
    """Seeded white noise (seed=42), 1 second."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(SR * DURATION)).astype(np.float32) * 0.5


@pytest.fixture()
def two_tone():
    """440 Hz + 880 Hz summed, each at 0.5 amplitude, 1 second."""
    t = np.arange(int(SR * DURATION)) / SR
    sig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    return sig.astype(np.float32)


@pytest.fixture()
def sweep():
    """Linear chirp from 20 Hz to 20 kHz, 2 seconds."""
    from scipy.signal import chirp

    t = np.arange(int(SR * 2.0)) / SR
    return chirp(t, f0=20, f1=20000, t1=2.0, method="linear").astype(np.float32)


@pytest.fixture()
def piano_like():
    """Piano-like tone: 440 Hz fundamental + decaying harmonics."""
    t = np.arange(int(SR * DURATION)) / SR
    signal = np.zeros_like(t)
    harmonics = [(1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15), (5, 0.08)]
    for n, amp in harmonics:
        freq = 440 * n
        if freq < SR / 2:
            signal += amp * np.sin(2 * np.pi * freq * t) * np.exp(-3.0 * t)
    # Normalize peak to 0.9
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal * 0.9 / peak
    return signal.astype(np.float32)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def peak_frequency(signal: np.ndarray, sr: int = SR) -> float:
    """Find the dominant frequency in a signal via FFT."""
    # Use a window to reduce spectral leakage
    windowed = signal * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    # Ignore DC bin
    spectrum[0] = 0
    return float(freqs[np.argmax(spectrum)])


def peak_frequencies(signal: np.ndarray, n_peaks: int = 2, sr: int = SR) -> list:
    """Find the top N peak frequencies."""
    windowed = signal * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    spectrum[0] = 0  # ignore DC

    peaks = []
    spec_copy = spectrum.copy()
    for _ in range(n_peaks):
        idx = np.argmax(spec_copy)
        peaks.append(float(freqs[idx]))
        # Zero out a neighborhood around the peak
        lo = max(0, idx - 20)
        hi = min(len(spec_copy), idx + 20)
        spec_copy[lo:hi] = 0
    return sorted(peaks)


def rms_db(signal: np.ndarray) -> float:
    """RMS level in dB (relative to 1.0)."""
    rms = np.sqrt(np.mean(signal.astype(np.float64) ** 2))
    if rms < 1e-20:
        return -200.0
    return float(20 * np.log10(rms))


def has_nan_or_inf(signal: np.ndarray) -> bool:
    """Check if signal contains NaN or Inf values."""
    return bool(np.any(np.isnan(signal)) or np.any(np.isinf(signal)))


def energy_ratio_db(output: np.ndarray, input_sig: np.ndarray) -> float:
    """Energy change from input to output, in dB."""
    return rms_db(output) - rms_db(input_sig)


def spectral_energy_above(signal: np.ndarray, freq_hz: float, sr: int = SR) -> float:
    """RMS energy of frequency content above freq_hz, in dB."""
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    mask = freqs >= freq_hz
    energy = np.sqrt(np.mean(spectrum[mask] ** 2)) if np.any(mask) else 1e-20
    return float(20 * np.log10(max(energy, 1e-20)))


def spectral_energy_below(signal: np.ndarray, freq_hz: float, sr: int = SR) -> float:
    """RMS energy of frequency content below freq_hz, in dB."""
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    mask = freqs <= freq_hz
    energy = np.sqrt(np.mean(spectrum[mask] ** 2)) if np.any(mask) else 1e-20
    return float(20 * np.log10(max(energy, 1e-20)))


def dc_component_db(signal: np.ndarray) -> float:
    """DC component magnitude in dB."""
    dc = np.abs(np.mean(signal.astype(np.float64)))
    if dc < 1e-20:
        return -200.0
    return float(20 * np.log10(dc))


def process_with_params(plugin, signal: np.ndarray, sr: int = SR, **params) -> np.ndarray:
    """Process a signal through the plugin with given parameters.

    Handles mono→stereo conversion (pedalboard expects [channels, samples]).
    Returns mono output (first channel).
    """
    # Reset plugin state
    plugin.reset()

    # Set parameters
    for name, value in params.items():
        setattr(plugin, name, value)

    # pedalboard expects shape (channels, samples)
    if signal.ndim == 1:
        audio = signal[np.newaxis, :]  # mono → (1, N)
    else:
        audio = signal

    output = plugin.process(audio, sample_rate=float(sr))

    # Return first channel as 1D
    if output.ndim == 2:
        return output[0]
    return output
