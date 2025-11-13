"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests."""
    return 44100


@pytest.fixture
def duration():
    """Standard duration for test signals (1 second)."""
    return 1.0


@pytest.fixture
def sine_440hz(sample_rate, duration):
    """Generate a 440 Hz sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def sine_880hz(sample_rate, duration):
    """Generate an 880 Hz sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * 880 * t)


@pytest.fixture
def harmonic_series(sample_rate, duration):
    """Generate harmonic series: 440 + 880 + 1320 Hz."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = (
        np.sin(2 * np.pi * 440 * t) +
        0.5 * np.sin(2 * np.pi * 880 * t) +
        0.25 * np.sin(2 * np.pi * 1320 * t)
    )
    return signal / np.max(np.abs(signal))


@pytest.fixture
def white_noise(sample_rate, duration):
    """Generate white noise."""
    n_samples = int(sample_rate * duration)
    return np.random.randn(n_samples) * 0.5


@pytest.fixture
def c_major_chord(sample_rate, duration):
    """Generate C major chord (C4 + E4 + G4)."""
    from harmonic_shifter.theory.tuning import midi_to_freq

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    c4 = np.sin(2 * np.pi * midi_to_freq(60) * t)
    e4 = np.sin(2 * np.pi * midi_to_freq(64) * t)
    g4 = np.sin(2 * np.pi * midi_to_freq(67) * t)
    chord = (c4 + e4 + g4) / 3
    return chord


@pytest.fixture
def fft_size():
    """Standard FFT size for tests."""
    return 4096


@pytest.fixture
def hop_size():
    """Standard hop size for tests."""
    return 1024
