# Project Specification: Harmonic-Preserving Frequency Shifter

## Project Overview

Build a Python-based audio frequency shifter that maintains harmonic relationships by quantizing shifted frequencies to a musical scale. This project serves as a prototype/research implementation before potential VST/C++ conversion.

## Project Structure

```
harmonic-frequency-shifter/
├── pyproject.toml              # Project metadata and dependencies
├── README.md                   # User-facing documentation
├── MATH_FOUNDATION.md          # Mathematical specification (already provided)
├── .gitignore
├── .python-version             # Python 3.11+
├── src/
│   └── harmonic_shifter/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── stft.py                 # STFT/ISTFT implementation
│       │   ├── frequency_shifter.py    # Core shifting logic
│       │   ├── quantizer.py            # Musical scale quantization
│       │   └── phase_vocoder.py        # Phase coherence
│       ├── audio/
│       │   ├── __init__.py
│       │   ├── io.py                   # Audio file I/O
│       │   └── buffer.py               # Circular buffer management
│       ├── theory/
│       │   ├── __init__.py
│       │   ├── scales.py               # Scale definitions
│       │   └── tuning.py               # Frequency/MIDI conversion
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── processor.py            # Main processing pipeline
│       │   └── peaks.py                # Spectral peak detection (optional)
│       └── utils/
│           ├── __init__.py
│           ├── validation.py           # Audio quality metrics
│           └── visualization.py        # Spectrogram plotting
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # Pytest fixtures
│   ├── unit/
│   │   ├── test_stft.py
│   │   ├── test_frequency_shifter.py
│   │   ├── test_quantizer.py
│   │   ├── test_phase_vocoder.py
│   │   └── test_tuning.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_audio_quality.py
│   └── fixtures/
│       ├── sine_440hz.wav              # Test signals
│       ├── harmonic_series.wav
│       └── white_noise.wav
├── examples/
│   ├── basic_usage.py
│   ├── batch_processing.py
│   └── parameter_sweep.py
└── docs/
    ├── API.md
    ├── ALGORITHM.md
    └── BENCHMARKS.md
```

## Technology Stack

### Core Dependencies

```toml
[project]
name = "harmonic-frequency-shifter"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "soundfile>=0.12.0",
    "librosa>=0.10.0",  # Optional: for audio utilities
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-benchmark>=4.0.0",
    "black>=23.7.0",
    "ruff>=0.0.282",
    "mypy>=1.5.0",
]
viz = [
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]
```

### Build System (uv compatible)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/harmonic_shifter"]
```

## Installation Instructions

### Option 1: Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <repo-url>
cd harmonic-frequency-shifter

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev,viz]"

# Run tests
pytest
```

### Option 2: Using pip

```bash
# Clone repository
git clone <repo-url>
cd harmonic-frequency-shifter

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e ".[dev,viz]"

# Run tests
pytest
```

### Option 3: Using poetry (Alternative)

```bash
# Install dependencies
poetry install --with dev,viz

# Run tests
poetry run pytest
```

## Core Module Specifications

### 1. STFT Module (`core/stft.py`)

**Purpose:** Implement windowed FFT and IFFT with overlap-add

**Key Functions:**

```python
def stft(
    signal: np.ndarray,
    fft_size: int = 4096,
    hop_size: int = 1024,
    window: str = 'hann'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Short-Time Fourier Transform

    Returns:
        magnitude: (n_frames, n_bins) - Magnitude spectrum
        phase: (n_frames, n_bins) - Phase spectrum in radians
    """

def istft(
    magnitude: np.ndarray,
    phase: np.ndarray,
    hop_size: int = 1024,
    window: str = 'hann'
) -> np.ndarray:
    """
    Inverse Short-Time Fourier Transform with overlap-add

    Returns:
        signal: reconstructed time-domain audio
    """
```

**Tests:**

- Perfect reconstruction (STFT → ISTFT)
- Energy conservation (Parseval's theorem)
- Window normalization correctness

### 2. Frequency Shifter (`core/frequency_shifter.py`)

**Purpose:** Shift frequencies in spectral domain

**Key Functions:**

```python
class FrequencyShifter:
    def __init__(self, sample_rate: int, fft_size: int):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.bin_resolution = sample_rate / fft_size

    def shift(
        self,
        magnitude: np.ndarray,
        phase: np.ndarray,
        shift_hz: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shift all frequencies by shift_hz in the spectral domain

        Args:
            magnitude: (n_frames, n_bins)
            phase: (n_frames, n_bins)
            shift_hz: Amount to shift in Hz (can be negative)

        Returns:
            shifted_magnitude: (n_frames, n_bins)
            shifted_phase: (n_frames, n_bins)
        """
```

**Tests:**

- Sine wave shift accuracy (within 1 Hz)
- Negative shift handling
- Aliasing prevention at Nyquist

### 3. Musical Quantizer (`core/quantizer.py`)

**Purpose:** Quantize frequencies to musical scales

**Key Functions:**

```python
class MusicalQuantizer:
    def __init__(self, root_midi: int, scale_type: str):
        self.root_midi = root_midi
        self.scale_degrees = SCALES[scale_type]

    def quantize_frequencies(
        self,
        frequencies: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Quantize frequencies to nearest scale notes

        Args:
            frequencies: Array of frequencies in Hz
            strength: 0.0 (no quantization) to 1.0 (full quantization)

        Returns:
            Quantized frequencies in Hz
        """

    def quantize_spectrum(
        self,
        magnitude: np.ndarray,
        phase: np.ndarray,
        sample_rate: int,
        fft_size: int,
        strength: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize entire spectrum to scale

        Returns:
            quantized_magnitude: (n_frames, n_bins)
            quantized_phase: (n_frames, n_bins)
        """
```

**Tests:**

- Exact scale note matching
- Strength parameter interpolation
- Edge cases (DC, Nyquist)
- All scale types

### 4. Phase Vocoder (`core/phase_vocoder.py`)

**Purpose:** Maintain phase coherence when moving energy between bins

**Key Functions:**

```python
def propagate_phase(
    phase_prev: np.ndarray,
    phase_curr: np.ndarray,
    hop_size: int,
    fft_size: int
) -> np.ndarray:
    """
    Calculate instantaneous phase for smooth transitions

    Args:
        phase_prev: Phase from previous frame (n_bins,)
        phase_curr: Phase from current frame (n_bins,)
        hop_size: Hop size in samples
        fft_size: FFT size

    Returns:
        instantaneous_phase: (n_bins,)
    """

def transfer_phase(
    phase: np.ndarray,
    source_bins: np.ndarray,
    target_bins: np.ndarray,
    frequencies: np.ndarray
) -> np.ndarray:
    """
    Transfer phase from source to target bins while maintaining coherence

    Args:
        phase: Original phase values (n_bins,)
        source_bins: Source bin indices
        target_bins: Target bin indices
        frequencies: Frequency ratios for phase scaling

    Returns:
        transferred_phase: Phase values at target bins
    """
```

**Tests:**

- Phase continuity across frames
- No discontinuities > π
- Proper phase scaling with frequency changes

### 5. Processing Pipeline (`processing/processor.py`)

**Purpose:** Main entry point orchestrating all components

**Key Class:**

```python
class HarmonicShifter:
    """
    Main processor for harmonic-preserving frequency shifting
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        fft_size: int = 4096,
        hop_size: int = 1024,
        window: str = 'hann'
    ):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window = window

        self.shifter = FrequencyShifter(sample_rate, fft_size)
        self.quantizer = None  # Set via set_scale()

    def set_scale(self, root_midi: int, scale_type: str):
        """
        Configure musical scale for quantization

        Args:
            root_midi: MIDI note number for scale root (0-127)
            scale_type: Scale name from SCALES dict
        """
        self.quantizer = MusicalQuantizer(root_midi, scale_type)

    def process(
        self,
        audio: np.ndarray,
        shift_hz: float,
        quantize_strength: float = 1.0
    ) -> np.ndarray:
        """
        Process audio with frequency shifting and scale quantization

        Args:
            audio: Input audio (mono, normalized to [-1, 1])
            shift_hz: Frequency shift in Hz (positive or negative)
            quantize_strength: 0-1, amount of scale quantization
                             0.0 = pure frequency shift (inharmonic)
                             1.0 = fully quantized to scale (harmonic)

        Returns:
            Processed audio (same shape as input)

        Raises:
            ValueError: If scale not set when quantize_strength > 0
        """

    def process_realtime(
        self,
        audio_chunk: np.ndarray,
        shift_hz: float,
        quantize_strength: float = 1.0
    ) -> np.ndarray:
        """
        Process single audio chunk for real-time use
        (maintains internal state for phase vocoder)
        """
```

### 6. Musical Theory (`theory/scales.py`, `theory/tuning.py`)

**scales.py:**

```python
from typing import Dict, List

SCALES: Dict[str, List[int]] = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'aeolian': [0, 2, 3, 5, 7, 8, 10],
    'locrian': [0, 1, 3, 5, 6, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
    'pentatonic_major': [0, 2, 4, 7, 9],
    'pentatonic_minor': [0, 3, 5, 7, 10],
    'blues': [0, 3, 5, 6, 7, 10],
    'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}

def get_scale_frequencies(
    root_midi: int,
    scale_type: str,
    octave_range: tuple = (0, 10)
) -> List[float]:
    """
    Return all frequencies in scale across MIDI range

    Args:
        root_midi: Root note MIDI number
        scale_type: Scale name from SCALES
        octave_range: (min_octave, max_octave) relative to root

    Returns:
        List of frequencies in Hz
    """

def get_scale_name(scale_type: str) -> str:
    """Get human-readable scale name"""
```

**tuning.py:**

```python
import numpy as np

def freq_to_midi(freq: float, a4_freq: float = 440.0) -> float:
    """
    Convert frequency to MIDI note number

    Args:
        freq: Frequency in Hz
        a4_freq: Reference frequency for A4 (default 440 Hz)

    Returns:
        MIDI note number (can be fractional)
    """
    return 69 + 12 * np.log2(freq / a4_freq)

def midi_to_freq(midi: float, a4_freq: float = 440.0) -> float:
    """
    Convert MIDI note number to frequency

    Args:
        midi: MIDI note number (can be fractional)
        a4_freq: Reference frequency for A4 (default 440 Hz)

    Returns:
        Frequency in Hz
    """
    return a4_freq * 2 ** ((midi - 69) / 12)

def quantize_to_scale(
    midi_note: float,
    root_midi: int,
    scale_degrees: List[int]
) -> int:
    """
    Quantize MIDI note to nearest scale degree

    Args:
        midi_note: Input MIDI note (can be fractional)
        root_midi: Root note of scale
        scale_degrees: List of semitones from root

    Returns:
        Quantized MIDI note number (integer)
    """

def cents_difference(freq1: float, freq2: float) -> float:
    """
    Calculate difference between two frequencies in cents

    Args:
        freq1, freq2: Frequencies in Hz

    Returns:
        Difference in cents (1 semitone = 100 cents)
    """
    return 1200 * np.log2(freq2 / freq1)
```

### 7. Audio I/O (`audio/io.py`)

**Purpose:** Read/write audio files with proper error handling

**Key Functions:**

```python
import soundfile as sf
import numpy as np
from typing import Tuple

def load_audio(
    filepath: str,
    sample_rate: int = 44100,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file, resample if needed, convert to mono if requested

    Args:
        filepath: Path to audio file
        sample_rate: Target sample rate (None to keep original)
        mono: Convert to mono if True

    Returns:
        audio: Audio data normalized to [-1, 1]
        sr: Sample rate

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not supported
    """

def save_audio(
    filepath: str,
    audio: np.ndarray,
    sample_rate: int = 44100,
    subtype: str = 'PCM_24'
):
    """
    Save audio to file

    Args:
        filepath: Output file path
        audio: Audio data (will be clipped to [-1, 1])
        sample_rate: Sample rate in Hz
        subtype: Bit depth ('PCM_16', 'PCM_24', 'FLOAT')
    """

def validate_audio(audio: np.ndarray) -> None:
    """
    Validate audio array format

    Raises:
        ValueError: If audio format is invalid
    """
```

### 8. Validation (`utils/validation.py`)

**Purpose:** Audio quality metrics for testing

**Key Functions:**

```python
import numpy as np
from typing import Dict

def compute_snr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio in dB

    Args:
        original: Reference signal
        processed: Processed signal

    Returns:
        SNR in dB (higher is better, >60 dB is excellent)
    """

def compute_thd(
    audio: np.ndarray,
    sample_rate: int,
    fundamental_freq: float,
    n_harmonics: int = 5
) -> float:
    """
    Calculate Total Harmonic Distortion percentage

    Args:
        audio: Audio signal
        sample_rate: Sample rate
        fundamental_freq: Fundamental frequency in Hz
        n_harmonics: Number of harmonics to analyze

    Returns:
        THD as percentage (0-100, lower is better, <1% is excellent)
    """

def measure_latency(fft_size: int, hop_size: int, sample_rate: int) -> float:
    """
    Calculate processing latency in milliseconds

    Returns:
        Latency in ms
    """

def check_scale_conformance(
    audio: np.ndarray,
    sample_rate: int,
    root_midi: int,
    scale_type: str,
    tolerance_cents: float = 50.0
) -> Dict[str, float]:
    """
    Measure how well audio conforms to musical scale

    Args:
        audio: Audio signal
        sample_rate: Sample rate
        root_midi: Scale root note
        scale_type: Scale name
        tolerance_cents: Tolerance window in cents

    Returns:
        Dictionary with:
            - 'conformance': % of energy within tolerance
            - 'mean_deviation': Average cents from nearest scale note
            - 'max_deviation': Maximum cents from scale
    """

def compute_spectral_centroid(magnitude: np.ndarray, frequencies: np.ndarray) -> float:
    """Calculate spectral centroid (brightness measure)"""

def compute_rms(audio: np.ndarray) -> float:
    """Calculate RMS level"""
```

## Test-Driven Development Approach

### Test Execution Order

```bash
# 1. Unit tests (run first, fast)
pytest tests/unit/ -v

# 2. Integration tests (slower, full pipeline)
pytest tests/integration/ -v

# 3. All tests with coverage
pytest --cov=harmonic_shifter --cov-report=html

# 4. Benchmarks (optional)
pytest tests/ --benchmark-only
```

### Test Coverage Requirements

- **Unit tests:** >90% coverage for core modules
- **Integration tests:** Complete pipeline validation
- **Audio quality tests:** All metrics within specification

### Critical Test Cases

#### Unit Tests

**test_stft.py:**

```python
import numpy as np
import pytest
from harmonic_shifter.core.stft import stft, istft

def test_perfect_reconstruction():
    """STFT → ISTFT should recover original signal within numerical precision"""
    signal = np.random.randn(44100)  # 1 second at 44.1kHz
    mag, phase = stft(signal, fft_size=2048, hop_size=512)
    reconstructed = istft(mag, phase, hop_size=512)

    # Trim edges (boundary effects)
    error = np.mean(np.abs(signal[2048:-2048] - reconstructed[2048:-2048]))
    assert error < 1e-6, f"Reconstruction error: {error}"

def test_parseval_theorem():
    """Energy conservation in frequency domain"""
    signal = np.random.randn(44100)
    mag, phase = stft(signal, fft_size=2048, hop_size=512)

    time_energy = np.sum(signal ** 2)
    freq_energy = np.sum(mag ** 2) / mag.shape[0]  # Average over frames

    relative_error = np.abs(time_energy - freq_energy) / time_energy
    assert relative_error < 0.01, f"Energy not conserved: {relative_error}"

@pytest.mark.parametrize("window", ['hann', 'hamming', 'blackman'])
def test_different_window_functions(window):
    """Test various window functions"""
    signal = np.random.randn(44100)
    mag, phase = stft(signal, window=window)
    reconstructed = istft(mag, phase, window=window)
    assert reconstructed.shape[0] > 0
```

**test_frequency_shifter.py:**

```python
import numpy as np
import pytest
from harmonic_shifter.core.frequency_shifter import FrequencyShifter
from harmonic_shifter.core.stft import stft, istft

def test_sine_wave_shift():
    """440 Hz + 100 Hz shift = 540 Hz output"""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.sin(2 * np.pi * 440 * t)

    shifter = FrequencyShifter(sr, fft_size=4096)
    mag, phase = stft(signal, fft_size=4096)

    shifted_mag, shifted_phase = shifter.shift(mag, phase, shift_hz=100)
    output = istft(shifted_mag, shifted_phase)

    # Analyze output frequency
    from scipy.fft import rfft, rfftfreq
    spectrum = np.abs(rfft(output))
    freqs = rfftfreq(len(output), 1/sr)
    peak_freq = freqs[np.argmax(spectrum)]

    assert np.abs(peak_freq - 540) < 5, f"Expected 540 Hz, got {peak_freq} Hz"

def test_negative_shift():
    """440 Hz - 100 Hz shift = 340 Hz output"""
    sr = 44100
    t = np.linspace(0, 1.0, sr)
    signal = np.sin(2 * np.pi * 440 * t)

    shifter = FrequencyShifter(sr, fft_size=4096)
    mag, phase = stft(signal, fft_size=4096)
    shifted_mag, shifted_phase = shifter.shift(mag, phase, shift_hz=-100)
    output = istft(shifted_mag, shifted_phase)

    from scipy.fft import rfft, rfftfreq
    spectrum = np.abs(rfft(output))
    freqs = rfftfreq(len(output), 1/sr)
    peak_freq = freqs[np.argmax(spectrum)]

    assert np.abs(peak_freq - 340) < 5, f"Expected 340 Hz, got {peak_freq} Hz"

def test_aliasing_prevention():
    """Frequencies > Nyquist are handled correctly"""
    sr = 44100
    shifter = FrequencyShifter(sr, fft_size=2048)

    # Create spectrum with energy near Nyquist
    mag = np.zeros((100, 1025))  # 100 frames, 1025 bins
    mag[:, -50:] = 1.0  # Energy near Nyquist
    phase = np.zeros_like(mag)

    # Try to shift beyond Nyquist
    shifted_mag, shifted_phase = shifter.shift(mag, phase, shift_hz=10000)

    # Should not raise error and should clip/handle properly
    assert np.all(np.isfinite(shifted_mag))
    assert np.all(np.isfinite(shifted_phase))
```

**test_quantizer.py:**

```python
import numpy as np
import pytest
from harmonic_shifter.core.quantizer import MusicalQuantizer
from harmonic_shifter.theory.tuning import midi_to_freq, freq_to_midi

def test_exact_scale_note():
    """Frequency on scale note should not change (or change minimally)"""
    quantizer = MusicalQuantizer(root_midi=60, scale_type='major')  # C major

    # C major scale: C, D, E, F, G, A, B
    c_freq = midi_to_freq(60)  # C4
    e_freq = midi_to_freq(64)  # E4
    g_freq = midi_to_freq(67)  # G4

    freqs = np.array([c_freq, e_freq, g_freq])
    quantized = quantizer.quantize_frequencies(freqs, strength=1.0)

    assert np.allclose(freqs, quantized, atol=1.0), "Scale notes changed"

def test_between_notes():
    """Quantize to nearest scale note"""
    quantizer = MusicalQuantizer(root_midi=60, scale_type='major')

    # 450 Hz is between C# (277.18) and D (293.66) at higher octave
    # In C major, should quantize to nearest scale note
    freq = np.array([450.0])
    quantized = quantizer.quantize_frequencies(freq, strength=1.0)

    quantized_midi = freq_to_midi(quantized[0])
    # Check that it's a valid scale note
    relative_note = (round(quantized_midi) - 60) % 12
    assert relative_note in [0, 2, 4, 5, 7, 9, 11], f"Not in C major scale: {relative_note}"

def test_strength_parameter():
    """strength=0 → no change, strength=1 → full quantization"""
    quantizer = MusicalQuantizer(root_midi=60, scale_type='major')

    # Frequency between notes
    original = np.array([450.0])

    # No quantization
    result_0 = quantizer.quantize_frequencies(original, strength=0.0)
    assert np.allclose(result_0, original), "strength=0 should not change freq"

    # Full quantization
    result_1 = quantizer.quantize_frequencies(original, strength=1.0)
    assert not np.allclose(result_1, original), "strength=1 should quantize"

    # Partial quantization
    result_05 = quantizer.quantize_frequencies(original, strength=0.5)
    assert np.allclose(result_05, (original + result_1) / 2, atol=5.0)

@pytest.mark.parametrize("scale", [
    'major', 'minor', 'pentatonic_major', 'pentatonic_minor',
    'dorian', 'mixolydian', 'blues'
])
def test_all_scales(scale):
    """Verify all scale definitions work"""
    quantizer = MusicalQuantizer(root_midi=60, scale_type=scale)
    freqs = np.array([440.0, 880.0, 1000.0])
    quantized = quantizer.quantize_frequencies(freqs, strength=1.0)

    assert len(quantized) == len(freqs)
    assert np.all(quantized > 0)
```

**test_tuning.py:**

```python
import numpy as np
import pytest
from harmonic_shifter.theory.tuning import (
    freq_to_midi, midi_to_freq, quantize_to_scale, cents_difference
)

def test_freq_midi_roundtrip():
    """freq → MIDI → freq should be identical"""
    test_freqs = [220.0, 440.0, 880.0, 1000.0]
    for freq in test_freqs:
        midi = freq_to_midi(freq)
        freq_back = midi_to_freq(midi)
        assert np.abs(freq - freq_back) < 0.01, f"Roundtrip failed for {freq} Hz"

def test_a4_reference():
    """MIDI 69 should be 440 Hz"""
    assert np.abs(midi_to_freq(69) - 440.0) < 0.01
    assert np.abs(freq_to_midi(440.0) - 69) < 0.01

def test_octave_doubling():
    """Each octave should double frequency"""
    c4 = midi_to_freq(60)  # C4
    c5 = midi_to_freq(72)  # C5
    c6 = midi_to_freq(84)  # C6

    assert np.abs(c5 / c4 - 2.0) < 0.001
    assert np.abs(c6 / c5 - 2.0) < 0.001

def test_cents_calculation():
    """Test cents difference calculation"""
    # Semitone should be 100 cents
    c4 = midi_to_freq(60)
    c_sharp_4 = midi_to_freq(61)
    cents = cents_difference(c4, c_sharp_4)
    assert np.abs(cents - 100) < 0.1

    # Octave should be 1200 cents
    c5 = midi_to_freq(72)
    cents = cents_difference(c4, c5)
    assert np.abs(cents - 1200) < 0.1
```

#### Integration Tests

**test_pipeline.py:**

```python
import numpy as np
import pytest
from harmonic_shifter.processing.processor import HarmonicShifter

def test_sine_wave_processing():
    """Single sine wave through full pipeline"""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.sin(2 * np.pi * 440 * t)

    processor = HarmonicShifter(sample_rate=sr, fft_size=4096)
    processor.set_scale(root_midi=60, scale_type='major')

    output = processor.process(signal, shift_hz=100, quantize_strength=1.0)

    assert output.shape == signal.shape
    assert np.all(np.isfinite(output))
    assert np.max(np.abs(output)) <= 1.0  # No clipping

def test_harmonic_series():
    """Multiple harmonics maintain relationships when quantized"""
    sr = 44100
    t = np.linspace(0, 1.0, sr)

    # Create harmonic series: 440 + 880 + 1320 Hz
    signal = (np.sin(2 * np.pi * 440 * t) +
              0.5 * np.sin(2 * np.pi * 880 * t) +
              0.25 * np.sin(2 * np.pi * 1320 * t))
    signal /= np.max(np.abs(signal))

    processor = HarmonicShifter(sample_rate=sr)
    processor.set_scale(root_midi=69, scale_type='major')  # A major

    output = processor.process(signal, shift_hz=50, quantize_strength=1.0)

    assert output.shape == signal.shape
    assert np.all(np.isfinite(output))

def test_polyphonic_material():
    """Chord processing - multiple simultaneous notes"""
    sr = 44100
    t = np.linspace(0, 1.0, sr)

    # C major chord: C4 + E4 + G4
    c4 = np.sin(2 * np.pi * midi_to_freq(60) * t)
    e4 = np.sin(2 * np.pi * midi_to_freq(64) * t)
    g4 = np.sin(2 * np.pi * midi_to_freq(67) * t)
    chord = (c4 + e4 + g4) / 3

    processor = HarmonicShifter(sample_rate=sr)
    processor.set_scale(root_midi=60, scale_type='major')

    output = processor.process(chord, shift_hz=100, quantize_strength=1.0)

    assert output.shape == chord.shape
    assert np.all(np.isfinite(output))

def test_white_noise():
    """Noise texture preservation"""
    sr = 44100
    noise = np.random.randn(sr) * 0.5  # 1 second

    processor = HarmonicShifter(sample_rate=sr)
    output = processor.process(noise, shift_hz=100, quantize_strength=0.0)

    # Should process without errors
    assert output.shape == noise.shape
    assert np.all(np.isfinite(output))
```

**test_audio_quality.py:**

```python
import numpy as np
import pytest
from harmonic_shifter.processing.processor import HarmonicShifter
from harmonic_shifter.utils.validation import (
    compute_snr, compute_thd, measure_latency, check_scale_conformance
)

def test_snr_threshold():
    """SNR > 60 dB for clean signals"""
    sr = 44100
    t = np.linspace(0, 1.0, sr)
    signal = np.sin(2 * np.pi * 440 * t)

    processor = HarmonicShifter(sample_rate=sr)
    output = processor.process(signal, shift_hz=0, quantize_strength=0.0)

    snr = compute_snr(signal, output)
    assert snr > 60, f"SNR too low: {snr} dB"

def test_thd_threshold():
    """THD < 1% for sine waves"""
    sr = 44100
    t = np.linspace(0, 1.0, sr)
    signal = np.sin(2 * np.pi * 440 * t)

    processor = HarmonicShifter(sample_rate=sr)
    output = processor.process(signal, shift_hz=100, quantize_strength=0.0)

    thd = compute_thd(output, sr, fundamental_freq=540)
    assert thd < 1.0, f"THD too high: {thd}%"

def test_latency_acceptable():
    """Latency < 150 ms"""
    latency = measure_latency(fft_size=4096, hop_size=1024, sample_rate=44100)
    assert latency < 150, f"Latency too high: {latency} ms"

def test_scale_conformance():
    """Quantized output matches scale (>95% energy)"""
    sr = 44100
    t = np.linspace(0, 1.0, sr)

    # Create signal with multiple frequencies
    signal = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 550 * t)
    signal /= np.max(np.abs(signal))

    processor = HarmonicShifter(sample_rate=sr)
    processor.set_scale(root_midi=69, scale_type='major')  # A major

    output = processor.process(signal, shift_hz=0, quantize_strength=1.0)

    metrics = check_scale_conformance(output, sr, 69, 'major', tolerance_cents=50)

    assert metrics['conformance'] > 0.95, \
        f"Only {metrics['conformance']*100}% energy in scale"
```

## Example Usage

### Basic Example (`examples/basic_usage.py`)

```python
"""
Basic usage example for HarmonicShifter
"""
from harmonic_shifter import HarmonicShifter
from harmonic_shifter.audio import load_audio, save_audio

def main():
    # Initialize processor
    processor = HarmonicShifter(
        sample_rate=44100,
        fft_size=4096,
        hop_size=1024
    )

    # Configure musical scale
    processor.set_scale(
        root_midi=60,  # C4
        scale_type='major'
    )

    # Load audio
    print("Loading audio...")
    audio, sr = load_audio('input.wav')

    # Process with frequency shift and quantization
    print("Processing...")
    output = processor.process(
        audio=audio,
        shift_hz=100,  # Shift up by 100 Hz
        quantize_strength=1.0  # Full quantization to scale
    )

    # Save result
    print("Saving output...")
    save_audio('output.wav', output, sr)
    print("Done!")

if __name__ == '__main__':
    main()
```

### Parameter Sweep (`examples/parameter_sweep.py`)

```python
"""
Test different shift amounts and quantization strengths
"""
from harmonic_shifter import HarmonicShifter
from harmonic_shifter.audio import load_audio, save_audio
import os

def main():
    processor = HarmonicShifter(sample_rate=44100)
    processor.set_scale(root_midi=60, scale_type='major')

    audio, sr = load_audio('input.wav')

    # Parameter ranges
    shift_values = [-200, -100, 0, 100, 200]  # Hz
    quantize_values = [0.0, 0.5, 1.0]  # strength

    os.makedirs('output', exist_ok=True)

    for shift in shift_values:
        for quant in quantize_values:
            print(f"Processing: shift={shift}Hz, quantize={quant}")

            output = processor.process(
                audio=audio,
                shift_hz=shift,
                quantize_strength=quant
            )

            filename = f'output/shift{shift:+04d}_quant{int(quant*100):02d}.wav'
            save_audio(filename, output, sr)

    print("All variations saved to output/ directory")

if __name__ == '__main__':
    main()
```

### Batch Processing (`examples/batch_processing.py`)

```python
"""
Process multiple files with same settings
"""
from harmonic_shifter import HarmonicShifter
from harmonic_shifter.audio import load_audio, save_audio
from pathlib import Path

def process_directory(
    input_dir: str,
    output_dir: str,
    shift_hz: float = 100,
    quantize_strength: float = 1.0,
    scale_root: int = 60,
    scale_type: str = 'major'
):
    processor = HarmonicShifter(sample_rate=44100)
    processor.set_scale(root_midi=scale_root, scale_type=scale_type)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    audio_files = list(input_path.glob('*.wav'))

    for i, filepath in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Processing {filepath.name}...")

        audio, sr = load_audio(str(filepath))
        output = processor.process(audio, shift_hz, quantize_strength)

        output_file = output_path / f"shifted_{filepath.name}"
        save_audio(str(output_file), output, sr)

    print(f"Processed {len(audio_files)} files")

if __name__ == '__main__':
    process_directory(
        input_dir='input_samples',
        output_dir='output_samples',
        shift_hz=150,
        quantize_strength=1.0,
        scale_root=57,  # A3
        scale_type='minor'
    )
```

## Performance Targets

### Computational Requirements

|FFT Size|Hop Size|Latency (44.1kHz)|CPU (estimate)|RAM  |
|--------|--------|-----------------|--------------|-----|
|2048    |512     |~58 ms           |~3%           |~2 MB|
|4096    |1024    |~116 ms          |~5%           |~4 MB|
|8192    |2048    |~232 ms          |~8%           |~8 MB|

### Memory Requirements

```
STFT buffer: ~(fft_size * n_frames * 8 bytes)
Example: 4096 * 100 frames * 8 bytes = ~3.2 MB
```

## Known Challenges & Solutions

### Challenge 1: Phase Discontinuities

**Problem:** Artifacts (metallic sound) when bins are reassigned
**Solution:** Phase vocoder with instantaneous frequency tracking
**Implementation:** See `core/phase_vocoder.py`

### Challenge 2: Low-Frequency Resolution

**Problem:** 10 Hz bins too coarse for bass notes (< 100 Hz)
**Solution:** Use larger FFT (8192) for better frequency resolution
**Trade-off:** Increased latency

### Challenge 3: Transient Smearing

**Problem:** FFT smears drum hits and percussive attacks
**Solution:** Optional transient detection and bypass
**Status:** Future enhancement

### Challenge 4: Polyphonic Quantization

**Problem:** Multiple notes may quantize to same frequency
**Solution:** Peak-based processing with harmonic tracking
**Status:** Optional feature in `processing/peaks.py`

## Documentation Requirements

### API Documentation

Each module should have:

- Docstrings (Google style)
- Type hints on all functions
- Usage examples in docstrings
- Performance notes (complexity, memory)

Example:

```python
def process(
    self,
    audio: np.ndarray,
    shift_hz: float,
    quantize_strength: float = 1.0
) -> np.ndarray:
    """
    Process audio with frequency shifting and scale quantization.

    This method applies frequency shifting in the spectral domain, followed
    by optional quantization to a musical scale. The algorithm maintains
    phase coherence using phase vocoder techniques.

    Args:
        audio: Input audio signal (mono, normalized to [-1, 1])
        shift_hz: Frequency shift in Hz (positive or negative)
        quantize_strength: Scale quantization amount (0.0 to 1.0)
            - 0.0: Pure frequency shift (inharmonic)
            - 1.0: Fully quantized to scale (harmonic)
            - 0.5: 50% blend between shifted and quantized

    Returns:
        Processed audio signal (same shape as input)

    Raises:
        ValueError: If scale not set when quantize_strength > 0
        ValueError: If audio shape is invalid

    Example:
        >>> processor = HarmonicShifter(sample_rate=44100)
        >>> processor.set_scale(root_midi=60, scale_type='major')
        >>> output = processor.process(audio, shift_hz=100, quantize_strength=1.0)

    Performance:
        - Time complexity: O(N log N) per frame
        - Space complexity: O(N * frames)
        - Typical latency: 100-150ms
    """
```

### Algorithm Documentation

`docs/ALGORITHM.md` should explain:

- Step-by-step processing flow with diagrams
- Mathematical operations (reference MATH_FOUNDATION.md)
- Design decisions and rationale
- Trade-offs and limitations

### Benchmarks

`docs/BENCHMARKS.md` should include:

- CPU usage measurements (profiling data)
- Memory profiling results
- Latency measurements across configurations
- Audio quality metrics on test signals

## Success Criteria

### Minimum Viable Product (MVP)

- [ ] Pure sine wave shifts accurately (within 5 Hz)
- [ ] Harmonic series maintains relationships
- [ ] All unit tests pass with >90% coverage
- [ ] Basic examples run without errors
- [ ] Documentation complete (docstrings + README)

### Full Release v0.1.0

- [ ] All scales implemented and tested
- [ ] SNR > 60 dB on test signals
- [ ] THD < 1% on sine waves
- [ ] Python implementation performs adequately
- [ ] Clean codebase (black, ruff, mypy pass)
- [ ] Example scripts demonstrate all features

### Future Enhancements (v0.2.0+)

- [ ] Real-time processing with callback interface
- [ ] CLI tool for batch processing
- [ ] Optional GUI for parameter control
- [ ] Peak-based processing mode
- [ ] Transient preservation
- [ ] Multi-channel support

### Long-term (v1.0.0)

- [ ] VST/AU plugin (C++ port)
- [ ] GPU acceleration (CUDA/OpenCL)
- [ ] Microtonal scale support
- [ ] Advanced harmonic tracking

## Git Workflow

```bash
# Feature branch development
git checkout -b feature/stft-implementation
# ... develop and test ...
git add src/harmonic_shifter/core/stft.py tests/unit/test_stft.py
git commit -m "feat: implement STFT with perfect reconstruction"

# Bug fixes
git checkout -b fix/phase-discontinuity
git commit -m "fix: resolve phase discontinuities in frequency shifter"

# Documentation
git commit -m "docs: add API documentation for quantizer module"

# Tests
git commit -m "test: add integration tests for full pipeline"
```

### Conventional Commits

- `feat:` new feature
- `fix:` bug fix
- `test:` add/update tests
- `docs:` documentation
- `refactor:` code restructure (no behavior change)
- `perf:` performance improvement
- `style:` formatting changes
- `chore:` maintenance tasks

## CI/CD (Future)

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install uv
        run: pip install uv
      - name: Install dependencies
        run: uv pip install -e ".[dev]"
      - name: Run tests
        run: pytest --cov --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Implementation Questions for Claude Code

When starting with Claude Code, clarify these decisions:

1. **Windowing:** Use Hann (default) or Blackman-Harris (better quality)?
1. **Peak detection:** Implement in MVP or defer to v0.2.0?
1. **Transient handling:** Include separation or keep simple for MVP?
1. **Visualization:** Generate spectrograms for debugging?
1. **CLI interface:** Add command-line tool in MVP or Python API only?
1. **Test fixtures:** Generate synthetic test audio or provide sample files?

## Project Metadata

### pyproject.toml Template

```toml
[project]
name = "harmonic-frequency-shifter"
version = "0.1.0"
description = "Audio frequency shifter with musical scale quantization"
authors = [{name = "Your Name", email = "your.email@example.com"}]
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
keywords = ["audio", "dsp", "frequency-shifter", "music", "signal-processing"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Multimedia :: Sound/Audio",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "numpy>=1.24.0,<2.0.0",
    "scipy>=1.11.0",
    "soundfile>=0.12.0",
    "librosa>=0.10.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-benchmark>=4.0.0",
    "black>=23.7.0",
    "ruff>=0.0.282",
    "mypy>=1.5.0",
    "types-numpy",
]
viz = [
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/harmonic-frequency-shifter"
Documentation = "https://harmonic-frequency-shifter.readthedocs.io"
Repository = "https://github.com/yourusername/harmonic-frequency-shifter"
Issues = "https://github.com/yourusername/harmonic-frequency-shifter/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/harmonic_shifter"]

[tool.black]
line-length = 88
target-version = ['py311']

[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --strict-markers"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
]

[tool.coverage.run]
source = ["src/harmonic_shifter"]
omit = ["*/tests/*", "*/examples/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

## Contact & Contributions

This is a research prototype for exploring harmonic-preserving frequency shifting.

**Contributions welcome for:**

- Additional musical scales (microtonal, world music, custom tunings)
- GPU acceleration (CUDA/OpenCL implementation)
- Real-time processing optimization
- VST/AU plugin port documentation
- Alternative windowing functions
- Improved phase vocoder algorithms

-----

**Document Version:** 1.0
**Last Updated:** 2025-11-13
**Status:** Ready for Claude Code scaffolding

## Appendix: Quick Start Guide for Claude Code

### Step 1: Provide Context Files

```bash
# Ensure these files are available:
# 1. MATH_FOUNDATION.md - Mathematical specifications
# 2. PROJECT_SPEC.md - This file
```

### Step 2: Prompt for Claude Code

```
Please scaffold a Python project for a harmonic-preserving frequency shifter
following the PROJECT_SPEC.md file. Use MATH_FOUNDATION.md for mathematical
implementation details.

Start by:
1. Creating the complete directory structure
2. Setting up pyproject.toml with all dependencies
3. Creating test stubs for all modules (TDD approach)
4. Implementing the core STFT module first with full tests
5. Then implementing frequency_shifter, quantizer, and tuning modules
6. Finally, integrate everything in the processor module

Use pytest for testing, black for formatting, and include type hints throughout.
Focus on clean, readable code with comprehensive docstrings.
```

### Step 3: Iterative Development

After initial scaffolding, guide Claude Code through:

1. Implementing and testing each module individually
1. Running tests after each module (pytest tests/unit/)
1. Integration testing (pytest tests/integration/)
1. Creating example scripts
1. Writing documentation

### Expected Timeline

- **Day 1-2:** Project setup + STFT implementation + tests
- **Day 3-4:** Frequency shifter + quantizer + tuning modules + tests
- **Day 5-6:** Phase vocoder + processor integration + tests
- **Day 7:** Examples + documentation + polish

-----

**End of Project Specification**
