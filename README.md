# Harmonic-Preserving Frequency Shifter

A Python library for frequency shifting audio while maintaining musical harmonic relationships through intelligent scale quantization.

## Overview

This project implements a novel audio effect that combines:

- **Frequency shifting** (linear Hz offset) - shifts all frequencies by a fixed amount
- **Musical scale quantization** - snaps shifted frequencies to notes in a musical scale
- **Phase vocoder** - maintains phase coherence for artifact-free processing

Unlike traditional pitch shifters (which preserve harmonic relationships but change pitch), this tool allows you to shift frequencies linearly while keeping the output musical by quantizing to a chosen scale.

## Features

- 🎵 **Musical Scale Support** - Major, minor, pentatonic, blues, and modal scales
- 🔧 **Adjustable Quantization** - Blend from pure frequency shift to full scale quantization
- 🎚️ **High-Quality Processing** - Phase vocoder ensures minimal artifacts
- 📊 **Test-Driven** - Comprehensive test suite with >90% coverage
- 📈 **Research-Grade** - Based on solid DSP mathematical foundations

## Installation

### Using uv (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/yourusername/harmonic-frequency-shifter.git
cd harmonic-frequency-shifter
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev,viz]"
```

### Using pip

```bash
git clone https://github.com/yourusername/harmonic-frequency-shifter.git
cd harmonic-frequency-shifter
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,viz]"
```

## Quick Start

```python
from harmonic_shifter import HarmonicShifter
from harmonic_shifter.audio import load_audio, save_audio

# Initialize processor
processor = HarmonicShifter(
    sample_rate=44100,
    fft_size=4096,
    hop_size=1024
)

# Set musical scale (C major)
processor.set_scale(root_midi=60, scale_type='major')

# Load and process audio
audio, sr = load_audio('input.wav')
output = processor.process(
    audio=audio,
    shift_hz=100,  # Shift up by 100 Hz
    quantize_strength=1.0  # Full quantization to scale
)

# Save result
save_audio('output.wav', output, sr)
```

## Use Cases

### Creative Sound Design

```python
# Metallic/robotic vocal effects
output = processor.process(vocals, shift_hz=150, quantize_strength=0.0)

# Re-harmonize to different scale
processor.set_scale(root_midi=57, scale_type='minor')  # A minor
output = processor.process(audio, shift_hz=50, quantize_strength=1.0)

# Subtle detuning/chorus
output = processor.process(audio, shift_hz=5, quantize_strength=0.5)
```

### Music Production

```python
# Shift drums while keeping them in key
processor.set_scale(root_midi=60, scale_type='pentatonic_minor')
drums_shifted = processor.process(drums, shift_hz=80, quantize_strength=0.8)

# Create harmonic variations
for shift in [-100, 0, 100, 200]:
    variant = processor.process(melody, shift_hz=shift, quantize_strength=1.0)
    save_audio(f'melody_shift_{shift}.wav', variant, sr)
```

## Parameters

### HarmonicShifter Parameters

|Parameter    |Type|Range                  |Default|Description                 |
|-------------|----|-----------------------|-------|----------------------------|
|`sample_rate`|int |8000-192000            |44100  |Audio sample rate           |
|`fft_size`   |int |1024-8192              |4096   |FFT window size (power of 2)|
|`hop_size`   |int |256-4096               |1024   |Hop size between frames     |
|`window`     |str |hann, hamming, blackman|'hann' |Window function             |

### Processing Parameters

|Parameter          |Type |Range             |Default|Description                 |
|-------------------|-----|------------------|-------|----------------------------|
|`shift_hz`         |float|-1000 to +1000    |0      |Frequency shift amount in Hz|
|`quantize_strength`|float|0.0-1.0           |1.0    |Scale quantization amount   |
|`root_midi`        |int  |0-127             |60 (C4)|Root note of scale          |
|`scale_type`       |str  |major, minor, etc.|'major'|Musical scale               |

### Quantization Strength Guide

- **0.0** - Pure frequency shift (inharmonic, metallic)
- **0.25** - Slight musical pull
- **0.50** - Balanced between shifted and quantized
- **0.75** - Mostly musical
- **1.0** - Fully quantized to scale (harmonic)

## Supported Scales

- **Western:** major, minor, harmonic_minor, melodic_minor
- **Modes:** dorian, phrygian, lydian, mixolydian, aeolian, locrian
- **Pentatonic:** pentatonic_major, pentatonic_minor
- **Blues:** blues
- **Chromatic:** chromatic

See `src/harmonic_shifter/theory/scales.py` for the complete list.

## Architecture

```
Input Audio
    ↓
[STFT] → Magnitude & Phase
    ↓
[Frequency Shifter] → Bin reassignment
    ↓
[Musical Quantizer] → Scale quantization
    ↓
[Phase Vocoder] → Phase coherence
    ↓
[ISTFT] → Output Audio
```

## Performance

|FFT Size|Hop Size|Latency|CPU   |Quality  |
|--------|--------|-------|------|---------|
|2048    |512     |~58 ms |Low   |Good     |
|4096    |1024    |~116 ms|Medium|Excellent|
|8192    |2048    |~232 ms|High  |Best     |

## Documentation

- **[Mathematical Foundation](MATH_FOUNDATION.md)** - Detailed algorithm mathematics
- **[Project Specification](PROJECT_SPEC.md)** - Complete implementation spec
- **[API Documentation](docs/API.md)** - Full API reference
- **[Algorithm Details](docs/ALGORITHM.md)** - Processing pipeline explanation
- **[Benchmarks](docs/BENCHMARKS.md)** - Performance measurements

## Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest --cov=harmonic_shifter --cov-report=html

# Specific test
pytest tests/unit/test_stft.py::test_perfect_reconstruction
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

### Project Structure

```
src/harmonic_shifter/
├── core/              # Core DSP algorithms
│   ├── stft.py
│   ├── frequency_shifter.py
│   ├── quantizer.py
│   └── phase_vocoder.py
├── theory/            # Musical theory
│   ├── scales.py
│   └── tuning.py
├── processing/        # Main processor
│   └── processor.py
├── audio/             # I/O utilities
│   └── io.py
└── utils/             # Validation & visualization
    ├── validation.py
    └── visualization.py
```

## Examples

Check the `examples/` directory for more:

- `basic_usage.py` - Simple frequency shifting
- `parameter_sweep.py` - Test different parameters
- `batch_processing.py` - Process multiple files

## Roadmap

### v0.1.0 (Current)

- [x] Core STFT/ISTFT implementation
- [x] Frequency shifter
- [x] Musical quantizer
- [x] Phase vocoder
- [x] All major scales
- [x] Test suite (>90% coverage)

### v0.2.0 (Planned)

- [ ] Real-time processing mode
- [ ] CLI tool
- [ ] Spectral peak detection
- [ ] Transient preservation
- [ ] Multi-channel support

### v1.0.0 (Future)

- [ ] VST/AU plugin (C++ port)
- [ ] GPU acceleration
- [ ] Microtonal scales
- [ ] Advanced harmonic tracking

## Contributing

Contributions are welcome! Areas of interest:

- Additional musical scales (world music, microtonal)
- Performance optimizations
- GUI development
- VST port assistance
- Documentation improvements

Please ensure:

- Tests pass (`pytest`)
- Code is formatted (`black src/ tests/`)
- Type hints are included
- Documentation is updated

## Known Limitations

1. **Latency** - ~100-230ms depending on FFT size (not suitable for live performance)
1. **Transients** - Percussive material may smear slightly (FFT artifact)
1. **Low frequencies** - Bass notes may have coarse quantization (<100 Hz)
1. **Mono only** - Currently only processes mono audio

## Theory Background

This project implements a hybrid approach:

1. **Frequency Shifting** uses single-sideband modulation in the frequency domain
1. **Scale Quantization** maps frequencies to nearest musical scale notes
1. **Phase Vocoder** maintains phase coherence to prevent artifacts

Traditional frequency shifters create inharmonic, metallic sounds. By adding musical quantization, we maintain the frequency-shifting character while keeping the output musical.

See [MATH_FOUNDATION.md](MATH_FOUNDATION.md) for complete mathematical details.

## Research Applications

This algorithm could be useful for:

- Music information retrieval (MIR) research
- Audio effect development
- Creative music production tools
- Sound design for film/games
- Educational demonstrations of DSP concepts

## Citation

If you use this in academic work, please cite:

```bibtex
@software{harmonic_frequency_shifter,
  title={Harmonic-Preserving Frequency Shifter},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/harmonic-frequency-shifter}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Based on established DSP techniques:

- Phase vocoder (Laroche & Dolson, 1999)
- STFT overlap-add (Allen & Rabiner, 1977)
- Musical scale quantization (original contribution)

## Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/harmonic-frequency-shifter/issues)
- **Documentation:** [Read the Docs](https://harmonic-frequency-shifter.readthedocs.io)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/harmonic-frequency-shifter/discussions)

-----

**Status:** Alpha (v0.1.0)
**Python:** 3.11+
**License:** MIT
**Author:** Your Name

**⭐ Star this repo if you find it useful!**
