# Harmonic-Preserving Frequency Shifter

A VST3/AU audio plugin for frequency shifting with musical scale quantization.

![Plugin Screenshot](plugin.png)

## Download

Get the latest release from the [Releases page](https://github.com/ludzeller/frequency-shifter/releases).

| Platform | Format | Download |
|----------|--------|----------|
| macOS (Universal) | VST3 | [Latest Release](https://github.com/ludzeller/frequency-shifter/releases/latest) |
| macOS (Universal) | AU | [Latest Release](https://github.com/ludzeller/frequency-shifter/releases/latest) |
| Windows | VST3 | [Latest Release](https://github.com/ludzeller/frequency-shifter/releases/latest) |

## What It Does

This plugin performs **frequency shifting** - moving all frequencies in your audio by a fixed Hz amount - while keeping the output **musical** through intelligent scale quantization.

Unlike pitch shifting (which preserves harmonic relationships), frequency shifting creates unique, often metallic or otherworldly tones. By adding scale quantization, you get the best of both worlds: the character of frequency shifting with musical coherence.

## Features

- **Frequency Shift**: ±20,000 Hz range with linear or logarithmic control
- **Musical Quantization**: Snap frequencies to any musical scale
- **22 Scale Types**: Major, minor, modes, pentatonic, blues, chromatic, world scales
- **Phase Vocoder**: High-quality processing with identity phase locking (Laroche & Dolson)
- **Quality Modes**: Low Latency, Balanced, or Quality presets
- **Real-time Spectrum Analyzer**: Visualize your frequency content
- **Dry/Wet Mix**: Blend processed and original signals
- **Stereo Support**: Full stereo/multi-channel processing

## Installation

### macOS

**VST3:**
```bash
# System-wide
/Library/Audio/Plug-Ins/VST3/

# User only
~/Library/Audio/Plug-Ins/VST3/
```

**AU (Audio Unit):**
```bash
# System-wide
/Library/Audio/Plug-Ins/Components/

# User only
~/Library/Audio/Plug-Ins/Components/
```

### Windows

**VST3:**
```
C:\Program Files\Common Files\VST3\
```

## Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| **Shift (Hz)** | ±20,000 Hz | Amount to shift frequencies. Positive = up, negative = down |
| **Quantize** | 0-100% | How strongly to snap to scale notes |
| **Root Note** | C, C#, D... B | The root of your scale |
| **Scale** | 22 types | Choose from Major, Minor, Dorian, Pentatonic, Blues, and more |
| **Dry/Wet** | 0-100% | Mix between original and processed audio |
| **Quality** | 3 modes | Low Latency (~58ms), Balanced (~116ms), Quality (~232ms) |

## Creative Tips

- **Metallic vocals**: Shift by 50-200 Hz with 0% quantization
- **Re-harmonize**: Use 100% quantization to force audio into a new scale
- **Subtle detuning**: Small shifts (5-20 Hz) with 50% quantization for chorus-like effects
- **Robotic sounds**: Large shifts with the Chromatic scale

## System Requirements

- **macOS**: 11.0+ (Intel and Apple Silicon universal binary)
- **Windows**: Windows 10+
- **DAW**: Any VST3 or AU compatible host

## Building from Source

### Requirements

- CMake 3.22+
- C++20 compiler (Clang, GCC, or MSVC)
- Git

### Build Steps

```bash
# Clone the repository
git clone https://github.com/ludzeller/frequency-shifter.git
cd frequency-shifter/plugin

# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release

# Plugins will be in:
# build/FrequencyShifter_artefacts/Release/VST3/
# build/FrequencyShifter_artefacts/Release/AU/  (macOS only)
```

### macOS Universal Binary

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="11.0"
```

## Releases

To create a new release:

1. **Tag the version:**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. The GitHub Actions release workflow will automatically:
   - Build plugins for macOS (VST3, AU) and Windows (VST3)
   - Create a GitHub Release with downloadable zip files

## Project Structure

```
frequency-shifter/
├── plugin/                 # JUCE plugin source
│   ├── src/
│   │   ├── PluginProcessor.cpp/h   # Audio processing
│   │   ├── PluginEditor.cpp/h      # GUI
│   │   └── dsp/                    # DSP modules
│   │       ├── STFT.cpp/h
│   │       ├── PhaseVocoder.cpp/h
│   │       ├── FrequencyShifter.cpp/h
│   │       ├── MusicalQuantizer.cpp/h
│   │       └── Scales.h
│   └── CMakeLists.txt
├── website/                # GitHub Pages documentation
├── docs/                   # Technical documentation
│   ├── ALGORITHM.md
│   └── PHASE_VOCODER.md
├── legacy/                 # Python prototype (reference)
└── .github/workflows/      # CI/CD
    ├── build.yml          # Build on push/PR
    ├── pages.yml          # Deploy documentation
    └── release.yml        # Create releases
```

## Algorithm

The plugin uses a sophisticated DSP pipeline:

```
Audio Input
    ↓
[STFT Analysis] → Convert to frequency domain
    ↓
[Frequency Shift] → Move all bins by Hz offset
    ↓
[Scale Quantization] → Snap to musical notes
    ↓
[Phase Vocoder] → Maintain phase coherence
    ↓
[ISTFT Synthesis] → Convert back to audio
```

For detailed algorithm documentation, see:
- [Algorithm Details](docs/ALGORITHM.md)
- [Phase Vocoder](docs/PHASE_VOCODER.md)
- [Mathematical Foundation](MATH_FOUNDATION.md)

## Documentation

Visit the [GitHub Pages site](https://ludzeller.github.io/frequency-shifter/) for:
- Full algorithm documentation
- Parameter guides
- Technical deep-dives

## Research References

Based on established DSP techniques:

- **Laroche, J., & Dolson, M. (1999)** - "Improved phase vocoder time-scale modification of audio" - Identity phase locking
- **Zölzer, U. (2011)** - "DAFX: Digital Audio Effects" - STFT/ISTFT, window normalization
- **Smith, J. O. (2011)** - "Spectral Audio Signal Processing" - Phase vocoder theory

## Legacy Python Prototype

The `legacy/` folder contains the original Python prototype used for algorithm development and testing. This is kept for reference but the C++ plugin is the production version.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/ludzeller/frequency-shifter) for:
- Issue reporting
- Pull requests
- Feature discussions

---

**Version:** 0.1.0
**Formats:** VST3, AU
**Platforms:** macOS, Windows
**License:** MIT
