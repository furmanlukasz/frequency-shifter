# Frequency Shifter VST/AU Plugin

A cross-platform audio plugin for harmonic-preserving frequency shifting with musical scale quantization.

## Features

- **Frequency Shifting**: Shift all frequencies by a fixed Hz amount (±20,000 Hz range)
- **Logarithmic Scale Mode**: Toggle between linear and logarithmic frequency control
- **Musical Quantization**: Snap shifted frequencies to musical scales (22 scales)
- **Enhanced Phase Vocoder**: Identity phase locking (Laroche & Dolson) for minimal artifacts
- **Quality/Latency Modes**: Choose between Low Latency, Balanced, or Quality processing
- **Real-time Spectrum Analyzer**: Visualize frequency content with toggle
- **Stereo Support**: Full stereo/multi-channel processing
- **Cross-Platform**: Builds for VST3, AU (macOS), AUv3 (iOS), and Standalone

## Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| **Shift (Hz)** | -20,000 to +20,000 Hz | Frequency shift amount |
| **Log Scale** | On/Off | Toggle logarithmic dial response for fine control near zero |
| **Quantize** | 0-100% | Scale quantization strength (0% = pure shift, 100% = fully quantized) |
| **Root Note** | C, C#, D, D#, E, F, F#, G, G#, A, A#, B | Root note for the scale |
| **Scale** | 22 options | Musical scale type (see below) |
| **Quality** | 3 modes | Low Latency (~58ms) / Balanced (~116ms) / Quality (~232ms) |
| **Dry/Wet** | 0-100% | Mix between original and processed audio |
| **Spectrum** | On/Off | Toggle real-time spectrum analyzer display |

### Available Scales

- **Western**: Major, Minor, Harmonic Minor, Melodic Minor
- **Modes**: Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian
- **Pentatonic**: Major Pentatonic, Minor Pentatonic
- **Blues & Jazz**: Blues, Bebop Dominant
- **Symmetric**: Chromatic, Whole Tone, Diminished (Half-Whole), Diminished (Whole-Half)
- **World**: Arabic, Japanese, Hungarian Minor, Spanish Phrygian

### Quality Modes

| Mode | FFT Size | Hop Size | Latency | Best For |
|------|----------|----------|---------|----------|
| Low Latency | 2048 | 512 | ~58 ms | Live performance, monitoring |
| Balanced | 4096 | 1024 | ~116 ms | General use (default) |
| Quality | 8192 | 2048 | ~232 ms | Offline rendering, bass-heavy material |

## Building

### Prerequisites

- CMake 3.22+
- C++20 compatible compiler
- Git

### macOS (Universal Binary - Apple Silicon + Intel)

```bash
cd plugin

# Configure for universal binary
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="11.0"

# Build
cmake --build build --config Release

# Plugins output:
# build/FrequencyShifter_artefacts/Release/VST3/Frequency Shifter.vst3
# build/FrequencyShifter_artefacts/Release/AU/Frequency Shifter.component
```

### Windows

```bash
cd plugin
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Plugin output:
# build/FrequencyShifter_artefacts/Release/VST3/Frequency Shifter.vst3
```

### Install for Testing

**macOS:**
```bash
# VST3
cp -R "build/FrequencyShifter_artefacts/Release/VST3/Frequency Shifter.vst3" \
  ~/Library/Audio/Plug-Ins/VST3/

# AU
cp -R "build/FrequencyShifter_artefacts/Release/AU/Frequency Shifter.component" \
  ~/Library/Audio/Plug-Ins/Components/
```

**Windows:**
Copy to `C:\Program Files\Common Files\VST3\`

## Project Structure

```
plugin/
├── CMakeLists.txt           # Build configuration (JUCE, plugin settings)
├── src/
│   ├── PluginProcessor.cpp  # Audio processing, parameter handling
│   ├── PluginProcessor.h
│   ├── PluginEditor.cpp     # GUI, spectrum analyzer
│   ├── PluginEditor.h
│   └── dsp/                 # DSP algorithms
│       ├── STFT.cpp/h           # Short-Time Fourier Transform
│       ├── PhaseVocoder.cpp/h   # Phase vocoder (identity phase locking)
│       ├── FrequencyShifter.cpp/h   # Spectral bin shifting
│       ├── MusicalQuantizer.cpp/h   # Scale quantization
│       └── Scales.h             # Musical scale definitions (22 scales)
```

## DSP Algorithm

The plugin implements a sophisticated frequency shifting pipeline:

```
Audio Input
    ↓
[STFT Analysis]
    ├── Apply window function (Hann)
    └── Forward FFT
    ↓
[Phase Vocoder]
    ├── Peak detection
    ├── Instantaneous frequency estimation
    └── Identity phase locking (Laroche & Dolson)
    ↓
[Frequency Shifting]
    └── Reassign spectral bins by Hz offset
    ↓
[Musical Quantization]
    └── Snap frequencies to scale notes (with strength control)
    ↓
[ISTFT Synthesis]
    ├── Inverse FFT
    └── Overlap-add reconstruction
    ↓
Audio Output (mixed with dry signal)
```

## License

MIT License - See LICENSE file for details.

## Credits

- Phase vocoder algorithm based on Laroche & Dolson (1999) - "Improved phase vocoder time-scale modification of audio"
- JUCE framework for cross-platform audio plugin development
- Built from the Python `harmonic_shifter` prototype
