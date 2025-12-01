# Frequency Shifter VST/AU Plugin

A cross-platform audio plugin for harmonic-preserving frequency shifting.

## Features

- **Frequency Shifting**: Shift all frequencies by a fixed Hz amount (-2000 to +2000 Hz)
- **Musical Quantization**: Snap shifted frequencies to musical scales (20+ scales)
- **Enhanced Phase Vocoder**: Reduces metallic artifacts using Laroche & Dolson's algorithm
- **Cross-Platform**: Builds for VST3, AU (macOS), AUv3 (iOS), and Standalone

## Building

### Prerequisites

- CMake 3.22+
- Ninja (recommended)
- Xcode (macOS) or Visual Studio (Windows)
- C++20 compatible compiler

### macOS (Apple Silicon / Intel)

```bash
# Install dependencies (if not already installed)
brew install cmake ninja

# Configure and build
cd plugin
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Plugin outputs will be in:
# build/FrequencyShifter_artefacts/Release/
#   - VST3/Frequency Shifter.vst3
#   - AU/Frequency Shifter.component
#   - Standalone/Frequency Shifter.app
```

### VS Code Development

1. Open `FrequencyShifter.code-workspace` in VS Code
2. Install recommended extensions when prompted
3. Use CMake Tools sidebar to configure and build
4. Press `Cmd+Shift+B` to build (default task)

### Debugging

The standalone app can be debugged directly. For plugin debugging:
1. Build Debug configuration
2. Load the plugin in your DAW
3. Attach debugger to the DAW process

## Project Structure

```
plugin/
├── CMakeLists.txt          # Build configuration
├── src/
│   ├── PluginProcessor.cpp # Main audio processor
│   ├── PluginProcessor.h
│   ├── PluginEditor.cpp    # GUI
│   ├── PluginEditor.h
│   └── dsp/                # DSP algorithms
│       ├── STFT.cpp/h      # Short-Time Fourier Transform
│       ├── PhaseVocoder.cpp/h  # Phase vocoder (Laroche & Dolson)
│       ├── FrequencyShifter.cpp/h  # Spectral bin shifting
│       ├── MusicalQuantizer.cpp/h  # Scale quantization
│       └── Scales.h        # Musical scale definitions
└── .vscode/                # VS Code configuration
```

## Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| Shift (Hz) | -2000 to +2000 | Frequency shift amount |
| Quantize | 0-100% | Scale quantization strength |
| Root Note | C0-B8 | Root note for scale |
| Scale | 22 options | Musical scale type |
| Dry/Wet | 0-100% | Mix between original and processed |
| Enhanced Mode | On/Off | Enable phase vocoder |

## DSP Algorithm

The plugin implements a sophisticated frequency shifting pipeline:

1. **STFT Analysis**: Window audio into overlapping frames, compute FFT
2. **Phase Vocoder**: Estimate instantaneous frequencies, apply peak detection and phase locking
3. **Frequency Shifting**: Reassign spectral bins by shift amount
4. **Musical Quantization**: Optionally snap frequencies to scale notes
5. **ISTFT Synthesis**: Overlap-add reconstruction

## License

MIT License - See LICENSE file for details.

## Credits

Based on the Python `harmonic_shifter` library implementation.
Phase vocoder algorithm based on Laroche & Dolson (1999).
