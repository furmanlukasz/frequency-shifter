# Frequency Shifter Plugin - Architecture Reference

## Overview

A VST3/AU audio plugin built with JUCE 8.0.4 that performs **harmonic-preserving frequency shifting** with musical scale quantization. Unlike simple pitch shifting, this plugin shifts all frequencies by a fixed Hz amount while maintaining harmonic relationships through advanced phase vocoder techniques.

## Core Concepts

### Frequency Shifting vs Pitch Shifting
- **Pitch shifting**: Multiplies all frequencies by a ratio (e.g., 2x = octave up). Preserves harmonic ratios.
- **Frequency shifting**: Adds a fixed Hz offset to all frequencies. Creates inharmonic overtones and unique timbral effects.

### Processing Pipeline
```
Input Audio
    ↓
STFT Analysis (windowed FFT)
    ↓
Phase Vocoder (phase coherence)
    ↓
Frequency Shifter (bin reassignment)
    ↓
Musical Quantizer (scale snapping + drift)
    ↓
Spectral Mask (frequency-selective wet/dry)
    ↓
Spectral Delay (per-bin delays)
    ↓
STFT Synthesis (overlap-add)
    ↓
Dry/Wet Mix
    ↓
Output Audio
```

## Directory Structure

```
plugin/
├── CMakeLists.txt              # Build config (JUCE 8.0.4, VST3/AU/AUv3)
├── src/
│   ├── PluginProcessor.h/cpp   # Main audio processor
│   ├── PluginEditor.h/cpp      # GUI implementation
│   └── dsp/
│       ├── STFT.h/cpp          # Short-Time Fourier Transform
│       ├── PhaseVocoder.h/cpp  # Phase coherence (Laroche & Dolson)
│       ├── FrequencyShifter.h/cpp  # Bin-based frequency shifting
│       ├── MusicalQuantizer.h/cpp  # Scale-aware frequency snapping
│       ├── Scales.h            # Scale definitions & tuning utilities
│       ├── DriftModulator.h    # LFO/Perlin/Stochastic modulation
│       ├── SpectralMask.h      # Frequency-selective processing
│       └── SpectralDelay.h     # Per-bin delay lines
```

## DSP Components

### 1. STFT (`dsp/STFT.h/cpp`)

**Purpose**: Time-frequency analysis via windowed FFT

**Key Features**:
- Configurable FFT size (1024/2048/4096) and hop size
- Window types: Hann, Hamming, Blackman
- Manual Cooley-Tukey FFT implementation (no external dependencies)
- Returns magnitude/phase representation

**API**:
```cpp
STFT(int fftSize = 4096, int hopSize = 1024, WindowType = Hann);
void prepare(double sampleRate);
pair<vector<float>, vector<float>> forward(vector<float>& frame);  // → (mag, phase)
vector<float> inverse(vector<float>& mag, vector<float>& phase);
```

### 2. Phase Vocoder (`dsp/PhaseVocoder.h/cpp`)

**Purpose**: Maintain phase coherence during frequency modification to reduce metallic artifacts

**Algorithm**: Based on Laroche & Dolson (1999)
1. **Peak detection**: Find spectral peaks above threshold
2. **Instantaneous frequency**: Estimate true frequency from phase derivatives
3. **Phase locking**: Lock non-peak bins to nearest peak's phase (vertical coherence)
4. **Phase synthesis**: Propagate phase based on shifted instantaneous frequencies

**Parameters**:
- `peakThresholdDb`: Peak detection sensitivity (-40dB default)
- `regionSize`: Phase locking influence radius (4 bins default)
- `usePhaseLocking`: Enable/disable phase locking

### 3. Frequency Shifter (`dsp/FrequencyShifter.h/cpp`)

**Purpose**: Linear frequency shifting by bin reassignment

**Operation**:
- Calculates bin shift: `binShift = round(shiftHz / binResolution)`
- Moves magnitude/phase from source to target bins
- Energy outside valid range is discarded (anti-aliasing)

**Range**: ±20,000 Hz

### 4. Musical Quantizer (`dsp/MusicalQuantizer.h/cpp`)

**Purpose**: Snap frequencies to musical scale notes

**Features**:
- MIDI-based scale quantization
- Configurable root note and scale type (22 scales)
- Strength control (0% = bypass, 100% = full snap)
- Optional per-bin drift modulation (in cents)

**Key Method**:
```cpp
pair<vector<float>, vector<float>> quantizeSpectrum(
    magnitude, phase, sampleRate, fftSize,
    strength,           // 0-1 blending
    driftCents          // optional per-bin drift
);
```

### 5. Scales (`dsp/Scales.h`)

**Provides**:
- 22 scale types (Major, Minor modes, Pentatonic, Blues, Exotic scales)
- MIDI ↔ Frequency conversion utilities
- Scale degree lookup
- Note name utilities

**Scale Types**: Major, Minor, Natural/Harmonic/Melodic Minor, All 7 Modes, Pentatonic Major/Minor, Blues, Chromatic, Whole Tone, Diminished variants, Arabic, Japanese, Spanish

### 6. Drift Modulator (`dsp/DriftModulator.h`)

**Purpose**: Add organic pitch variation to quantized notes

**Modes**:
1. **LFO**: Sine/Triangle oscillation (predictable, continuous)
2. **Perlin**: Smooth pseudo-random noise (organic, continuous)
3. **Stochastic**: Event-based modulation with 3 sub-types:
   - **Poisson**: Random events with smooth attack/decay (bubbles)
   - **Random Walk**: Direction changes with momentum
   - **Jump Diffusion**: Small Brownian motion + occasional jumps

**Parameters**:
- `rate`: Modulation speed (0.1-10 Hz)
- `depth`: Modulation amount (0-100%, maps to ±50 cents)
- `density`: Event frequency for stochastic modes
- `smoothness`: Attack/decay characteristics

### 7. Spectral Mask (`dsp/SpectralMask.h`)

**Purpose**: Frequency-selective wet/dry blending

**Modes**:
- **LowPass**: Process frequencies below cutoff
- **HighPass**: Process frequencies above cutoff
- **BandPass**: Process frequencies between low and high cutoffs

**Parameters**:
- `lowFreq` / `highFreq`: Cutoff frequencies (20-20000 Hz, log scale)
- `transition`: Crossover width in octaves (0.1-4.0)

**Uses Hermite smoothstep** for smooth crossover curves.

### 8. Spectral Delay (`dsp/SpectralDelay.h`)

**Purpose**: Per-frequency-bin delay for diffusion effects

**Features**:
- Base delay time (10-2000ms)
- Frequency slope (low vs high freq delay differential)
- Feedback with high-frequency damping
- Additive wet/dry mix
- Gain control (-12 to +24 dB)

## Plugin Processor (`PluginProcessor.h/cpp`)

### Parameters (AudioProcessorValueTreeState)

| Parameter ID | Type | Range | Default | Description |
|-------------|------|-------|---------|-------------|
| `shiftHz` | float | -20000 to 20000 | 0 | Frequency shift in Hz |
| `quantizeStrength` | float | 0-100% | 0 | Scale quantization strength |
| `rootNote` | choice | C-B (0-11) | C | Scale root note |
| `scaleType` | choice | 0-21 | Major | Scale type |
| `dryWet` | float | 0-100% | 100 | Dry/wet mix |
| `phaseVocoder` | bool | - | true | Enhanced mode toggle |
| `qualityMode` | choice | 0-2 | Quality | FFT size/latency tradeoff |
| `logScale` | bool | - | false | Log scale for shift control |
| `driftAmount` | float | 0-100% | 0 | Drift modulation depth |
| `driftRate` | float | 0.1-10 Hz | 1 | Drift modulation speed |
| `driftMode` | choice | 0-2 | LFO | LFO/Perlin/Stochastic |
| `stochasticType` | choice | 0-2 | Poisson | Stochastic sub-type |
| `stochasticDensity` | float | 0-100% | 50 | Event frequency |
| `stochasticSmoothness` | float | 0-100% | 50 | Envelope smoothness |
| `maskEnabled` | bool | - | false | Enable spectral mask |
| `maskMode` | choice | 0-2 | BandPass | LP/HP/BP mask mode |
| `maskLowFreq` | float | 20-20000 Hz | 200 | Low cutoff |
| `maskHighFreq` | float | 20-20000 Hz | 5000 | High cutoff |
| `maskTransition` | float | 0.1-4 oct | 1 | Transition width |
| `delayEnabled` | bool | - | false | Enable spectral delay |
| `delayTime` | float | 10-2000 ms | 200 | Base delay time |
| `delaySlope` | float | -100 to 100% | 0 | Frequency slope |
| `delayFeedback` | float | 0-95% | 30 | Feedback amount |
| `delayDamping` | float | 0-100% | 30 | HF damping |
| `delayMix` | float | 0-100% | 50 | Delay wet level |
| `delayGain` | float | -12 to 24 dB | 0 | Delay output gain |

### Quality Modes (Latency/Quality Tradeoff)

| Mode | FFT Size | Hop Size | Latency |
|------|----------|----------|---------|
| Low Latency | 1024 | 256 | ~23ms |
| Balanced | 2048 | 512 | ~46ms |
| Quality | 4096 | 1024 | ~93ms |

### Processing Flow in `processBlock()`

1. Check for DSP reinitialization (quality mode change)
2. Update spectral mask if parameters changed
3. For each channel:
   - Store dry signal for mixing
   - Accumulate samples in circular buffer
   - When hop size samples accumulated:
     - Forward STFT → (magnitude, phase)
     - Apply Phase Vocoder (if enabled)
     - Apply Frequency Shifting
     - Apply Musical Quantization (with drift)
     - Apply Spectral Mask
     - Apply Spectral Delay
     - Inverse STFT with overlap-add
   - Apply dry/wet mix

## Editor/GUI (`PluginEditor.h/cpp`)

### Components
- **Spectrum Analyzer**: Real-time FFT visualization (30fps)
- **Main Shift Knob**: Large rotary with symmetric log scale
- **Scale Controls**: Root note and scale type dropdowns
- **Quality/Mode**: Latency selector, enhanced mode toggle
- **Drift Section**: Amount, rate, mode, stochastic parameters
- **Mask Section**: Enable, mode, low/high freq, transition
- **Delay Section**: Enable, time, slope, feedback, damping, mix, gain

### Color Scheme (Catppuccin-inspired)
```cpp
background = 0xFF1E1E2E;
panelBackground = 0xFF2A2A3E;
accent = 0xFF7AA2F7;
accentSecondary = 0xFF9ECE6A;
text = 0xFFCDD6F4;
textDim = 0xFF6C7086;
```

### UI Size
- Default: 640 x 610
- With Spectrum: 640 x 770

## Build System

### Requirements
- CMake 3.22+
- C++20
- JUCE 8.0.4 (fetched automatically)

### Build Commands
```bash
cd plugin
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Output Formats
- VST3
- AU (macOS)
- AUv3 (iOS/macOS)
- Standalone

### macOS Settings
- Deployment target: macOS 11.0
- Architecture: Universal (arm64 + x86_64)
- Plugins auto-copied after build

## Key Implementation Details

### Thread Safety
- All parameter values use `std::atomic<>` for audio thread access
- SpinLock for spectrum data visualization
- Parameter listener pattern for real-time updates

### Memory Management
- Pre-allocated buffers sized for max FFT (4096)
- Per-channel STFT/PhaseVocoder/FrequencyShifter instances
- Shared MusicalQuantizer and DriftModulator

### Overlap-Add
- Circular input/output buffers (2x FFT size)
- Synthesis windows applied during inverse STFT
- Proper normalization for energy conservation

## Future Enhancement Areas

Potential areas for new features:
1. **Formant preservation** - Maintain vocal characteristics during shifting
2. **Pitch tracking** - Auto-tune to detected pitch
3. **MIDI control** - Real-time parameter modulation
4. **Sidechain input** - Modulate based on external audio
5. **Preset system** - Save/recall parameter states
6. **Multi-band processing** - Independent shift per frequency band
7. **Modulation matrix** - Route LFOs/envelopes to parameters
8. **Visualization enhancements** - Mask overlay, before/after comparison
