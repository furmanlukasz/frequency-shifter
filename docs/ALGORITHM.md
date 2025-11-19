# Algorithm Overview: Harmonic-Preserving Frequency Shifter

## Introduction

This document provides a comprehensive overview of the harmonic-preserving frequency shifter algorithm, explaining the theory, implementation, and research foundations.

## Table of Contents

1. [Core Concept](#core-concept)
2. [Processing Pipeline](#processing-pipeline)
3. [Key Components](#key-components)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Implementation Details](#implementation-details)
6. [Research References](#research-references)

---

## Core Concept

### The Problem

Traditional frequency shifting adds a fixed Hz offset to all frequencies in an audio signal:

```
f_output = f_input + Δf
```

This destroys harmonic relationships. For example:
- A harmonic series: 440 Hz, 880 Hz (2× harmonic), 1320 Hz (3× harmonic)
- After +100 Hz shift: 540 Hz, 980 Hz, 1420 Hz
- **No longer harmonically related** → sounds metallic/inharmonic

### Our Solution

We combine three techniques:

1. **Spectral Frequency Shifting** - Linear Hz offset in frequency domain
2. **Musical Scale Quantization** - Snap shifted frequencies to nearest scale notes
3. **Enhanced Phase Vocoder** - Maintain phase coherence to reduce artifacts

This creates frequency-shifted audio that:
- Maintains musical harmonic relationships
- Sounds natural (not metallic)
- Can be quantized to specific musical scales

---

## Processing Pipeline

```
Input Audio (time domain)
    ↓
┌─────────────────────────────────────────────────┐
│ STEP 1: Analysis (STFT)                         │
│ - Apply windowing (Hann, Hamming, Blackman)    │
│ - Compute FFT per frame                         │
│ - Extract magnitude and phase spectra           │
└─────────────────────────────────────────────────┘
    ↓
    magnitude[n_frames, n_bins]
    phase[n_frames, n_bins]
    ↓
┌─────────────────────────────────────────────────┐
│ STEP 2: Enhanced Phase Vocoder (Frame-by-Frame)│
│                                                  │
│ For each frame:                                 │
│   a) Peak Detection                             │
│      - Find spectral peaks (harmonics/formants) │
│      - Identify regions of influence            │
│                                                  │
│   b) Instantaneous Frequency Estimation         │
│      - Compute phase deviation from expected    │
│      - Calculate true instantaneous frequency   │
│                                                  │
│   c) Phase Locking (Vertical Coherence)         │
│      - Lock phases around peaks                 │
│      - Preserve harmonic structure              │
│                                                  │
│   d) Frequency Modification                     │
│      - Apply Hz shift to inst. frequencies      │
│      - Reassign magnitude to new bins           │
│                                                  │
│   e) Phase Synthesis                            │
│      - Synthesize coherent phase for new bins   │
│      - Maintain temporal phase continuity       │
└─────────────────────────────────────────────────┘
    ↓
    shifted_magnitude[n_frames, n_bins]
    synthesized_phase[n_frames, n_bins]
    ↓
┌─────────────────────────────────────────────────┐
│ STEP 3: Musical Quantization (Optional)         │
│ - Convert bin frequencies to MIDI notes         │
│ - Quantize to nearest scale degree              │
│ - Redistribute energy to quantized bins         │
│ - Blend with original (quantize_strength param) │
└─────────────────────────────────────────────────┘
    ↓
    final_magnitude[n_frames, n_bins]
    final_phase[n_frames, n_bins]
    ↓
┌─────────────────────────────────────────────────┐
│ STEP 4: Synthesis (ISTFT)                       │
│ - Inverse FFT per frame                         │
│ - Apply synthesis window                        │
│ - Overlap-add frames                            │
│ - Window normalization                          │
└─────────────────────────────────────────────────┘
    ↓
Output Audio (time domain)
```

---

## Key Components

### 1. Short-Time Fourier Transform (STFT)

**Purpose:** Convert audio from time domain to time-frequency representation

**Implementation:** `src/harmonic_shifter/core/stft.py`

**Key Parameters:**
- `fft_size`: Window size (default: 4096 samples)
- `hop_size`: Frame advance (default: 1024 samples)
- `window`: Window function (default: Hann)

**Process:**
```python
X[k, m] = Σ(n=0 to N-1) x[n + mH] · w[n] · e^(-j2πkn/N)
```

Where:
- `k` = frequency bin index
- `m` = frame index
- `H` = hop size
- `w[n]` = window function

**Perfect Reconstruction:**
The ISTFT perfectly reconstructs the original signal when no modifications are made, verified by test with error < 1e-6.

---

### 2. Enhanced Phase Vocoder

**Purpose:** Maintain phase coherence during spectral modifications to reduce artifacts

**Implementation:** `src/harmonic_shifter/core/phase_vocoder.py`

**Research Foundation:**
Based on Laroche & Dolson (1999) "Improved phase vocoder time-scale modification of audio"

#### 2a. Peak Detection

**Function:** `detect_peaks(magnitude, threshold_db=-40)`

Identifies spectral peaks (harmonics, formants) that need special phase treatment.

**Algorithm:**
```python
1. Convert magnitude to dB scale
2. Set threshold (max_peak + threshold_db)
3. For each bin i:
   if mag[i] > threshold AND
      mag[i] > mag[i-1] AND
      mag[i] > mag[i+1]:
      mark as peak
```

**Why It Matters:**
Peaks represent harmonically important content. Preserving their phase relationships maintains audio quality.

#### 2b. Instantaneous Frequency Estimation

**Function:** `compute_instantaneous_frequency(phase_prev, phase_curr, hop_size, sample_rate)`

Computes the **true** frequency in each bin, accounting for phase deviation.

**Algorithm:**
```python
# Expected phase advance for bin k
expected_advance[k] = 2π × f[k] × hop_size / sample_rate

# Actual phase difference
phase_diff = phase_curr - phase_prev

# Wrap to [-π, π]
phase_diff = angle(e^(j·phase_diff))

# Phase deviation
deviation = phase_diff - expected_advance

# Wrap deviation to [-π, π]
deviation = angle(e^(j·deviation))

# Instantaneous frequency
inst_freq[k] = f[k] + deviation × sample_rate / (2π × hop_size)
```

**Why It Matters:**
Standard FFT bins assume frequencies are exact multiples of sample_rate/fft_size. Real audio has frequencies that fall between bins. Instantaneous frequency reveals the **actual** frequency, enabling accurate phase synthesis after modification.

#### 2c. Vertical Phase Locking (Identity Phase Locking)

**Function:** `phase_lock_vertical(phase, magnitude, peaks, region_size=4)`

Locks phases around peaks to preserve harmonic structure.

**Algorithm:**
```python
For each peak at index p:
    # Define region of influence
    region = [p - region_size, p + region_size]

    # Lock phases in region relative to peak
    peak_phase = phase[p]

    for bin i in region:
        phase_offset = original_phase[i] - peak_phase
        locked_phase[i] = peak_phase + phase_offset
```

**Why It Matters:**
This is **Laroche & Dolson's key contribution**. Bins near peaks represent harmonics of the same fundamental. Locking their phases maintains the harmonic structure, preventing the "phasiness" artifact.

#### 2d. Phase Synthesis for Shifted Frequencies

**Function:** `synthesize_phase_for_modification(inst_freq, phase_prev, hop_size, sample_rate, frequency_map)`

Generates coherent phase values for frequency-modified spectrum.

**Algorithm:**
```python
# Compute phase advance based on new frequencies
phase_advance = 2π × frequency_map × hop_size / sample_rate

# Synthesize phase from previous frame
new_phase = phase_prev + phase_advance

# Wrap to [-π, π]
new_phase = angle(e^(j·new_phase))
```

**Why It Matters:**
When we shift frequencies, we can't just copy the old phase. We must synthesize new phase values that are consistent with the **new** frequencies and maintain temporal continuity.

---

### 3. Frequency Shifting

**Purpose:** Move all frequencies by a fixed Hz offset

**Implementation:** `src/harmonic_shifter/core/frequency_shifter.py`

**Algorithm:**
```python
bin_shift = round(shift_hz / bin_resolution)

for k in range(n_bins):
    k_new = k + bin_shift
    if 0 <= k_new < n_bins:
        magnitude_shifted[k_new] = magnitude[k]
        phase_shifted[k_new] = phase[k]  # With phase vocoder
```

**Aliasing Prevention:**
Bins that would shift beyond Nyquist (sample_rate/2) are discarded.

---

### 4. Musical Quantization

**Purpose:** Snap shifted frequencies to nearest notes in a musical scale

**Implementation:** `src/harmonic_shifter/core/quantizer.py`

**Algorithm:**
```python
# For each frequency bin:
1. Convert bin frequency to MIDI note number:
   midi = 69 + 12 × log₂(freq / 440)

2. Find relative position in scale:
   relative = (midi - root) mod 12

3. Find nearest scale degree:
   closest = argmin(|relative - scale_degrees|)

4. Quantize MIDI note:
   quantized_midi = root + octave × 12 + closest_degree

5. Convert back to frequency:
   quantized_freq = 440 × 2^((quantized_midi - 69) / 12)

6. Blend with original (based on strength parameter):
   final_freq = (1 - strength) × original + strength × quantized
```

**Supported Scales:**
- Western: major, minor, harmonic_minor, melodic_minor
- Modes: dorian, phrygian, lydian, mixolydian, aeolian, locrian
- Pentatonic: pentatonic_major, pentatonic_minor
- Blues, chromatic, whole tone, diminished
- Exotic: arabic, japanese, spanish

**Energy Redistribution:**
When multiple bins quantize to the same frequency, their energies are summed to conserve total power.

---

## Mathematical Foundation

### Phase Vocoder Equations

#### Expected Phase Advance
```
φ_expected[k] = 2π × k × hop_size / fft_size
```

#### Phase Deviation (Unwrapped)
```
Δφ[k] = (φ_curr[k] - φ_prev[k] - φ_expected[k]) mod 2π
```
Wrapped to [-π, π]

#### Instantaneous Frequency
```
f_inst[k] = (k × sample_rate / fft_size) + (Δφ[k] × sample_rate) / (2π × hop_size)
```

#### Phase Synthesis
```
φ_synth[k] = φ_prev_synth[k] + 2π × f_new[k] × hop_size / sample_rate
```

### Musical Quantization Equations

#### Frequency to MIDI
```
MIDI = 69 + 12 × log₂(f / 440)
```

#### MIDI to Frequency
```
f = 440 × 2^((MIDI - 69) / 12)
```

#### Cents (Pitch Difference)
```
cents = 1200 × log₂(f₂ / f₁)
```

### STFT/ISTFT Equations

#### Analysis (STFT)
```
X[k, m] = Σ(n=0 to N-1) x[n + mH] · w[n] · e^(-j2πkn/N)
```

#### Synthesis (ISTFT)
```
y[n] = Σ(m) IFFT(X[k, m]) · w[n - mH] / Σ(m) w²[n - mH]
```

The division by window sum ensures perfect reconstruction.

---

## Implementation Details

### Frame-by-Frame Processing Flow

The enhanced processor works frame-by-frame to maintain phase coherence:

```python
synth_phase_prev = phase[0]  # Initialize

for frame_idx in range(n_frames):
    if frame_idx == 0:
        # First frame: no previous for analysis
        process_magnitude_only()
    else:
        # Analyze phase relationships
        inst_freq, locked_phase = propagate_phase_enhanced(
            mag_prev, phase_prev,
            mag_curr, phase_curr,
            hop_size, sample_rate
        )

        # Apply frequency shift
        shifted_freq = inst_freq + shift_hz

        # Synthesize coherent phase
        phase_advance = 2π × shifted_freq × hop_size / sample_rate
        synth_phase_curr = synth_phase_prev + phase_advance
        synth_phase_curr = wrap_to_[-π, π]

        # Shift bins
        reassign_magnitude_to_shifted_bins()

        # Update state
        synth_phase_prev = synth_phase_curr
```

### Parameter Selection Guidelines

#### FFT Size vs. Frequency Resolution

```
frequency_resolution = sample_rate / fft_size
```

| FFT Size | Resolution @ 44.1kHz | Best For |
|----------|---------------------|----------|
| 2048 | ~21.5 Hz | Fast processing, less latency |
| 4096 | ~10.8 Hz | **Recommended** - good balance |
| 8192 | ~5.4 Hz | Bass-heavy material, high quality |

**Trade-off:** Larger FFT = better frequency resolution but higher latency

#### Hop Size vs. Latency

```
latency_ms = (fft_size + hop_size) / sample_rate × 1000
```

| Hop Size | Overlap | Latency @ 4096 FFT | Quality |
|----------|---------|-------------------|---------|
| 2048 (N/2) | 50% | ~139 ms | Good |
| 1024 (N/4) | 75% | **~116 ms** | **Recommended** |
| 512 (N/8) | 87.5% | ~104 ms | Best (slower) |

**Trade-off:** Smaller hop = better quality but slower processing

#### Window Functions

| Window | Sidelobe Level | Best For |
|--------|---------------|----------|
| Hann | -31 dB | **General purpose** |
| Hamming | -43 dB | Sharper spectral features |
| Blackman | -58 dB | Maximum sidelobe rejection |

**Recommendation:** Use Hann for most applications

---

## Performance Characteristics

### Computational Complexity

**Per Frame:**
- STFT: O(N log N) - FFT
- Peak Detection: O(N) - linear scan
- Phase Vocoder: O(N) - per-bin calculations
- Quantization: O(N × log(scale_size))
- ISTFT: O(N log N) - IFFT

**Total:** O(frames × N log N)

For 1 second of audio at 44.1kHz with N=4096, H=1024:
- Frames: ~43
- Operations: ~43 × 4096 × log₂(4096) ≈ 2.1M operations

### Memory Usage

```
STFT Buffer: n_frames × n_bins × 16 bytes (complex)
            = 43 × 2049 × 16 ≈ 1.4 MB

Working Memory: 4 × n_bins × 8 bytes (temp arrays)
              = 4 × 2049 × 8 ≈ 64 KB

Total: ~1.5 MB per second of audio
```

### Latency Analysis

```
Processing Latency = FFT Size + Hop Size (samples)

At 44.1kHz with FFT=4096, Hop=1024:
Latency = (4096 + 1024) / 44100 = 116.1 ms
```

**Acceptable for:**
- Creative effects
- Offline processing
- Non-real-time applications

**Not suitable for:**
- Live performance (needs <10ms)
- Real-time monitoring
- Gaming audio

---

## Research References

### Core Algorithm

1. **Laroche, J., & Dolson, M. (1999)**
   "Improved phase vocoder time-scale modification of audio"
   *IEEE Transactions on Speech and Audio Processing*

   **Contributions used:**
   - Identity phase locking (vertical coherence)
   - Peak-based processing
   - Regions of influence around peaks

2. **Zölzer, U. (2011)**
   "DAFX: Digital Audio Effects" (2nd ed.)
   *Wiley*

   **Contributions used:**
   - STFT/ISTFT perfect reconstruction
   - Window overlap-add normalization
   - Frequency shifting in spectral domain

3. **Smith, J. O. (2011)**
   "Spectral Audio Signal Processing"
   *W3K Publishing*

   **Contributions used:**
   - Phase unwrapping techniques
   - Instantaneous frequency estimation
   - FFT bin frequency interpolation

### Musical Theory

4. **Equal Temperament Tuning System**
   - MIDI note to frequency mapping
   - Logarithmic frequency scaling
   - Cent-based pitch deviation measurement

### Additional Research

5. **Phase Vocoder Improvements (2022)**
   "Phase Vocoder Done Right" - Průša & Holighaus
   *ArXiv preprint*

   Modern perspectives on phase coherence and artifact reduction

---

## Comparison with Alternatives

### vs. Pitch Shifting (Time-Stretching)

| Feature | Frequency Shifting | Pitch Shifting |
|---------|-------------------|----------------|
| Operation | f → f + Δf (linear) | f → f × ratio (multiplicative) |
| Harmonics | Destroyed | Preserved |
| Formants | Shifted | Shifted (unless formant-corrected) |
| Tempo | Unchanged | Can change |
| Use Case | Creative effects | Key/tempo changes |

### vs. Traditional Frequency Shifter

| Feature | Our Implementation | Traditional |
|---------|-------------------|-------------|
| Phase Coherence | ✓ Enhanced vocoder | ✗ None |
| Artifacts | Low (phase locked) | High (metallic) |
| Musical Quantization | ✓ Yes | ✗ No |
| Quality | High | Low |
| Complexity | O(N log N) | O(N log N) |

---

## Known Limitations

### 1. Latency
~100-230ms depending on FFT size
- **Not suitable** for real-time performance
- **Acceptable** for creative effects, offline processing

### 2. Transient Smearing
FFT analysis smears percussive transients
- **Mitigation:** Use smaller FFT/hop for transient material
- **Future:** Implement transient/tonal separation

### 3. Low-Frequency Resolution
At 4096 FFT, resolution is ~10.8 Hz
- **Issue:** Bass notes <100 Hz have coarse quantization
- **Mitigation:** Use 8192 FFT for bass-heavy material

### 4. Extreme Shifts
Very large shifts (>500 Hz) can cause:
- Aliasing if approaching Nyquist
- Loss of spectral content
- Increased artifacts

**Recommendation:** Keep shifts in ±250 Hz range for best quality

---

## Future Enhancements

### Planned (v0.2.0)
1. **Transient/Tonal Separation**
   - Separate percussive from harmonic content
   - Process each differently
   - Reduces transient smearing

2. **Multi-Resolution Processing**
   - Different FFT sizes for different frequency bands
   - Better bass + treble quality simultaneously

3. **GPU Acceleration**
   - CUDA/OpenCL implementation
   - Real-time capable

### Research Directions
1. **Neural Phase Prediction**
   - ML model to predict optimal phases
   - Learn from training data

2. **Harmonic Tracking**
   - Track fundamental + harmonics explicitly
   - Shift each harmonic's phase independently

3. **Spectral Envelope Preservation**
   - Maintain formant structure
   - Better for vocals

---

## Conclusion

This implementation combines:
- State-of-the-art phase vocoder techniques (Laroche & Dolson)
- Musical theory (equal temperament, scale quantization)
- Robust DSP practices (STFT with perfect reconstruction)

The result is a frequency shifter that:
- ✓ Maintains audio quality (minimal artifacts)
- ✓ Preserves musical relationships (scale quantization)
- ✓ Provides creative control (adjustable quantization strength)
- ✓ Has solid theoretical foundation (research-based)

**Use Cases:**
- Creative sound design
- Harmonic shifting effects
- Musical reharmonization
- Educational DSP demonstrations

---

*Document Version: 1.0*
*Last Updated: 2025-01-19*
*Author: Harmonic Frequency Shifter Team*
