---
layout: default
title: Algorithm Details
---

# Algorithm Documentation

This page provides a comprehensive overview of the harmonic-preserving frequency shifter algorithm.

## Table of Contents

1. [Core Concept](#core-concept)
2. [Processing Pipeline](#processing-pipeline)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Key Components](#key-components)
5. [Parameter Guide](#parameter-guide)
6. [Research References](#research-references)

---

## Core Concept

### The Problem

Traditional frequency shifting adds a fixed Hz offset to all frequencies:

```
f_output = f_input + Δf
```

This **destroys harmonic relationships**. For example:

| Original | After +100 Hz Shift |
|----------|---------------------|
| 440 Hz (fundamental) | 540 Hz |
| 880 Hz (2nd harmonic) | 980 Hz |
| 1320 Hz (3rd harmonic) | 1420 Hz |

The shifted frequencies are **no longer harmonically related**, resulting in a metallic, inharmonic sound.

### Our Solution

We combine three techniques:

1. **Spectral Frequency Shifting** - Linear Hz offset in frequency domain
2. **Musical Scale Quantization** - Snap shifted frequencies to nearest scale notes
3. **Enhanced Phase Vocoder** - Maintain phase coherence to reduce artifacts

---

## Processing Pipeline

```
Input Audio (time domain)
    ↓
┌─────────────────────────────────────────────────┐
│ STEP 1: Analysis (STFT)                         │
│ - Apply windowing (Hann, Hamming, Blackman)     │
│ - Compute FFT per frame                         │
│ - Extract magnitude and phase spectra           │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ STEP 2: Frequency Shifting                      │
│ - Calculate bin shift from Hz offset            │
│ - Reassign magnitudes to new bins               │
│ - Handle aliasing at Nyquist boundary           │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ STEP 3: Musical Quantization (Optional)         │
│ - Convert bin frequencies to MIDI notes         │
│ - Quantize to nearest scale degree              │
│ - Redistribute energy to quantized bins         │
│ - Blend with original (quantize_strength param) │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ STEP 4: Phase Vocoder                           │
│ - Detect spectral peaks                         │
│ - Compute instantaneous frequencies             │
│ - Apply identity phase locking                  │
│ - Synthesize coherent phases                    │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ STEP 5: Synthesis (ISTFT)                       │
│ - Inverse FFT per frame                         │
│ - Apply synthesis window                        │
│ - Overlap-add frames                            │
│ - Window normalization                          │
└─────────────────────────────────────────────────┘
    ↓
Output Audio (time domain)
```

---

## Mathematical Foundation

### Short-Time Fourier Transform

**Forward Transform:**

```
X[k, m] = Σ(n=0 to N-1) x[n + mH] · w[n] · e^(-j2πkn/N)
```

Where:
- `k` = frequency bin index (0 to N-1)
- `m` = frame index
- `H` = hop size (samples between frames)
- `N` = FFT size
- `w[n]` = window function

**Magnitude and Phase:**

```
|X[k, m]| = sqrt(Re(X)² + Im(X)²)
φ[k, m] = atan2(Im(X), Re(X))
```

**Frequency Resolution:**

```
Δf = sample_rate / N
f[k] = k · Δf
```

Example: At 44.1kHz with 4096 FFT → ~10.77 Hz per bin

### Frequency Shifting

For each frequency bin `k`:

```
f_shifted = f[k] + shift_hz
k_new = round(f_shifted / Δf)
```

Magnitude redistribution with energy conservation:

```
|Y[k_new]| = sqrt(Σ |X[k_source]|²)
```

### Musical Quantization

**Frequency to MIDI:**

```
midi = 69 + 12 × log₂(f / 440)
```

**Scale Quantization:**

```python
relative_note = (midi - root) mod 12
closest_degree = argmin(|relative_note - scale_degrees|)
quantized_midi = root + octave × 12 + closest_degree
```

**MIDI to Frequency:**

```
f = 440 × 2^((midi - 69) / 12)
```

**Quantization Strength:**

```
f_final = (1 - α) × f_shifted + α × f_quantized
```

Where `α ∈ [0, 1]`:
- `α = 0`: Pure frequency shift (inharmonic)
- `α = 1`: Fully quantized to scale (harmonic)

### Phase Vocoder Equations

**Expected Phase Advance:**

```
φ_expected[k] = 2π × k × hop_size / fft_size
```

**Instantaneous Frequency:**

```
Δφ = (φ_curr - φ_prev - φ_expected) mod 2π
f_inst[k] = f_bin[k] + Δφ × sample_rate / (2π × hop_size)
```

**Phase Synthesis:**

```
φ_synth[k] = φ_prev_synth[k] + 2π × f_new[k] × hop_size / sample_rate
```

---

## Key Components

### 1. STFT (Short-Time Fourier Transform)

Converts audio from time domain to time-frequency representation.

**Key Parameters:**

| Parameter | Values | Trade-off |
|-----------|--------|-----------|
| FFT Size | 2048, 4096, 8192 | Larger = better frequency resolution, more latency |
| Hop Size | N/4 recommended | Smaller = better quality, more computation |
| Window | Hann (default) | Good balance of frequency/time resolution |

### 2. Frequency Shifter

Moves all frequency content by a fixed Hz amount.

**Algorithm:**
```
bin_shift = round(shift_hz / bin_resolution)
for each bin k:
    k_new = k + bin_shift
    if 0 <= k_new < num_bins:
        magnitude_new[k_new] = magnitude[k]
```

### 3. Musical Quantizer

Snaps frequencies to the nearest notes in a musical scale.

**Supported Scales:**

| Category | Scales |
|----------|--------|
| Western | Major, Minor, Harmonic Minor, Melodic Minor |
| Modes | Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian |
| Pentatonic | Major Pentatonic, Minor Pentatonic |
| Other | Blues, Chromatic, Whole Tone, Diminished |
| World | Arabic, Japanese, Spanish |

### 4. Phase Vocoder

Maintains phase coherence during spectral modifications.

**Key Techniques:**

1. **Peak Detection**: Identify spectral peaks (harmonics, formants)
2. **Identity Phase Locking**: Lock phases around peaks (Laroche & Dolson)
3. **Instantaneous Frequency**: Calculate true frequency in each bin
4. **Phase Synthesis**: Generate coherent phases for modified spectrum

---

## Parameter Guide

### Quality Modes

| Mode | FFT Size | Hop Size | Latency | Best For |
|------|----------|----------|---------|----------|
| Low Latency | 2048 | 512 | ~58 ms | Live use |
| Balanced | 4096 | 1024 | ~116 ms | General purpose |
| Quality | 8192 | 2048 | ~232 ms | Offline, bass-heavy |

### Recommended Settings

**Metallic/Robotic Effects:**
- Shift: 50-200 Hz
- Quantize: 0%
- Quality: Low Latency or Balanced

**Re-harmonization:**
- Shift: Any amount
- Quantize: 100%
- Scale: Choose your target key
- Quality: Balanced or Quality

**Subtle Chorus/Detuning:**
- Shift: 5-20 Hz
- Quantize: 30-50%
- Quality: Balanced

**Bass-Heavy Material:**
- Quality: Quality mode (8192 FFT)
- Better frequency resolution at low frequencies

---

## Research References

### Core Algorithm

1. **Laroche, J., & Dolson, M. (1999)**
   "Improved phase vocoder time-scale modification of audio"
   *IEEE Transactions on Speech and Audio Processing*

   Key contributions: Identity phase locking, peak-based processing

2. **Zölzer, U. (2011)**
   "DAFX: Digital Audio Effects" (2nd ed.)
   *Wiley*

   Comprehensive reference for STFT/ISTFT, window normalization

3. **Smith, J. O. (2011)**
   "Spectral Audio Signal Processing"
   *W3K Publishing* - [Online](https://ccrma.stanford.edu/~jos/sasp/)

   Phase unwrapping, instantaneous frequency estimation

### Additional Resources

- Flanagan & Golden (1966) - Original phase vocoder concept
- Dolson (1986) - "The phase vocoder: A tutorial"
- Průša & Holighaus (2022) - "Phase Vocoder Done Right"

---

## Performance Characteristics

### Computational Complexity

Per frame: O(N log N) for FFT operations

For 1 second of audio at 44.1kHz with N=4096, H=1024:
- Frames: ~43
- Operations: ~2.1M

### Latency

```
latency = (fft_size + hop_size) / sample_rate
```

| FFT Size | Hop Size | Latency |
|----------|----------|---------|
| 2048 | 512 | ~58 ms |
| 4096 | 1024 | ~116 ms |
| 8192 | 2048 | ~232 ms |

### Known Limitations

1. **Latency**: Not suitable for live performance (needs <10ms)
2. **Transients**: Percussive material may smear slightly
3. **Low Frequencies**: Coarse quantization below 100 Hz with small FFT
4. **Extreme Shifts**: Best quality within ±500 Hz range

---

[Back to Home](index.html)
