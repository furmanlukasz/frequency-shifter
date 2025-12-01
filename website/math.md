---
layout: default
title: Mathematical Foundation
---

# Mathematical Foundation

Complete mathematical specification for the harmonic-preserving frequency shifter.

## 1. Core Concepts

### Frequency Shifting vs Pitch Shifting

**Frequency Shifting (Linear)**
```
f_out = f_in + Δf
```
- Adds/subtracts fixed Hz offset
- Destroys harmonic relationships
- Creates metallic/inharmonic sounds

**Pitch Shifting (Multiplicative)**
```
f_out = f_in × ratio
```
- Scales by ratio
- Preserves harmonic relationships
- Changes perceived pitch

**Our Hybrid Approach**
1. Apply frequency shift in spectral domain
2. Quantize shifted frequencies to musical scale
3. Preserve harmonic coherence through phase vocoder

---

## 2. Short-Time Fourier Transform (STFT)

### Forward Transform

For audio signal `x[n]`, apply windowed FFT:

```
X[k, m] = Σ(n=0 to N-1) x[n + mH] · w[n] · e^(-j2πkn/N)
```

Where:
- `k` = frequency bin index (0 to N-1)
- `m` = frame index
- `H` = hop size (samples between frames)
- `N` = FFT size
- `w[n]` = window function

### Magnitude and Phase

```
|X[k, m]| = sqrt(Re(X)² + Im(X)²)
φ[k, m] = atan2(Im(X), Re(X))
```

### Frequency Resolution

```
Δf = sample_rate / N
f[k] = k · Δf
```

**Example:**
- Sample rate: 44100 Hz
- FFT size: 4096
- Resolution: 44100/4096 ≈ 10.77 Hz/bin

---

## 3. Frequency Shifting

### Linear Shift Operation

For each frequency bin `k` at frequency `f[k]`:

```
f_shifted[k] = f[k] + shift_hz
k_new = round(f_shifted[k] / Δf)
```

### Magnitude Redistribution

When multiple bins map to the same target:

```
|Y[k_target]| = sqrt(Σ |X[k_source]|²)
```

This maintains RMS power (energy conservation).

---

## 4. Musical Quantization

### Frequency to MIDI Conversion

```
midi = 69 + 12 · log₂(f / 440)
```

Where:
- 69 = MIDI note for A4 (440 Hz)
- 440 = reference frequency (Hz)

### Scale Quantization Algorithm

```python
function quantize_to_scale(midi_note, root, scale_degrees):
    relative_note = (midi_note - root) mod 12
    closest_degree = argmin(|relative_note - scale_degrees|)
    octave = floor((midi_note - root) / 12)
    return root + octave × 12 + scale_degrees[closest_degree]
```

### MIDI to Frequency Conversion

```
f = 440 · 2^((midi - 69) / 12)
```

### Quantization Strength

Interpolate between shifted and quantized:

```
f_final = (1 - α) · f_shifted + α · f_quantized
```

Where `α ∈ [0, 1]`:
- `α = 0`: pure frequency shift
- `α = 1`: fully quantized

---

## 5. Phase Vocoder

### Phase Propagation

When moving energy between bins:

```
Δφ[k] = φ[k, m] - φ[k, m-1] - 2πkH/N
Δφ_wrapped = ((Δφ[k] + π) mod 2π) - π
φ_instantaneous[k] = 2πkH/N + Δφ_wrapped
```

### Instantaneous Frequency

```
f_inst[k] = (k · sample_rate / N) + (Δφ_wrapped · sample_rate) / (2π · H)
```

### Phase Synthesis

```
φ_synth[k] = φ_prev[k] + 2π · f_new[k] · H / sample_rate
```

### Phase Transfer to New Bin

```
φ[k_new, m] = φ[k, m-1] + φ_instantaneous[k] · (f[k_new] / f[k])
```

---

## 6. Overlap-Add Reconstruction

### Inverse STFT

```
y[n] = Σ(m) IFFT(Y[k, m]) · w[n - mH]
```

### Window Normalization

For perfect reconstruction with overlap factor `R = N/H`:

```
w_normalized[n] = w[n] / (Σ(m) w²[n - mH])
```

Common overlap factors:
- 2× (H = N/2): Hann window
- 4× (H = N/4): Better for modification
- 8× (H = N/8): Highest quality

---

## 7. Energy Conservation

### Parseval's Theorem

Total energy must be conserved:

```
E_time = Σ |x[n]|²
E_freq = (1/N) · Σ |X[k]|²
```

### Normalization After Binning

```
|Y[k_target]| = sqrt(Σ |X[k_source]|²)
```

---

## 8. Scale Definitions

### Common Scales (semitones from root)

| Scale | Degrees |
|-------|---------|
| Major | [0, 2, 4, 5, 7, 9, 11] |
| Minor | [0, 2, 3, 5, 7, 8, 10] |
| Harmonic Minor | [0, 2, 3, 5, 7, 8, 11] |
| Melodic Minor | [0, 2, 3, 5, 7, 9, 11] |
| Dorian | [0, 2, 3, 5, 7, 9, 10] |
| Phrygian | [0, 1, 3, 5, 7, 8, 10] |
| Lydian | [0, 2, 4, 6, 7, 9, 11] |
| Mixolydian | [0, 2, 4, 5, 7, 9, 10] |
| Aeolian | [0, 2, 3, 5, 7, 8, 10] |
| Locrian | [0, 1, 3, 5, 6, 8, 10] |
| Pentatonic Major | [0, 2, 4, 7, 9] |
| Pentatonic Minor | [0, 3, 5, 7, 10] |
| Blues | [0, 3, 5, 6, 7, 10] |
| Chromatic | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] |
| Whole Tone | [0, 2, 4, 6, 8, 10] |
| Diminished | [0, 2, 3, 5, 6, 8, 9, 11] |

---

## 9. Performance Metrics

### Latency

```
latency_samples = N + H
latency_ms = (N + H) / sample_rate × 1000
```

Example: N=4096, H=1024, fs=44100 → ~116ms

### Computational Complexity

```
O(N log N) per frame
frames_per_second = sample_rate / H
```

---

## 10. Edge Cases

### DC and Nyquist

- DC bin (k=0): Leave unshifted
- Nyquist bin (k=N/2): Handle carefully

### Aliasing Prevention

```
if f_shifted > sample_rate / 2:
    f_shifted = sample_rate / 2 - Δf  # clip
```

---

## 11. Quality Metrics

### Target Specifications

- Frequency accuracy: Within 1 cent of target
- Energy conservation: Within 0.1 dB
- Phase continuity: No discontinuities > π
- THD: < 1%
- SNR: > 60 dB

---

## References

1. Laroche, J., & Dolson, M. (1999). "Improved phase vocoder time-scale modification of audio." *IEEE Transactions on Speech and Audio Processing.*

2. Zölzer, U. (2011). "DAFX: Digital Audio Effects" (2nd ed.). *Wiley.*

3. Smith, J. O. (2011). "Spectral Audio Signal Processing." *W3K Publishing.*

---

[Back to Home](index.html) | [Algorithm Details](algorithm.html)
