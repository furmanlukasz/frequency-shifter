# Mathematical Foundation: Harmonic-Preserving Frequency Shifter

## 1. Core Concepts

### 1.1 Frequency Shifting vs Pitch Shifting

**Frequency Shifting (Linear)**

- Adds/subtracts fixed Hz offset: `f_out = f_in + Δf`
- Destroys harmonic relationships
- Example: 440Hz + 100Hz → 540Hz, 880Hz + 100Hz → 980Hz (no longer octave)

**Pitch Shifting (Multiplicative)**

- Scales by ratio: `f_out = f_in × ratio`
- Preserves harmonic relationships
- Example: 440Hz × 2 → 880Hz, 880Hz × 2 → 1760Hz (maintains octave)

**This Project: Hybrid Approach**

- Apply frequency shift in spectral domain
- Quantize shifted frequencies to musical scale
- Preserve harmonic coherence through intelligent binning

## 2. Short-Time Fourier Transform (STFT)

### 2.1 Forward Transform

For audio signal `x[n]`, apply windowed FFT:

```
X[k, m] = Σ(n=0 to N-1) x[n + mH] · w[n] · e^(-j2πkn/N)
```

Where:

- `k` = frequency bin index (0 to N-1)
- `m` = frame index
- `H` = hop size (samples between frames)
- `N` = FFT size
- `w[n]` = window function (Hann, Blackman-Harris, etc.)

### 2.2 Magnitude and Phase Extraction

```
|X[k, m]| = sqrt(Re(X[k,m])² + Im(X[k,m])²)
φ[k, m] = atan2(Im(X[k,m]), Re(X[k,m]))
```

### 2.3 Frequency Resolution

```
Δf = sample_rate / N
f[k] = k · Δf
```

**Example:**

- Sample rate: 44100 Hz
- FFT size: 4096
- Resolution: 44100/4096 ≈ 10.77 Hz/bin

## 3. Frequency Shifting in Spectral Domain

### 3.1 Linear Shift Operation

For each frequency bin `k` at frequency `f[k]`:

```
f_shifted[k] = f[k] + shift_hz
k_new = round(f_shifted[k] / Δf)
```

### 3.2 Magnitude Redistribution

When shifting from bin `k` to bin `k_new`:

```
Y[k_new, m] += X[k, m]
```

Handle overlapping contributions through accumulation:

```
magnitude_accum[k_new] = sqrt(Σ |X[k, m]|²)
phase_accum[k_new] = weighted_average(φ[k, m], |X[k, m]|)
```

## 4. Musical Quantization

### 4.1 Frequency to MIDI Conversion

```
midi_note = 69 + 12 · log₂(f / 440)
```

Where:

- 69 = MIDI note for A4 (440 Hz)
- 440 = reference frequency (Hz)

### 4.2 Scale Quantization

Given scale degrees relative to root (e.g., Major scale: [0, 2, 4, 5, 7, 9, 11]):

```
function quantize_to_scale(midi_note, root, scale_degrees):
    relative_note = (midi_note - root) mod 12
    closest_degree = argmin(|relative_note - scale_degrees|)
    octave = floor((midi_note - root) / 12)
    return root + octave * 12 + scale_degrees[closest_degree]
```

### 4.3 MIDI to Frequency Conversion

```
f = 440 · 2^((midi_note - 69) / 12)
```

### 4.4 Quantization Strength Parameter

Interpolate between shifted and quantized frequencies:

```
f_final = (1 - α) · f_shifted + α · f_quantized
```

Where `α ∈ [0, 1]` is quantization strength:

- `α = 0`: pure frequency shift (inharmonic)
- `α = 1`: fully quantized to scale (harmonic)

## 5. Phase Coherence (Phase Vocoder)

### 5.1 Phase Propagation

When moving energy between bins, maintain phase relationships:

```
Δφ[k] = φ[k, m] - φ[k, m-1] - 2πkH/N
Δφ_wrapped = ((Δφ[k] + π) mod 2π) - π
φ_instantaneous[k] = 2πkH/N + Δφ_wrapped
```

### 5.2 Phase Transfer to New Bin

```
φ[k_new, m] = φ[k, m-1] + φ_instantaneous[k] · (f[k_new] / f[k])
```

This preserves frequency-dependent phase evolution.

## 6. Overlap-Add Reconstruction

### 6.1 Inverse STFT

```
y[n] = Σ(m) IFFT(Y[k, m]) · w[n - mH]
```

### 6.2 Window Normalization

For perfect reconstruction with overlap factor `R = N/H`:

```
w_normalized[n] = w[n] / (Σ(m) w[n - mH]²)
```

Common overlap factors:

- 2× (H = N/2): Hann window
- 4× (H = N/4): Better for modification
- 8× (H = N/8): Highest quality, more computation

## 7. Spectral Peak Detection (Optional Enhancement)

### 7.1 Peak Finding

Identify local maxima in magnitude spectrum:

```
is_peak[k] = |X[k]| > |X[k-1]| AND |X[k]| > |X[k+1]| AND |X[k]| > threshold
```

### 7.2 Peak Refinement (Parabolic Interpolation)

For better frequency accuracy:

```
α = |X[k-1]|
β = |X[k]|
γ = |X[k+1]|

δ = 0.5 · (α - γ) / (α - 2β + γ)
f_peak = (k + δ) · Δf
```

### 7.3 Peak-Based Processing

Only shift and quantize spectral peaks, leaving noise floor unprocessed.

## 8. Energy Conservation

### 8.1 Parseval's Theorem

Total energy must be conserved:

```
E_time = Σ |x[n]|²
E_freq = (1/N) · Σ |X[k]|²
```

### 8.2 Normalization After Binning

When multiple bins contribute to one quantized frequency:

```
|Y[k_target]| = sqrt(Σ(all contributing bins) |X[k_source]|²)
```

This maintains RMS power.

## 9. Parameter Specifications

### 9.1 Required Parameters

|Parameter        |Range         |Recommended Default    |Units     |
|-----------------|--------------|-----------------------|----------|
|FFT Size         |1024-8192     |4096                   |samples   |
|Hop Size         |N/2 to N/8    |N/4                    |samples   |
|Window Type      |-             |Hann or Blackman-Harris|-         |
|Shift Amount     |-1000 to +1000|0                      |Hz        |
|Root Note        |0-127         |60 (C4)                |MIDI      |
|Scale Type       |-             |Major                  |-         |
|Quantize Strength|0-1           |1.0                    |normalized|

### 9.2 Performance Metrics

**Latency:**

```
latency_samples = N + H
latency_ms = (N + H) / sample_rate · 1000
```

Example: N=4096, H=1024, fs=44100 → ~116ms

**CPU Complexity:**

```
O(N log N) per frame
frames_per_second = sample_rate / H
```

## 10. Scale Definitions

### 10.1 Common Scales (semitones from root)

```python
SCALES = {
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
```

## 11. Edge Cases and Considerations

### 11.1 DC and Nyquist Bins

- DC bin (k=0): Usually leave unshifted
- Nyquist bin (k=N/2): Handle carefully to avoid aliasing

### 11.2 Aliasing Prevention

After shifting, check if frequency exceeds Nyquist:

```
if f_shifted > sample_rate / 2:
    # Either clip, wrap, or discard
    f_shifted = min(f_shifted, sample_rate / 2 - Δf)
```

### 11.3 Transient Handling

For percussive/transient material:

- Consider transient/tonal separation
- Apply lighter processing to transients
- Use shorter FFT for transient frames

## 12. Testing Validation Criteria

### 12.1 Unit Tests

1. **Frequency accuracy:** Quantized frequencies must match scale frequencies within 1 cent
1. **Energy conservation:** RMS before/after within 0.1 dB
1. **Phase continuity:** No discontinuities > π at frame boundaries
1. **Scale conformance:** All output frequencies within 50 cents of scale notes (when quantize=1.0)

### 12.2 Integration Tests

1. **Sine wave:** Pure tone shifted and quantized correctly
1. **Harmonic series:** All harmonics maintain relative amplitude
1. **White noise:** Spectral shape preserved after shift
1. **Musical content:** Pitched instruments stay recognizable

### 12.3 Audio Quality Metrics

- THD (Total Harmonic Distortion) < 1%
- SNR > 60 dB
- Latency < 150ms (acceptable for creative effects)
- CPU usage < 10% single core (at 44.1kHz, 512 buffer)

## 13. Implementation Pipeline

```
Input Audio
    ↓
[Buffer] → [Windowing] → [FFT]
    ↓
[Magnitude/Phase Extraction]
    ↓
[Frequency Shifting] (bin reassignment)
    ↓
[MIDI Conversion] → [Scale Quantization]
    ↓
[Frequency → Bin Mapping]
    ↓
[Magnitude Accumulation] + [Phase Vocoder]
    ↓
[IFFT] → [Overlap-Add] → [Output Buffer]
    ↓
Output Audio
```

## References

- Laroche, J., & Dolson, M. (1999). "Improved phase vocoder time-scale modification of audio." IEEE Transactions on Speech and Audio Processing.
- Zölzer, U. (2011). "DAFX: Digital Audio Effects" (2nd ed.). Wiley.
- Smith, J. O. (2011). "Spectral Audio Signal Processing." W3K Publishing.

-----

**Document Version:** 1.0
**Last Updated:** 2025-11-13
**Author:** Mathematical specification for harmonic-preserving frequency shifter
