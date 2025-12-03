# DSP Research Findings: Frequency Shifter Plugin Improvements

## Executive Summary

This document presents research findings on potential improvements to the frequency shifter plugin, focusing on:
1. STFT vs Wavelet/CQT transforms
2. Transient preservation techniques
3. Phase vocoder algorithm improvements
4. Implementation-specific recommendations

**TL;DR Verdict**: Stick with STFT, but implement transient detection + HPSS or phase gradient integration (PGHI/RTPGHI) for significant quality improvements.

---

## 1. STFT vs Wavelets vs Constant-Q Transform

### Current State of the Art

| Transform | Pros | Cons | Best For |
|-----------|------|------|----------|
| **STFT** | Well-understood, efficient, invertible, real-time capable | Fixed time-freq resolution | General audio, real-time |
| **CWT (Wavelets)** | Variable resolution matching auditory perception | Computationally expensive, poor invertibility, no phase info (DWT) | Analysis only |
| **CQT** | Log frequency spacing (musical), good transient preservation at HF | Complex inversion (~55dB SNR), requires octave-by-octave processing | Pitch shifting (not freq shifting) |

### Key Research Findings

From [Nathan Ho's Wavelet Introduction](https://nathan.ho.name/posts/wavelets/):
> "For both audio analysis alone and analysis-resynthesis, the short-time Fourier transform and the phase vocoder are still the gold standard."

From [Semantic Scholar - Wavelet-based Pitch-shifting](https://www.semanticscholar.org/paper/A-Wavelet-based-Pitch-shifting-Method-Sklar/7e7954f765b800c6102acc997a5ecc576c582081):
> "Despite theoretical advantages, little has emerged in audio processing literature due to inherent limitations: lack of phase information from DWT, and high computational cost and lack of available inverse transform implementations for CWT."

### CQT Consideration

From [DAFx12 - Pitch Shifting with CQT](https://dafx12.york.ac.uk/papers/dafx12_submission_81.pdf):
> "CQT is specifically attractive for **pitch shifting** because it can be implemented by frequency translation... Furthermore, the high time resolution at high frequencies improves transient preservation."

**However**: CQT is designed for **pitch shifting** (multiplicative scaling), not **frequency shifting** (additive offset). In CQT, shifting bins translates to multiplying frequencies (log scale). For true frequency shifting (adding Hz), STFT's linear bin spacing is actually correct.

### Recommendation: STFT

**Verdict: Keep STFT** for frequency shifting. The linear frequency spacing of STFT is mathematically correct for adding a fixed Hz offset. Wavelets/CQT would be beneficial for pitch shifting but are suboptimal for frequency shifting.

**Potential Hybrid**: Consider CQT for visualization/analysis while keeping STFT for processing.

---

## 2. Transient Preservation Techniques

### The Problem

From [Wikipedia - Phase Vocoder](https://en.wikipedia.org/wiki/Phase_vocoder):
> "The phase vocoder handles sinusoid components well, but early implementations introduced considerable smearing on transient ('beat') waveforms... which renders the results 'phasey and diffuse.'"

### Current Implementation Gap

Your current `PhaseVocoder.cpp` implements Laroche & Dolson's identity phase locking but **lacks explicit transient detection**. This is a significant improvement opportunity.

### Solution A: Röbel Transient Detection (IRCAM)

From [IRCAM Research - Röbel 2003](https://www.researchgate.net/publication/242019287_A_new_approach_to_transient_processing_in_the_phase_vocoder):

**Algorithm**:
1. Calculate Center of Gravity (COG) for each spectral peak
2. If COG exceeds threshold → transient detected
3. Reset phase vocoder state at transient frames
4. Process at bin level (preserves nearby stationary components)

**Key Insight**:
> "Both the transient detection and transient processing algorithms operate on the level of spectral bins, which reduces possible artifacts in stationary signal components close to spectral peaks classified as transient."

**Implementation Sketch**:
```cpp
// In PhaseVocoder::process()
float computeCOG(int peakBin, const vector<float>& magnitude, const vector<float>& phase) {
    // Center of gravity measures energy distribution around peak
    // High COG = attack transient (energy arriving)
    // Low COG = steady state or decay
    float weightedSum = 0.0f, totalWeight = 0.0f;
    for (int i = peakBin - 2; i <= peakBin + 2; ++i) {
        if (i >= 0 && i < numBins) {
            float energy = magnitude[i] * magnitude[i];
            weightedSum += i * energy;
            totalWeight += energy;
        }
    }
    return (totalWeight > 0) ? weightedSum / totalWeight - peakBin : 0.0f;
}

bool isTransient(int bin, float cog) {
    return std::abs(cog) > cogThreshold;  // ~0.3-0.5
}
```

### Solution B: Harmonic-Percussive Source Separation (HPSS)

From [ResearchGate - HPSS for TSM](https://www.researchgate.net/publication/260507822_Improving_Time-Scale_Modification_of_Music_Signals_Using_Harmonic-Percussive_Separation):

**Concept**:
1. Separate signal into harmonic (horizontal spectrogram structure) and percussive (vertical structure) components
2. Process harmonic with phase vocoder (large window)
3. Process percussive with simple OLA or pass-through (short window)
4. Recombine

**Algorithm (Median Filtering)**:
```cpp
// Harmonic-enhanced spectrogram: median filter along time axis
// Percussive-enhanced spectrogram: median filter along frequency axis
// Create soft masks from enhanced spectrograms
float H = medianFilterTime(spectrogram, bin, frame, kernelSize);
float P = medianFilterFreq(spectrogram, bin, frame, kernelSize);
float maskH = H*H / (H*H + P*P + epsilon);
float maskP = 1.0f - maskH;
```

**Benefits**:
- No explicit transient detection needed
- Handles both drums AND other transients (plucks, consonants)
- Well-suited for music with mixed content

### Solution C: Phase Gradient Heap Integration (PGHI/RTPGHI)

From [Průša & Holighaus 2022 - Phase Vocoder Done Right](https://arxiv.org/abs/2202.07382):

**Revolutionary Approach**:
> "The method does not require explicit peak picking and tracking nor does it require detection of transients and their separate treatment. Yet, the method does not suffer from the typical phase vocoder artifacts even for extreme time stretching factors."

**How it Works**:
1. Estimate phase gradients in time and frequency using centered finite differences
2. Integrate gradients starting from highest-magnitude bins (heap)
3. Automatically enforces both horizontal AND vertical phase coherence

**This is the state-of-the-art** as of 2022 and has an [open-source implementation](https://github.com/ltfat/pvdoneright).

### Recommendation: Transients

**Priority 1**: Implement Röbel-style transient detection with phase reset - relatively simple addition to existing code.

**Priority 2**: Consider HPSS pre-processing as an optional mode for drum-heavy material.

**Priority 3 (Advanced)**: Investigate RTPGHI algorithm for highest quality - would require significant refactoring but eliminates need for peak tracking AND transient detection.

---

## 3. Phase Vocoder Algorithm Improvements

### Your Current Implementation Assessment

**Strengths** (already in `PhaseVocoder.cpp`):
- Peak detection with dB threshold
- Identity phase locking (Laroche & Dolson 1999)
- Region of influence concept
- Proper instantaneous frequency estimation

**Gaps Identified**:

| Feature | Status | Impact |
|---------|--------|--------|
| Transient detection | Missing | High - causes smearing |
| Multi-resolution analysis | Missing | Medium - HF transients vs LF resolution |
| Sinusoid/noise separation | Missing | Medium - noisy components need random phase |
| Phase gradient integration | Missing | High - state-of-the-art alternative |

### Improvement 1: Random Phase for Noise

From [Laroche & Dolson 1999](https://www.ee.columbia.edu/~dpwe/papers/LaroD99-pvoc.pdf):
> "For large modification factors, locking the entire phase spectrum sounds 'rigid'. The deterministic and stochastic components of a sound are separated in the frequency domain, and only the phases of sinusoids are locked, while the remaining phases are set to random numbers."

**Implementation**:
```cpp
// In phaseLockVertical()
for (int i = 0; i < numBins; ++i) {
    bool isNearPeak = false;
    for (int peakIdx : peakIndices) {
        if (std::abs(i - peakIdx) <= regionSize) {
            isNearPeak = true;
            break;
        }
    }
    if (!isNearPeak && magnitude[i] < noiseThreshold) {
        // Randomize phase for noise-like components
        lockedPhase[i] = randomPhase();  // uniform [-pi, pi]
    }
}
```

### Improvement 2: Multi-Resolution Peak Picking

From [PhaVoRIT Research](https://www.researchgate.net/publication/228872152_PhaVoRIT_A_Phase_Vocoder_for_Real-Time_Interactive_Time-Stretching):
> "The peak detection function is made frequency dependent... closely spaced spectral peaks need to be processed separately in low-frequency regions but can be combined in high-frequency regions."

**Implementation**:
```cpp
int getRegionSize(int bin, int numBins) {
    // Logarithmic scaling: fewer peaks tracked at high frequencies
    float normalizedBin = float(bin) / numBins;
    int baseRegion = 4;
    int maxRegion = 16;
    return baseRegion + int(normalizedBin * normalizedBin * (maxRegion - baseRegion));
}
```

### Improvement 3: PVSOLA (Phase Vocoder with Synchronized Overlap-Add)

From [IRCAM DAFx11 - PVSOLA](http://recherche.ircam.fr/pub/dafx11/Papers/57_e.pdf):

Combines phase vocoder with time-domain SOLA for better coherence. The method can be understood as a frequency domain SOLA algorithm using phase vocoder for phase synchronization.

---

## 4. Multi-Resolution STFT Considerations

### The Time-Frequency Tradeoff

| FFT Size | Freq Resolution | Time Resolution | Best For |
|----------|-----------------|-----------------|----------|
| 1024 | ~43 Hz | ~23 ms | Transients, drums |
| 2048 | ~21 Hz | ~46 ms | Balanced |
| 4096 | ~11 Hz | ~93 ms | Sustained tones, bass |

### Multi-Resolution Approach

From [Phase Vocoder Research](https://en.wikipedia.org/wiki/Phase_vocoder):
> "The input signal is subdivided in three frequency bands and a different STFT window length is used in each band (multiresolution discrete Fourier transform)."

**Implementation Options**:

**Option A: Band-Split Processing**
```
Low band (20-500 Hz):   FFT 8192, excellent freq resolution
Mid band (500-4000 Hz): FFT 2048, balanced
High band (4000+ Hz):   FFT 512, excellent time resolution
```

**Option B: Adaptive Window (simpler)**
- Detect transient frames globally
- Use shorter window (1024) during transients
- Use longer window (4096) for sustained portions

---

## 5. Specific Implementation Recommendations

### Priority 1: Transient Detection (High Impact, Moderate Effort)

Add to `PhaseVocoder.h`:
```cpp
struct TransientDetector {
    float cogThreshold = 0.4f;
    float energyRatioThreshold = 3.0f;
    int lookbackFrames = 2;

    std::vector<float> previousEnergy;

    bool detectTransient(const std::vector<float>& magnitude);
    void reset();
};
```

Modify `PhaseVocoder::process()`:
```cpp
if (transientDetector.detectTransient(magnitude)) {
    // Reset phase accumulator - use input phase directly
    prevSynthPhase = phase;
    return phase;  // Skip phase vocoding for this frame
}
```

### Priority 2: Noise Component Handling (Medium Impact, Low Effort)

In `phaseLockVertical()`:
- Identify bins far from peaks with low magnitude
- Assign random phases to these "noise" bins
- Prevents "rigid" sound at high modification factors

### Priority 3: Multi-Resolution Peak Tracking (Medium Impact, Medium Effort)

Make `regionSize` frequency-dependent:
- Small regions (2-4 bins) for low frequencies
- Large regions (8-16 bins) for high frequencies
- Matches human auditory frequency discrimination

### Priority 4: Optional HPSS Pre-Processing (High Impact, High Effort)

Add as optional processing mode:
- Median filtering for H/P separation
- Route percussive component through bypass or simple OLA
- Route harmonic component through full phase vocoder

### Priority 5: PGHI Integration (Highest Quality, Highest Effort)

Consider as future major version upgrade:
- Replaces current phase locking entirely
- No peak tracking needed
- State-of-the-art quality
- Reference: https://github.com/ltfat/pvdoneright

---

## 6. Quick Wins for Your Implementation

### A. Fix Phase Locking for Frequency Shifting

Your current phase locking is designed for time-stretching. For frequency shifting, you should apply frequency shift BEFORE phase locking:

```cpp
// Current flow (suboptimal for freq shift):
// 1. Phase lock
// 2. Frequency shift

// Better flow:
// 1. Frequency shift (move bins)
// 2. THEN phase lock on shifted spectrum
```

### B. Add Soft Phase Transition at Mask Boundaries

In `SpectralMask::applyMaskToPhase()`, the current hard threshold (0.5) can cause clicks:
```cpp
// Current:
if (mask < 0.5f) wetPhase[bin] = dryPhase[bin];

// Better - smooth transition:
float smoothMask = smoothstep(mask);
wetPhase[bin] = circularInterpolate(dryPhase[bin], wetPhase[bin], smoothMask);
```

### C. Energy Conservation in Quantizer

Your quantizer accumulates magnitude but this can create energy buildup:
```cpp
// Add normalization after quantization
float inputEnergy = std::accumulate(magnitude.begin(), magnitude.end(), 0.0f,
    [](float sum, float m) { return sum + m*m; });
float outputEnergy = std::accumulate(quantizedMagnitude.begin(), quantizedMagnitude.end(), 0.0f,
    [](float sum, float m) { return sum + m*m; });
float normFactor = std::sqrt(inputEnergy / (outputEnergy + 1e-10f));
for (auto& m : quantizedMagnitude) m *= normFactor;
```

---

## 7. Literature References

### Foundational Papers
1. Laroche, J., & Dolson, M. (1999). [Improved phase vocoder time-scale modification of audio](https://www.ee.columbia.edu/~dpwe/papers/LaroD99-pvoc.pdf). IEEE TSAP.

2. Röbel, A. (2003). [A new approach to transient processing in the phase vocoder](https://www.researchgate.net/publication/242019287_A_new_approach_to_transient_processing_in_the_phase_vocoder). DAFx-03.

3. Průša, Z., & Holighaus, N. (2017/2022). [Phase Vocoder Done Right](https://arxiv.org/abs/2202.07382). EUSIPCO.

### Time-Scale Modification
4. Driedger, J., & Müller, M. (2014). [Improving Time-Scale Modification of Music Signals Using Harmonic-Percussive Separation](https://www.researchgate.net/publication/260507822_Improving_Time-Scale_Modification_of_Music_Signals_Using_Harmonic-Percussive_Separation). IEEE SPL.

5. Bonada, J. (2000). [Automatic Technique in Frequency Domain for Near-Lossless Time-Scale Modification of Audio](http://mtg.upf.edu/files/publications/ICMC2000-bonada.pdf). ICMC.

### CQT and Wavelets
6. Schörkhuber, C., et al. (2013). [Audio Pitch Shifting Using the Constant-Q Transform](https://dafx12.york.ac.uk/papers/dafx12_submission_81.pdf). JAES.

7. [Wavelet Transform vs STFT](https://dsp.stackexchange.com/questions/79586/advantage-of-stft-over-wavelet-transform) - DSP StackExchange discussion.

### Implementation Resources
8. [LTFAT Phase Vocoder](https://github.com/ltfat/pvdoneright) - Reference implementation of PGHI
9. [SiTraNo](https://github.com/himynameisfuego/SiTraNo) - STN decomposition tool
10. [Phase Vocoder Bela Tutorial](https://learn.bela.io/tutorials/c-plus-plus-for-real-time-audio-programming/phase-vocoder-part-1/)

---

## 8. Recommended Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
- [ ] Add transient detection using energy ratio method
- [ ] Implement phase reset at detected transients
- [ ] Add random phase for low-magnitude bins

### Phase 2: Quality Improvements (3-5 days)
- [ ] Frequency-dependent region size for peak detection
- [ ] Energy normalization in quantizer
- [ ] Soft phase interpolation in mask

### Phase 3: Advanced Features (1-2 weeks)
- [ ] HPSS pre-processing option
- [ ] Multi-resolution STFT mode (experimental)
- [ ] PVSOLA exploration

### Phase 4: State-of-the-Art (Future)
- [ ] RTPGHI algorithm investigation
- [ ] Neural audio codec integration (research)

---

## Conclusion

Your current implementation is solid and follows established practices (Laroche & Dolson phase locking). The main gaps are:

1. **Transient detection** - Biggest quality win for the effort
2. **Noise/sinusoid separation** - Prevents "rigid" artifacts
3. **STFT is correct** - Don't switch to wavelets for frequency shifting

The STFT remains the right choice for frequency shifting specifically because of its linear frequency spacing. Wavelets and CQT are better suited for pitch shifting (multiplicative) operations.

For maximum quality improvement with minimum refactoring, start with Röbel-style transient detection and phase reset.
