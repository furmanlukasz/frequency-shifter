#pragma once

#include <vector>
#include <cmath>
#include <numbers>

namespace fshift
{

/**
 * PhaseVocoder - Enhanced phase vocoder for maintaining phase coherence.
 *
 * This implementation uses advanced techniques from Laroche & Dolson (1999)
 * to reduce metallic artifacts in frequency-modified audio:
 *
 * 1. Peak detection and phase locking (vertical coherence)
 * 2. Proper instantaneous frequency estimation
 * 3. Identity phase locking for regions of influence around peaks
 * 4. Smooth phase propagation across frames
 *
 * Reference:
 * Laroche, J., & Dolson, M. (1999). "Improved phase vocoder time-scale
 * modification of audio." IEEE Transactions on Speech and Audio Processing.
 */
class PhaseVocoder
{
public:
    /**
     * Construct phase vocoder processor.
     *
     * @param fftSize FFT size
     * @param hopSize Hop size in samples
     * @param sampleRate Sample rate in Hz
     */
    PhaseVocoder(int fftSize, int hopSize, double sampleRate);

    ~PhaseVocoder() = default;

    /**
     * Reset internal state for new audio stream.
     */
    void reset();

    /**
     * Process a single frame with phase vocoder.
     *
     * @param magnitude Current frame magnitude spectrum
     * @param phase Current frame phase spectrum
     * @param shiftHz Frequency shift amount in Hz
     * @return Synthesized phase for the shifted spectrum
     */
    std::vector<float> process(const std::vector<float>& magnitude,
                               const std::vector<float>& phase,
                               float shiftHz);

    /**
     * Set peak detection threshold in dB.
     */
    void setPeakThresholdDb(float threshold) { peakThresholdDb = threshold; }

    /**
     * Set region of influence size for phase locking.
     */
    void setRegionSize(int size) { regionSize = size; }

    /**
     * Enable or disable phase locking.
     */
    void setUsePhaseLocking(bool enabled) { usePhaseLocking = enabled; }

private:
    /**
     * Detect spectral peaks in magnitude spectrum.
     */
    std::vector<bool> detectPeaks(const std::vector<float>& magnitude);

    /**
     * Compute instantaneous frequency for each bin.
     */
    std::vector<float> computeInstantaneousFrequency(const std::vector<float>& phasePrev,
                                                      const std::vector<float>& phaseCurr);

    /**
     * Apply vertical phase locking (Laroche & Dolson's identity phase locking).
     */
    std::vector<float> phaseLockVertical(const std::vector<float>& phase,
                                          [[maybe_unused]] const std::vector<float>& magnitude,
                                          const std::vector<bool>& peaks);

    /**
     * Synthesize phase for frequency-modified spectrum.
     */
    std::vector<float> synthesizePhase(const std::vector<float>& instFreq,
                                        const std::vector<float>& phasePrevSynth,
                                        float shiftHz);

    /**
     * Wrap phase to [-pi, pi] range.
     */
    float wrapPhase(float phase);

    int hopSize;
    int numBins;
    double sampleRate;

    // Phase vocoder state
    std::vector<float> prevMagnitude;
    std::vector<float> prevPhase;
    std::vector<float> prevSynthPhase;
    bool firstFrame;

    // Parameters
    float peakThresholdDb;
    int regionSize;
    bool usePhaseLocking;

    // Pre-computed values
    std::vector<float> binFrequencies;
    std::vector<float> expectedPhaseAdvance;
};

} // namespace fshift
