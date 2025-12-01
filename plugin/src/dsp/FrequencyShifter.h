#pragma once

#include <vector>
#include <cmath>
#include <utility>

namespace fshift
{

/**
 * FrequencyShifter - Frequency shifting in the spectral domain.
 *
 * Implements linear frequency shifting by reassigning FFT bins.
 * All frequencies are shifted by a fixed Hz amount.
 *
 * Based on the Python implementation in harmonic_shifter/core/frequency_shifter.py
 */
class FrequencyShifter
{
public:
    /**
     * Construct frequency shifter.
     *
     * @param sampleRate Audio sample rate in Hz
     * @param fftSize FFT size used for analysis
     */
    FrequencyShifter(double sampleRate, int fftSize);

    ~FrequencyShifter() = default;

    /**
     * Shift all frequencies by shiftHz in the spectral domain.
     *
     * @param magnitude Magnitude spectrum (numBins elements)
     * @param phase Phase spectrum in radians (numBins elements)
     * @param shiftHz Amount to shift in Hz (can be negative)
     * @return Pair of (shifted_magnitude, shifted_phase)
     */
    std::pair<std::vector<float>, std::vector<float>> shift(
        const std::vector<float>& magnitude,
        const std::vector<float>& phase,
        float shiftHz);

    /**
     * Get the bin index corresponding to a frequency in Hz.
     *
     * @param frequencyHz Frequency in Hz
     * @return Bin index (can be fractional)
     */
    float frequencyToBin(float frequencyHz) const;

    /**
     * Get the frequency in Hz corresponding to a bin index.
     *
     * @param bin Bin index
     * @return Frequency in Hz
     */
    float binToFrequency(int bin) const;

    /**
     * Get shifted frequency values for each bin.
     *
     * @param shiftHz Frequency shift amount in Hz
     * @return Vector of shifted frequencies for each bin
     */
    std::vector<float> getShiftedFrequencies(float shiftHz) const;

    // Getters
    double getSampleRate() const { return sampleRate; }
    int getFFTSize() const { return fftSize; }
    int getNumBins() const { return numBins; }
    float getBinResolution() const { return binResolution; }

private:
    double sampleRate;
    int fftSize;
    int numBins;
    float binResolution;

    // Pre-computed original frequencies
    std::vector<float> originalFrequencies;
};

} // namespace fshift
