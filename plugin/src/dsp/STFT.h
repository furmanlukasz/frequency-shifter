#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

namespace fshift
{

/**
 * Window function types for STFT analysis/synthesis.
 */
enum class WindowType
{
    Hann,
    Hamming,
    Blackman
};

/**
 * STFT - Short-Time Fourier Transform implementation.
 *
 * Provides windowed FFT analysis and overlap-add synthesis for
 * time-frequency processing of audio signals.
 *
 * Based on the Python implementation in harmonic_shifter/core/stft.py
 */
class STFT
{
public:
    /**
     * Construct STFT processor.
     *
     * @param fftSize FFT window size (must be power of 2)
     * @param hopSize Hop size between frames in samples
     * @param windowType Window function type
     */
    STFT(int fftSize = 4096, int hopSize = 1024, WindowType windowType = WindowType::Hann);

    ~STFT() = default;

    /**
     * Prepare the STFT processor for a given sample rate.
     */
    void prepare(double sampleRate);

    /**
     * Reset internal state.
     */
    void reset();

    /**
     * Perform forward STFT on an input frame.
     *
     * @param inputFrame Time-domain samples (fftSize samples)
     * @return Pair of (magnitude, phase) vectors
     */
    std::pair<std::vector<float>, std::vector<float>> forward(const std::vector<float>& inputFrame);

    /**
     * Perform inverse STFT to reconstruct time-domain signal.
     *
     * @param magnitude Magnitude spectrum
     * @param phase Phase spectrum in radians
     * @return Time-domain frame (fftSize samples)
     */
    std::vector<float> inverse(const std::vector<float>& magnitude, const std::vector<float>& phase);

    /**
     * Get frequency values for each FFT bin.
     *
     * @return Vector of frequency values in Hz
     */
    std::vector<float> getFrequencyBins() const;

    // Getters
    int getFFTSize() const { return fftSize; }
    int getHopSize() const { return hopSize; }
    int getNumBins() const { return numBins; }
    double getSampleRate() const { return sampleRate; }
    float getBinResolution() const { return binResolution; }

private:
    /**
     * Create window function.
     */
    void createWindow();

    /**
     * Perform FFT using Cooley-Tukey algorithm.
     */
    void fft(std::vector<std::complex<float>>& x);

    /**
     * Perform inverse FFT.
     */
    void ifft(std::vector<std::complex<float>>& x);

    /**
     * Bit-reversal permutation.
     */
    void bitReverse(std::vector<std::complex<float>>& x);

    int fftSize;
    int hopSize;
    int numBins;
    WindowType windowType;
    double sampleRate;
    float binResolution;

    std::vector<float> window;
    std::vector<float> windowSquared;
    std::vector<std::complex<float>> fftBuffer;

    // Pre-computed twiddle factors for FFT
    std::vector<std::complex<float>> twiddleFactors;
};

} // namespace fshift
