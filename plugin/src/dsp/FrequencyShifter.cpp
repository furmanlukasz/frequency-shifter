#include "FrequencyShifter.h"
#include <algorithm>
#include <cmath>

namespace fshift
{

FrequencyShifter::FrequencyShifter(double sampleRate, int fftSize)
    : sampleRate(sampleRate),
      fftSize(fftSize),
      numBins(fftSize / 2 + 1),
      binResolution(static_cast<float>(sampleRate) / static_cast<float>(fftSize))
{
    // Pre-compute original frequencies for each bin
    originalFrequencies.resize(numBins);
    for (int i = 0; i < numBins; ++i)
    {
        originalFrequencies[i] = static_cast<float>(i) * binResolution;
    }
}

float FrequencyShifter::frequencyToBin(float frequencyHz) const
{
    return frequencyHz / binResolution;
}

float FrequencyShifter::binToFrequency(int bin) const
{
    return static_cast<float>(bin) * binResolution;
}

std::vector<float> FrequencyShifter::getShiftedFrequencies(float shiftHz) const
{
    std::vector<float> shifted(numBins);
    for (int i = 0; i < numBins; ++i)
    {
        shifted[i] = originalFrequencies[i] + shiftHz;
    }
    return shifted;
}

std::pair<std::vector<float>, std::vector<float>> FrequencyShifter::shift(
    const std::vector<float>& magnitude,
    const std::vector<float>& phase,
    float shiftHz)
{
    // Initialize output arrays with zeros
    std::vector<float> shiftedMagnitude(numBins, 0.0f);
    std::vector<float> shiftedPhase(numBins, 0.0f);

    // Calculate bin shift
    int binShift = static_cast<int>(std::round(shiftHz / binResolution));

    // Shift each bin
    for (int k = 0; k < numBins; ++k)
    {
        // Calculate target bin
        int kNew = k + binShift;

        // Check if target bin is within valid range
        if (kNew >= 0 && kNew < numBins)
        {
            // Copy magnitude and phase to new bin
            shiftedMagnitude[kNew] = magnitude[k];
            shiftedPhase[kNew] = phase[k];
        }
        // If target bin is outside range, energy is discarded (anti-aliasing)
    }

    return { shiftedMagnitude, shiftedPhase };
}

} // namespace fshift
