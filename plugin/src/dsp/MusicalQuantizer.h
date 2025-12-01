#pragma once

#include <vector>
#include <utility>
#include "Scales.h"

namespace fshift
{

/**
 * MusicalQuantizer - Musical scale quantization for frequency spectra.
 *
 * Takes a frequency spectrum and quantizes frequencies to the nearest
 * notes in a specified musical scale.
 *
 * Based on the Python implementation in harmonic_shifter/core/quantizer.py
 */
class MusicalQuantizer
{
public:
    /**
     * Construct musical quantizer.
     *
     * @param rootMidi MIDI note number for scale root (0-127)
     * @param scaleType Scale type from ScaleType enum
     */
    MusicalQuantizer(int rootMidi, ScaleType scaleType);

    ~MusicalQuantizer() = default;

    /**
     * Set the root note.
     *
     * @param rootMidi MIDI note number for scale root (0-127)
     */
    void setRootNote(int rootMidi);

    /**
     * Set the scale type.
     *
     * @param scaleType Scale type from ScaleType enum
     */
    void setScaleType(ScaleType scaleType);

    /**
     * Quantize frequencies to nearest scale notes.
     *
     * @param frequencies Array of frequencies in Hz
     * @param strength Quantization strength (0.0 = no quantization, 1.0 = full)
     * @return Quantized frequencies in Hz
     */
    std::vector<float> quantizeFrequencies(const std::vector<float>& frequencies, float strength = 1.0f);

    /**
     * Quantize entire spectrum to scale.
     *
     * Maps energy from each frequency bin to the nearest scale frequency.
     *
     * @param magnitude Magnitude spectrum
     * @param phase Phase spectrum
     * @param sampleRate Sample rate in Hz
     * @param fftSize FFT size
     * @param strength Quantization strength (0-1)
     * @return Pair of (quantized_magnitude, quantized_phase)
     */
    std::pair<std::vector<float>, std::vector<float>> quantizeSpectrum(
        const std::vector<float>& magnitude,
        const std::vector<float>& phase,
        double sampleRate,
        int fftSize,
        float strength = 1.0f);

    /**
     * Get all scale frequencies in a given range.
     *
     * @param minFreq Minimum frequency in Hz
     * @param maxFreq Maximum frequency in Hz
     * @return Vector of frequencies in scale within range
     */
    std::vector<float> getScaleFrequencies(float minFreq = 20.0f, float maxFreq = 20000.0f) const;

    // Getters
    int getRootMidi() const { return rootMidi; }
    ScaleType getScaleType() const { return scaleType; }
    const std::vector<int>& getScaleDegrees() const { return scaleDegrees; }

private:
    /**
     * Quantize a single frequency to the scale.
     */
    float quantizeFrequency(float frequency, float strength) const;

    int rootMidi;
    ScaleType scaleType;
    std::vector<int> scaleDegrees;
};

} // namespace fshift
