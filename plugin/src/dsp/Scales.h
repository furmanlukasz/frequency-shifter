#pragma once

#include <vector>
#include <string>
#include <array>
#include <cmath>

namespace fshift
{

/**
 * Musical scale types.
 */
enum class ScaleType
{
    Major = 0,
    Minor,
    NaturalMinor,
    HarmonicMinor,
    MelodicMinor,
    Ionian,
    Dorian,
    Phrygian,
    Lydian,
    Mixolydian,
    Aeolian,
    Locrian,
    PentatonicMajor,
    PentatonicMinor,
    Blues,
    Chromatic,
    WholeTone,
    Diminished,
    HalfWholeDiminished,
    Arabic,
    Japanese,
    Spanish,
    COUNT  // Keep last for iteration
};

/**
 * Get the scale degrees (semitones from root) for a given scale type.
 */
inline std::vector<int> getScaleDegrees(ScaleType type)
{
    switch (type)
    {
        case ScaleType::Major:
        case ScaleType::Ionian:
            return { 0, 2, 4, 5, 7, 9, 11 };

        case ScaleType::Minor:
        case ScaleType::NaturalMinor:
        case ScaleType::Aeolian:
            return { 0, 2, 3, 5, 7, 8, 10 };

        case ScaleType::HarmonicMinor:
            return { 0, 2, 3, 5, 7, 8, 11 };

        case ScaleType::MelodicMinor:
            return { 0, 2, 3, 5, 7, 9, 11 };

        case ScaleType::Dorian:
            return { 0, 2, 3, 5, 7, 9, 10 };

        case ScaleType::Phrygian:
            return { 0, 1, 3, 5, 7, 8, 10 };

        case ScaleType::Lydian:
            return { 0, 2, 4, 6, 7, 9, 11 };

        case ScaleType::Mixolydian:
            return { 0, 2, 4, 5, 7, 9, 10 };

        case ScaleType::Locrian:
            return { 0, 1, 3, 5, 6, 8, 10 };

        case ScaleType::PentatonicMajor:
            return { 0, 2, 4, 7, 9 };

        case ScaleType::PentatonicMinor:
            return { 0, 3, 5, 7, 10 };

        case ScaleType::Blues:
            return { 0, 3, 5, 6, 7, 10 };

        case ScaleType::Chromatic:
            return { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

        case ScaleType::WholeTone:
            return { 0, 2, 4, 6, 8, 10 };

        case ScaleType::Diminished:
            return { 0, 2, 3, 5, 6, 8, 9, 11 };

        case ScaleType::HalfWholeDiminished:
            return { 0, 1, 3, 4, 6, 7, 9, 10 };

        case ScaleType::Arabic:
            return { 0, 1, 4, 5, 7, 8, 11 };

        case ScaleType::Japanese:
            return { 0, 1, 5, 7, 8 };

        case ScaleType::Spanish:
            return { 0, 1, 3, 4, 5, 6, 8, 10 };

        case ScaleType::COUNT:
        default:
            return { 0, 2, 4, 5, 7, 9, 11 };  // Default to major
    }
}

/**
 * Get human-readable name for a scale type.
 */
inline std::string getScaleName(ScaleType type)
{
    switch (type)
    {
        case ScaleType::Major: return "Major";
        case ScaleType::Minor: return "Minor";
        case ScaleType::NaturalMinor: return "Natural Minor";
        case ScaleType::HarmonicMinor: return "Harmonic Minor";
        case ScaleType::MelodicMinor: return "Melodic Minor";
        case ScaleType::Ionian: return "Ionian";
        case ScaleType::Dorian: return "Dorian";
        case ScaleType::Phrygian: return "Phrygian";
        case ScaleType::Lydian: return "Lydian";
        case ScaleType::Mixolydian: return "Mixolydian";
        case ScaleType::Aeolian: return "Aeolian";
        case ScaleType::Locrian: return "Locrian";
        case ScaleType::PentatonicMajor: return "Pentatonic Major";
        case ScaleType::PentatonicMinor: return "Pentatonic Minor";
        case ScaleType::Blues: return "Blues";
        case ScaleType::Chromatic: return "Chromatic";
        case ScaleType::WholeTone: return "Whole Tone";
        case ScaleType::Diminished: return "Diminished";
        case ScaleType::HalfWholeDiminished: return "Half-Whole Dim";
        case ScaleType::Arabic: return "Arabic";
        case ScaleType::Japanese: return "Japanese";
        case ScaleType::Spanish: return "Spanish";
        case ScaleType::COUNT:
        default: return "Unknown";
    }
}

/**
 * Get all scale names as a vector.
 */
inline std::vector<std::string> getScaleNames()
{
    std::vector<std::string> names;
    for (int i = 0; i < static_cast<int>(ScaleType::COUNT); ++i)
    {
        names.push_back(getScaleName(static_cast<ScaleType>(i)));
    }
    return names;
}

/**
 * Tuning utilities - MIDI/frequency conversion.
 */
namespace tuning
{

/**
 * Convert frequency to MIDI note number.
 * Uses standard equal temperament: midi = 69 + 12 * log2(freq / 440)
 *
 * @param freq Frequency in Hz
 * @param a4Freq Reference frequency for A4 (default 440 Hz)
 * @return MIDI note number (can be fractional for microtonal pitches)
 */
inline float freqToMidi(float freq, float a4Freq = 440.0f)
{
    if (freq <= 0.0f)
        return 0.0f;
    return 69.0f + 12.0f * std::log2(freq / a4Freq);
}

/**
 * Convert MIDI note number to frequency.
 * Uses standard equal temperament: freq = 440 * 2^((midi - 69) / 12)
 *
 * @param midi MIDI note number (can be fractional)
 * @param a4Freq Reference frequency for A4 (default 440 Hz)
 * @return Frequency in Hz
 */
inline float midiToFreq(float midi, float a4Freq = 440.0f)
{
    return a4Freq * std::pow(2.0f, (midi - 69.0f) / 12.0f);
}

/**
 * Quantize a MIDI note to the nearest scale degree.
 *
 * @param midiNote Input MIDI note (can be fractional)
 * @param rootMidi Root note of scale (MIDI number)
 * @param scaleDegrees Scale degrees (semitones from root)
 * @return Quantized MIDI note number (integer)
 */
inline int quantizeToScale(float midiNote, int rootMidi, const std::vector<int>& scaleDegrees)
{
    // Calculate relative note within octave
    float relativeNote = std::fmod(midiNote - static_cast<float>(rootMidi), 12.0f);
    if (relativeNote < 0.0f)
        relativeNote += 12.0f;

    // Find closest scale degree
    float minDiff = 12.0f;
    int closestDegree = 0;

    for (int degree : scaleDegrees)
    {
        float diff = std::abs(static_cast<float>(degree) - relativeNote);
        // Handle wraparound
        float wrapDiff = std::abs(static_cast<float>(degree) - (relativeNote - 12.0f));
        diff = std::min(diff, wrapDiff);

        if (diff < minDiff)
        {
            minDiff = diff;
            closestDegree = degree;
        }
    }

    // Calculate which octave we're in
    int octave = static_cast<int>(std::floor((midiNote - static_cast<float>(rootMidi)) / 12.0f));

    // Handle edge case where we wrapped around to lower octave
    if (relativeNote < static_cast<float>(closestDegree) && closestDegree > 6)
    {
        octave -= 1;
    }

    return rootMidi + octave * 12 + closestDegree;
}

/**
 * Calculate difference between two frequencies in cents.
 *
 * @param freq1 First frequency in Hz
 * @param freq2 Second frequency in Hz
 * @return Difference in cents (1 semitone = 100 cents)
 */
inline float centsDifference(float freq1, float freq2)
{
    if (freq1 <= 0.0f || freq2 <= 0.0f)
        return 0.0f;
    return 1200.0f * std::log2(freq2 / freq1);
}

/**
 * Get note name from MIDI number.
 *
 * @param midi MIDI note number (0-127)
 * @return Note name (e.g., "C4", "A#5")
 */
inline std::string midiToNoteName(int midi)
{
    static const std::array<const char*, 12> noteNames = {
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    };

    if (midi < 0 || midi > 127)
        return "?";

    int octave = (midi / 12) - 1;
    auto noteIdx = static_cast<std::size_t>(midi % 12);

    return std::string(noteNames[noteIdx]) + std::to_string(octave);
}

} // namespace tuning

} // namespace fshift
