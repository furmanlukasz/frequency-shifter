#pragma once

#include <JuceHeader.h>
#include <array>

/**
 * PianoKeyboardComponent - One-octave clickable keyboard for scale note selection.
 *
 * Displays 12 pitch classes (C through B) as a compact piano keyboard.
 * Users click keys to toggle notes on/off for quantization.
 * Wired to 12 AudioParameterBool params (scaleNote0..scaleNote11).
 */
class PianoKeyboardComponent : public juce::Component,
                                private juce::AudioProcessorValueTreeState::Listener
{
public:
    PianoKeyboardComponent(juce::AudioProcessorValueTreeState& apvts,
                           const juce::String& paramPrefix);
    ~PianoKeyboardComponent() override;

    void paint(juce::Graphics& g) override;
    void mouseDown(const juce::MouseEvent& event) override;

private:
    void parameterChanged(const juce::String& parameterID, float newValue) override;

    // Returns the key index (0-11) at the given point, or -1 if none
    int getKeyAtPoint(juce::Point<float> point) const;

    // Get the rectangle for a white key by its sequential index (0-6)
    juce::Rectangle<float> getWhiteKeyRect(int whiteIndex) const;

    // Get the rectangle for a black key by its pitch class (1,3,6,8,10)
    juce::Rectangle<float> getBlackKeyRect(int pitchClass) const;

    juce::AudioProcessorValueTreeState& apvts;
    juce::String paramPrefix;
    std::array<bool, 12> noteStates{};

    // White key pitch classes: C=0, D=2, E=4, F=5, G=7, A=9, B=11
    static constexpr int whiteKeyPitchClass[7] = { 0, 2, 4, 5, 7, 9, 11 };
    // Black key pitch classes: C#=1, D#=3, F#=6, G#=8, A#=10
    static constexpr int blackKeyPitchClass[5] = { 1, 3, 6, 8, 10 };
    // Which white key index each black key sits after (for positioning)
    static constexpr int blackKeyAfterWhite[5] = { 0, 1, 3, 4, 5 };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PianoKeyboardComponent)
};
