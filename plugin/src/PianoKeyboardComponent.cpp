#include "PianoKeyboardComponent.h"

// Colors matching the Holy Shifter theme
namespace KeyColors
{
    static constexpr juce::uint32 whiteKeyOff     = 0xFF1E1E22;  // Dark surface
    static constexpr juce::uint32 whiteKeyOn      = 0xFFC9A96E;  // Accent gold
    static constexpr juce::uint32 blackKeyOff     = 0xFF0A0A0C;  // Background dark
    static constexpr juce::uint32 blackKeyOn      = 0xFFC9A96E;  // Accent gold
    static constexpr juce::uint32 keyBorder       = 0xFF2A2A2E;  // Subtle border
    static constexpr juce::uint32 blackKeyBorder  = 0xFF1A1A1D;  // Darker border for black keys
}

PianoKeyboardComponent::PianoKeyboardComponent(juce::AudioProcessorValueTreeState& apvtsRef,
                                               const juce::String& prefix)
    : apvts(apvtsRef), paramPrefix(prefix)
{
    noteStates.fill(true);  // Match default (all ON)

    for (int i = 0; i < 12; ++i)
    {
        auto paramId = paramPrefix + juce::String(i);
        apvts.addParameterListener(paramId, this);

        // Read initial state
        if (auto* param = apvts.getRawParameterValue(paramId))
            noteStates[i] = param->load() > 0.5f;
    }
}

PianoKeyboardComponent::~PianoKeyboardComponent()
{
    for (int i = 0; i < 12; ++i)
        apvts.removeParameterListener(paramPrefix + juce::String(i), this);
}

void PianoKeyboardComponent::parameterChanged(const juce::String& parameterID, float newValue)
{
    auto idx = parameterID.substring(paramPrefix.length()).getIntValue();
    if (idx >= 0 && idx < 12)
    {
        noteStates[idx] = newValue > 0.5f;
        juce::MessageManager::callAsync([this]() { repaint(); });
    }
}

juce::Rectangle<float> PianoKeyboardComponent::getWhiteKeyRect(int whiteIndex) const
{
    float w = static_cast<float>(getWidth()) / 7.0f;
    float h = static_cast<float>(getHeight());
    return { whiteIndex * w, 0.0f, w, h };
}

juce::Rectangle<float> PianoKeyboardComponent::getBlackKeyRect(int pitchClass) const
{
    float whiteW = static_cast<float>(getWidth()) / 7.0f;
    float blackW = whiteW * 0.6f;
    float blackH = static_cast<float>(getHeight()) * 0.58f;

    // Find which white key this black key is after
    int whiteIdx = -1;
    for (int i = 0; i < 5; ++i)
    {
        if (blackKeyPitchClass[i] == pitchClass)
        {
            whiteIdx = blackKeyAfterWhite[i];
            break;
        }
    }
    if (whiteIdx < 0)
        return {};

    float x = (whiteIdx + 1) * whiteW - blackW * 0.5f;
    return { x, 0.0f, blackW, blackH };
}

int PianoKeyboardComponent::getKeyAtPoint(juce::Point<float> point) const
{
    // Check black keys first (they overlap white keys)
    for (int i = 0; i < 5; ++i)
    {
        int pc = blackKeyPitchClass[i];
        if (getBlackKeyRect(pc).contains(point))
            return pc;
    }

    // Then white keys
    for (int i = 0; i < 7; ++i)
    {
        if (getWhiteKeyRect(i).contains(point))
            return whiteKeyPitchClass[i];
    }

    return -1;
}

void PianoKeyboardComponent::paint(juce::Graphics& g)
{
    float cornerSize = 2.0f;

    // Draw white keys
    for (int i = 0; i < 7; ++i)
    {
        int pc = whiteKeyPitchClass[i];
        auto rect = getWhiteKeyRect(i).reduced(0.5f);

        g.setColour(juce::Colour(noteStates[pc] ? KeyColors::whiteKeyOn : KeyColors::whiteKeyOff));
        g.fillRoundedRectangle(rect, cornerSize);

        g.setColour(juce::Colour(KeyColors::keyBorder));
        g.drawRoundedRectangle(rect, cornerSize, 0.5f);
    }

    // Draw black keys on top
    for (int i = 0; i < 5; ++i)
    {
        int pc = blackKeyPitchClass[i];
        auto rect = getBlackKeyRect(pc);

        g.setColour(juce::Colour(noteStates[pc] ? KeyColors::blackKeyOn : KeyColors::blackKeyOff));
        g.fillRoundedRectangle(rect, cornerSize);

        g.setColour(juce::Colour(KeyColors::blackKeyBorder));
        g.drawRoundedRectangle(rect, cornerSize, 0.5f);
    }
}

void PianoKeyboardComponent::mouseDown(const juce::MouseEvent& event)
{
    int key = getKeyAtPoint(event.position);
    if (key < 0 || key >= 12)
        return;

    auto paramId = paramPrefix + juce::String(key);
    if (auto* param = apvts.getParameter(paramId))
    {
        bool newState = !noteStates[key];
        param->setValueNotifyingHost(newState ? 1.0f : 0.0f);
    }
}
