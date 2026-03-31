#pragma once

#include <JuceHeader.h>

/**
 * PresetManager - Handles saving, loading, and browsing presets.
 *
 * Operates on top of AudioProcessorValueTreeState. Presets are XML files
 * stored on disk (user presets) or embedded in the binary (factory presets).
 * DAW automation and state save/restore are unaffected.
 */
class PresetManager
{
public:
    explicit PresetManager(juce::AudioProcessorValueTreeState& apvts);

    // Preset I/O
    void savePreset(const juce::String& name);
    void loadPreset(const juce::String& name);
    void deletePreset(const juce::String& name);

    // Navigation
    void loadNextPreset();
    void loadPreviousPreset();

    // Query
    juce::StringArray getAllPresetNames() const;
    juce::String getCurrentPresetName() const;
    bool isFactoryPreset(const juce::String& name) const;
    int getCurrentPresetIndex() const;
    int getTotalPresetCount() const;

private:
    juce::AudioProcessorValueTreeState& apvts;
    juce::String currentPresetName;

    struct FactoryPreset
    {
        juce::String name;
        juce::String xmlData;
    };
    std::vector<FactoryPreset> factoryPresets;

    void initFactoryPresets();
    juce::File getUserPresetDirectory() const;
    juce::File getPresetFile(const juce::String& name) const;
    juce::StringArray getUserPresetNames() const;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PresetManager)
};
