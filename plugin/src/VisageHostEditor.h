#pragma once

#include <JuceHeader.h>
#include <visage/app.h>
#include "ui/HolyShifterUI.h"

class FrequencyShifterProcessor;

/**
 * Thin JUCE editor shell that embeds the Visage-rendered UI.
 * Uses a Timer to poll parameter changes for DAW automation and preset updates.
 */
class VisageHostEditor : public juce::AudioProcessorEditor,
                         private juce::Timer
{
public:
    explicit VisageHostEditor(FrequencyShifterProcessor& processor);
    ~VisageHostEditor() override;

    void paint(juce::Graphics& g) override {
        if (!windowShown_) {
            g.fillAll(juce::Colours::black);
            g.setColour(juce::Colours::white);
            g.drawText("Initializing UI...", getLocalBounds(), juce::Justification::centred);
        }
    }
    void resized() override;
    void parentHierarchyChanged() override;

    static constexpr int kBaseWidth = 700;
    static constexpr int kBaseHeight = 928;

private:
    void timerCallback() override;
    void tryCreateVisageWindow();

    FrequencyShifterProcessor& processor_;
    std::unique_ptr<HolyShifterUI> ui_;
    std::unique_ptr<visage::ApplicationWindow> visageWindow_;
    bool windowShown_ = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VisageHostEditor)
};
