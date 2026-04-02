#pragma once

#include "HolyTheme.h"
#include "controls/HolyRotaryKnob.h"
#include "controls/HolySlider.h"
#include "controls/HolyToggle.h"
#include "controls/HolyComboBox.h"
#include "controls/HolyPianoKeyboard.h"
#include <visage_ui/frame.h>
#include <visage_widgets/button.h>
#include <JuceHeader.h>

class FrequencyShifterProcessor;

class HolyPresetDropdown : public visage::Frame
{
public:
    HolyPresetDropdown() { setVisible(false); }

    void showFor(FrequencyShifterProcessor* proc, int x, int y, int w);
    void hide();
    bool isOpen() const { return isVisible(); }

    void draw(visage::Canvas& canvas) override;
    void mouseDown(const visage::MouseEvent& e) override;
    void mouseMove(const visage::MouseEvent& e) override;

private:
    FrequencyShifterProcessor* processor_ = nullptr;
    std::vector<std::string> presetNames_;
    int hoveredIndex_ = -1;
    static constexpr int kItemHeight = 24;
    std::function<void()> onPresetChanged_;
    friend class HolyShifterUI;

    VISAGE_LEAK_CHECKER(HolyPresetDropdown)
};

class HolyShifterUI : public visage::Frame
{
public:
    explicit HolyShifterUI(FrequencyShifterProcessor& processor);

    void resized() override;
    void draw(visage::Canvas& canvas) override;
    void pollState();  // Called by timer to sync preset name and redraw

    static constexpr int kBaseW = 700;
    static constexpr int kBaseH = 928;

private:
    void drawStrip(visage::Canvas& canvas, int y, int h,
                   const std::string& label, bool dimmed = false);
    void updateControlsForMode();

    FrequencyShifterProcessor& processor_;
    juce::AudioProcessorValueTreeState& apvts_;

    // Title bar
    HolyToggle warmToggle_;

    // Preset strip
    visage::UiButton presetPrevBtn_;
    visage::UiButton presetNextBtn_;
    visage::Frame presetNameArea_;  // clickable area that opens preset dropdown
    std::string currentPresetName_;

    // Main shift knob
    HolyRotaryKnob shiftKnob_;

    // Spectral panel
    HolyComboBox processingModeCombo_;
    HolyPianoKeyboard pianoKeyboard_;
    HolySlider quantizeSlider_;
    HolySlider preserveSlider_;
    HolySlider transientsSlider_;
    HolySlider sensitivitySlider_;

    // Smear & Enhance
    HolyToggle phaseVocoderToggle_;
    HolySlider smearSlider_;

    // Freq Modulation
    HolySlider lfoDepthSlider_;
    HolyComboBox lfoDepthModeCombo_;
    HolySlider lfoRateSlider_;
    HolyToggle lfoSyncToggle_;
    HolyComboBox lfoDivisionCombo_;
    HolyComboBox lfoShapeCombo_;

    // Delay
    HolyToggle delayEnabledToggle_;
    HolySlider delayTimeSlider_;
    HolyToggle delaySyncToggle_;
    HolyComboBox delayDivisionCombo_;
    HolySlider delayFeedbackSlider_;
    HolySlider delayDampingSlider_;
    HolySlider delaySlopeSlider_;
    HolySlider delayDiffuseSlider_;
    HolyToggle stereoDecorrelateToggle_;

    // Delay Modulation
    HolySlider dlyLfoDepthSlider_;
    HolySlider dlyLfoRateSlider_;
    HolyToggle dlyLfoSyncToggle_;
    HolyComboBox dlyLfoDivisionCombo_;
    HolyComboBox dlyLfoShapeCombo_;

    // Mask
    HolyToggle maskEnabledToggle_;
    HolyComboBox maskModeCombo_;
    HolySlider maskTransitionSlider_;
    HolySlider maskLowFreqSlider_;
    HolySlider maskHighFreqSlider_;

    // Mix
    HolySlider dryWetSlider_;

    // Shared dropdown overlay (must be last child for z-ordering)
    HolyDropdownOverlay dropdownOverlay_;
    HolyPresetDropdown presetDropdown_;

    VISAGE_LEAK_CHECKER(HolyShifterUI)
};
