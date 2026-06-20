#pragma once

#include "HolyTheme.h"
#include "controls/HolyRotaryKnob.h"
#include "controls/HolySlider.h"
#include "controls/HolyToggle.h"
#include "controls/HolyComboBox.h"
#include "controls/HolyPianoKeyboard.h"
#include "controls/HolySegmentedControl.h"
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

// ─────────────────────────────────────────────────────────────────────────────
// Minimal themed single-line text input. Visage ships a TextEditor widget, but it
// renders from a theme palette this UI never configures — every Holy* control draws
// with raw colours instead, so we do the same here for a consistent look. Handles
// typing, backspace, enter/escape and paste: enough to name a preset.
// ─────────────────────────────────────────────────────────────────────────────
class HolyTextField : public visage::Frame
{
public:
    HolyTextField() { setAcceptsKeystrokes(true); }

    void setText(const std::string& t) { value_ = t; redraw(); }
    const std::string& getText() const { return value_; }
    void setPlaceholder(const std::string& p) { placeholder_ = p; redraw(); }
    void setMaxLength(int n) { maxLen_ = n; }

    std::function<void()> onSubmit;   // Enter / Return
    std::function<void()> onCancel;   // Escape

    void draw(visage::Canvas& canvas) override;
    void mouseDown(const visage::MouseEvent&) override { requestKeyboardFocus(); redraw(); }
    bool keyPress(const visage::KeyEvent& e) override;
    void textInput(const std::string& text) override { appendFiltered(text); }
    bool receivesTextInput() override { return true; }
    void focusChanged(bool is_focused, bool) override { focused_ = is_focused; redraw(); }

private:
    void appendFiltered(const std::string& text);

    std::string value_;
    std::string placeholder_;
    int maxLen_ = 40;
    bool focused_ = false;

    VISAGE_LEAK_CHECKER(HolyTextField)
};

// Small themed push-button — preset SAVE/DELETE actions and the modal's confirm/
// cancel. Two styles: Primary (gold fill) and Outline.
class HolyTextButton : public visage::Frame
{
public:
    enum class Style { Outline, Primary };

    void configure(const std::string& text, Style style) { text_ = text; style_ = style; redraw(); }
    void setText(const std::string& t) { text_ = t; redraw(); }
    void setEnabledState(bool e) { if (enabled_ != e) { enabled_ = e; redraw(); } }

    std::function<void()> onClick;

    void draw(visage::Canvas& canvas) override;
    void mouseEnter(const visage::MouseEvent&) override { hovered_ = true; redraw(); }
    void mouseExit(const visage::MouseEvent&) override { hovered_ = false; redraw(); }
    void mouseDown(const visage::MouseEvent&) override { if (enabled_ && onClick) onClick(); }

private:
    std::string text_;
    Style style_ = Style::Outline;
    bool hovered_ = false;
    bool enabled_ = true;

    VISAGE_LEAK_CHECKER(HolyTextButton)
};

// Modal overlay covering the whole UI: a centred panel that either prompts for a
// preset name (Save) or confirms a delete. Click-outside or Escape cancels.
class HolyModalDialog : public visage::Frame
{
public:
    HolyModalDialog();

    void openSave(FrequencyShifterProcessor* proc, const std::string& suggestedName);
    void openConfirmDelete(FrequencyShifterProcessor* proc, const std::string& presetName);
    void close();
    bool isOpen() const { return isVisible(); }

    std::function<void()> onChanged;   // fired after a successful save or delete

    void draw(visage::Canvas& canvas) override;
    void mouseDown(const visage::MouseEvent& e) override;
    void resized() override { layoutPanel(); }

private:
    enum class Mode { Save, ConfirmDelete };

    void layoutPanel();
    void commitSave();
    void commitDelete();

    static constexpr int kPanelW = 320;
    static constexpr int kPanelH = 168;

    Mode mode_ = Mode::Save;
    FrequencyShifterProcessor* processor_ = nullptr;
    std::string suggested_;
    std::string targetName_;
    std::string status_;

    HolyTextField nameField_;
    HolyTextButton primaryBtn_;
    HolyTextButton cancelBtn_;

    VISAGE_LEAK_CHECKER(HolyModalDialog)
};

class HolyShifterUI : public visage::Frame
{
public:
    explicit HolyShifterUI(FrequencyShifterProcessor& processor);

    void resized() override;
    void draw(visage::Canvas& canvas) override;
    void pollState();

    static constexpr int kBaseW = 700;
    static constexpr int kBaseH = 928;

private:
    void drawStrip(visage::Canvas& canvas, int y, int h,
                   const std::string& label, bool dimmed = false);
    void updateControlsForMode();
    void updateDelaySyncUI();
    void updateLfoSyncUI();
    void updateDlyLfoSyncUI();
    void updateLfoEnableUI();  // R3: grey each LFO's controls when its toggle is off

    FrequencyShifterProcessor& processor_;
    juce::AudioProcessorValueTreeState& apvts_;

    // Title bar
    HolyToggle warmToggle_;

    // Mode selector (segmented control)
    HolySegmentedControl modeSelector_;

    // Preset strip
    visage::UiButton presetPrevBtn_;
    visage::UiButton presetNextBtn_;
    visage::Frame presetNameArea_;
    HolyTextButton presetSaveBtn_;
    HolyTextButton presetDeleteBtn_;
    std::string currentPresetName_;
    void updatePresetStrip();   // sync name + disable DELETE for factory/empty presets

    // Main shift knob
    HolyRotaryKnob shiftKnob_;

    // Spectral panel (includes keyboard, sliders, Smear)
    HolyPianoKeyboard pianoKeyboard_;
    HolySlider quantizeSlider_;
    HolySlider preserveSlider_;
    HolySlider transientsSlider_;
    HolySlider sensitivitySlider_;
    HolySlider smearSlider_;

    // Peak-region snapping + sines/noise split (provisional placement in Spectral panel)
    HolyToggle peakSnapToggle_;
    HolySlider noiseSlider_;
    HolySlider peakSensSlider_;

    // Freq Modulation
    HolySlider lfoDepthSlider_;
    HolyComboBox lfoDepthModeCombo_;   // Hz/Degrees (hidden, kept for param binding)
    HolyComboBox lfoShapeCombo_;       // Sine/Tri/Saw/etc — on Rate row (per Figma)
    HolySlider lfoRateSlider_;         // dual-purpose: Hz when free, divisions when synced
    HolyToggle lfoSyncToggle_;
    HolyToggle lfoEnabledToggle_;      // R3: LFO on/off (section header pill)

    // Delay
    HolyToggle delayEnabledToggle_;
    HolySlider delayTimeSlider_;       // dual-purpose: ms when free, divisions when synced
    HolyToggle delaySyncToggle_;
    HolySlider delayFeedbackSlider_;
    HolySlider delayDampingSlider_;
    HolySlider delaySlopeSlider_;
    HolyToggle stereoDecorrelateToggle_;

    // Delay Modulation
    HolySlider dlyLfoDepthSlider_;
    HolySlider dlyLfoRateSlider_;       // dual-purpose: Hz when free, divisions when synced
    HolyToggle dlyLfoSyncToggle_;
    HolyComboBox dlyLfoShapeCombo_;
    HolyToggle dlyLfoEnabledToggle_;    // R3: delay LFO on/off (section header pill)

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

    // Save / delete-confirm modal — added LAST so it paints above (and intercepts
    // clicks from) everything, including the dropdowns.
    HolyModalDialog presetModal_;

    VISAGE_LEAK_CHECKER(HolyShifterUI)
};
