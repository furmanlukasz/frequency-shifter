#include "PluginEditor.h"
#include <cmath>

//==============================================================================
// HolyShifterLookAndFeel Implementation
//==============================================================================

FrequencyShifterEditor::HolyShifterLookAndFeel::HolyShifterLookAndFeel()
{
    // Set default colors for Holy Shifter theme
    setColour(juce::Slider::rotarySliderFillColourId, juce::Colour(Colors::accent));
    setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colour(Colors::track));
    setColour(juce::Slider::thumbColourId, juce::Colour(Colors::accent));
    setColour(juce::Slider::trackColourId, juce::Colour(Colors::track));
    setColour(juce::Slider::textBoxTextColourId, juce::Colour(Colors::text));
    setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);

    setColour(juce::ComboBox::backgroundColourId, juce::Colour(Colors::raised));
    setColour(juce::ComboBox::textColourId, juce::Colour(Colors::text));
    setColour(juce::ComboBox::outlineColourId, juce::Colour(Colors::border));
    setColour(juce::ComboBox::arrowColourId, juce::Colour(Colors::textMuted));

    setColour(juce::PopupMenu::backgroundColourId, juce::Colour(Colors::raised));
    setColour(juce::PopupMenu::textColourId, juce::Colour(Colors::text));
    setColour(juce::PopupMenu::highlightedBackgroundColourId, juce::Colour(Colors::accentDim));
    setColour(juce::PopupMenu::highlightedTextColourId, juce::Colour(Colors::text));

    setColour(juce::ToggleButton::textColourId, juce::Colour(Colors::text));
    setColour(juce::ToggleButton::tickColourId, juce::Colour(Colors::accent));
    setColour(juce::ToggleButton::tickDisabledColourId, juce::Colour(Colors::textMuted));

    setColour(juce::Label::textColourId, juce::Colour(Colors::text));
}

void FrequencyShifterEditor::HolyShifterLookAndFeel::drawRotarySlider(
    juce::Graphics& g, int x, int y, int width, int height,
    float sliderPosProportional, float rotaryStartAngle, float rotaryEndAngle,
    juce::Slider& slider)
{
    const float radius = static_cast<float>(juce::jmin(width / 2, height / 2)) - 18.0f;
    const float centreX = static_cast<float>(x) + static_cast<float>(width) * 0.5f;
    const float centreY = static_cast<float>(y) + static_cast<float>(height) * 0.5f;
    const float angle = rotaryStartAngle + sliderPosProportional * (rotaryEndAngle - rotaryStartAngle);

    // Check if bipolar (shift knob has min < 0 and max > 0)
    double minVal = slider.getMinimum();
    double maxVal = slider.getMaximum();
    bool isBipolar = (minVal < 0 && maxVal > 0);
    float centerAngle = rotaryStartAngle + 0.5f * (rotaryEndAngle - rotaryStartAngle);

    // Subtle outer ring
    g.setColour(juce::Colour(Colors::borderDim));
    g.drawEllipse(centreX - radius - 8.0f, centreY - radius - 8.0f,
                  (radius + 8.0f) * 2.0f, (radius + 8.0f) * 2.0f, 0.5f);

    // Background arc (track)
    g.setColour(juce::Colour(Colors::track));
    juce::Path backgroundArc;
    backgroundArc.addCentredArc(centreX, centreY, radius, radius, 0.0f,
                                 rotaryStartAngle, rotaryEndAngle, true);
    g.strokePath(backgroundArc, juce::PathStrokeType(2.5f, juce::PathStrokeType::curved,
                                                      juce::PathStrokeType::rounded));

    // Value arc with glow effect
    if (isBipolar)
    {
        // Bipolar: draw from center to current position
        float startAngle = (sliderPosProportional >= 0.5f) ? centerAngle : angle;
        float endAngle = (sliderPosProportional >= 0.5f) ? angle : centerAngle;

        if (std::abs(endAngle - startAngle) > 0.01f)
        {
            g.setColour(juce::Colour(Colors::accent));
            juce::Path valueArc;
            valueArc.addCentredArc(centreX, centreY, radius, radius, 0.0f,
                                    startAngle, endAngle, true);
            g.strokePath(valueArc, juce::PathStrokeType(2.5f, juce::PathStrokeType::curved,
                                                         juce::PathStrokeType::rounded));
        }
    }
    else
    {
        // Unipolar: draw from start to current position
        if (sliderPosProportional > 0.0f)
        {
            g.setColour(juce::Colour(Colors::accent));
            juce::Path valueArc;
            valueArc.addCentredArc(centreX, centreY, radius, radius, 0.0f,
                                    rotaryStartAngle, angle, true);
            g.strokePath(valueArc, juce::PathStrokeType(2.5f, juce::PathStrokeType::curved,
                                                         juce::PathStrokeType::rounded));
        }
    }

    // Tick marks at 0%, 25%, 50%, 75%, 100%
    for (int i = 0; i <= 4; ++i)
    {
        float tickNorm = static_cast<float>(i) / 4.0f;
        float tickAngle = rotaryStartAngle + tickNorm * (rotaryEndAngle - rotaryStartAngle);
        float tickAngleRad = tickAngle - juce::MathConstants<float>::halfPi;

        float innerR = radius + 7.0f;
        float outerR = radius + 12.0f;

        float x1 = centreX + innerR * std::cos(tickAngleRad);
        float y1 = centreY + innerR * std::sin(tickAngleRad);
        float x2 = centreX + outerR * std::cos(tickAngleRad);
        float y2 = centreY + outerR * std::sin(tickAngleRad);

        // Center tick (50%) is brighter for bipolar knobs
        bool isCenterTick = (i == 2) && isBipolar;
        g.setColour(juce::Colour(isCenterTick ? Colors::textSec : Colors::textMuted));
        g.drawLine(x1, y1, x2, y2, 0.8f);
    }

    // Indicator dot
    float indicatorAngleRad = angle - juce::MathConstants<float>::halfPi;
    float dotX = centreX + radius * std::cos(indicatorAngleRad);
    float dotY = centreY + radius * std::sin(indicatorAngleRad);
    g.setColour(juce::Colour(Colors::accent));
    g.fillEllipse(dotX - 4.0f, dotY - 4.0f, 8.0f, 8.0f);

    // Value text in center
    g.setColour(juce::Colour(Colors::text));
    g.setFont(juce::FontOptions(32.0f).withStyle("Light"));

    double value = slider.getValue();
    juce::String valueText;
    if (std::abs(value) >= 100.0)
        valueText = juce::String(static_cast<int>(value));
    else
        valueText = juce::String(value, 1);

    g.drawText(valueText, static_cast<int>(centreX - 50), static_cast<int>(centreY - 16),
               100, 32, juce::Justification::centred, false);

    // Unit text below value
    g.setColour(juce::Colour(Colors::textMuted));
    g.setFont(juce::FontOptions(11.0f));
    g.drawText("HZ", static_cast<int>(centreX - 20), static_cast<int>(centreY + 16),
               40, 14, juce::Justification::centred, false);
}

void FrequencyShifterEditor::HolyShifterLookAndFeel::drawLinearSlider(
    juce::Graphics& g, int x, int y, int width, int height,
    float sliderPos, float minSliderPos, float maxSliderPos,
    juce::Slider::SliderStyle style, juce::Slider& slider)
{
    if (style == juce::Slider::LinearHorizontal)
    {
        const float trackY = static_cast<float>(y) + static_cast<float>(height) * 0.5f;
        const float trackHeight = 1.5f;

        // Background track
        g.setColour(juce::Colour(Colors::track));
        g.fillRoundedRectangle(static_cast<float>(x), trackY - trackHeight * 0.5f,
                                static_cast<float>(width), trackHeight, 1.0f);

        // Value track with accent color
        float valueWidth = sliderPos - static_cast<float>(x);
        if (valueWidth > 0)
        {
            g.setColour(juce::Colour(Colors::accent));
            g.fillRoundedRectangle(static_cast<float>(x), trackY - trackHeight * 0.5f,
                                    valueWidth, trackHeight, 1.0f);
        }

        // Thumb (small circle)
        const float thumbRadius = 3.5f;
        g.setColour(juce::Colour(Colors::accent));
        g.fillEllipse(sliderPos - thumbRadius, trackY - thumbRadius,
                      thumbRadius * 2.0f, thumbRadius * 2.0f);
    }
    else
    {
        juce::LookAndFeel_V4::drawLinearSlider(g, x, y, width, height, sliderPos,
                                                minSliderPos, maxSliderPos, style, slider);
    }
}

void FrequencyShifterEditor::HolyShifterLookAndFeel::drawToggleButton(
    juce::Graphics& g, juce::ToggleButton& button,
    bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown)
{
    (void)shouldDrawButtonAsHighlighted;
    (void)shouldDrawButtonAsDown;

    auto bounds = button.getLocalBounds().toFloat();
    bool isOn = button.getToggleState();

    // Pill-style toggle dimensions
    const float toggleWidth = 30.0f;
    const float toggleHeight = 15.0f;
    const float dotSize = 11.0f;

    // Draw toggle pill
    float toggleX = 0.0f;
    float toggleY = (bounds.getHeight() - toggleHeight) * 0.5f;

    g.setColour(juce::Colour(isOn ? Colors::accentDim : Colors::track));
    g.fillRoundedRectangle(toggleX, toggleY, toggleWidth, toggleHeight, toggleHeight * 0.5f);

    // Draw toggle dot
    float dotX = isOn ? (toggleX + toggleWidth - dotSize - 2.0f) : (toggleX + 2.0f);
    float dotY = toggleY + (toggleHeight - dotSize) * 0.5f;
    g.setColour(juce::Colour(isOn ? Colors::accent : Colors::textMuted));
    g.fillEllipse(dotX, dotY, dotSize, dotSize);

    // Draw label text
    g.setColour(juce::Colour(isOn ? Colors::text : Colors::textSec));
    g.setFont(juce::FontOptions(11.0f));

    auto textBounds = bounds.withLeft(toggleWidth + 6.0f);
    g.drawText(button.getButtonText(), textBounds, juce::Justification::centredLeft, false);
}

void FrequencyShifterEditor::HolyShifterLookAndFeel::drawComboBox(
    juce::Graphics& g, int width, int height, bool isButtonDown,
    int buttonX, int buttonY, int buttonW, int buttonH,
    juce::ComboBox& box)
{
    (void)isButtonDown;
    (void)buttonX;
    (void)buttonY;
    (void)buttonW;
    (void)buttonH;

    auto bounds = juce::Rectangle<float>(0, 0, static_cast<float>(width), static_cast<float>(height));

    // Background
    g.setColour(juce::Colour(Colors::raised));
    g.fillRoundedRectangle(bounds, 3.0f);

    // Border
    g.setColour(juce::Colour(Colors::border));
    g.drawRoundedRectangle(bounds.reduced(0.5f), 3.0f, 1.0f);

    // Arrow
    g.setColour(juce::Colour(Colors::textMuted));
    float arrowX = static_cast<float>(width) - 12.0f;
    float arrowY = static_cast<float>(height) * 0.5f;
    juce::Path arrow;
    arrow.addTriangle(arrowX - 3.0f, arrowY - 2.0f,
                      arrowX + 3.0f, arrowY - 2.0f,
                      arrowX, arrowY + 2.0f);
    g.fillPath(arrow);
}

void FrequencyShifterEditor::HolyShifterLookAndFeel::drawPopupMenuItem(
    juce::Graphics& g, const juce::Rectangle<int>& area,
    bool isSeparator, bool isActive, bool isHighlighted,
    bool isTicked, bool hasSubMenu,
    const juce::String& text, const juce::String& shortcutKeyText,
    const juce::Drawable* icon, const juce::Colour* textColour)
{
    (void)isSeparator;
    (void)isActive;
    (void)isTicked;
    (void)hasSubMenu;
    (void)shortcutKeyText;
    (void)icon;
    (void)textColour;

    if (isHighlighted)
    {
        g.setColour(juce::Colour(Colors::accentDim));
        g.fillRect(area);
    }

    g.setColour(juce::Colour(Colors::text));
    g.setFont(juce::FontOptions(13.0f));
    g.drawText(text, area.reduced(8, 0), juce::Justification::centredLeft, true);
}

//==============================================================================
// FrequencyShifterEditor Implementation
//==============================================================================

FrequencyShifterEditor::FrequencyShifterEditor(FrequencyShifterProcessor& p)
    : AudioProcessorEditor(&p), audioProcessor(p)
{
    setLookAndFeel(&holyLookAndFeel);

    // === Preset Controls ===
    presetPrevButton.setButtonText("<");
    presetPrevButton.setColour(juce::TextButton::buttonColourId, juce::Colours::transparentBlack);
    presetPrevButton.setColour(juce::TextButton::textColourOffId, juce::Colour(Colors::textSec));
    presetPrevButton.onClick = [this]()
    {
        audioProcessor.getPresetManager().loadPreviousPreset();
        refreshPresetList();
    };
    addAndMakeVisible(presetPrevButton);

    presetNextButton.setButtonText(">");
    presetNextButton.setColour(juce::TextButton::buttonColourId, juce::Colours::transparentBlack);
    presetNextButton.setColour(juce::TextButton::textColourOffId, juce::Colour(Colors::textSec));
    presetNextButton.onClick = [this]()
    {
        audioProcessor.getPresetManager().loadNextPreset();
        refreshPresetList();
    };
    addAndMakeVisible(presetNextButton);

    presetComboBox.setColour(juce::ComboBox::backgroundColourId, juce::Colour(Colors::raised));
    presetComboBox.setColour(juce::ComboBox::textColourId, juce::Colour(Colors::text));
    presetComboBox.setColour(juce::ComboBox::outlineColourId, juce::Colour(Colors::border));
    presetComboBox.onChange = [this]()
    {
        auto idx = presetComboBox.getSelectedItemIndex();
        if (idx >= 0)
        {
            auto names = audioProcessor.getPresetManager().getAllPresetNames();
            if (idx < names.size())
                audioProcessor.getPresetManager().loadPreset(names[idx]);
        }
    };
    addAndMakeVisible(presetComboBox);

    presetSaveButton.setButtonText("Save");
    presetSaveButton.setColour(juce::TextButton::buttonColourId, juce::Colours::transparentBlack);
    presetSaveButton.setColour(juce::TextButton::textColourOffId, juce::Colour(Colors::textSec));
    presetSaveButton.onClick = [this]()
    {
        auto& pm = audioProcessor.getPresetManager();
        auto name = pm.getCurrentPresetName();
        if (name.isEmpty() || pm.isFactoryPreset(name))
        {
            // Can't overwrite factory preset -- redirect to Save As
            presetSaveAsButton.triggerClick();
            return;
        }
        pm.savePreset(name);
    };
    addAndMakeVisible(presetSaveButton);

    presetSaveAsButton.setButtonText("Save As");
    presetSaveAsButton.setColour(juce::TextButton::buttonColourId, juce::Colours::transparentBlack);
    presetSaveAsButton.setColour(juce::TextButton::textColourOffId, juce::Colour(Colors::textSec));
    presetSaveAsButton.onClick = [this]()
    {
        auto dialog = std::make_shared<juce::AlertWindow>("Save Preset",
            "Enter a name for the preset:", juce::MessageBoxIconType::NoIcon);
        dialog->addTextEditor("name", "", "Preset Name:");
        dialog->addButton("Save", 1);
        dialog->addButton("Cancel", 0);

        // Style the dialog to match theme
        dialog->setColour(juce::AlertWindow::backgroundColourId, juce::Colour(Colors::surface));
        dialog->setColour(juce::AlertWindow::textColourId, juce::Colour(Colors::text));
        dialog->setColour(juce::AlertWindow::outlineColourId, juce::Colour(Colors::border));

        dialog->enterModalState(true, juce::ModalCallbackFunction::create(
            [this, dialog](int result)
            {
                if (result == 1)
                {
                    auto name = dialog->getTextEditorContents("name").trim();
                    if (name.isNotEmpty() && !audioProcessor.getPresetManager().isFactoryPreset(name))
                    {
                        audioProcessor.getPresetManager().savePreset(name);
                        refreshPresetList();
                    }
                }
            }), true);
    };
    addAndMakeVisible(presetSaveAsButton);

    presetDeleteButton.setButtonText("Del");
    presetDeleteButton.setColour(juce::TextButton::buttonColourId, juce::Colours::transparentBlack);
    presetDeleteButton.setColour(juce::TextButton::textColourOffId, juce::Colour(Colors::textSec));
    presetDeleteButton.onClick = [this]()
    {
        auto& pm = audioProcessor.getPresetManager();
        auto name = pm.getCurrentPresetName();
        if (name.isEmpty() || pm.isFactoryPreset(name))
            return;

        auto options = juce::MessageBoxOptions()
            .withTitle("Delete Preset")
            .withMessage("Delete \"" + name + "\"?")
            .withButton("Delete")
            .withButton("Cancel")
            .withIconType(juce::MessageBoxIconType::WarningIcon);

        juce::AlertWindow::showAsync(options, [this, name](int result)
        {
            if (result == 1)
            {
                audioProcessor.getPresetManager().deletePreset(name);
                refreshPresetList();
            }
        });
    };
    addAndMakeVisible(presetDeleteButton);

    refreshPresetList();

    // Processing mode combo (Classic vs Spectral)
    // Parameter order: 0=Classic, 1=Spectral (must match processor's StringArray)
    processingModeCombo.addItem("Classic", 1);   // ID 1 -> Index 0
    processingModeCombo.addItem("Spectral", 2);  // ID 2 -> Index 1
    processingModeCombo.onChange = [this]() { updateControlsForMode(); };
    addAndMakeVisible(processingModeCombo);
    processingModeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_PROCESSING_MODE, processingModeCombo);

    // WARM toggle
    warmButton.setButtonText("Warm");
    addAndMakeVisible(warmButton);
    warmAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_WARM, warmButton);

    // Main shift knob with logarithmic scale
    shiftSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    shiftSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);

    auto rangeToValue = [](double, double, double normalised) -> double
    {
        double symNorm = normalised * 2.0 - 1.0;
        double sign = symNorm >= 0.0 ? 1.0 : -1.0;
        double absNorm = std::abs(symNorm);
        constexpr double logScale = 10.0;
        constexpr double maxShift = 5000.0;
        double logMax = std::log(1.0 + maxShift / logScale);
        double absVal = logScale * (std::exp(absNorm * logMax) - 1.0);
        return sign * absVal;
    };

    auto valueToRange = [](double, double, double value) -> double
    {
        double sign = value >= 0.0 ? 1.0 : -1.0;
        double absVal = std::abs(value);
        constexpr double logScale = 10.0;
        constexpr double maxShift = 5000.0;
        double logMax = std::log(1.0 + maxShift / logScale);
        double normalized = sign * std::log(1.0 + absVal / logScale) / logMax;
        return (normalized + 1.0) * 0.5;
    };

    auto snapToLegalValue = [](double, double, double value) -> double
    {
        if (std::abs(value) < 100.0)
            return std::round(value * 10.0) / 10.0;
        return std::round(value);
    };

    shiftSlider.setNormalisableRange(
        juce::NormalisableRange<double>(-5000.0, 5000.0, rangeToValue, valueToRange, snapToLegalValue));

    addAndMakeVisible(shiftSlider);
    shiftSlider.addListener(this);

    // Initialize shift slider from persisted parameter value
    // (No SliderAttachment is used here because the slider has a custom log-scale range
    // that differs from the parameter's linear range, so we sync manually)
    float currentShiftHz = *audioProcessor.getValueTreeState()
        .getRawParameterValue(FrequencyShifterProcessor::PARAM_SHIFT_HZ);
    shiftSlider.setValue(static_cast<double>(currentShiftHz), juce::dontSendNotification);

    // Piano keyboard for scale note selection
    pianoKeyboard = std::make_unique<PianoKeyboardComponent>(
        audioProcessor.getValueTreeState(),
        FrequencyShifterProcessor::PARAM_SCALE_NOTE_PREFIX);
    addAndMakeVisible(*pianoKeyboard);

    // Quantize slider
    setupHorizontalSlider(quantizeSlider);
    quantizeSlider.setTextValueSuffix("");
    addAndMakeVisible(quantizeSlider);
    quantizeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_QUANTIZE_STRENGTH, quantizeSlider);

    setupLabel(quantizeLabel, "Quantize");
    addAndMakeVisible(quantizeLabel);

    // Preserve slider
    setupHorizontalSlider(preserveSlider);
    addAndMakeVisible(preserveSlider);
    preserveAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_PRESERVE, preserveSlider);

    setupLabel(preserveLabel, "Envelope");
    addAndMakeVisible(preserveLabel);

    // Transients slider
    setupHorizontalSlider(transientsSlider);
    addAndMakeVisible(transientsSlider);
    transientsAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_TRANSIENTS, transientsSlider);

    setupLabel(transientsLabel, "Transients");
    addAndMakeVisible(transientsLabel);

    // Sensitivity slider
    setupHorizontalSlider(sensitivitySlider);
    sensitivitySlider.setNumDecimalPlacesToDisplay(0);
    addAndMakeVisible(sensitivitySlider);
    sensitivityAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_SENSITIVITY, sensitivitySlider);

    setupLabel(sensitivityLabel, "Sensitivity");
    addAndMakeVisible(sensitivityLabel);

    // Enhanced mode toggle
    phaseVocoderButton.setButtonText("Enhanced");
    addAndMakeVisible(phaseVocoderButton);
    phaseVocoderAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_PHASE_VOCODER, phaseVocoderButton);

    // SMEAR slider
    setupHorizontalSlider(smearSlider);
    smearSlider.setTextValueSuffix(" ms");
    addAndMakeVisible(smearSlider);
    smearAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_SMEAR, smearSlider);

    setupLabel(smearLabel, "Smear");
    addAndMakeVisible(smearLabel);

    // === LFO Modulation Controls ===

    setupHorizontalSlider(lfoDepthSlider);
    lfoDepthSlider.setNumDecimalPlacesToDisplay(1);
    addAndMakeVisible(lfoDepthSlider);
    lfoDepthAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_LFO_DEPTH, lfoDepthSlider);

    setupLabel(lfoDepthLabel, "Depth");
    addAndMakeVisible(lfoDepthLabel);

    lfoDepthModeCombo.addItem("Hz", 1);
    lfoDepthModeCombo.addItem("Degrees", 2);
    addAndMakeVisible(lfoDepthModeCombo);
    lfoDepthModeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_LFO_DEPTH_MODE, lfoDepthModeCombo);

    setupHorizontalSlider(lfoRateSlider);
    lfoRateSlider.setTextValueSuffix(" Hz");
    lfoRateSlider.setNumDecimalPlacesToDisplay(1);  // Reduced from 2 for cleaner display
    addAndMakeVisible(lfoRateSlider);
    lfoRateAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_LFO_RATE, lfoRateSlider);

    setupLabel(lfoRateLabel, "Rate");
    addAndMakeVisible(lfoRateLabel);

    lfoSyncButton.setButtonText("Sync");
    addAndMakeVisible(lfoSyncButton);
    lfoSyncAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_LFO_SYNC, lfoSyncButton);

    lfoDivisionCombo.addItem("4/1", 1);
    lfoDivisionCombo.addItem("2/1", 2);
    lfoDivisionCombo.addItem("1/1", 3);
    lfoDivisionCombo.addItem("1/2", 4);
    lfoDivisionCombo.addItem("1/4", 5);
    lfoDivisionCombo.addItem("1/8", 6);
    lfoDivisionCombo.addItem("1/16", 7);
    lfoDivisionCombo.addItem("1/32", 8);
    addAndMakeVisible(lfoDivisionCombo);
    lfoDivisionAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_LFO_DIVISION, lfoDivisionCombo);

    lfoShapeCombo.addItem("Sine", 1);
    lfoShapeCombo.addItem("Triangle", 2);
    lfoShapeCombo.addItem("Saw", 3);
    lfoShapeCombo.addItem("Inv Saw", 4);
    lfoShapeCombo.addItem("Random", 5);
    addAndMakeVisible(lfoShapeCombo);
    lfoShapeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_LFO_SHAPE, lfoShapeCombo);

    lfoSyncButton.onClick = [this]() { updateLfoSyncUI(); };
    updateLfoSyncUI();

    // === Delay Controls ===

    delayEnabledButton.setButtonText("Delay");
    addAndMakeVisible(delayEnabledButton);
    delayEnabledAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_DELAY_ENABLED, delayEnabledButton);

    setupHorizontalSlider(delayTimeSlider);
    delayTimeSlider.setTextValueSuffix(" ms");
    delayTimeSlider.setNumDecimalPlacesToDisplay(1);
    addAndMakeVisible(delayTimeSlider);
    delayTimeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_DELAY_TIME, delayTimeSlider);

    setupLabel(delayTimeLabel, "Time");
    addAndMakeVisible(delayTimeLabel);

    delaySyncButton.setButtonText("Sync");
    delaySyncButton.onClick = [this]() { updateDelaySyncUI(); };
    addAndMakeVisible(delaySyncButton);
    delaySyncAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_DELAY_SYNC, delaySyncButton);

    delayDivisionCombo.addItem("1/32", 1);
    delayDivisionCombo.addItem("1/16T", 2);
    delayDivisionCombo.addItem("1/16", 3);
    delayDivisionCombo.addItem("1/16D", 4);
    delayDivisionCombo.addItem("1/8T", 5);
    delayDivisionCombo.addItem("1/8", 6);
    delayDivisionCombo.addItem("1/8D", 7);
    delayDivisionCombo.addItem("1/4T", 8);
    delayDivisionCombo.addItem("1/4", 9);
    delayDivisionCombo.addItem("1/4D", 10);
    delayDivisionCombo.addItem("1/2T", 11);
    delayDivisionCombo.addItem("1/2", 12);
    delayDivisionCombo.addItem("1/2D", 13);
    delayDivisionCombo.addItem("1/1", 14);
    delayDivisionCombo.addItem("2/1", 15);
    delayDivisionCombo.addItem("4/1", 16);
    addAndMakeVisible(delayDivisionCombo);
    delayDivisionAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_DELAY_DIVISION, delayDivisionCombo);

    setupHorizontalSlider(delayFeedbackSlider);
    delayFeedbackSlider.setNumDecimalPlacesToDisplay(1);
    addAndMakeVisible(delayFeedbackSlider);
    delayFeedbackAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_DELAY_FEEDBACK, delayFeedbackSlider);

    setupLabel(delayFeedbackLabel, "Feedback");
    addAndMakeVisible(delayFeedbackLabel);

    setupHorizontalSlider(delayDampingSlider);
    delayDampingSlider.setNumDecimalPlacesToDisplay(1);
    addAndMakeVisible(delayDampingSlider);
    delayDampingAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_DELAY_DAMPING, delayDampingSlider);

    setupLabel(delayDampingLabel, "Damping");
    addAndMakeVisible(delayDampingLabel);

    setupHorizontalSlider(delaySlopeSlider);
    delaySlopeSlider.setNumDecimalPlacesToDisplay(1);
    addAndMakeVisible(delaySlopeSlider);
    delaySlopeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_DELAY_SLOPE, delaySlopeSlider);

    setupLabel(delaySlopeLabel, "Slope");
    addAndMakeVisible(delaySlopeLabel);

    setupHorizontalSlider(delayDiffuseSlider);
    delayDiffuseSlider.setNumDecimalPlacesToDisplay(1);
    addAndMakeVisible(delayDiffuseSlider);
    delayDiffuseAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_DELAY_DIFFUSE, delayDiffuseSlider);

    setupLabel(delayDiffuseLabel, "Diffuse");
    addAndMakeVisible(delayDiffuseLabel);

    stereoDecorrelateToggle.setButtonText("L/R Decorr");
    stereoDecorrelateToggle.onClick = [this]() {
        audioProcessor.setStereoDecorrelate(stereoDecorrelateToggle.getToggleState());
    };
    addAndMakeVisible(stereoDecorrelateToggle);

    // === Delay Time LFO Controls ===

    setupHorizontalSlider(dlyLfoDepthSlider);
    dlyLfoDepthSlider.setNumDecimalPlacesToDisplay(1);
    dlyLfoDepthSlider.setTextValueSuffix(" ms");
    addAndMakeVisible(dlyLfoDepthSlider);
    dlyLfoDepthAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_DLY_LFO_DEPTH, dlyLfoDepthSlider);

    setupLabel(dlyLfoDepthLabel, "Depth");
    addAndMakeVisible(dlyLfoDepthLabel);

    setupHorizontalSlider(dlyLfoRateSlider);
    dlyLfoRateSlider.setTextValueSuffix(" Hz");
    dlyLfoRateSlider.setNumDecimalPlacesToDisplay(1);  // Reduced from 2 for cleaner display
    addAndMakeVisible(dlyLfoRateSlider);
    dlyLfoRateAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_DLY_LFO_RATE, dlyLfoRateSlider);

    setupLabel(dlyLfoRateLabel, "Rate");
    addAndMakeVisible(dlyLfoRateLabel);

    dlyLfoSyncButton.setButtonText("Sync");
    addAndMakeVisible(dlyLfoSyncButton);
    dlyLfoSyncAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_DLY_LFO_SYNC, dlyLfoSyncButton);

    dlyLfoDivisionCombo.addItem("4/1", 1);
    dlyLfoDivisionCombo.addItem("2/1", 2);
    dlyLfoDivisionCombo.addItem("1/1", 3);
    dlyLfoDivisionCombo.addItem("1/2", 4);
    dlyLfoDivisionCombo.addItem("1/4", 5);
    dlyLfoDivisionCombo.addItem("1/8", 6);
    dlyLfoDivisionCombo.addItem("1/16", 7);
    dlyLfoDivisionCombo.addItem("1/32", 8);
    addAndMakeVisible(dlyLfoDivisionCombo);
    dlyLfoDivisionAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_DLY_LFO_DIVISION, dlyLfoDivisionCombo);

    dlyLfoShapeCombo.addItem("Sine", 1);
    dlyLfoShapeCombo.addItem("Triangle", 2);
    dlyLfoShapeCombo.addItem("Saw", 3);
    dlyLfoShapeCombo.addItem("Inv Saw", 4);
    dlyLfoShapeCombo.addItem("Random", 5);
    addAndMakeVisible(dlyLfoShapeCombo);
    dlyLfoShapeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_DLY_LFO_SHAPE, dlyLfoShapeCombo);

    dlyLfoSyncButton.onClick = [this]() { updateDlyLfoSyncUI(); };
    updateDlyLfoSyncUI();

    // === Spectral Mask Controls ===

    maskEnabledButton.setButtonText("Mask");
    addAndMakeVisible(maskEnabledButton);
    maskEnabledAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_MASK_ENABLED, maskEnabledButton);

    maskModeCombo.addItem("Low Pass", 1);
    maskModeCombo.addItem("High Pass", 2);
    maskModeCombo.addItem("Band Pass", 3);
    addAndMakeVisible(maskModeCombo);
    maskModeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_MASK_MODE, maskModeCombo);

    setupHorizontalSlider(maskLowFreqSlider);
    maskLowFreqSlider.setNumDecimalPlacesToDisplay(0);
    addAndMakeVisible(maskLowFreqSlider);
    maskLowFreqAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_MASK_LOW_FREQ, maskLowFreqSlider);

    setupLabel(maskLowFreqLabel, "Low");
    addAndMakeVisible(maskLowFreqLabel);

    setupHorizontalSlider(maskHighFreqSlider);
    maskHighFreqSlider.setNumDecimalPlacesToDisplay(0);
    addAndMakeVisible(maskHighFreqSlider);
    maskHighFreqAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_MASK_HIGH_FREQ, maskHighFreqSlider);

    setupLabel(maskHighFreqLabel, "High");
    addAndMakeVisible(maskHighFreqLabel);

    setupHorizontalSlider(maskTransitionSlider);
    maskTransitionSlider.setNumDecimalPlacesToDisplay(2);
    maskTransitionSlider.setTextValueSuffix(" oct");
    addAndMakeVisible(maskTransitionSlider);
    maskTransitionAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_MASK_TRANSITION, maskTransitionSlider);

    setupLabel(maskTransitionLabel, "Transition");
    addAndMakeVisible(maskTransitionLabel);

    // === Dry/Wet Mix ===

    setupHorizontalSlider(dryWetSlider);
    addAndMakeVisible(dryWetSlider);
    dryWetAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getValueTreeState(), FrequencyShifterProcessor::PARAM_DRY_WET, dryWetSlider);

    setupLabel(dryWetLabel, "Dry / Wet");
    addAndMakeVisible(dryWetLabel);

    // Set editor size (v109: +28px for preset strip)
    setSize(700, 928);

    // Initialize UI states
    updateDelaySyncUI();
    updateControlsForMode();
}

FrequencyShifterEditor::~FrequencyShifterEditor()
{
    shiftSlider.removeListener(this);
    setLookAndFeel(nullptr);
}

void FrequencyShifterEditor::refreshPresetList()
{
    auto& pm = audioProcessor.getPresetManager();
    auto names = pm.getAllPresetNames();
    auto current = pm.getCurrentPresetName();

    presetComboBox.clear(juce::dontSendNotification);
    for (int i = 0; i < names.size(); ++i)
        presetComboBox.addItem(names[i], i + 1);

    int idx = names.indexOf(current);
    if (idx >= 0)
        presetComboBox.setSelectedId(idx + 1, juce::dontSendNotification);

    // Disable delete for factory presets
    presetDeleteButton.setEnabled(!pm.isFactoryPreset(current));
    presetDeleteButton.setAlpha(pm.isFactoryPreset(current) ? 0.3f : 1.0f);
}

void FrequencyShifterEditor::setupLabel(juce::Label& label, const juce::String& text, bool isSection)
{
    label.setText(text, juce::dontSendNotification);
    if (isSection)
    {
        label.setFont(juce::FontOptions(10.0f));
        label.setColour(juce::Label::textColourId, juce::Colour(Colors::textMuted));
    }
    else
    {
        label.setFont(juce::FontOptions(11.0f));
        label.setColour(juce::Label::textColourId, juce::Colour(Colors::textSec));
    }
    label.setJustificationType(juce::Justification::centredRight);
}

void FrequencyShifterEditor::setupSlider(juce::Slider& slider, juce::Slider::SliderStyle style)
{
    slider.setSliderStyle(style);
    slider.setColour(juce::Slider::textBoxTextColourId, juce::Colour(Colors::text));
    slider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
}

void FrequencyShifterEditor::setupHorizontalSlider(juce::Slider& slider)
{
    slider.setSliderStyle(juce::Slider::LinearHorizontal);
    slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 58, 22);
    slider.setNumDecimalPlacesToDisplay(1);
    slider.setColour(juce::Slider::textBoxTextColourId, juce::Colour(Colors::text));
    slider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
}

void FrequencyShifterEditor::drawStrip(juce::Graphics& g, int y, int height,
                                        const juce::String& label, bool hasBorder, bool dimmed)
{
    auto stripBounds = juce::Rectangle<float>(0, static_cast<float>(y),
                                               static_cast<float>(getWidth()), static_cast<float>(height));

    // Strip background
    g.setColour(juce::Colour(Colors::strip).withAlpha(dimmed ? 0.3f : 1.0f));
    g.fillRect(stripBounds);

    // Top border
    if (hasBorder)
    {
        g.setColour(juce::Colour(Colors::stripBorder));
        g.drawHorizontalLine(y, 0, static_cast<float>(getWidth()));
    }

    // Gold accent bar on left edge
    g.setColour(juce::Colour(dimmed ? Colors::textMuted : Colors::accent).withAlpha(dimmed ? 0.3f : 1.0f));
    g.fillRect(0, y, 3, height);
}

void FrequencyShifterEditor::paint(juce::Graphics& g)
{
    // Main background
    g.fillAll(juce::Colour(Colors::background));

    // Top accent line (gold gradient)
    {
        juce::ColourGradient gradient(
            juce::Colours::transparentBlack, 0, 0,
            juce::Colours::transparentBlack, static_cast<float>(getWidth()), 0,
            false);
        gradient.addColour(0.08, juce::Colours::transparentBlack);
        gradient.addColour(0.3, juce::Colour(Colors::accentDim));
        gradient.addColour(0.5, juce::Colour(Colors::accent));
        gradient.addColour(0.7, juce::Colour(Colors::accentDim));
        gradient.addColour(0.92, juce::Colours::transparentBlack);
        g.setGradientFill(gradient);
        g.fillRect(0, 0, getWidth(), 1);
    }

    // Title - "Holy Shifter" with serif font
    g.setColour(juce::Colour(Colors::text));
    g.setFont(juce::FontOptions(28.0f).withStyle("Light"));
    g.drawText("H O L Y   S H I F T E R", 28, 16, 400, 32, juce::Justification::centredLeft, false);

    // Subtitle
    g.setColour(juce::Colour(Colors::textMuted));
    g.setFont(juce::FontOptions(12.0f).withStyle("Italic"));
    g.drawText("Frequency Shifter with Harmonic Quantisation", 28, 48, 400, 16,
               juce::Justification::centredLeft, false);

    // Preset strip background
    {
        const int psY = 56;
        g.setColour(juce::Colour(Colors::strip));
        g.fillRect(0, psY, getWidth(), 30);
        g.setColour(juce::Colour(Colors::stripBorder));
        g.drawHorizontalLine(psY, 0, static_cast<float>(getWidth()));
        g.drawHorizontalLine(psY + 30, 0, static_cast<float>(getWidth()));
    }

    const int presetStripOffset = 28;

    // Check processing mode for dimming
    bool isSpectral = (processingModeCombo.getSelectedId() == 2);

    // Spectral panel background - shifted down
    g.setColour(juce::Colour(Colors::panelBg));
    g.fillRoundedRectangle(240.0f, 76.0f + presetStripOffset, 432.0f, 180.0f, 6.0f);
    g.setColour(juce::Colour(Colors::panelBorder));
    g.drawRoundedRectangle(240.0f, 76.0f + presetStripOffset, 432.0f, 180.0f, 6.0f, 1.0f);

    // Strip sections - shifted down
    int stripY = 268 + presetStripOffset;

    // Smear & Enhance strip
    drawStrip(g, stripY, 56, "Smear & Enhance", true, !isSpectral);
    g.setColour(juce::Colour(!isSpectral ? Colors::textMuted : Colors::accent));
    g.setFont(juce::FontOptions(10.0f));
    g.drawText("SMEAR & ENHANCE", 14, stripY + 4, 180, 14, juce::Justification::centredLeft, false);
    stripY += 60;  // 56 + 4px gap

    // Freq Modulation strip
    drawStrip(g, stripY, 78, "Freq Modulation", true, false);
    g.setColour(juce::Colour(Colors::accent));
    g.drawText("FREQ MODULATION", 14, stripY + 4, 180, 14, juce::Justification::centredLeft, false);
    stripY += 82;  // 78 + 4px gap

    // Delay strip
    drawStrip(g, stripY, 144, "Delay", true, false);
    g.setColour(juce::Colour(Colors::accent));
    g.drawText("DELAY", 14, stripY + 4, 120, 14, juce::Justification::centredLeft, false);
    stripY += 148;  // 144 + 4px gap

    // Delay Modulation strip
    drawStrip(g, stripY, 78, "Delay Modulation", true, false);
    g.setColour(juce::Colour(Colors::accent));
    g.drawText("DELAY MODULATION", 14, stripY + 4, 180, 14, juce::Justification::centredLeft, false);
    stripY += 82;  // 78 + 4px gap

    // Mask strip
    drawStrip(g, stripY, 86, "Mask", true, !isSpectral);
    g.setColour(juce::Colour(!isSpectral ? Colors::textMuted : Colors::accent));
    g.drawText("MASK", 14, stripY + 4, 120, 14, juce::Justification::centredLeft, false);
    stripY += 90;  // 86 + 4px gap

    // Mix strip
    drawStrip(g, stripY, 50, "Mix", true, false);
    stripY += 50;

    // Bottom accent line
    {
        juce::ColourGradient gradient(
            juce::Colours::transparentBlack, 0, static_cast<float>(getHeight() - 1),
            juce::Colours::transparentBlack, static_cast<float>(getWidth()), static_cast<float>(getHeight() - 1),
            false);
        gradient.addColour(0.15, juce::Colours::transparentBlack);
        gradient.addColour(0.5, juce::Colour(Colors::borderDim));
        gradient.addColour(0.85, juce::Colours::transparentBlack);
        g.setGradientFill(gradient);
        g.fillRect(0, getHeight() - 1, getWidth(), 1);
    }
}

void FrequencyShifterEditor::resized()
{
    const int margin = 28;
    const int presetStripOffset = 28;  // Height of preset strip

    // Title bar controls (stay in place)
    warmButton.setBounds(getWidth() - margin - 80, 24, 80, 24);

    // Preset strip (y=58, between title and content)
    int presetY = 60;
    presetPrevButton.setBounds(margin, presetY, 24, 24);
    presetNextButton.setBounds(margin + 26, presetY, 24, 24);
    presetComboBox.setBounds(margin + 56, presetY, 340, 24);
    presetSaveButton.setBounds(margin + 404, presetY, 50, 24);
    presetSaveAsButton.setBounds(margin + 458, presetY, 65, 24);
    presetDeleteButton.setBounds(margin + 527, presetY, 40, 24);

    // Main shift knob (left side) - shifted down
    shiftSlider.setBounds(28, 76 + presetStripOffset, 200, 200);

    // Spectral panel controls (right side) - shifted down
    processingModeCombo.setBounds(252, 84 + presetStripOffset, 100, 24);

    int panelX = 252;
    int panelY = 118 + presetStripOffset;
    int panelRowGap = 30;

    pianoKeyboard->setBounds(panelX, panelY, 320, 46);
    panelY += 52;

    quantizeLabel.setBounds(panelX, panelY, 60, 22);
    quantizeSlider.setBounds(panelX + 65, panelY, 280, 22);
    panelY += panelRowGap;

    preserveLabel.setBounds(panelX, panelY, 60, 22);
    preserveSlider.setBounds(panelX + 65, panelY, 280, 22);
    panelY += panelRowGap;

    transientsLabel.setBounds(panelX, panelY, 65, 22);
    transientsSlider.setBounds(panelX + 68, panelY, 140, 22);
    sensitivityLabel.setBounds(panelX + 215, panelY, 65, 22);
    sensitivitySlider.setBounds(panelX + 282, panelY, 100, 22);

    // Strip sections - shifted down
    int stripY = 268 + presetStripOffset;
    int stripPadding = 24;

    // Smear & Enhance strip
    phaseVocoderButton.setBounds(margin, stripY + stripPadding, 100, 24);
    smearLabel.setBounds(margin + 110, stripY + stripPadding, 45, 22);
    smearSlider.setBounds(margin + 160, stripY + stripPadding, getWidth() - margin * 2 - 170, 22);
    stripY += 60;

    // Freq Modulation strip
    int lfoY = stripY + stripPadding;
    lfoDepthLabel.setBounds(margin, lfoY, 45, 22);
    lfoDepthSlider.setBounds(margin + 50, lfoY, getWidth() - margin * 2 - 190, 22);
    lfoDepthModeCombo.setBounds(getWidth() - margin - 130, lfoY, 75, 24);

    lfoY += 30;
    lfoRateLabel.setBounds(margin, lfoY, 45, 22);
    lfoRateSlider.setBounds(margin + 50, lfoY, getWidth() - margin * 2 - 280, 22);
    lfoSyncButton.setBounds(getWidth() - margin - 220, lfoY, 75, 24);
    lfoDivisionCombo.setBounds(getWidth() - margin - 140, lfoY, 58, 24);
    lfoShapeCombo.setBounds(getWidth() - margin - 78, lfoY, 78, 24);
    stripY += 82;

    // Delay strip
    int delY = stripY + stripPadding;
    delayEnabledButton.setBounds(margin, delY, 75, 24);
    delayTimeLabel.setBounds(margin + 85, delY, 40, 22);
    delayTimeSlider.setBounds(margin + 130, delY, getWidth() - margin * 2 - 290, 22);
    delaySyncButton.setBounds(getWidth() - margin - 150, delY, 75, 24);
    delayDivisionCombo.setBounds(getWidth() - margin - 68, delY, 68, 24);

    delY += 30;
    delayFeedbackLabel.setBounds(margin, delY, 58, 22);
    delayFeedbackSlider.setBounds(margin + 62, delY, (getWidth() - margin * 2 - 130) / 2, 22);
    int dampX = margin + 62 + (getWidth() - margin * 2 - 130) / 2 + 8;
    delayDampingLabel.setBounds(dampX, delY, 58, 22);
    delayDampingSlider.setBounds(dampX + 60, delY, getWidth() - margin - dampX - 60, 22);

    delY += 30;
    delaySlopeLabel.setBounds(margin, delY, 45, 22);
    delaySlopeSlider.setBounds(margin + 50, delY, (getWidth() - margin * 2 - 118) / 2, 22);
    int diffX = margin + 50 + (getWidth() - margin * 2 - 118) / 2 + 8;
    delayDiffuseLabel.setBounds(diffX, delY, 52, 22);
    delayDiffuseSlider.setBounds(diffX + 55, delY, getWidth() - margin - diffX - 55, 22);

    delY += 30;
    stereoDecorrelateToggle.setBounds(getWidth() - margin - 110, delY, 110, 22);
    stripY += 148;

    // Delay Modulation strip
    int dlyLfoY = stripY + stripPadding;
    dlyLfoDepthLabel.setBounds(margin, dlyLfoY, 45, 22);
    dlyLfoDepthSlider.setBounds(margin + 50, dlyLfoY, getWidth() - margin * 2 - 60, 22);

    dlyLfoY += 30;
    dlyLfoRateLabel.setBounds(margin, dlyLfoY, 45, 22);
    dlyLfoRateSlider.setBounds(margin + 50, dlyLfoY, getWidth() - margin * 2 - 280, 22);
    dlyLfoSyncButton.setBounds(getWidth() - margin - 220, dlyLfoY, 75, 24);
    dlyLfoDivisionCombo.setBounds(getWidth() - margin - 140, dlyLfoY, 58, 24);
    dlyLfoShapeCombo.setBounds(getWidth() - margin - 78, dlyLfoY, 78, 24);
    stripY += 82;

    // Mask strip
    int maskY = stripY + stripPadding;
    maskEnabledButton.setBounds(margin, maskY, 75, 24);
    maskModeCombo.setBounds(margin + 85, maskY, 95, 24);
    maskTransitionLabel.setBounds(margin + 195, maskY, 65, 22);
    maskTransitionSlider.setBounds(margin + 262, maskY, getWidth() - margin * 2 - 270, 22);

    maskY += 30;
    maskLowFreqLabel.setBounds(margin, maskY, 30, 22);
    maskLowFreqSlider.setBounds(margin + 35, maskY, (getWidth() - margin * 2 - 75) / 2, 22);
    int highX = margin + 35 + (getWidth() - margin * 2 - 75) / 2 + 10;
    maskHighFreqLabel.setBounds(highX, maskY, 35, 22);
    maskHighFreqSlider.setBounds(highX + 38, maskY, getWidth() - margin - highX - 38, 22);
    stripY += 90;

    // Mix strip
    int mixY = stripY + 14;
    dryWetLabel.setBounds(margin, mixY, 65, 22);
    dryWetSlider.setBounds(margin + 70, mixY, getWidth() - margin * 2 - 80, 22);
    stripY += 50;
}

void FrequencyShifterEditor::sliderValueChanged(juce::Slider* slider)
{
    if (slider == &shiftSlider)
    {
        float value = static_cast<float>(shiftSlider.getValue());
        auto* param = audioProcessor.getValueTreeState().getParameter(FrequencyShifterProcessor::PARAM_SHIFT_HZ);
        if (param != nullptr)
        {
            // Convert to normalized 0-1 range for the parameter (-20000 to +20000)
            float normalized = (value + 20000.0f) / 40000.0f;
            param->setValueNotifyingHost(normalized);
        }
    }
}

void FrequencyShifterEditor::updateDelaySyncUI()
{
    bool syncEnabled = delaySyncButton.getToggleState();

    delayTimeSlider.setEnabled(!syncEnabled);
    delayTimeSlider.setAlpha(syncEnabled ? 0.35f : 1.0f);
    delayTimeLabel.setAlpha(syncEnabled ? 0.35f : 1.0f);

    delayDivisionCombo.setEnabled(syncEnabled);
    delayDivisionCombo.setAlpha(syncEnabled ? 1.0f : 0.35f);
}

void FrequencyShifterEditor::updateLfoSyncUI()
{
    bool syncEnabled = lfoSyncButton.getToggleState();

    lfoRateSlider.setEnabled(!syncEnabled);
    lfoRateSlider.setAlpha(syncEnabled ? 0.35f : 1.0f);
    lfoRateLabel.setAlpha(syncEnabled ? 0.35f : 1.0f);

    lfoDivisionCombo.setEnabled(syncEnabled);
    lfoDivisionCombo.setAlpha(syncEnabled ? 1.0f : 0.35f);
}

void FrequencyShifterEditor::updateDlyLfoSyncUI()
{
    bool syncEnabled = dlyLfoSyncButton.getToggleState();

    dlyLfoRateSlider.setEnabled(!syncEnabled);
    dlyLfoRateSlider.setAlpha(syncEnabled ? 0.35f : 1.0f);
    dlyLfoRateLabel.setAlpha(syncEnabled ? 0.35f : 1.0f);

    dlyLfoDivisionCombo.setEnabled(syncEnabled);
    dlyLfoDivisionCombo.setAlpha(syncEnabled ? 1.0f : 0.35f);
}

void FrequencyShifterEditor::updateControlsForMode()
{
    // Classic is ID=1, Spectral is ID=2 (matching processor's 0=Classic, 1=Spectral)
    bool isClassic = (processingModeCombo.getSelectedId() == 1);
    float disabledAlpha = 0.25f;
    float enabledAlpha = 1.0f;

    // SMEAR - Spectral only
    smearSlider.setEnabled(!isClassic);
    smearSlider.setAlpha(isClassic ? disabledAlpha : enabledAlpha);
    smearLabel.setAlpha(isClassic ? disabledAlpha : enabledAlpha);

    // Quantize, Root, Scale - Spectral only
    quantizeSlider.setEnabled(!isClassic);
    quantizeSlider.setAlpha(isClassic ? disabledAlpha : enabledAlpha);
    quantizeLabel.setAlpha(isClassic ? disabledAlpha : enabledAlpha);

    if (pianoKeyboard)
    {
        pianoKeyboard->setEnabled(!isClassic);
        pianoKeyboard->setAlpha(isClassic ? disabledAlpha : enabledAlpha);
    }

    // PRESERVE, TRANSIENTS, SENSITIVITY - Spectral only
    preserveSlider.setEnabled(!isClassic);
    preserveSlider.setAlpha(isClassic ? disabledAlpha : enabledAlpha);
    preserveLabel.setAlpha(isClassic ? disabledAlpha : enabledAlpha);

    transientsSlider.setEnabled(!isClassic);
    transientsSlider.setAlpha(isClassic ? disabledAlpha : enabledAlpha);
    transientsLabel.setAlpha(isClassic ? disabledAlpha : enabledAlpha);

    sensitivitySlider.setEnabled(!isClassic);
    sensitivitySlider.setAlpha(isClassic ? disabledAlpha : enabledAlpha);
    sensitivityLabel.setAlpha(isClassic ? disabledAlpha : enabledAlpha);

    // LFO Depth Mode - Spectral only
    lfoDepthModeCombo.setEnabled(!isClassic);
    lfoDepthModeCombo.setAlpha(isClassic ? disabledAlpha : enabledAlpha);

    // Enhanced Mode (Phase Vocoder) - Spectral only
    phaseVocoderButton.setEnabled(!isClassic);
    phaseVocoderButton.setAlpha(isClassic ? disabledAlpha : enabledAlpha);

    // Mask controls - all Spectral only
    maskEnabledButton.setEnabled(!isClassic);
    maskEnabledButton.setAlpha(isClassic ? disabledAlpha : enabledAlpha);

    maskModeCombo.setEnabled(!isClassic);
    maskModeCombo.setAlpha(isClassic ? disabledAlpha : enabledAlpha);

    maskLowFreqSlider.setEnabled(!isClassic);
    maskLowFreqSlider.setAlpha(isClassic ? disabledAlpha : enabledAlpha);
    maskLowFreqLabel.setAlpha(isClassic ? disabledAlpha : enabledAlpha);

    maskHighFreqSlider.setEnabled(!isClassic);
    maskHighFreqSlider.setAlpha(isClassic ? disabledAlpha : enabledAlpha);
    maskHighFreqLabel.setAlpha(isClassic ? disabledAlpha : enabledAlpha);

    maskTransitionSlider.setEnabled(!isClassic);
    maskTransitionSlider.setAlpha(isClassic ? disabledAlpha : enabledAlpha);
    maskTransitionLabel.setAlpha(isClassic ? disabledAlpha : enabledAlpha);

    // SLOPE, DIFFUSE - Spectral delay features
    delaySlopeSlider.setEnabled(!isClassic);
    delaySlopeSlider.setAlpha(isClassic ? disabledAlpha : enabledAlpha);
    delaySlopeLabel.setAlpha(isClassic ? disabledAlpha : enabledAlpha);

    delayDiffuseSlider.setEnabled(!isClassic);
    delayDiffuseSlider.setAlpha(isClassic ? disabledAlpha : enabledAlpha);
    delayDiffuseLabel.setAlpha(isClassic ? disabledAlpha : enabledAlpha);

    // Trigger repaint to update strip dimming
    repaint();
}
