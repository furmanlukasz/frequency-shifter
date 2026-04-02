#include "HolyShifterUI.h"
#include "../PluginProcessor.h"
#include "../dsp/Scales.h"

HolyShifterUI::HolyShifterUI(FrequencyShifterProcessor& processor)
    : processor_(processor),
      apvts_(processor.getValueTreeState()),
      presetPrevBtn_("<"),
      presetNextBtn_(">"),
      warmToggle_("Warm"),
      phaseVocoderToggle_("Enhanced"),
      lfoSyncToggle_("Sync"),
      delayEnabledToggle_("Delay"),
      delaySyncToggle_("Sync"),
      dlyLfoSyncToggle_("Sync"),
      maskEnabledToggle_("Mask")
{
    // Wire all controls to parameters
    shiftKnob_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_SHIFT_HZ);
    shiftKnob_.setBipolar(true);
    shiftKnob_.setUnit("HZ");

    // Custom symmetric log mapping: knob range ±5000 Hz, param range ±20000 Hz
    constexpr double logScale = 10.0;
    constexpr double maxShift = 5000.0;
    double logMax = std::log(1.0 + maxShift / logScale);

    // knobNorm (0-1) -> display Hz value (±5000)
    shiftKnob_.setCustomMapping(
        [logScale, maxShift, logMax](float knobNorm) -> float {
            double symNorm = knobNorm * 2.0 - 1.0;
            double sign = symNorm >= 0.0 ? 1.0 : -1.0;
            double absNorm = std::abs(symNorm);
            double absVal = logScale * (std::exp(absNorm * logMax) - 1.0);
            return static_cast<float>(sign * absVal);
        },
        // knobNorm (0-1) -> paramNorm (0-1) [maps ±5000 display to ±20000 param]
        [logScale, maxShift, logMax](float knobNorm) -> float {
            double symNorm = knobNorm * 2.0 - 1.0;
            double sign = symNorm >= 0.0 ? 1.0 : -1.0;
            double absNorm = std::abs(symNorm);
            double absVal = logScale * (std::exp(absNorm * logMax) - 1.0);
            double paramValue = sign * absVal;
            return static_cast<float>((paramValue + 20000.0) / 40000.0);
        },
        // paramNorm (0-1) -> knobNorm (0-1)
        [logScale, logMax](float paramNorm) -> float {
            double paramValue = paramNorm * 40000.0 - 20000.0;
            double sign = paramValue >= 0.0 ? 1.0 : -1.0;
            double absVal = std::abs(paramValue);
            // Clamp to ±5000 display range
            absVal = std::min(absVal, 5000.0);
            double normalized = sign * std::log(1.0 + absVal / logScale) / logMax;
            return static_cast<float>((normalized + 1.0) * 0.5);
        }
    );
    addChild(&shiftKnob_);

    warmToggle_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_WARM);
    addChild(&warmToggle_);

    // Preset strip
    presetPrevBtn_.setText("<");
    presetPrevBtn_.onToggle() = [this](visage::Button*, bool) {
        processor_.getPresetManager().loadPreviousPreset();
        currentPresetName_ = processor_.getPresetManager().getCurrentPresetName().toStdString();
        redraw();
    };
    addChild(&presetPrevBtn_);

    presetNextBtn_.setText(">");
    presetNextBtn_.onToggle() = [this](visage::Button*, bool) {
        processor_.getPresetManager().loadNextPreset();
        currentPresetName_ = processor_.getPresetManager().getCurrentPresetName().toStdString();
        redraw();
    };
    addChild(&presetNextBtn_);

    currentPresetName_ = processor_.getPresetManager().getCurrentPresetName().toStdString();

    // Preset name clickable area (opens dropdown)
    presetNameArea_.onMouseDown() = [this](const visage::MouseEvent&) {
        if (presetDropdown_.isOpen())
        {
            presetDropdown_.hide();
        }
        else
        {
            auto pos = presetNameArea_.positionInWindow();
            presetDropdown_.showFor(&processor_,
                static_cast<int>(pos.x), static_cast<int>(pos.y) + presetNameArea_.height(),
                presetNameArea_.width());
        }
    };
    addChild(&presetNameArea_);

    // Preset dropdown (added later for z-order, see end of constructor)
    presetDropdown_.onPresetChanged_ = [this]() {
        currentPresetName_ = processor_.getPresetManager().getCurrentPresetName().toStdString();
        redraw();
    };

    // Processing mode combo
    processingModeCombo_.addItem("Classic");
    processingModeCombo_.addItem("Spectral");
    processingModeCombo_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_PROCESSING_MODE);
    addChild(&processingModeCombo_);

    // Piano keyboard
    pianoKeyboard_.setAttachments(apvts_, FrequencyShifterProcessor::PARAM_SCALE_NOTE_PREFIX);
    addChild(&pianoKeyboard_);

    // Spectral panel sliders
    quantizeSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_QUANTIZE_STRENGTH);
    addChild(&quantizeSlider_);

    preserveSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_PRESERVE);
    addChild(&preserveSlider_);

    transientsSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_TRANSIENTS);
    addChild(&transientsSlider_);

    sensitivitySlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_SENSITIVITY);
    sensitivitySlider_.setDecimals(0);
    addChild(&sensitivitySlider_);

    // Smear & Enhance
    phaseVocoderToggle_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_PHASE_VOCODER);
    addChild(&phaseVocoderToggle_);

    smearSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_SMEAR);
    smearSlider_.setSuffix(" ms");
    addChild(&smearSlider_);

    // LFO
    lfoDepthSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_LFO_DEPTH);
    addChild(&lfoDepthSlider_);

    lfoDepthModeCombo_.addItem("Hz");
    lfoDepthModeCombo_.addItem("Degrees");
    lfoDepthModeCombo_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_LFO_DEPTH_MODE);
    addChild(&lfoDepthModeCombo_);

    lfoRateSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_LFO_RATE);
    lfoRateSlider_.setSuffix(" Hz");
    addChild(&lfoRateSlider_);

    lfoSyncToggle_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_LFO_SYNC);
    addChild(&lfoSyncToggle_);

    for (auto& name : { "4/1", "2/1", "1/1", "1/2", "1/4", "1/8", "1/16", "1/32",
                         "1/4T", "1/8T", "1/16T", "1/4.", "1/8.", "1/16." })
        lfoDivisionCombo_.addItem(name);
    lfoDivisionCombo_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_LFO_DIVISION);
    addChild(&lfoDivisionCombo_);

    for (auto& name : { "Sine", "Triangle", "Saw", "Inv Saw", "Random" })
        lfoShapeCombo_.addItem(name);
    lfoShapeCombo_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_LFO_SHAPE);
    addChild(&lfoShapeCombo_);

    // Delay
    delayEnabledToggle_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DELAY_ENABLED);
    addChild(&delayEnabledToggle_);

    delayTimeSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DELAY_TIME);
    delayTimeSlider_.setSuffix(" ms");
    addChild(&delayTimeSlider_);

    delaySyncToggle_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DELAY_SYNC);
    addChild(&delaySyncToggle_);

    for (auto& name : { "1/32", "1/16T", "1/16", "1/16D", "1/8T", "1/8", "1/8D", "1/4T",
                         "1/4", "1/4D", "1/2T", "1/2", "1/2D", "1/1", "2/1", "4/1" })
        delayDivisionCombo_.addItem(name);
    delayDivisionCombo_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DELAY_DIVISION);
    addChild(&delayDivisionCombo_);

    delayFeedbackSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DELAY_FEEDBACK);
    addChild(&delayFeedbackSlider_);

    delayDampingSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DELAY_DAMPING);
    addChild(&delayDampingSlider_);

    delaySlopeSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DELAY_SLOPE);
    addChild(&delaySlopeSlider_);

    delayDiffuseSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DELAY_DIFFUSE);
    addChild(&delayDiffuseSlider_);

    stereoDecorrelateToggle_.setLabel("L/R Decorr");
    stereoDecorrelateToggle_.onToggle = [this](bool on) {
        processor_.setStereoDecorrelate(on);
    };
    addChild(&stereoDecorrelateToggle_);

    // Delay Modulation
    dlyLfoDepthSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DLY_LFO_DEPTH);
    dlyLfoDepthSlider_.setSuffix(" ms");
    addChild(&dlyLfoDepthSlider_);

    dlyLfoRateSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DLY_LFO_RATE);
    dlyLfoRateSlider_.setSuffix(" Hz");
    addChild(&dlyLfoRateSlider_);

    dlyLfoSyncToggle_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DLY_LFO_SYNC);
    addChild(&dlyLfoSyncToggle_);

    for (auto& name : { "4/1", "2/1", "1/1", "1/2", "1/4", "1/8", "1/16", "1/32",
                         "1/4T", "1/8T", "1/16T", "1/4.", "1/8.", "1/16." })
        dlyLfoDivisionCombo_.addItem(name);
    dlyLfoDivisionCombo_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DLY_LFO_DIVISION);
    addChild(&dlyLfoDivisionCombo_);

    for (auto& name : { "Sine", "Triangle", "Saw", "Inv Saw", "Random" })
        dlyLfoShapeCombo_.addItem(name);
    dlyLfoShapeCombo_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DLY_LFO_SHAPE);
    addChild(&dlyLfoShapeCombo_);

    // Mask
    maskEnabledToggle_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_MASK_ENABLED);
    addChild(&maskEnabledToggle_);

    for (auto& name : { "Low Pass", "High Pass", "Band Pass" })
        maskModeCombo_.addItem(name);
    maskModeCombo_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_MASK_MODE);
    addChild(&maskModeCombo_);

    maskTransitionSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_MASK_TRANSITION);
    maskTransitionSlider_.setSuffix(" oct");
    maskTransitionSlider_.setDecimals(2);
    addChild(&maskTransitionSlider_);

    maskLowFreqSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_MASK_LOW_FREQ);
    maskLowFreqSlider_.setDecimals(0);
    addChild(&maskLowFreqSlider_);

    maskHighFreqSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_MASK_HIGH_FREQ);
    maskHighFreqSlider_.setDecimals(0);
    addChild(&maskHighFreqSlider_);

    // Mix
    dryWetSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DRY_WET);
    addChild(&dryWetSlider_);

    // Shared dropdown overlay — must be added LAST for z-ordering
    addChild(&dropdownOverlay_);
    addChild(&presetDropdown_);
    HolyComboBox::setSharedDropdown(&dropdownOverlay_);
}

// === HolyPresetDropdown ===

void HolyPresetDropdown::showFor(FrequencyShifterProcessor* proc, int x, int y, int w)
{
    processor_ = proc;
    presetNames_.clear();
    auto names = proc->getPresetManager().getAllPresetNames();
    for (int i = 0; i < names.size(); ++i)
        presetNames_.push_back(names[i].toStdString());

    int dropH = static_cast<int>(presetNames_.size()) * kItemHeight + 4;
    if (parent())
        dropH = (dropH < parent()->height() - y) ? dropH : parent()->height() - y;

    setBounds(x, y, w, dropH);
    hoveredIndex_ = -1;
    setVisible(true);
    redraw();
}

void HolyPresetDropdown::hide()
{
    setVisible(false);
    processor_ = nullptr;
    if (parent())
        parent()->redraw();
}

void HolyPresetDropdown::draw(visage::Canvas& canvas)
{
    float w = static_cast<float>(width());
    float h = static_cast<float>(height());

    canvas.setColor(holy::colors::raised);
    canvas.roundedRectangle(0, 0, w, h, 3.0f);
    canvas.setColor(holy::colors::border);
    canvas.roundedRectangleBorder(0, 0, w, h, 3.0f, 1.0f);

    auto font = holy::makeFont(13.0f);
    std::string currentName = processor_ ?
        processor_->getPresetManager().getCurrentPresetName().toStdString() : "";

    for (int i = 0; i < static_cast<int>(presetNames_.size()); ++i)
    {
        int itemY = 2 + i * kItemHeight;
        if (itemY + kItemHeight > static_cast<int>(h))
            break;

        if (i == hoveredIndex_)
        {
            canvas.setColor(holy::colors::accentDim);
            canvas.fill(2, itemY, static_cast<int>(w) - 4, kItemHeight);
        }
        else if (presetNames_[i] == currentName)
        {
            canvas.setColor(0xFF1A1A1Du);
            canvas.fill(2, itemY, static_cast<int>(w) - 4, kItemHeight);
        }

        canvas.setColor(holy::colors::text);
        canvas.text(presetNames_[i].c_str(), font, visage::Font::kLeft,
                    8, itemY, static_cast<int>(w) - 16, kItemHeight);
    }
}

void HolyPresetDropdown::mouseDown(const visage::MouseEvent& e)
{
    if (!processor_)
    {
        hide();
        return;
    }

    int itemIdx = static_cast<int>((e.position.y - 2) / kItemHeight);
    if (itemIdx >= 0 && itemIdx < static_cast<int>(presetNames_.size()))
    {
        processor_->getPresetManager().loadPreset(presetNames_[itemIdx]);
        if (onPresetChanged_)
            onPresetChanged_();
    }
    hide();
}

void HolyPresetDropdown::mouseMove(const visage::MouseEvent& e)
{
    int newHover = static_cast<int>((e.position.y - 2) / kItemHeight);
    if (newHover < 0 || newHover >= static_cast<int>(presetNames_.size()))
        newHover = -1;
    if (newHover != hoveredIndex_)
    {
        hoveredIndex_ = newHover;
        redraw();
    }
}

void HolyShifterUI::pollState()
{
    currentPresetName_ = processor_.getPresetManager().getCurrentPresetName().toStdString();
    redrawAll();
}

void HolyShifterUI::drawStrip(visage::Canvas& canvas, int y, int h,
                                const std::string& label, bool dimmed)
{
    unsigned int stripAlpha = dimmed ? 0x4D0E0E10u : holy::colors::strip;
    canvas.setColor(stripAlpha);
    canvas.fill(0, y, width(), h);

    canvas.setColor(holy::colors::stripBorder);
    canvas.segment(0.0f, static_cast<float>(y), static_cast<float>(width()), static_cast<float>(y), 1.0f, false);

    // Gold accent bar on left edge
    unsigned int accentColor = dimmed ? 0x4DC9A96Eu : holy::colors::accent;
    canvas.setColor(accentColor);
    canvas.fill(0, y, 3, h);

    // Section label
    auto labelFont = holy::makeFont(10.0f);
    canvas.setColor(dimmed ? holy::colors::textMuted : holy::colors::accent);
    canvas.text(label.c_str(), labelFont, visage::Font::kLeft, 14, y + 4, 180, 14);
}

void HolyShifterUI::draw(visage::Canvas& canvas)
{
    int w = width();
    int h = height();

    // Background
    canvas.setColor(holy::colors::background);
    canvas.fill(0, 0, w, h);

    // Title
    auto titleFont = holy::makeFont(28.0f);
    canvas.setColor(holy::colors::text);
    canvas.text("H O L Y   S H I F T E R", titleFont, visage::Font::kLeft, 28, 16, 400, 32);

    // Subtitle
    auto subtitleFont = holy::makeFont(12.0f);
    canvas.setColor(holy::colors::textMuted);
    canvas.text("Frequency Shifter with Harmonic Quantisation", subtitleFont,
                visage::Font::kLeft, 28, 48, 400, 16);

    // Preset strip background (below subtitle)
    canvas.setColor(holy::colors::strip);
    canvas.fill(0, 68, w, 30);
    canvas.setColor(holy::colors::stripBorder);
    canvas.segment(0.0f, 68.0f, static_cast<float>(w), 68.0f, 1.0f, false);
    canvas.segment(0.0f, 98.0f, static_cast<float>(w), 98.0f, 1.0f, false);

    // Preset name
    auto presetFont = holy::makeFont(13.0f);
    canvas.setColor(holy::colors::text);
    canvas.text(currentPresetName_.c_str(), presetFont, visage::Font::kLeft, 84, 72, 340, 24);

    bool isSpectral = (processingModeCombo_.getSelectedIndex() == 1);

    // Spectral panel background
    canvas.setColor(holy::colors::panelBg);
    canvas.roundedRectangle(240.0f, 104.0f, 432.0f, 180.0f, 6.0f);
    canvas.setColor(holy::colors::panelBorder);
    canvas.roundedRectangleBorder(240.0f, 104.0f, 432.0f, 180.0f, 6.0f, 1.0f);

    // Strip sections
    int stripY = 296;
    drawStrip(canvas, stripY, 56, "SMEAR & ENHANCE", !isSpectral);
    stripY += 60;
    drawStrip(canvas, stripY, 78, "FREQ MODULATION");
    stripY += 82;
    drawStrip(canvas, stripY, 144, "DELAY");
    stripY += 148;
    drawStrip(canvas, stripY, 78, "DELAY MODULATION");
    stripY += 82;
    drawStrip(canvas, stripY, 86, "MASK", !isSpectral);
    stripY += 90;
    drawStrip(canvas, stripY, 50, "MIX");

    // === Labels (drawn on top of strips) ===
    auto labelFont = holy::makeFont(11.0f);
    canvas.setColor(holy::colors::textSec);

    int margin = 28;
    int panelX = 252;

    // Spectral panel labels
    int panelY = 198;
    int panelRowGap = 30;
    canvas.text("Quantize", labelFont, visage::Font::kRight, panelX, panelY, 60, 22);
    panelY += panelRowGap;
    canvas.text("Envelope", labelFont, visage::Font::kRight, panelX, panelY, 60, 22);
    panelY += panelRowGap;
    canvas.text("Transients", labelFont, visage::Font::kRight, panelX, panelY, 65, 22);
    canvas.text("Sensitivity", labelFont, visage::Font::kRight, panelX + 215, panelY, 65, 22);

    // Smear strip label
    stripY = 296;
    canvas.text("Smear", labelFont, visage::Font::kRight, margin + 110, stripY + 24, 45, 22);

    // Freq Modulation labels
    stripY = 356;
    canvas.text("Depth", labelFont, visage::Font::kRight, margin, stripY + 24, 45, 22);
    canvas.text("Rate", labelFont, visage::Font::kRight, margin, stripY + 54, 45, 22);

    // Delay labels
    stripY = 438;
    canvas.text("Time", labelFont, visage::Font::kRight, margin + 85, stripY + 24, 40, 22);
    canvas.text("Feedback", labelFont, visage::Font::kRight, margin, stripY + 54, 58, 22);
    int dampLabelX = margin + 62 + (w - margin * 2 - 130) / 2 + 8;
    canvas.text("Damping", labelFont, visage::Font::kLeft, dampLabelX, stripY + 54, 58, 22);
    canvas.text("Slope", labelFont, visage::Font::kRight, margin, stripY + 84, 45, 22);
    int diffLabelX = margin + 50 + (w - margin * 2 - 118) / 2 + 8;
    canvas.text("Diffuse", labelFont, visage::Font::kLeft, diffLabelX, stripY + 84, 52, 22);

    // Delay Modulation labels
    stripY = 586;
    canvas.text("Depth", labelFont, visage::Font::kRight, margin, stripY + 24, 45, 22);
    canvas.text("Rate", labelFont, visage::Font::kRight, margin, stripY + 54, 45, 22);

    // Mask labels
    stripY = 668;
    canvas.text("Transition", labelFont, visage::Font::kRight, margin + 195, stripY + 24, 65, 22);
    canvas.text("Low", labelFont, visage::Font::kRight, margin, stripY + 54, 30, 22);
    canvas.text("High", labelFont, visage::Font::kRight, margin + 35 + (w - margin * 2 - 75) / 2 + 10, stripY + 54, 35, 22);

    // Mix label
    stripY = 758;
    canvas.text("Dry / Wet", labelFont, visage::Font::kRight, margin, stripY + 14, 65, 22);
}

void HolyShifterUI::resized()
{
    int margin = 28;

    // Title bar
    warmToggle_.setBounds(width() - margin - 80, 24, 80, 24);

    // Preset strip (below subtitle at y=68)
    int presetY = 71;
    presetPrevBtn_.setBounds(margin, presetY, 24, 24);
    presetNextBtn_.setBounds(margin + 26, presetY, 24, 24);
    presetNameArea_.setBounds(margin + 56, presetY, 340, 24);

    // Main shift knob
    shiftKnob_.setBounds(28, 104, 200, 200);

    // Spectral panel
    processingModeCombo_.setBounds(252, 112, 100, 24);

    int panelX = 252;
    int panelY = 146;
    int panelRowGap = 30;

    pianoKeyboard_.setBounds(panelX, panelY, 320, 46);
    panelY += 52;

    quantizeSlider_.setBounds(panelX + 65, panelY, 280, 22);
    panelY += panelRowGap;

    preserveSlider_.setBounds(panelX + 65, panelY, 280, 22);
    panelY += panelRowGap;

    transientsSlider_.setBounds(panelX + 68, panelY, 140, 22);
    sensitivitySlider_.setBounds(panelX + 282, panelY, 100, 22);

    // Strip sections
    int stripY = 296;
    int stripPad = 24;

    // Smear & Enhance
    phaseVocoderToggle_.setBounds(margin, stripY + stripPad, 100, 24);
    smearSlider_.setBounds(margin + 160, stripY + stripPad, width() - margin * 2 - 170, 22);
    stripY += 60;

    // Freq Modulation
    int lfoY = stripY + stripPad;
    lfoDepthSlider_.setBounds(margin + 50, lfoY, width() - margin * 2 - 190, 22);
    lfoDepthModeCombo_.setBounds(width() - margin - 130, lfoY, 75, 24);
    lfoY += 30;
    lfoRateSlider_.setBounds(margin + 50, lfoY, width() - margin * 2 - 280, 22);
    lfoSyncToggle_.setBounds(width() - margin - 220, lfoY, 75, 24);
    lfoDivisionCombo_.setBounds(width() - margin - 140, lfoY, 58, 24);
    lfoShapeCombo_.setBounds(width() - margin - 78, lfoY, 78, 24);
    stripY += 82;

    // Delay
    int delY = stripY + stripPad;
    delayEnabledToggle_.setBounds(margin, delY, 75, 24);
    delayTimeSlider_.setBounds(margin + 130, delY, width() - margin * 2 - 290, 22);
    delaySyncToggle_.setBounds(width() - margin - 150, delY, 75, 24);
    delayDivisionCombo_.setBounds(width() - margin - 68, delY, 68, 24);
    delY += 30;
    int halfW = (width() - margin * 2 - 130) / 2;
    delayFeedbackSlider_.setBounds(margin + 62, delY, halfW, 22);
    int dampX = margin + 62 + halfW + 8;
    delayDampingSlider_.setBounds(dampX + 60, delY, width() - margin - dampX - 60, 22);
    delY += 30;
    delaySlopeSlider_.setBounds(margin + 50, delY, halfW, 22);
    int diffX = margin + 50 + halfW + 8;
    delayDiffuseSlider_.setBounds(diffX + 55, delY, width() - margin - diffX - 55, 22);
    delY += 30;
    stereoDecorrelateToggle_.setBounds(width() - margin - 110, delY, 110, 22);
    stripY += 148;

    // Delay Modulation
    int dlyLfoY = stripY + stripPad;
    dlyLfoDepthSlider_.setBounds(margin + 50, dlyLfoY, width() - margin * 2 - 60, 22);
    dlyLfoY += 30;
    dlyLfoRateSlider_.setBounds(margin + 50, dlyLfoY, width() - margin * 2 - 280, 22);
    dlyLfoSyncToggle_.setBounds(width() - margin - 220, dlyLfoY, 75, 24);
    dlyLfoDivisionCombo_.setBounds(width() - margin - 140, dlyLfoY, 58, 24);
    dlyLfoShapeCombo_.setBounds(width() - margin - 78, dlyLfoY, 78, 24);
    stripY += 82;

    // Mask
    int maskY = stripY + stripPad;
    maskEnabledToggle_.setBounds(margin, maskY, 75, 24);
    maskModeCombo_.setBounds(margin + 85, maskY, 95, 24);
    maskTransitionSlider_.setBounds(margin + 262, maskY, width() - margin * 2 - 270, 22);
    maskY += 30;
    int maskHalfW = (width() - margin * 2 - 75) / 2;
    maskLowFreqSlider_.setBounds(margin + 35, maskY, maskHalfW, 22);
    int highX = margin + 35 + maskHalfW + 10;
    maskHighFreqSlider_.setBounds(highX + 38, maskY, width() - margin - highX - 38, 22);
    stripY += 90;

    // Mix
    int mixY = stripY + 14;
    dryWetSlider_.setBounds(margin + 70, mixY, width() - margin * 2 - 80, 22);
}
