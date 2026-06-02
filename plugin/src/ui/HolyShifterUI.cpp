#include "HolyShifterUI.h"
#include "../PluginProcessor.h"
#include "../dsp/Scales.h"
#include "embedded/holy_images.h"   // embedded Heathen Machines logo + pagan backdrop (holy::images::*_png)

// ─────────────────────────────────────────────────────────────────────────────
// Pagan "grimoire" background texture
// ─────────────────────────────────────────────────────────────────────────────
// The embedded artwork (assets/pagan-background.png) is painted behind everything,
// then knocked back by a translucent scrim so the runes read as a subtle backdrop
// while the gold controls stay legible. The section panels — opaque near-black fills
// in the flat design — become translucent here so the texture bleeds through them.
// These alphas are the single tuning surface for the effect's intensity.
namespace paganbg {
    constexpr unsigned int kScrim      = 0xA40A0A0Cu; // dark scrim over the artwork (~64%)
    constexpr unsigned int kStripAlpha = 0x4Du;       // Freq/Delay/DelayMod section strips
    constexpr unsigned int kStripDim   = 0x26u;       // …same strips when dimmed (Classic mode)
    constexpr unsigned int kModeAlpha  = 0x73u;       // CLASSIC/SPECTRAL selector bar
    constexpr unsigned int kPanelAlpha = 0xCCu;       // Spectral Controls panel (stays a raised box)
    constexpr unsigned int kMaskAlpha  = 0x80u;       // Mask section
    constexpr unsigned int kMixAlpha   = 0x99u;       // Dry/Wet footer

    // Replace just the alpha byte of an 0xAARRGGBB color.
    constexpr unsigned int withAlpha(unsigned int argb, unsigned int a) {
        return (a << 24) | (argb & 0x00FFFFFFu);
    }
}

// Unified division labels for all sync sliders (matches both arrays in PluginProcessor.h)
static const std::vector<std::string> kDivisionLabels = {
    "1/32", "1/16", "1/16 D", "1/8", "1/8 D", "1/4", "1/4 D",
    "1/2", "1/1", "2/1", "3/1", "4/1", "8/1", "16/1"
};

HolyShifterUI::HolyShifterUI(FrequencyShifterProcessor& processor)
    : processor_(processor),
      apvts_(processor.getValueTreeState()),
      warmToggle_("WARM"),
      presetPrevBtn_("<"),
      presetNextBtn_(">"),
      lfoSyncToggle_("Sync"),
      lfoEnabledToggle_(""),
      delayEnabledToggle_(""),       // label is the strip header
      delaySyncToggle_("Sync"),
      dlyLfoSyncToggle_("Sync"),
      dlyLfoEnabledToggle_(""),
      maskEnabledToggle_("")         // label is the strip header
{
    // === Shift Knob ===
    shiftKnob_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_SHIFT_HZ);
    shiftKnob_.setBipolar(true);
    shiftKnob_.setUnit("HZ");

    constexpr double logScale = 10.0;
    constexpr double maxShift = 5000.0;
    double logMax = std::log(1.0 + maxShift / logScale);

    shiftKnob_.setCustomMapping(
        [logMax](float knobNorm) -> float {
            double symNorm = knobNorm * 2.0 - 1.0;
            double sign = symNorm >= 0.0 ? 1.0 : -1.0;
            double absNorm = std::abs(symNorm);
            double absVal = logScale * (std::exp(absNorm * logMax) - 1.0);
            return static_cast<float>(sign * absVal);
        },
        [logMax](float knobNorm) -> float {
            double symNorm = knobNorm * 2.0 - 1.0;
            double sign = symNorm >= 0.0 ? 1.0 : -1.0;
            double absNorm = std::abs(symNorm);
            double absVal = logScale * (std::exp(absNorm * logMax) - 1.0);
            double paramValue = sign * absVal;
            return static_cast<float>((paramValue + 20000.0) / 40000.0);
        },
        [logMax](float paramNorm) -> float {
            double paramValue = paramNorm * 40000.0 - 20000.0;
            double sign = paramValue >= 0.0 ? 1.0 : -1.0;
            double absVal = std::abs(paramValue);
            absVal = std::min(absVal, 5000.0);
            double normalized = sign * std::log(1.0 + absVal / logScale) / logMax;
            return static_cast<float>((normalized + 1.0) * 0.5);
        }
    );
    addChild(&shiftKnob_);

    // === Title bar ===
    warmToggle_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_WARM);
    addChild(&warmToggle_);

    // === Mode selector ===
    modeSelector_.addSegment("CLASSIC");
    modeSelector_.addSegment("SPECTRAL");
    modeSelector_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_PROCESSING_MODE);
    modeSelector_.onChange = [this](int) { updateControlsForMode(); };
    addChild(&modeSelector_);

    // === Preset strip ===
    presetPrevBtn_.setText("<");
    presetPrevBtn_.onToggle() = [this](visage::Button*, bool) {
        processor_.getPresetManager().loadPreviousPreset();
        currentPresetName_ = processor_.getPresetManager().getCurrentPresetName().toStdString();
        updatePresetStrip();
        redraw();
    };
    addChild(&presetPrevBtn_);

    presetNextBtn_.setText(">");
    presetNextBtn_.onToggle() = [this](visage::Button*, bool) {
        processor_.getPresetManager().loadNextPreset();
        currentPresetName_ = processor_.getPresetManager().getCurrentPresetName().toStdString();
        updatePresetStrip();
        redraw();
    };
    addChild(&presetNextBtn_);

    currentPresetName_ = processor_.getPresetManager().getCurrentPresetName().toStdString();

    presetNameArea_.onMouseDown() = [this](const visage::MouseEvent&) {
        if (presetDropdown_.isOpen())
            presetDropdown_.hide();
        else
        {
            auto pos = presetNameArea_.positionInWindow();
            presetDropdown_.showFor(&processor_,
                static_cast<int>(pos.x),
                static_cast<int>(pos.y) + static_cast<int>(presetNameArea_.height()),
                static_cast<int>(presetNameArea_.width()));
        }
    };
    addChild(&presetNameArea_);

    presetDropdown_.onPresetChanged_ = [this]() {
        currentPresetName_ = processor_.getPresetManager().getCurrentPresetName().toStdString();
        updatePresetStrip();
        redraw();
    };

    // SAVE — name the current state and write it as a user preset.
    presetSaveBtn_.configure("SAVE", HolyTextButton::Style::Primary);
    presetSaveBtn_.onClick = [this]() {
        auto& pm = processor_.getPresetManager();
        std::string base = currentPresetName_.empty() ? std::string("Untitled") : currentPresetName_;
        if (pm.isFactoryPreset(juce::String(base)))
            base += " Custom";   // factory names are read-only; suggest a saveable variant
        presetModal_.openSave(&processor_, base);
    };
    addChild(&presetSaveBtn_);

    // DELETE — remove the current user preset (disabled for factory/empty).
    presetDeleteBtn_.configure("DELETE", HolyTextButton::Style::Outline);
    presetDeleteBtn_.onClick = [this]() {
        if (currentPresetName_.empty()
            || processor_.getPresetManager().isFactoryPreset(juce::String(currentPresetName_)))
            return;
        presetModal_.openConfirmDelete(&processor_, currentPresetName_);
    };
    addChild(&presetDeleteBtn_);

    presetModal_.onChanged = [this]() {
        currentPresetName_ = processor_.getPresetManager().getCurrentPresetName().toStdString();
        updatePresetStrip();
        redrawAll();
    };

    // === Spectral panel controls ===
    pianoKeyboard_.setAttachments(apvts_, FrequencyShifterProcessor::PARAM_SCALE_NOTE_PREFIX);
    addChild(&pianoKeyboard_);

    quantizeSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_QUANTIZE_STRENGTH);
    addChild(&quantizeSlider_);

    preserveSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_PRESERVE);
    addChild(&preserveSlider_);

    transientsSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_TRANSIENTS);
    addChild(&transientsSlider_);

    sensitivitySlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_SENSITIVITY);
    sensitivitySlider_.setDecimals(0);
    addChild(&sensitivitySlider_);

    smearSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_SMEAR);
    smearSlider_.setSuffix(" ms");
    addChild(&smearSlider_);

    // === Freq Modulation ===
    lfoDepthSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_LFO_DEPTH);
    addChild(&lfoDepthSlider_);

    // DepthMode combo — hidden, param still bound
    lfoDepthModeCombo_.addItem("Hz");
    lfoDepthModeCombo_.addItem("Degrees");
    lfoDepthModeCombo_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_LFO_DEPTH_MODE);
    addChild(&lfoDepthModeCombo_);

    // Shape combo — on the Depth row
    for (auto& name : { "Sine", "Triangle", "Saw", "Inv Saw", "Random" })
        lfoShapeCombo_.addItem(name);
    lfoShapeCombo_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_LFO_SHAPE);
    addChild(&lfoShapeCombo_);

    lfoRateSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_LFO_RATE);
    lfoRateSlider_.setSuffix(" Hz");
    lfoRateSlider_.setSyncAttachment(apvts_, FrequencyShifterProcessor::PARAM_LFO_DIVISION);
    lfoRateSlider_.setSyncLabels(kDivisionLabels);
    addChild(&lfoRateSlider_);

    lfoSyncToggle_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_LFO_SYNC);
    addChild(&lfoSyncToggle_);

    lfoEnabledToggle_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_LFO_ENABLED);
    addChild(&lfoEnabledToggle_);

    // === Delay (dual-purpose Time slider) ===
    delayEnabledToggle_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DELAY_ENABLED);
    addChild(&delayEnabledToggle_);

    delayTimeSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DELAY_TIME);
    delayTimeSlider_.setSuffix(" ms");
    delayTimeSlider_.setSyncAttachment(apvts_, FrequencyShifterProcessor::PARAM_DELAY_DIVISION);
    delayTimeSlider_.setSyncLabels(kDivisionLabels);
    addChild(&delayTimeSlider_);

    delaySyncToggle_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DELAY_SYNC);
    addChild(&delaySyncToggle_);

    delayFeedbackSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DELAY_FEEDBACK);
    addChild(&delayFeedbackSlider_);

    delayDampingSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DELAY_DAMPING);
    addChild(&delayDampingSlider_);

    delaySlopeSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DELAY_SLOPE);
    addChild(&delaySlopeSlider_);

    stereoDecorrelateToggle_.setLabel("L/R Decorr");
    stereoDecorrelateToggle_.setLabelColor(holy::colors::accent);  // Figma: #c9a96e
    stereoDecorrelateToggle_.onToggle = [this](bool on) {
        processor_.setStereoDecorrelate(on);
    };
    addChild(&stereoDecorrelateToggle_);

    // === Delay Modulation ===
    dlyLfoDepthSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DLY_LFO_DEPTH);
    dlyLfoDepthSlider_.setSuffix(" ms");
    addChild(&dlyLfoDepthSlider_);

    dlyLfoRateSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DLY_LFO_RATE);
    dlyLfoRateSlider_.setSuffix(" Hz");
    dlyLfoRateSlider_.setSyncAttachment(apvts_, FrequencyShifterProcessor::PARAM_DLY_LFO_DIVISION);
    dlyLfoRateSlider_.setSyncLabels(kDivisionLabels);
    addChild(&dlyLfoRateSlider_);

    dlyLfoSyncToggle_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DLY_LFO_SYNC);
    addChild(&dlyLfoSyncToggle_);

    for (auto& name : { "Sine", "Triangle", "Saw", "Inv Saw", "Random" })
        dlyLfoShapeCombo_.addItem(name);
    dlyLfoShapeCombo_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DLY_LFO_SHAPE);
    addChild(&dlyLfoShapeCombo_);

    dlyLfoEnabledToggle_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DLY_LFO_ENABLED);
    addChild(&dlyLfoEnabledToggle_);

    // === Mask ===
    maskEnabledToggle_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_MASK_ENABLED);
    addChild(&maskEnabledToggle_);

    for (auto& name : { "Low Pass", "High Pass", "Band Pass" })
        maskModeCombo_.addItem(name);
    maskModeCombo_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_MASK_MODE);
    // Figma (Mask dropdown only): muted tan #a59777 @ 9px. Waveform combos keep defaults.
    maskModeCombo_.setTextColor(0xFFA59777u);
    maskModeCombo_.setTextSize(9.0f);
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

    // === Mix ===
    dryWetSlider_.setAttachment(apvts_, FrequencyShifterProcessor::PARAM_DRY_WET);
    addChild(&dryWetSlider_);

    // Shared dropdown overlay — must be added LAST for z-ordering
    addChild(&dropdownOverlay_);
    addChild(&presetDropdown_);
    addChild(&presetModal_);   // above the dropdowns
    HolyComboBox::setSharedDropdown(&dropdownOverlay_);

    updateControlsForMode();
    updateDelaySyncUI();
    updateLfoSyncUI();
    updateDlyLfoSyncUI();
    updateLfoEnableUI();
    updatePresetStrip();
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
        dropH = static_cast<int>((dropH < parent()->height() - y) ? dropH : parent()->height() - y);

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
        else if (presetNames_[static_cast<size_t>(i)] == currentName)
        {
            canvas.setColor(0xFF1A1A1Du);
            canvas.fill(2, itemY, static_cast<int>(w) - 4, kItemHeight);
        }

        canvas.setColor(holy::colors::text);
        canvas.text(presetNames_[static_cast<size_t>(i)].c_str(), font, visage::Font::kLeft,
                    8, itemY, static_cast<int>(w) - 16, kItemHeight);
    }
}

void HolyPresetDropdown::mouseDown(const visage::MouseEvent& e)
{
    if (!processor_) { hide(); return; }
    int itemIdx = static_cast<int>((e.position.y - 2) / kItemHeight);
    if (itemIdx >= 0 && itemIdx < static_cast<int>(presetNames_.size()))
    {
        processor_->getPresetManager().loadPreset(presetNames_[static_cast<size_t>(itemIdx)]);
        if (onPresetChanged_) onPresetChanged_();
    }
    hide();
}

void HolyPresetDropdown::mouseMove(const visage::MouseEvent& e)
{
    int newHover = static_cast<int>((e.position.y - 2) / kItemHeight);
    if (newHover < 0 || newHover >= static_cast<int>(presetNames_.size()))
        newHover = -1;
    if (newHover != hoveredIndex_) { hoveredIndex_ = newHover; redraw(); }
}

// === Sync UI ===

void HolyShifterUI::updateDelaySyncUI()
{
    bool synced = delaySyncToggle_.isOn();
    delayTimeSlider_.setSynced(synced);
}

void HolyShifterUI::updateLfoSyncUI()
{
    bool synced = lfoSyncToggle_.isOn();
    lfoRateSlider_.setSynced(synced);
}

void HolyShifterUI::updateDlyLfoSyncUI()
{
    bool synced = dlyLfoSyncToggle_.isOn();
    dlyLfoRateSlider_.setSynced(synced);
}

void HolyShifterUI::updateLfoEnableUI()
{
    // R3: dim each LFO's own controls when its enable toggle is off.
    bool freqOn = lfoEnabledToggle_.isOn();
    lfoDepthSlider_.setDimmed(!freqOn);
    lfoRateSlider_.setDimmed(!freqOn);
    lfoShapeCombo_.setDimmed(!freqOn);
    lfoSyncToggle_.setDimmed(!freqOn);

    bool dlyOn = dlyLfoEnabledToggle_.isOn();
    dlyLfoDepthSlider_.setDimmed(!dlyOn);
    dlyLfoRateSlider_.setDimmed(!dlyOn);
    dlyLfoShapeCombo_.setDimmed(!dlyOn);
    dlyLfoSyncToggle_.setDimmed(!dlyOn);
}

void HolyShifterUI::updatePresetStrip()
{
    // Factory presets are read-only, so DELETE only applies to user presets.
    bool deletable = !currentPresetName_.empty()
                     && !processor_.getPresetManager().isFactoryPreset(juce::String(currentPresetName_));
    presetDeleteBtn_.setEnabledState(deletable);
}

void HolyShifterUI::pollState()
{
    currentPresetName_ = processor_.getPresetManager().getCurrentPresetName().toStdString();
    updatePresetStrip();
    updateControlsForMode();
    updateDelaySyncUI();
    updateLfoSyncUI();
    updateDlyLfoSyncUI();
    updateLfoEnableUI();
    redrawAll();
}

void HolyShifterUI::updateControlsForMode()
{
    bool isClassic = (modeSelector_.getSelectedIndex() == 0);

    pianoKeyboard_.setDimmed(isClassic);
    quantizeSlider_.setDimmed(isClassic);
    preserveSlider_.setDimmed(isClassic);
    transientsSlider_.setDimmed(isClassic);
    sensitivitySlider_.setDimmed(isClassic);
    smearSlider_.setDimmed(isClassic);
    lfoDepthModeCombo_.setDimmed(isClassic);
    maskEnabledToggle_.setDimmed(isClassic);
    maskModeCombo_.setDimmed(isClassic);
    maskTransitionSlider_.setDimmed(isClassic);
    maskLowFreqSlider_.setDimmed(isClassic);
    maskHighFreqSlider_.setDimmed(isClassic);

    // R4: Slope only affects Spectral mode — grey it in Classic.
    delaySlopeSlider_.setDimmed(isClassic);
}

// === Strip drawing ===

void HolyShifterUI::drawStrip(visage::Canvas& canvas, int y, int h,
                                const std::string& label, bool dimmed)
{
    unsigned int stripCol = dimmed ? paganbg::withAlpha(holy::colors::strip, paganbg::kStripDim)
                                   : paganbg::withAlpha(holy::colors::strip, paganbg::kStripAlpha);
    canvas.setColor(stripCol);
    canvas.fill(0, y, width(), h);

    // Top border line (1px, Figma: #1a1a1d)
    canvas.setColor(holy::colors::stripBorder);
    canvas.segment(0.0f, static_cast<float>(y), static_cast<float>(width()),
                   static_cast<float>(y), 1.0f, false);

    // Section label (Figma: 8px Inter Medium, tracking 1.6px, gold, at x=14, y=y+6)
    auto labelFont = holy::makeFont(8.0f, holy::FontWeight::Medium);
    canvas.setColor(dimmed ? holy::colors::textMuted : holy::colors::accent);
    canvas.text(label.c_str(), labelFont, visage::Font::kLeft, 14, y + 6, 180, 10);
}

// === Gradient accent line helper ===
static void drawGradientAccentLine(visage::Canvas& canvas, int y, int w, float maxAlpha)
{
    // Approximate Figma gradient: transparent → rgba(107,93,61,0.6) → #C9A96E → rgba(107,93,61,0.6) → transparent
    // Using 20 segments for smooth approximation
    constexpr int segs = 20;
    float segW = static_cast<float>(w) / segs;
    for (int s = 0; s < segs; ++s)
    {
        float t = (static_cast<float>(s) + 0.5f) / static_cast<float>(segs);
        float alpha;
        if (t < 0.3f)
            alpha = (t / 0.3f) * 0.6f;
        else if (t < 0.5f)
            alpha = 0.6f + ((t - 0.3f) / 0.2f) * 0.4f;
        else if (t < 0.7f)
            alpha = 1.0f - ((t - 0.5f) / 0.2f) * 0.4f;
        else
            alpha = 0.6f * (1.0f - (t - 0.7f) / 0.3f);
        alpha *= maxAlpha;
        unsigned int a = static_cast<unsigned int>(alpha * 255.0f);
        canvas.setColor((a << 24) | 0x00C9A96Eu);
        int sx = static_cast<int>(s * segW);
        int ex = static_cast<int>((s + 1) * segW);
        canvas.fill(sx, y, ex - sx, 1);
    }
}

// === Draw — all positions from Figma metadata (pixel-perfect) ===

void HolyShifterUI::draw(visage::Canvas& canvas)
{
    int w = static_cast<int>(width());
    int h = static_cast<int>(height());

    // Background — pagan artwork backdrop (replaces the flat #0a0a0c fill of the original design).
    // Base fill first (shows through transparent edges / if the image ever fails to decode), then
    // the artwork stretched to the fixed 700×928 window, then a translucent scrim to knock it back.
    canvas.setColor(holy::colors::background);
    canvas.fill(0, 0, w, h);
    canvas.setColor(0xFFFFFFFFu);   // white brush = untinted artwork at full alpha
    canvas.image(holy::images::pagan_background_png, 0, 0, w, h);
    canvas.setColor(paganbg::kScrim);
    canvas.fill(0, 0, w, h);

    // Top accent gradient line (Figma: y=-1, 1px, gradient gold)
    drawGradientAccentLine(canvas, 0, w, 1.0f);

    // Title (Figma: Inter Thin 26px, #e8e4db, tracking 9.1px, at x=27, y=13)
    auto titleFont = holy::makeFont(26.0f, holy::FontWeight::Thin);
    canvas.setColor(holy::colors::text);
    canvas.text("H O L Y   S H I F T E R", titleFont, visage::Font::kLeft, 27, 13, 440, 31);

    // Subtitle (Figma: Inter Regular 10px, #3e3a34, at x=29, y=45)
    auto subtitleFont = holy::makeFont(10.0f);
    canvas.setColor(holy::colors::textMuted);
    canvas.text("Frequency Shifter with Harmonic Quantisation", subtitleFont,
                visage::Font::kLeft, 29, 45, 400, 12);

    // Heathen Machines logo — Figma node 174:40 (top-right, 608,18, 53x53). Transparent PNG,
    // drawn aspect-preserved at 51x53. White brush = untinted, full alpha.
    canvas.setColor(0xFFFFFFFFu);
    canvas.image(holy::images::heathen_machines_logo_png, 609, 18, 51, 53);

    // Preset separator line (Figma: y=94, 1px, #1a1a1d — no bg fill for preset area)
    canvas.setColor(holy::colors::stripBorder);
    canvas.segment(0.0f, 94.0f, static_cast<float>(w), 94.0f, 1.0f, false);

    // Preset name (Figma: Inter Regular 12px, #e8e4db, at x=91, y=69)
    auto presetFont = holy::makeFont(12.0f);
    canvas.setColor(holy::colors::text);
    canvas.text(currentPresetName_.c_str(), presetFont, visage::Font::kLeft, 91, 69, 300, 20);

    // Mode selector strip (Figma: bg #0c0c0e, y=101, h=36, top line #1a1a1d)
    canvas.setColor(paganbg::withAlpha(holy::colors::modeSelectorBg, paganbg::kModeAlpha));
    canvas.fill(0, 101, w, 36);
    canvas.setColor(holy::colors::stripBorder);
    canvas.segment(0.0f, 101.0f, static_cast<float>(w), 101.0f, 1.0f, false);

    bool isSpectral = (modeSelector_.getSelectedIndex() == 1);
    bool isClassic = !isSpectral;

    // Spectral panel (Figma: x=245, y=158, w=430, h=240; vertical gradient #19191d→#101013, border #1c1c20)
    {
        unsigned int gTop = holy::dimColor(paganbg::withAlpha(holy::colors::panelGradTop, paganbg::kPanelAlpha), isClassic);
        unsigned int gBot = holy::dimColor(paganbg::withAlpha(holy::colors::panelGradBot, paganbg::kPanelAlpha), isClassic);
        canvas.setColor(visage::Brush::vertical(visage::Color(gTop), visage::Color(gBot)));
        canvas.roundedRectangle(245.0f, 158.0f, 430.0f, 240.0f, 6.0f);
    }
    // Border
    canvas.setColor(holy::dimColor(holy::colors::panelBorder, isClassic));
    canvas.roundedRectangleBorder(245.0f, 158.0f, 430.0f, 240.0f, 6.0f, 1.0f);
    // Top gold accent hairline inside panel (Figma: 2px, subtle gold rgba(201,169,110,0.12))
    canvas.setColor(holy::dimColor(0x1FC9A96Eu, isClassic));
    canvas.fill(246, 158, 428, 2);

    // "SPECTRAL CONTROLS" header (Figma: 8px, tracking 1.6px, gold, at panel-rel 13,9 → abs 258,167)
    auto panelHeaderFont = holy::makeFont(8.0f, holy::FontWeight::Medium);
    canvas.setColor(holy::dimColor(holy::colors::accent, isClassic));
    canvas.text("SPECTRAL CONTROLS", panelHeaderFont, visage::Font::kLeft, 258, 167, 150, 10);

    // === Strip sections — exact Figma positions ===
    drawStrip(canvas, 405, 110, "FREQ MODULATION");

    // Freq Mod has drop shadow (Figma: 0px 4px 4px rgba(0,0,0,0.25)) — approximate
    canvas.setColor(0x10000000u);
    canvas.fill(0, 515, w, 2);

    drawStrip(canvas, 517, 134, "DELAY");
    drawStrip(canvas, 651,  92, "DELAY MODULATION");

    // Mask strip (Figma: bg #19191d, different from other strips)
    canvas.setColor(holy::dimColor(paganbg::withAlpha(holy::colors::maskBg, paganbg::kMaskAlpha), isClassic));
    canvas.fill(0, 761, w, 90);
    canvas.setColor(holy::colors::stripBorder);
    canvas.segment(0.0f, 761.0f, static_cast<float>(w), 761.0f, 1.0f, false);
    {
        auto maskLabelFont = holy::makeFont(8.0f, holy::FontWeight::Medium);
        canvas.setColor(isClassic ? holy::colors::textMuted : holy::colors::accent);
        canvas.text("MASK", maskLabelFont, visage::Font::kLeft, 14, 767, 50, 10);
    }

    // Mix footer (Figma: vertical gradient #0f0f11→#0a0a0c, h=72; no section label in the design)
    canvas.setColor(visage::Brush::vertical(
        visage::Color(paganbg::withAlpha(holy::colors::mixGradTop, paganbg::kMixAlpha)),
        visage::Color(paganbg::withAlpha(holy::colors::background, paganbg::kMixAlpha))));
    canvas.fill(0, 851, w, 72);
    canvas.setColor(holy::colors::stripBorder);
    canvas.segment(0.0f, 851.0f, static_cast<float>(w), 851.0f, 1.0f, false);

    // Section divider bars — Figma nodes 32:15 (y=516) and 32:18 (y=760): SOLID 2px #48402F,
    // full width, no fade. These clearly separate Freq-Mod↔Delay and Delay-Mod↔Mask.
    // (Previously drawn as a center-fading gradient accent that didn't read as a divider.)
    canvas.setColor(0xFF48402Fu);
    canvas.fill(0, 516, w, 2);
    canvas.fill(0, 760, w, 2);

    // Bottom accent gradient line (Figma: y=926, 1px, 50% intensity)
    drawGradientAccentLine(canvas, 926, w, 0.5f);

    // === Labels — exact Figma positions ===
    auto labelFont = holy::makeFont(10.0f);

    // Spectral panel labels (panel at 243, 137; all dimmed in Classic mode)
    canvas.setColor(holy::dimColor(holy::colors::textSec, isClassic));
    canvas.text("Quantize",   labelFont, visage::Font::kRight, 254, 237, 62, 20);
    canvas.text("Envelope",   labelFont, visage::Font::kRight, 254, 267, 62, 20);
    canvas.text("Transients", labelFont, visage::Font::kRight, 254, 297, 62, 20);
    canvas.text("Sens",       labelFont, visage::Font::kRight, 443, 317, 30, 20);

    // "Smear" label (Figma: panel-rel x=30, y=200 → absolute 275, 382)
    canvas.setColor(holy::dimColor(holy::colors::textSec, isClassic));
    canvas.text("Smear", labelFont, visage::Font::kLeft, 270, 358, 40, 20);

    // Smear tooltip (Figma: panel-rel x=142, y=219 → absolute 387, 396; Inter Light 8px)
    auto tooltipFont = holy::makeFont(8.0f, holy::FontWeight::Light);
    canvas.setColor(holy::dimColor(holy::colors::textSec, isClassic));
    canvas.text("(Adjust with Care when track playing)", tooltipFont, visage::Font::kLeft,
                387, 377, 250, 10);

    // Freq Modulation labels (strip at y=405)
    canvas.setColor(holy::colors::textSec);
    canvas.text("Depth", labelFont, visage::Font::kRight, 27, 432, 42, 20);
    canvas.text("Rate",  labelFont, visage::Font::kRight, 27, 462, 42, 20);

    // Delay labels (strip at y=517)
    canvas.text("Time",     labelFont, visage::Font::kRight, 116, 543, 36, 20);
    canvas.text("Feedback", labelFont, visage::Font::kRight, 28,  577, 60, 20);
    canvas.text("Damping",  labelFont, visage::Font::kRight, 348, 577, 58, 20);
    canvas.text("Slope",    labelFont, visage::Font::kRight, 28,  611, 42, 20);

    // Delay Modulation labels (strip at y=651)
    canvas.text("Depth", labelFont, visage::Font::kRight, 28, 679, 42, 20);
    canvas.text("Rate",  labelFont, visage::Font::kRight, 28, 709, 42, 20);

    // Mask labels (dimmed in Classic)
    canvas.setColor(holy::dimColor(holy::colors::textSec, isClassic));
    canvas.text("Transition", labelFont, visage::Font::kRight, 218, 787, 60, 20);
    canvas.text("Low",        labelFont, visage::Font::kRight, 28,  819, 28, 20);
    canvas.text("High",       labelFont, visage::Font::kRight, 369, 819, 30, 20);

    // Mix label (Figma: Inter Medium 10px, tracking 1px, #8a857d)
    canvas.setColor(holy::colors::textSec);
    canvas.text("DRY / WET", labelFont, visage::Font::kRight, 28, 883, 72, 20);
}

// === Layout — exact Figma coordinates (pixel-perfect) ===

void HolyShifterUI::resized()
{
    // === Preset area (Figma: ◂ at x=46 centered, ▸ at x=70 centered, name at x=91, y=69) ===
    presetPrevBtn_.setBounds(36, 71, 20, 20);
    presetNextBtn_.setBounds(60, 71, 20, 20);
    presetNameArea_.setBounds(91, 69, 300, 20);

    // SAVE / DELETE sit to the right of the name, clear of the logo (x=609).
    presetSaveBtn_.setBounds(452, 70, 58, 22);
    presetDeleteBtn_.setBounds(518, 70, 72, 22);

    // Modal overlay covers the whole window (hidden until opened).
    presetModal_.setBounds(0, 0, kBaseW, kBaseH);

    // === Mode Selector row (Figma: frame y=101, h=36) ===
    modeSelector_.setBounds(28, 106, 220, 26);   // Figma: (28, 106, 220, 26)
    warmToggle_.setBounds(591, 107, 80, 24);      // Figma: (591, 107, 80, 24)

    // === Shift Knob (Figma: 27, 168, 210, 218) ===
    shiftKnob_.setBounds(27, 168, 210, 218);

    // === Spectral Panel (Figma: panel at 245, 158, 430, 240) ===
    int px = 245;
    int py = 158;

    pianoKeyboard_.setBounds(px + 13, py + 27, 400, 42);       // Figma: (13, 27, 400, 42)
    quantizeSlider_.setBounds(px + 77, py + 79, 334, 20);      // track at 77, value text extends to ~411
    preserveSlider_.setBounds(px + 77, py + 109, 334, 20);     // Envelope row
    transientsSlider_.setBounds(px + 77, py + 139, 334, 20);   // Transients row
    sensitivitySlider_.setBounds(px + 234, py + 159, 177, 20); // Sens track at 234, w=120
    smearSlider_.setBounds(px + 77, py + 200, 343, 20);        // Smear track at 77, w=286 (reflowed up after panel resize)

    // DepthMode combo — hidden
    lfoDepthModeCombo_.setBounds(0, 0, 0, 0);

    // === Freq Modulation (Figma: strip at y=405, h=110) ===
    int fmY = 405;

    lfoDepthSlider_.setBounds(75, fmY + 26, 430, 22);          // Depth: track at 75, w=385
    lfoRateSlider_.setBounds(75, fmY + 57, 356, 22);           // Rate: track at 75, w=290 (dual-purpose)
    lfoSyncToggle_.setBounds(451, fmY + 57, 75, 24);           // Sync toggle
    lfoShapeCombo_.setBounds(579, fmY + 58, 76, 22);           // Shape combo on Rate row (Figma: x=579)
    lfoEnabledToggle_.setBounds(119, fmY + 6, 30, 15);         // Figma x119 — after "FREQ MODULATION"

    // === Delay (Figma: strip at y=517, h=134) ===
    int dlY = 517;

    delayEnabledToggle_.setBounds(55, dlY + 6, 30, 15);        // Figma x55 — after "DELAY"
    delayTimeSlider_.setBounds(158, dlY + 26, 346, 22);        // Time: track at 158, w=280
    delaySyncToggle_.setBounds(523, dlY + 26, 80, 24);         // Sync toggle
    delayFeedbackSlider_.setBounds(94, dlY + 60, 238, 22);     // Feedback: track at 94, w=200
    delayDampingSlider_.setBounds(416, dlY + 60, 260, 22);     // Damping: track at 416, w=190
    delaySlopeSlider_.setBounds(76, dlY + 94, 256, 22);        // Slope: track at 76, w=218
    stereoDecorrelateToggle_.setBounds(582, dlY + 114, 110, 18); // L/R Decorr

    // === Delay Modulation (Figma: strip at y=651, h=92) ===
    int dmY = 651;

    dlyLfoDepthSlider_.setBounds(76, dmY + 28, 448, 22);       // Depth: track at 76, w=389 (Figma updated)
    dlyLfoRateSlider_.setBounds(76, dmY + 58, 356, 22);        // Rate: track at 76, w=290 (dual-purpose)
    dlyLfoSyncToggle_.setBounds(452, dmY + 58, 75, 24);        // Sync
    dlyLfoShapeCombo_.setBounds(584, dmY + 56, 76, 22);        // Shape (Figma: x=584)
    dlyLfoEnabledToggle_.setBounds(120, dmY + 6, 30, 15);      // Figma x120 — after "DELAY MODULATION"

    // === Mask (Figma: strip at y=761, h=90) ===
    int mkY = 761;

    maskEnabledToggle_.setBounds(56, mkY + 6, 30, 15);         // Figma x56 — tucked next to "MASK"
    maskModeCombo_.setBounds(88, mkY + 23, 90, 22);            // Band Pass combo
    maskTransitionSlider_.setBounds(284, mkY + 26, 191, 22);   // Transition slider
    maskLowFreqSlider_.setBounds(62, mkY + 58, 301, 22);       // Low freq slider
    maskHighFreqSlider_.setBounds(405, mkY + 58, 261, 22);     // High freq slider

    // === Mix Footer (Figma: strip at y=851, h=72) ===
    dryWetSlider_.setBounds(108, 851 + 28, 545, 22);           // DRY/WET: track at 108, w=494
}

// ═════════════════════════════════════════════════════════════════════════════
// HolyTextField — themed single-line text input for the preset name
// ═════════════════════════════════════════════════════════════════════════════

void HolyTextField::appendFiltered(const std::string& text)
{
    bool changed = false;
    for (char c : text)
    {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc < 0x20 || uc == 0x7F)        // drop control chars (newline, tab, …)
            continue;
        if (static_cast<int>(value_.size()) >= maxLen_)
            break;
        value_.push_back(c);
        changed = true;
    }
    if (changed)
        redraw();
}

bool HolyTextField::keyPress(const visage::KeyEvent& e)
{
    using K = visage::KeyCode;
    K k = e.keyCode();

    if (k == K::Backspace || k == K::Delete || k == K::KPBackspace)
    {
        if (!value_.empty())
        {
            value_.pop_back();   // also drop any trailing UTF-8 continuation bytes
            while (!value_.empty() && (static_cast<unsigned char>(value_.back()) & 0xC0) == 0x80)
                value_.pop_back();
            redraw();
        }
        return true;
    }
    if (k == K::Return || k == K::Return2 || k == K::KPEnter)
    {
        if (onSubmit) onSubmit();
        return true;
    }
    if (k == K::Escape)
    {
        if (onCancel) onCancel();
        return true;
    }
    // Cmd/Ctrl+V paste — while a modifier is held no textInput is delivered.
    if (e.isMainModifier() && (static_cast<int>(k) == 'v' || static_cast<int>(k) == 'V'))
    {
        appendFiltered(readClipboardText());
        return true;
    }
    return false;   // ordinary characters arrive via textInput()
}

void HolyTextField::draw(visage::Canvas& canvas)
{
    float w = static_cast<float>(width());
    float h = static_cast<float>(height());

    canvas.setColor(holy::colors::background);
    canvas.roundedRectangle(0, 0, w, h, 3.0f);
    canvas.setColor(focused_ ? holy::colors::accent : holy::colors::border);
    canvas.roundedRectangleBorder(0, 0, w, h, 3.0f, 1.0f);

    auto font = holy::makeFont(13.0f);
    bool empty = value_.empty();
    const std::string& shown = empty ? placeholder_ : value_;
    canvas.setColor(empty ? holy::colors::textMuted : holy::colors::text);
    canvas.text(shown.c_str(), font, visage::Font::kLeft,
                8, 0, static_cast<int>(w) - 16, static_cast<int>(h));

    // Caret just past the text (cosmetic — ASCII width is fine for preset names).
    if (focused_)
    {
        float cx = 9.0f;
        if (!empty)
        {
            std::u32string wide;
            wide.reserve(value_.size());
            for (char c : value_)
                wide.push_back(static_cast<char32_t>(static_cast<unsigned char>(c)));
            cx = 8.0f + font.stringWidth(wide) + 1.0f;
        }
        if (cx < w - 4.0f)
        {
            canvas.setColor(holy::colors::accent);
            canvas.fill(static_cast<int>(cx), static_cast<int>(h * 0.5f - 7.0f), 1, 14);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// HolyTextButton — themed push button (Primary = gold fill, Outline = bordered)
// ═════════════════════════════════════════════════════════════════════════════

void HolyTextButton::draw(visage::Canvas& canvas)
{
    float w = static_cast<float>(width());
    float h = static_cast<float>(height());
    bool dim = !enabled_;
    auto font = holy::makeFont(10.0f, holy::FontWeight::Medium);

    if (style_ == Style::Primary)
    {
        unsigned int bg = (hovered_ && enabled_) ? 0xFFD9BC82u : holy::colors::accent;
        canvas.setColor(holy::dimColor(bg, dim));
        canvas.roundedRectangle(0, 0, w, h, 4.0f);
        canvas.setColor(holy::dimColor(holy::colors::background, dim));   // dark text on gold
        canvas.text(text_.c_str(), font, visage::Font::kCenter,
                    0, 0, static_cast<int>(w), static_cast<int>(h));
    }
    else
    {
        canvas.setColor(holy::dimColor(holy::colors::raised, dim));
        canvas.roundedRectangle(0, 0, w, h, 4.0f);
        unsigned int border = (hovered_ && enabled_) ? holy::colors::accent : holy::colors::border;
        canvas.setColor(holy::dimColor(border, dim));
        canvas.roundedRectangleBorder(0, 0, w, h, 4.0f, 1.0f);
        unsigned int tc = (hovered_ && enabled_) ? holy::colors::accent : holy::colors::textSec;
        canvas.setColor(holy::dimColor(tc, dim));
        canvas.text(text_.c_str(), font, visage::Font::kCenter,
                    0, 0, static_cast<int>(w), static_cast<int>(h));
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// HolyModalDialog — centred Save / Confirm-Delete overlay
// ═════════════════════════════════════════════════════════════════════════════

HolyModalDialog::HolyModalDialog()
{
    setVisible(false);

    nameField_.setMaxLength(40);
    nameField_.onSubmit = [this]() { if (mode_ == Mode::Save) commitSave(); };
    nameField_.onCancel = [this]() { close(); };
    addChild(&nameField_);

    primaryBtn_.configure("Save", HolyTextButton::Style::Primary);
    primaryBtn_.onClick = [this]() { if (mode_ == Mode::Save) commitSave(); else commitDelete(); };
    addChild(&primaryBtn_);

    cancelBtn_.configure("Cancel", HolyTextButton::Style::Outline);
    cancelBtn_.onClick = [this]() { close(); };
    addChild(&cancelBtn_);
}

void HolyModalDialog::layoutPanel()
{
    int px = (static_cast<int>(width()) - kPanelW) / 2;
    int py = (static_cast<int>(height()) - kPanelH) / 2;

    nameField_.setBounds(px + 20, py + 60, kPanelW - 40, 30);

    const int bw = 74, bh = 28, gap = 8;
    primaryBtn_.setBounds(px + kPanelW - 20 - bw, py + kPanelH - 46, bw, bh);
    cancelBtn_.setBounds(px + kPanelW - 20 - bw - gap - bw, py + kPanelH - 46, bw, bh);
}

void HolyModalDialog::openSave(FrequencyShifterProcessor* proc, const std::string& suggestedName)
{
    mode_ = Mode::Save;
    processor_ = proc;
    suggested_ = suggestedName;
    status_.clear();

    nameField_.setText("");
    nameField_.setPlaceholder(suggestedName);
    nameField_.setVisible(true);
    primaryBtn_.setText("Save");

    if (auto* par = parent())
        setBounds(0, 0, par->width(), par->height());
    setVisible(true);
    layoutPanel();
    nameField_.requestKeyboardFocus();
    redraw();
}

void HolyModalDialog::openConfirmDelete(FrequencyShifterProcessor* proc, const std::string& presetName)
{
    mode_ = Mode::ConfirmDelete;
    processor_ = proc;
    targetName_ = presetName;
    status_.clear();

    nameField_.setVisible(false);
    primaryBtn_.setText("Delete");

    if (auto* par = parent())
        setBounds(0, 0, par->width(), par->height());
    setVisible(true);
    layoutPanel();
    redraw();
}

void HolyModalDialog::close()
{
    setVisible(false);
    processor_ = nullptr;
    if (parent())
        parent()->redraw();
}

void HolyModalDialog::commitSave()
{
    if (!processor_)
        return;

    juce::String name = juce::String(nameField_.getText()).trim();
    if (name.isEmpty())
        name = juce::String(suggested_).trim();

    juce::String legal = juce::File::createLegalFileName(name).trim();
    if (legal.isEmpty())
    {
        status_ = "Please enter a name";
        redraw();
        return;
    }
    if (processor_->getPresetManager().isFactoryPreset(legal))
    {
        status_ = "\"" + legal.toStdString() + "\" is a factory preset - choose another name";
        redraw();
        return;
    }

    processor_->getPresetManager().savePreset(legal);
    if (onChanged) onChanged();
    close();
}

void HolyModalDialog::commitDelete()
{
    if (processor_)
        processor_->getPresetManager().deletePreset(juce::String(targetName_));
    if (onChanged) onChanged();
    close();
}

void HolyModalDialog::mouseDown(const visage::MouseEvent& e)
{
    // The field and buttons handle their own clicks; this fires for the scrim or the
    // bare panel. Outside the panel = cancel; on the panel itself = swallow.
    int px = (static_cast<int>(width()) - kPanelW) / 2;
    int py = (static_cast<int>(height()) - kPanelH) / 2;
    bool insidePanel = e.position.x >= px && e.position.x <= px + kPanelW
                    && e.position.y >= py && e.position.y <= py + kPanelH;
    if (!insidePanel)
        close();
}

void HolyModalDialog::draw(visage::Canvas& canvas)
{
    int w = static_cast<int>(width());
    int h = static_cast<int>(height());

    // Scrim over the whole UI (children — panel contents — paint on top of this).
    canvas.setColor(0xC8070709u);
    canvas.fill(0, 0, w, h);

    int px = (w - kPanelW) / 2;
    int py = (h - kPanelH) / 2;
    float fpx = static_cast<float>(px), fpy = static_cast<float>(py);
    float fpw = static_cast<float>(kPanelW), fph = static_cast<float>(kPanelH);

    canvas.setColor(holy::colors::raised);
    canvas.roundedRectangle(fpx, fpy, fpw, fph, 6.0f);
    canvas.setColor(holy::colors::panelBorder);
    canvas.roundedRectangleBorder(fpx, fpy, fpw, fph, 6.0f, 1.0f);
    canvas.setColor(0x1FC9A96Eu);   // top gold hairline (matches the Spectral panel)
    canvas.fill(px + 1, py, kPanelW - 2, 2);

    auto titleFont = holy::makeFont(11.0f, holy::FontWeight::Medium);
    canvas.setColor(holy::colors::accent);
    canvas.text(mode_ == Mode::Save ? "SAVE PRESET" : "DELETE PRESET",
                titleFont, visage::Font::kLeft, px + 20, py + 18, kPanelW - 40, 14);

    auto bodyFont = holy::makeFont(11.0f);
    auto smallFont = holy::makeFont(9.0f);

    if (mode_ == Mode::Save)
    {
        canvas.setColor(holy::colors::textSec);
        canvas.text("Name", bodyFont, visage::Font::kLeft, px + 20, py + 44, 200, 12);
        if (!status_.empty())
        {
            canvas.setColor(0xFFCF8E6Cu);   // soft terracotta warning
            canvas.text(status_.c_str(), smallFont, visage::Font::kLeft,
                        px + 20, py + 94, kPanelW - 40, 12);
        }
    }
    else
    {
        canvas.setColor(holy::colors::text);
        std::string msg = "Delete \"" + targetName_ + "\"?";
        canvas.text(msg.c_str(), bodyFont, visage::Font::kLeft, px + 20, py + 56, kPanelW - 40, 14);
        canvas.setColor(holy::colors::textSec);
        canvas.text("This deletes the saved preset file.", smallFont, visage::Font::kLeft,
                    px + 20, py + 78, kPanelW - 40, 12);
    }
}
