#pragma once

#include "VisageControl.h"
#include "../HolyTheme.h"
#include <vector>
#include <string>

class HolySlider : public VisageControl
{
public:
    HolySlider();

    void draw(visage::Canvas& canvas) override;
    void mouseDown(const visage::MouseEvent& e) override;
    void mouseDrag(const visage::MouseEvent& e) override;
    void mouseUp(const visage::MouseEvent& e) override;

    void setSuffix(const std::string& suffix) { suffix_ = suffix; }
    void setDecimals(int d) { decimals_ = d; }

    // Adaptive precision for log-scaled readouts (freq/delay modulation depth & rate):
    // show extra decimals as the value shrinks below 0.1 (→2dp) and 0.01 (→3dp) so the
    // small end of the range stays readable instead of collapsing to "0.0".
    void setAdaptiveDecimals(bool enabled) { adaptiveDecimals_ = enabled; }

    // Reverse the synced (division) mapping so the slow divisions (e.g. 16/1) sit on the
    // left and the fast ones (1/32) on the right — matching the free-mode rate direction
    // (low rate left, high rate right). Used by the modulation rate sliders.
    void setSyncReversed(bool reversed) { syncReversed_ = reversed; }

    // For ms-valued sliders whose readout can reach 4 digits (e.g. delay time): once the
    // value reaches thresholdMs, render it in seconds instead so it fits the narrow text
    // field (e.g. "1.27 s" rather than a clipped "1274.3 ms"). 0 disables.
    void setSecondsAbove(float thresholdMs) { secondsAboveMs_ = thresholdMs; }

    // --- Dual-purpose sync slider support ---
    // Sets a secondary attachment used when sync mode is active.
    // The secondary param should be a Choice parameter (e.g. delayDivision).
    void setSyncAttachment(juce::AudioProcessorValueTreeState& apvts,
                           const juce::String& paramId);

    // Provide the list of division labels for sync mode display.
    void setSyncLabels(const std::vector<std::string>& labels);

    // Toggle between free (primary attachment) and synced (secondary attachment).
    // When synced, the slider snaps to discrete positions and shows division labels.
    void setSynced(bool synced);
    bool isSynced() const { return synced_; }

private:
    std::string formatValue() const;
    float getSliderWidth() const;

    // Param-normalised value <-> on-screen fill position (0=left,1=right). Identity unless
    // a reversed sync mapping is active, in which case left/right are flipped.
    float paramNormToVisual(float paramNorm) const;
    float visualToParamNorm(float visual) const;

    // Returns the active attachment (primary or sync)
    VisageParamAttachment* activeAttachment() const;

    std::unique_ptr<VisageParamAttachment> syncAttachment_;
    std::vector<std::string> syncLabels_;
    std::string suffix_;
    int decimals_ = 1;
    bool adaptiveDecimals_ = false;
    bool syncReversed_ = false;
    float secondsAboveMs_ = 0.0f;
    bool synced_ = false;
    bool dragging_ = false;
    bool fineMode_ = false;  // Shift held at mouseDown: delta-drag at 10x reduced sensitivity.
    float dragCurrentNorm_ = 0.0f;
    float dragStartNorm_ = 0.0f;
    float dragStartX_ = 0.0f;

    static constexpr float kTextWidth = 58.0f;
    static constexpr float kFineSensitivityScale = 10.0f;

    VISAGE_LEAK_CHECKER(HolySlider)
};
