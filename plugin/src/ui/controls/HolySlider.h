#pragma once

#include "../VisageParamAttachment.h"
#include "../HolyTheme.h"
#include <visage_ui/frame.h>
#include <memory>
#include <vector>
#include <string>

class HolySlider : public visage::Frame
{
public:
    HolySlider();

    void setAttachment(juce::AudioProcessorValueTreeState& apvts,
                       const juce::String& paramId);

    void draw(visage::Canvas& canvas) override;
    void mouseDown(const visage::MouseEvent& e) override;
    void mouseDrag(const visage::MouseEvent& e) override;
    void mouseUp(const visage::MouseEvent& e) override;

    void setSuffix(const std::string& suffix) { suffix_ = suffix; }
    void setDecimals(int d) { decimals_ = d; }
    void setDimmed(bool d) { if (dimmed_ != d) { dimmed_ = d; redraw(); } }
    bool isDimmed() const { return dimmed_; }

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

    // Returns the active attachment (primary or sync)
    VisageParamAttachment* activeAttachment() const;

    std::unique_ptr<VisageParamAttachment> attachment_;
    std::unique_ptr<VisageParamAttachment> syncAttachment_;
    std::vector<std::string> syncLabels_;
    std::string suffix_;
    int decimals_ = 1;
    bool dimmed_ = false;
    bool synced_ = false;
    bool dragging_ = false;
    float dragCurrentNorm_ = 0.0f;
    float dragStartNorm_ = 0.0f;
    float dragStartX_ = 0.0f;

    static constexpr float kTextWidth = 58.0f;

    VISAGE_LEAK_CHECKER(HolySlider)
};
