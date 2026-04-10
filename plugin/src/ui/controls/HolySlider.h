#pragma once

#include "../VisageParamAttachment.h"
#include "../HolyTheme.h"
#include <visage_ui/frame.h>
#include <memory>

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

private:
    std::string formatValue() const;
    float getSliderWidth() const;

    std::unique_ptr<VisageParamAttachment> attachment_;
    std::string suffix_;
    int decimals_ = 1;
    bool dimmed_ = false;
    bool dragging_ = false;
    float dragCurrentNorm_ = 0.0f;
    float dragStartNorm_ = 0.0f;
    float dragStartX_ = 0.0f;

    static constexpr float kTextWidth = 58.0f;

    VISAGE_LEAK_CHECKER(HolySlider)
};
