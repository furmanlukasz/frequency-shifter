#pragma once

#include "../VisageParamAttachment.h"
#include "../HolyTheme.h"
#include <visage_ui/frame.h>
#include <memory>

class HolyToggle : public visage::Frame
{
public:
    explicit HolyToggle(const std::string& label = "");

    void setAttachment(juce::AudioProcessorValueTreeState& apvts,
                       const juce::String& paramId);

    void draw(visage::Canvas& canvas) override;
    void mouseDown(const visage::MouseEvent& e) override;

    void setLabel(const std::string& label) { label_ = label; redraw(); }
    bool isOn() const;

    // Callback for non-param toggles (e.g. stereo decorrelate)
    std::function<void(bool)> onToggle;

private:
    std::unique_ptr<VisageParamAttachment> attachment_;
    std::string label_;
    bool manualState_ = false;

    VISAGE_LEAK_CHECKER(HolyToggle)
};
