#pragma once

#include "VisageControl.h"
#include "../HolyTheme.h"
#include <functional>
#include <string>

class HolyToggle : public VisageControl
{
public:
    explicit HolyToggle(const std::string& label = "");

    void draw(visage::Canvas& canvas) override;
    void mouseDown(const visage::MouseEvent& e) override;

    void setLabel(const std::string& label) { label_ = label; redraw(); }
    void setLabelColor(unsigned int color) { labelColor_ = color; }
    bool isOn() const;

    // Callback for non-param toggles (e.g. stereo decorrelate)
    std::function<void(bool)> onToggle;

private:
    std::string label_;
    unsigned int labelColor_ = 0; // 0 = use default
    bool manualState_ = false;

    VISAGE_LEAK_CHECKER(HolyToggle)
};
