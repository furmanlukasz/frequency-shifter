#pragma once

#include "VisageControl.h"
#include "../HolyTheme.h"
#include <functional>
#include <vector>
#include <string>

/**
 * Segmented control (pill-style tab selector) for mode switching.
 * Displays N segments in a rounded pill; the selected segment is highlighted
 * with the accent color. Connects to a JUCE Choice parameter.
 */
class HolySegmentedControl : public VisageControl
{
public:
    HolySegmentedControl();

    void addSegment(const std::string& label);

    void draw(visage::Canvas& canvas) override;
    void mouseDown(const visage::MouseEvent& e) override;

    int getSelectedIndex() const;

    // Called after selection changes
    std::function<void(int)> onChange;

private:
    std::vector<std::string> segments_;

    VISAGE_LEAK_CHECKER(HolySegmentedControl)
};
