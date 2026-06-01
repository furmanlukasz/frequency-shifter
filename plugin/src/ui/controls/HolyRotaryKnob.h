#pragma once

#include "VisageControl.h"
#include "../HolyTheme.h"
#include <functional>
#include <string>

class HolyRotaryKnob : public VisageControl
{
public:
    HolyRotaryKnob();

    void draw(visage::Canvas& canvas) override;
    void mouseDown(const visage::MouseEvent& e) override;
    void mouseDrag(const visage::MouseEvent& e) override;
    void mouseUp(const visage::MouseEvent& e) override;

    void setBipolar(bool bipolar) { bipolar_ = bipolar; }
    void setUnit(const std::string& unit) { unit_ = unit; }

    // Custom display range with log mapping (e.g. ±5000 Hz log scale)
    // knobNorm (0-1) <-> display value <-> param norm (0-1)
    using DisplayMapper = std::function<float(float knobNorm)>;   // knobNorm -> displayValue
    using ParamMapper = std::function<float(float knobNorm)>;     // knobNorm -> paramNorm
    using InverseMapper = std::function<float(float paramNorm)>;  // paramNorm -> knobNorm

    void setCustomMapping(DisplayMapper displayFn, ParamMapper toParamFn, InverseMapper fromParamFn)
    {
        displayMapper_ = std::move(displayFn);
        toParamMapper_ = std::move(toParamFn);
        fromParamMapper_ = std::move(fromParamFn);
    }

private:
    float normValueToAngle(float norm) const;
    std::string formatValue(float realValue) const;
    void updateFromMousePosition(float mx, float my);

    bool bipolar_ = true;
    std::string unit_ = "HZ";
    DisplayMapper displayMapper_;
    ParamMapper toParamMapper_;
    InverseMapper fromParamMapper_;
    float dragCurrentNorm_ = 0.0f;
    float dragKnobNorm_ = 0.0f;
    bool dragging_ = false;

    // Shift-for-fine: when shift is held at mouseDown, switch to vertical-delta
    // drag at reduced sensitivity for the whole gesture (matching DAW knob convention).
    bool fineMode_ = false;
    float fineStartY_ = 0.0f;
    float fineStartKnobNorm_ = 0.0f;

    static constexpr float kStartAngle = -2.35619f;  // -3*pi/4
    static constexpr float kEndAngle = 2.35619f;     // 3*pi/4
    static constexpr float kSensitivity = 300.0f;     // pixels for full range
    static constexpr float kFineSensitivityScale = 10.0f;  // shift-held drag is 10x finer

    VISAGE_LEAK_CHECKER(HolyRotaryKnob)
};
