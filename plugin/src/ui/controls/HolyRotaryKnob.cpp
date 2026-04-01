#include "HolyRotaryKnob.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

HolyRotaryKnob::HolyRotaryKnob() = default;

void HolyRotaryKnob::setAttachment(juce::AudioProcessorValueTreeState& apvts,
                                    const juce::String& paramId)
{
    attachment_ = std::make_unique<VisageParamAttachment>(apvts, paramId);
}

float HolyRotaryKnob::normValueToAngle(float norm) const
{
    return kStartAngle + norm * (kEndAngle - kStartAngle);
}

std::string HolyRotaryKnob::formatValue(float realValue) const
{
    char buf[32];
    if (std::abs(realValue) >= 100.0f)
        std::snprintf(buf, sizeof(buf), "%d", static_cast<int>(realValue));
    else
        std::snprintf(buf, sizeof(buf), "%.1f", realValue);
    return buf;
}

void HolyRotaryKnob::draw(visage::Canvas& canvas)
{
    float w = static_cast<float>(width());
    float h = static_cast<float>(height());
    float centreX = w * 0.5f;
    float centreY = h * 0.5f;
    float radius = std::min(w, h) * 0.36f;
    float arcWidth = 2.5f;

    float paramNorm = dragging_ ? dragCurrentNorm_
                                : (attachment_ ? attachment_->getNormalisedValue() : 0.5f);
    float knobNorm = fromParamMapper_ ? fromParamMapper_(paramNorm) : paramNorm;
    float angle = normValueToAngle(knobNorm);

    // Visage arc: 0 = right (3 o'clock), radians = half-sweep
    // Our angles: 0 = top (12 o'clock). Offset by -PI/2 to convert.
    constexpr float kArcOffset = -static_cast<float>(M_PI * 0.5);

    float halfSweep = (kEndAngle - kStartAngle) * 0.5f;
    float bgCenter = (kStartAngle + kEndAngle) * 0.5f + kArcOffset;

    // Background arc
    canvas.setColor(holy::colors::track);
    canvas.roundedArc(centreX - radius, centreY - radius,
                      radius * 2.0f, arcWidth, bgCenter, halfSweep);

    // Value arc: from zero (top) to dot position
    canvas.setColor(holy::colors::accent);
    if (bipolar_)
    {
        float midAngle = normValueToAngle(0.5f);  // = 0 (top)
        float valHalfSweep = std::abs(angle - midAngle) * 0.5f;
        float valCenter = (midAngle + angle) * 0.5f + kArcOffset;
        if (valHalfSweep > 0.005f)
            canvas.roundedArc(centreX - radius, centreY - radius,
                              radius * 2.0f, arcWidth, valCenter, valHalfSweep);
    }
    else
    {
        float valHalfSweep = (angle - kStartAngle) * 0.5f;
        float valCenter = (kStartAngle + angle) * 0.5f + kArcOffset;
        if (valHalfSweep > 0.005f)
            canvas.roundedArc(centreX - radius, centreY - radius,
                              radius * 2.0f, arcWidth, valCenter, valHalfSweep);
    }

    // 5 tick marks
    for (int i = 0; i < 5; ++i)
    {
        float tickNorm = i / 4.0f;
        float tickAngle = normValueToAngle(tickNorm);
        float innerR = radius + 7.0f;
        float outerR = radius + 12.0f;

        float x1 = centreX + innerR * std::sin(tickAngle);
        float y1 = centreY - innerR * std::cos(tickAngle);
        float x2 = centreX + outerR * std::sin(tickAngle);
        float y2 = centreY - outerR * std::cos(tickAngle);

        bool isCenterTick = (i == 2) && bipolar_;
        canvas.setColor(isCenterTick ? holy::colors::textSec : holy::colors::textMuted);
        canvas.segment(x1, y1, x2, y2, 0.8f, false);
    }

    // Indicator dot (bigger: 12px diameter)
    float dotSize = 12.0f;
    float dotX = centreX + radius * std::sin(angle);
    float dotY = centreY - radius * std::cos(angle);
    canvas.setColor(holy::colors::accent);
    canvas.circle(dotX - dotSize * 0.5f, dotY - dotSize * 0.5f, dotSize);

    // Value text
    float displayValue = displayMapper_ ? displayMapper_(knobNorm)
                                        : (attachment_ ? attachment_->getValue() : 0.0f);
    std::string valueText = formatValue(displayValue);
    auto valueFont = holy::makeFont(32.0f);
    canvas.setColor(holy::colors::text);
    canvas.text(valueText.c_str(), valueFont, visage::Font::kCenter,
                static_cast<int>(centreX - 50), static_cast<int>(centreY - 16), 100, 32);

    // Unit text
    auto unitFont = holy::makeFont(11.0f);
    canvas.setColor(holy::colors::textMuted);
    canvas.text(unit_.c_str(), unitFont, visage::Font::kCenter,
                static_cast<int>(centreX - 20), static_cast<int>(centreY + 16), 40, 14);
}

void HolyRotaryKnob::mouseDown(const visage::MouseEvent& e)
{
    dragging_ = true;
    float paramNorm = attachment_ ? attachment_->getNormalisedValue() : 0.5f;
    dragKnobNorm_ = fromParamMapper_ ? fromParamMapper_(paramNorm) : paramNorm;
    dragCurrentNorm_ = paramNorm;
    lastDragX_ = e.position.x;
    if (attachment_)
        attachment_->beginGesture();
    redraw();
}

void HolyRotaryKnob::mouseDrag(const visage::MouseEvent& e)
{
    if (!dragging_ || !attachment_)
        return;

    // Horizontal drag: right = increase (CW), left = decrease (CCW)
    float dx = e.position.x - lastDragX_;
    lastDragX_ = e.position.x;
    float sensitivity = e.isShiftDown() ? kSensitivity * 4.0f : kSensitivity;
    dragKnobNorm_ = std::clamp(dragKnobNorm_ + dx / sensitivity, 0.0f, 1.0f);
    dragCurrentNorm_ = toParamMapper_ ? toParamMapper_(dragKnobNorm_) : dragKnobNorm_;
    attachment_->setNormalisedValue(dragCurrentNorm_);
    redraw();
}

void HolyRotaryKnob::mouseUp(const visage::MouseEvent&)
{
    if (dragging_ && attachment_)
        attachment_->endGesture();
    dragging_ = false;
    redraw();
}
