#include "HolySlider.h"
#include <cmath>
#include <algorithm>

HolySlider::HolySlider() = default;

void HolySlider::setAttachment(juce::AudioProcessorValueTreeState& apvts,
                                const juce::String& paramId)
{
    attachment_ = std::make_unique<VisageParamAttachment>(apvts, paramId);
}

float HolySlider::getSliderWidth() const
{
    return std::max(1.0f, static_cast<float>(width()) - kTextWidth);
}

std::string HolySlider::formatValue() const
{
    if (!attachment_)
        return "0";
    float val = attachment_->getValue();
    char buf[32];
    if (decimals_ == 0)
        std::snprintf(buf, sizeof(buf), "%d", static_cast<int>(val));
    else
        std::snprintf(buf, sizeof(buf), "%.*f", decimals_, val);
    return std::string(buf) + suffix_;
}

void HolySlider::draw(visage::Canvas& canvas)
{
    float w = static_cast<float>(width());
    float h = static_cast<float>(height());
    float sliderW = getSliderWidth();

    float norm = dragging_ ? dragCurrentNorm_
                           : (attachment_ ? attachment_->getNormalisedValue() : 0.0f);

    float trackH = 1.5f;
    float trackY = h * 0.5f - trackH * 0.5f;

    // Background track
    canvas.setColor(holy::colors::track);
    canvas.roundedRectangle(0, trackY, sliderW, trackH, trackH * 0.5f);

    // Filled track
    float fillW = norm * sliderW;
    if (fillW > 0.5f)
    {
        canvas.setColor(holy::colors::accent);
        canvas.roundedRectangle(0, trackY, fillW, trackH, trackH * 0.5f);
    }

    // Thumb
    float thumbSize = 7.0f;
    float thumbX = fillW - thumbSize * 0.5f;
    float thumbY = h * 0.5f - thumbSize * 0.5f;
    canvas.setColor(holy::colors::accent);
    canvas.circle(thumbX, thumbY, thumbSize);

    // Value text
    std::string valText = formatValue();
    auto font = holy::makeFont(11.0f);
    canvas.setColor(holy::colors::text);
    canvas.text(valText.c_str(), font, visage::Font::kRight,
                static_cast<int>(sliderW + 4), 0,
                static_cast<int>(kTextWidth - 4), static_cast<int>(h));
}

void HolySlider::mouseDown(const visage::MouseEvent& e)
{
    if (!attachment_)
        return;

    dragging_ = true;
    attachment_->beginGesture();

    // e.position = frame-local coordinates
    float sliderW = getSliderWidth();
    dragCurrentNorm_ = std::clamp(e.position.x / sliderW, 0.0f, 1.0f);
    attachment_->setNormalisedValue(dragCurrentNorm_);
    redraw();
}

void HolySlider::mouseDrag(const visage::MouseEvent& e)
{
    if (!dragging_ || !attachment_)
        return;

    // e.position = frame-local coordinates (absolute within this frame)
    float sliderW = getSliderWidth();
    dragCurrentNorm_ = std::clamp(e.position.x / sliderW, 0.0f, 1.0f);
    attachment_->setNormalisedValue(dragCurrentNorm_);
    redraw();
}

void HolySlider::mouseUp(const visage::MouseEvent&)
{
    if (dragging_ && attachment_)
        attachment_->endGesture();
    dragging_ = false;
    redraw();
}
