#include "HolyToggle.h"

HolyToggle::HolyToggle(const std::string& label) : label_(label) {}

void HolyToggle::setAttachment(juce::AudioProcessorValueTreeState& apvts,
                                const juce::String& paramId)
{
    attachment_ = std::make_unique<VisageParamAttachment>(apvts, paramId);
}

bool HolyToggle::isOn() const
{
    if (attachment_)
        return attachment_->getNormalisedValue() > 0.5f;
    return manualState_;
}

void HolyToggle::draw(visage::Canvas& canvas)
{
    bool on = isOn();
    bool d = dimmed_;
    float h = static_cast<float>(height());

    // Pill dimensions
    float pillW = 30.0f;
    float pillH = 15.0f;
    float dotSize = 11.0f;
    float pillY = (h - pillH) * 0.5f;
    float pillR = pillH * 0.5f;

    // Pill background
    canvas.setColor(holy::dimColor(on ? holy::colors::accentDim : holy::colors::track, d));
    canvas.roundedRectangle(0, pillY, pillW, pillH, pillR);

    // Dot
    float dotY = pillY + (pillH - dotSize) * 0.5f;
    float dotX = on ? (pillW - dotSize - 2.0f) : 2.0f;
    canvas.setColor(holy::dimColor(on ? holy::colors::accent : holy::colors::textMuted, d));
    canvas.circle(dotX, dotY, dotSize);

    // Label (Figma: 9px Inter Medium, tracking varies)
    if (!label_.empty())
    {
        auto font = holy::makeFont(10.0f);
        unsigned int color = labelColor_ != 0 ? labelColor_
            : (on ? holy::colors::text : holy::colors::textSec);
        canvas.setColor(holy::dimColor(color, d));
        canvas.text(label_.c_str(), font, visage::Font::kLeft,
                    static_cast<int>(pillW + 6), 0,
                    width() - static_cast<int>(pillW + 6), static_cast<int>(h));
    }
}

void HolyToggle::mouseDown(const visage::MouseEvent&)
{
    if (dimmed_)
        return;
    if (attachment_)
    {
        bool newState = !isOn();
        attachment_->beginGesture();
        attachment_->setNormalisedValue(newState ? 1.0f : 0.0f);
        attachment_->endGesture();
    }
    else
    {
        manualState_ = !manualState_;
        if (onToggle)
            onToggle(manualState_);
    }
    redraw();
}
