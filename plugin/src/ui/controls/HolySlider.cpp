#include "HolySlider.h"
#include <cmath>
#include <algorithm>

HolySlider::HolySlider() = default;

void HolySlider::setAttachment(juce::AudioProcessorValueTreeState& apvts,
                                const juce::String& paramId)
{
    attachment_ = std::make_unique<VisageParamAttachment>(apvts, paramId);
}

void HolySlider::setSyncAttachment(juce::AudioProcessorValueTreeState& apvts,
                                    const juce::String& paramId)
{
    syncAttachment_ = std::make_unique<VisageParamAttachment>(apvts, paramId);
}

void HolySlider::setSyncLabels(const std::vector<std::string>& labels)
{
    syncLabels_ = labels;
}

void HolySlider::setSynced(bool synced)
{
    if (synced_ != synced)
    {
        synced_ = synced;
        redraw();
    }
}

VisageParamAttachment* HolySlider::activeAttachment() const
{
    if (synced_ && syncAttachment_)
        return syncAttachment_.get();
    return attachment_.get();
}

float HolySlider::getSliderWidth() const
{
    return std::max(1.0f, static_cast<float>(width()) - kTextWidth);
}

std::string HolySlider::formatValue() const
{
    auto* att = activeAttachment();
    if (!att)
        return "0";

    if (synced_ && !syncLabels_.empty())
    {
        // Show division label
        float norm = att->getNormalisedValue();
        int idx = static_cast<int>(std::round(norm * static_cast<float>(syncLabels_.size() - 1)));
        idx = std::clamp(idx, 0, static_cast<int>(syncLabels_.size()) - 1);
        return syncLabels_[idx];
    }

    float val = att->getValue();
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
    bool d = dimmed_;

    auto* att = activeAttachment();
    float norm = dragging_ ? dragCurrentNorm_
                           : (att ? att->getNormalisedValue() : 0.0f);

    float trackH = 1.5f;
    float trackY = h * 0.5f - trackH * 0.5f;

    // Background track
    canvas.setColor(holy::dimColor(holy::colors::track, d));
    canvas.roundedRectangle(0, trackY, sliderW, trackH, trackH * 0.5f);

    // Filled track
    float fillW = norm * sliderW;
    if (fillW > 0.5f)
    {
        canvas.setColor(holy::dimColor(holy::colors::accent, d));
        canvas.roundedRectangle(0, trackY, fillW, trackH, trackH * 0.5f);
    }

    // Snap tick marks when synced
    if (synced_ && !syncLabels_.empty())
    {
        int n = static_cast<int>(syncLabels_.size());
        for (int i = 0; i < n; ++i)
        {
            float tickX = (static_cast<float>(i) / static_cast<float>(n - 1)) * sliderW;
            canvas.setColor(holy::dimColor(holy::colors::textMuted, d));
            canvas.fill(static_cast<int>(tickX), static_cast<int>(trackY - 2.0f), 1, 5);
        }
    }

    // Thumb
    float thumbSize = 7.0f;
    float thumbX = fillW - thumbSize * 0.5f;
    float thumbY = h * 0.5f - thumbSize * 0.5f;
    canvas.setColor(holy::dimColor(holy::colors::accent, d));
    canvas.circle(thumbX, thumbY, thumbSize);

    // Value text
    std::string valText = formatValue();
    auto font = holy::makeFont(11.0f);
    canvas.setColor(holy::dimColor(holy::colors::text, d));
    canvas.text(valText.c_str(), font, visage::Font::kRight,
                static_cast<int>(sliderW + 4), 0,
                static_cast<int>(kTextWidth - 4), static_cast<int>(h));
}

void HolySlider::mouseDown(const visage::MouseEvent& e)
{
    auto* att = activeAttachment();
    if (!att || dimmed_)
        return;

    dragging_ = true;
    att->beginGesture();

    float sliderW = getSliderWidth();
    float rawNorm = std::clamp(e.position.x / sliderW, 0.0f, 1.0f);

    // Snap to nearest division when synced
    if (synced_ && !syncLabels_.empty())
    {
        int n = static_cast<int>(syncLabels_.size());
        int idx = static_cast<int>(std::round(rawNorm * static_cast<float>(n - 1)));
        idx = std::clamp(idx, 0, n - 1);
        rawNorm = static_cast<float>(idx) / static_cast<float>(n - 1);
    }

    dragCurrentNorm_ = rawNorm;
    att->setNormalisedValue(dragCurrentNorm_);
    redraw();
}

void HolySlider::mouseDrag(const visage::MouseEvent& e)
{
    auto* att = activeAttachment();
    if (!dragging_ || !att)
        return;

    float sliderW = getSliderWidth();
    float rawNorm = std::clamp(e.position.x / sliderW, 0.0f, 1.0f);

    // Snap to nearest division when synced
    if (synced_ && !syncLabels_.empty())
    {
        int n = static_cast<int>(syncLabels_.size());
        int idx = static_cast<int>(std::round(rawNorm * static_cast<float>(n - 1)));
        idx = std::clamp(idx, 0, n - 1);
        rawNorm = static_cast<float>(idx) / static_cast<float>(n - 1);
    }

    dragCurrentNorm_ = rawNorm;
    att->setNormalisedValue(dragCurrentNorm_);
    redraw();
}

void HolySlider::mouseUp(const visage::MouseEvent&)
{
    if (dragging_)
    {
        auto* att = activeAttachment();
        if (att)
            att->endGesture();
    }
    dragging_ = false;
    redraw();
}
