#include "HolySegmentedControl.h"
#include <algorithm>

HolySegmentedControl::HolySegmentedControl() = default;

void HolySegmentedControl::addSegment(const std::string& label)
{
    segments_.push_back(label);
}

int HolySegmentedControl::getSelectedIndex() const
{
    if (!attachment_ || segments_.empty())
        return 0;
    float norm = attachment_->getNormalisedValue();
    int idx = static_cast<int>(std::round(norm * static_cast<float>(segments_.size() - 1)));
    return std::clamp(idx, 0, static_cast<int>(segments_.size()) - 1);
}

void HolySegmentedControl::draw(visage::Canvas& canvas)
{
    float w = static_cast<float>(width());
    float h = static_cast<float>(height());
    int n = static_cast<int>(segments_.size());
    if (n == 0)
        return;

    float r = h * 0.5f;
    int selected = getSelectedIndex();

    // Outer pill background
    canvas.setColor(holy::colors::strip);
    canvas.roundedRectangle(0, 0, w, h, r);
    canvas.setColor(holy::colors::stripBorder);
    canvas.roundedRectangleBorder(0, 0, w, h, r, 1.0f);

    float segW = w / static_cast<float>(n);
    auto labelFont = holy::makeFont(13.0f);

    for (int i = 0; i < n; ++i)
    {
        float sx = segW * static_cast<float>(i);

        if (i == selected)
        {
            // Selected segment: accent-filled rounded rect
            float inset = 2.0f;
            float segR = r - inset;
            canvas.setColor(holy::colors::accent);
            canvas.roundedRectangle(sx + inset, inset, segW - inset * 2, h - inset * 2, segR);

            // Selected text: dark on accent
            canvas.setColor(holy::colors::background);
            canvas.text(segments_[i].c_str(), labelFont, visage::Font::kCenter,
                        static_cast<int>(sx), 0, static_cast<int>(segW), static_cast<int>(h));
        }
        else
        {
            // Unselected text: muted
            canvas.setColor(holy::colors::textSec);
            canvas.text(segments_[i].c_str(), labelFont, visage::Font::kCenter,
                        static_cast<int>(sx), 0, static_cast<int>(segW), static_cast<int>(h));
        }
    }
}

void HolySegmentedControl::mouseDown(const visage::MouseEvent& e)
{
    if (segments_.empty() || !attachment_)
        return;

    float segW = static_cast<float>(width()) / static_cast<float>(segments_.size());
    int clicked = static_cast<int>(e.position.x / segW);
    clicked = std::clamp(clicked, 0, static_cast<int>(segments_.size()) - 1);

    if (clicked != getSelectedIndex())
    {
        float norm = static_cast<float>(clicked) /
                     std::max(1.0f, static_cast<float>(segments_.size() - 1));
        attachment_->beginGesture();
        attachment_->setNormalisedValue(norm);
        attachment_->endGesture();
        redraw();

        if (onChange)
            onChange(clicked);
    }
}
