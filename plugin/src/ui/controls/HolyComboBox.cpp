#include "HolyComboBox.h"
#include <cmath>
#include <algorithm>

visage::Frame* HolyComboBox::sharedDropdown_ = nullptr;

// === HolyComboBox ===

HolyComboBox::HolyComboBox() = default;

void HolyComboBox::setSharedDropdown(visage::Frame* dropdown)
{
    sharedDropdown_ = dropdown;
}

visage::Frame* HolyComboBox::sharedDropdown()
{
    return sharedDropdown_;
}

void HolyComboBox::addItem(const std::string& name)
{
    items_.push_back(name);
}

void HolyComboBox::clearItems()
{
    items_.clear();
}

int HolyComboBox::getSelectedIndex() const
{
    if (!attachment_ || items_.empty())
        return 0;
    float norm = attachment_->getNormalisedValue();
    int idx = static_cast<int>(std::round(norm * static_cast<float>(items_.size() - 1)));
    return std::clamp(idx, 0, static_cast<int>(items_.size()) - 1);
}

std::string HolyComboBox::getSelectedText() const
{
    int idx = getSelectedIndex();
    if (idx >= 0 && idx < static_cast<int>(items_.size()))
        return items_[idx];
    return "";
}

void HolyComboBox::selectIndex(int idx)
{
    if (!attachment_ || items_.empty() || idx < 0 || idx >= static_cast<int>(items_.size()))
        return;
    float norm = static_cast<float>(idx) /
                 std::max(1.0f, static_cast<float>(items_.size() - 1));
    attachment_->beginGesture();
    attachment_->setNormalisedValue(norm);
    attachment_->endGesture();
    redraw();
}

void HolyComboBox::draw(visage::Canvas& canvas)
{
    float w = static_cast<float>(width());
    float h = static_cast<float>(height());
    bool d = dimmed_;

    canvas.setColor(holy::dimColor(holy::colors::raised, d));
    canvas.roundedRectangle(0, 0, w, h, 3.0f);
    canvas.setColor(holy::dimColor(holy::colors::border, d));
    canvas.roundedRectangleBorder(0, 0, w, h, 3.0f, 1.0f);

    std::string text = getSelectedText();
    auto font = holy::makeFont(13.0f);
    canvas.setColor(holy::dimColor(holy::colors::text, d));
    canvas.text(text.c_str(), font, visage::Font::kLeft,
                8, 0, static_cast<int>(w - 20), static_cast<int>(h));

    float arrowX = w - 14.0f;
    float arrowY = h * 0.5f - 2.0f;
    canvas.setColor(holy::dimColor(holy::colors::textMuted, d));
    canvas.triangleDown(arrowX, arrowY, 4.0f);
}

void HolyComboBox::mouseDown(const visage::MouseEvent&)
{
    if (items_.empty() || dimmed_)
        return;

    auto* overlay = dynamic_cast<HolyDropdownOverlay*>(sharedDropdown_);
    if (overlay)
    {
        if (overlay->isOpen())
            overlay->hide();
        else
            overlay->showFor(this);
    }
}

// === HolyDropdownOverlay ===

HolyDropdownOverlay::HolyDropdownOverlay()
{
    setVisible(false);
}

void HolyDropdownOverlay::showFor(HolyComboBox* combo)
{
    combo_ = combo;
    if (!combo_)
        return;

    hoveredIndex_ = -1;
    int numItems = static_cast<int>(combo_->getItems().size());
    int dropH = numItems * kItemHeight + 4;
    int dropW = combo_->width() > 120 ? combo_->width() : 120;

    // Position below the combo box in window coordinates
    auto comboPos = combo_->positionInWindow();
    int gx = static_cast<int>(comboPos.x);
    int gy = static_cast<int>(comboPos.y) + combo_->height();

    // Clamp to not go off-screen bottom
    if (parent())
    {
        int maxH = parent()->height() - gy;
        if (dropH > maxH)
            dropH = maxH;
    }

    setBounds(gx, gy, dropW, dropH);
    setVisible(true);
    redraw();
}

void HolyDropdownOverlay::hide()
{
    setVisible(false);
    combo_ = nullptr;
    if (parent())
        parent()->redraw();
}

void HolyDropdownOverlay::draw(visage::Canvas& canvas)
{
    if (!combo_)
        return;

    float w = static_cast<float>(width());
    float h = static_cast<float>(height());

    canvas.setColor(holy::colors::raised);
    canvas.roundedRectangle(0, 0, w, h, 3.0f);
    canvas.setColor(holy::colors::border);
    canvas.roundedRectangleBorder(0, 0, w, h, 3.0f, 1.0f);

    auto font = holy::makeFont(13.0f);
    const auto& items = combo_->getItems();
    int selected = combo_->getSelectedIndex();

    for (int i = 0; i < static_cast<int>(items.size()); ++i)
    {
        int itemY = 2 + i * kItemHeight;
        if (itemY + kItemHeight > static_cast<int>(h))
            break;

        if (i == hoveredIndex_)
        {
            canvas.setColor(holy::colors::accentDim);
            canvas.fill(2, itemY, static_cast<int>(w) - 4, kItemHeight);
        }
        else if (i == selected)
        {
            canvas.setColor(0xFF1A1A1Du);
            canvas.fill(2, itemY, static_cast<int>(w) - 4, kItemHeight);
        }

        canvas.setColor(holy::colors::text);
        canvas.text(items[i].c_str(), font, visage::Font::kLeft,
                    8, itemY, static_cast<int>(w) - 16, kItemHeight);
    }
}

void HolyDropdownOverlay::mouseDown(const visage::MouseEvent& e)
{
    if (!combo_)
    {
        hide();
        return;
    }

    int itemIdx = static_cast<int>((e.position.y - 2) / kItemHeight);
    int numItems = static_cast<int>(combo_->getItems().size());

    if (itemIdx >= 0 && itemIdx < numItems)
        combo_->selectIndex(itemIdx);

    hide();
}

void HolyDropdownOverlay::mouseMove(const visage::MouseEvent& e)
{
    int newHover = static_cast<int>((e.position.y - 2) / kItemHeight);
    if (!combo_ || newHover < 0 || newHover >= static_cast<int>(combo_->getItems().size()))
        newHover = -1;
    if (newHover != hoveredIndex_)
    {
        hoveredIndex_ = newHover;
        redraw();
    }
}
