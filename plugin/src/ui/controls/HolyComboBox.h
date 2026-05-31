#pragma once

#include "VisageControl.h"
#include "../HolyTheme.h"
#include <vector>
#include <string>

class HolyComboBox : public VisageControl
{
public:
    HolyComboBox();

    void addItem(const std::string& name);
    void clearItems();

    void draw(visage::Canvas& canvas) override;
    void mouseDown(const visage::MouseEvent& e) override;

    int getSelectedIndex() const;
    std::string getSelectedText() const;
    void selectIndex(int idx);
    const std::vector<std::string>& getItems() const { return items_; }

    // Provide a shared dropdown frame (owned by root UI)
    static void setSharedDropdown(visage::Frame* dropdown);
    static visage::Frame* sharedDropdown();

private:
    std::vector<std::string> items_;
    static visage::Frame* sharedDropdown_;

    VISAGE_LEAK_CHECKER(HolyComboBox)
};

/**
 * Shared dropdown overlay, added as child of root frame for z-ordering.
 * Only one can be visible at a time.
 */
class HolyDropdownOverlay : public visage::Frame
{
public:
    HolyDropdownOverlay();

    void showFor(HolyComboBox* combo);
    void hide();
    bool isOpen() const { return combo_ != nullptr && isVisible(); }

    void draw(visage::Canvas& canvas) override;
    void mouseDown(const visage::MouseEvent& e) override;
    void mouseMove(const visage::MouseEvent& e) override;

private:
    HolyComboBox* combo_ = nullptr;
    int hoveredIndex_ = -1;
    static constexpr int kItemHeight = 24;

    VISAGE_LEAK_CHECKER(HolyDropdownOverlay)
};
