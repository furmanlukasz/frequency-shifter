#pragma once

#include "../VisageParamAttachment.h"
#include "../HolyTheme.h"
#include <visage_ui/frame.h>
#include <array>
#include <memory>

class HolyPianoKeyboard : public visage::Frame
{
public:
    HolyPianoKeyboard();

    void setAttachments(juce::AudioProcessorValueTreeState& apvts,
                        const juce::String& paramPrefix);

    void draw(visage::Canvas& canvas) override;
    void mouseDown(const visage::MouseEvent& e) override;

private:
    int getKeyAtPoint(float x, float y) const;
    void getWhiteKeyRect(int whiteIndex, float& x, float& y, float& w, float& h) const;
    void getBlackKeyRect(int pitchClass, float& x, float& y, float& w, float& h) const;

    std::array<std::unique_ptr<VisageParamAttachment>, 12> attachments_;
    std::array<bool, 12> noteStates_{};

    static constexpr int whiteKeyPC_[7] = { 0, 2, 4, 5, 7, 9, 11 };
    static constexpr int blackKeyPC_[5] = { 1, 3, 6, 8, 10 };
    static constexpr int blackAfterWhite_[5] = { 0, 1, 3, 4, 5 };

    VISAGE_LEAK_CHECKER(HolyPianoKeyboard)
};
