#include "HolyPianoKeyboard.h"

HolyPianoKeyboard::HolyPianoKeyboard()
{
    noteStates_.fill(true);
}

void HolyPianoKeyboard::setAttachments(juce::AudioProcessorValueTreeState& apvts,
                                        const juce::String& paramPrefix)
{
    for (int i = 0; i < 12; ++i)
    {
        auto paramId = paramPrefix + juce::String(i);
        attachments_[i] = std::make_unique<VisageParamAttachment>(apvts, paramId);
        noteStates_[i] = attachments_[i]->getNormalisedValue() > 0.5f;
    }
}

void HolyPianoKeyboard::getWhiteKeyRect(int whiteIndex, float& rx, float& ry, float& rw, float& rh) const
{
    float w = static_cast<float>(width()) / 7.0f;
    rx = whiteIndex * w + 0.5f;
    ry = 0.5f;
    rw = w - 1.0f;
    rh = static_cast<float>(height()) - 1.0f;
}

void HolyPianoKeyboard::getBlackKeyRect(int pitchClass, float& rx, float& ry, float& rw, float& rh) const
{
    float whiteW = static_cast<float>(width()) / 7.0f;
    float blackW = whiteW * 0.6f;
    float blackH = static_cast<float>(height()) * 0.58f;

    int whiteIdx = -1;
    for (int i = 0; i < 5; ++i)
        if (blackKeyPC_[i] == pitchClass)
            whiteIdx = blackAfterWhite_[i];

    rx = (whiteIdx + 1) * whiteW - blackW * 0.5f;
    ry = 0;
    rw = blackW;
    rh = blackH;
}

int HolyPianoKeyboard::getKeyAtPoint(float px, float py) const
{
    // Check black keys first
    for (int i = 0; i < 5; ++i)
    {
        float kx, ky, kw, kh;
        getBlackKeyRect(blackKeyPC_[i], kx, ky, kw, kh);
        if (px >= kx && px < kx + kw && py >= ky && py < ky + kh)
            return blackKeyPC_[i];
    }
    // Then white keys
    for (int i = 0; i < 7; ++i)
    {
        float kx, ky, kw, kh;
        getWhiteKeyRect(i, kx, ky, kw, kh);
        if (px >= kx && px < kx + kw && py >= ky && py < ky + kh)
            return whiteKeyPC_[i];
    }
    return -1;
}

void HolyPianoKeyboard::draw(visage::Canvas& canvas)
{
    // Poll current state from parameters
    for (int i = 0; i < 12; ++i)
        if (attachments_[i])
            noteStates_[i] = attachments_[i]->getNormalisedValue() > 0.5f;

    bool d = dimmed_;
    float r = 2.0f;

    // White keys
    for (int i = 0; i < 7; ++i)
    {
        int pc = whiteKeyPC_[i];
        float kx, ky, kw, kh;
        getWhiteKeyRect(i, kx, ky, kw, kh);

        canvas.setColor(holy::dimColor(noteStates_[pc] ? holy::colors::accent : 0xFF1E1E22u, d));
        canvas.roundedRectangle(kx, ky, kw, kh, r);
        canvas.setColor(holy::dimColor(0xFF2A2A2Eu, d));
        canvas.roundedRectangleBorder(kx, ky, kw, kh, r, 0.5f);
    }

    // Black keys on top
    for (int i = 0; i < 5; ++i)
    {
        int pc = blackKeyPC_[i];
        float kx, ky, kw, kh;
        getBlackKeyRect(pc, kx, ky, kw, kh);

        canvas.setColor(holy::dimColor(noteStates_[pc] ? holy::colors::accent : holy::colors::background, d));
        canvas.roundedRectangle(kx, ky, kw, kh, r);
        canvas.setColor(holy::dimColor(0xFF1A1A1Du, d));
        canvas.roundedRectangleBorder(kx, ky, kw, kh, r, 0.5f);
    }
}

void HolyPianoKeyboard::mouseDown(const visage::MouseEvent& e)
{
    if (dimmed_)
        return;
    int key = getKeyAtPoint(e.position.x, e.position.y);
    if (key < 0 || key >= 12 || !attachments_[key])
        return;

    bool newState = !noteStates_[key];
    noteStates_[key] = newState;
    attachments_[key]->beginGesture();
    attachments_[key]->setNormalisedValue(newState ? 1.0f : 0.0f);
    attachments_[key]->endGesture();
    redraw();
}
