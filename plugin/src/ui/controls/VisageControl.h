#pragma once

#include "../VisageParamAttachment.h"
#include <visage_ui/frame.h>
#include <memory>

/**
 * Base for every Holy* control that binds to a single APVTS parameter.
 *
 * Owns the standard attachment + dimmed-state plumbing so each concrete
 * control only has to implement draw() and its own gesture logic.
 *
 * Controls that need additional attachments (e.g. HolySlider's sync attachment)
 * still declare those separately; this base only handles the primary binding.
 */
class VisageControl : public visage::Frame
{
public:
    // Standard single-parameter binding.
    void setAttachment(juce::AudioProcessorValueTreeState& apvts,
                       const juce::String& paramId)
    {
        attachment_ = std::make_unique<VisageParamAttachment>(apvts, paramId);
    }

    void setDimmed(bool d) { if (dimmed_ != d) { dimmed_ = d; redraw(); } }
    bool isDimmed() const { return dimmed_; }

protected:
    // Consumes a double-click as a reset-to-default. Returns true if handled —
    // callers should early-return without starting a drag gesture.
    bool handleDoubleClickReset(const visage::MouseEvent& e)
    {
        if (e.repeatClickCount() < 2 || !attachment_)
            return false;
        attachment_->beginGesture();
        attachment_->setNormalisedValue(attachment_->getDefaultNormalisedValue());
        attachment_->endGesture();
        redraw();
        return true;
    }

    std::unique_ptr<VisageParamAttachment> attachment_;
    bool dimmed_ = false;
};
