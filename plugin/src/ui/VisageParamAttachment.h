#pragma once

#include <JuceHeader.h>
#include <visage_ui/frame.h>
#include <functional>
#include <atomic>

/**
 * Bridges a JUCE AudioProcessorValueTreeState parameter to a Visage UI control.
 *
 * During drag (gesture active): uses locally cached value for responsive UI.
 * Otherwise: reads directly from param->getValue() every frame — always fresh,
 * picks up preset loads (replaceState) and DAW automation automatically.
 */
class VisageParamAttachment : public juce::AudioProcessorValueTreeState::Listener
{
public:
    VisageParamAttachment(juce::AudioProcessorValueTreeState& apvts,
                          const juce::String& paramId)
        : apvts_(apvts), paramId_(paramId)
    {
        param_ = apvts_.getParameter(paramId_);
        jassert(param_ != nullptr);
        apvts_.addParameterListener(paramId_, this);
    }

    ~VisageParamAttachment() override
    {
        apvts_.removeParameterListener(paramId_, this);
    }

    void beginGesture()
    {
        gestureActive_.store(true);
        gestureNorm_.store(param_ ? param_->getValue() : 0.0f);
        if (param_)
            param_->beginChangeGesture();
    }

    void endGesture()
    {
        if (param_)
            param_->endChangeGesture();
        gestureActive_.store(false);
    }

    void setNormalisedValue(float normValue)
    {
        gestureNorm_.store(normValue);
        if (param_)
            param_->setValueNotifyingHost(normValue);
    }

    // Always returns the true current value:
    // - During gesture: returns the local drag value (responsive)
    // - Otherwise: reads directly from the JUCE parameter (always fresh)
    float getNormalisedValue() const
    {
        if (gestureActive_.load())
            return gestureNorm_.load();
        if (param_)
            return param_->getValue();
        return 0.0f;
    }

    float getValue() const
    {
        if (param_)
            return param_->convertFrom0to1(getNormalisedValue());
        return 0.0f;
    }

    // Parameter default in normalized [0, 1] space — for double-click-to-default.
    float getDefaultNormalisedValue() const
    {
        return param_ ? param_->getDefaultValue() : 0.0f;
    }

    juce::RangedAudioParameter* getParameter() { return param_; }

private:
    void parameterChanged(const juce::String&, float) override
    {
        // No-op: we read directly from param_->getValue() now
    }

    juce::AudioProcessorValueTreeState& apvts_;
    juce::String paramId_;
    juce::RangedAudioParameter* param_ = nullptr;
    std::atomic<float> gestureNorm_{ 0.0f };
    std::atomic<bool> gestureActive_{ false };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VisageParamAttachment)
};
