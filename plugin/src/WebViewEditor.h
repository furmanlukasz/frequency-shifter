#pragma once
// Resizable, responsive WebView UI (juce::WebBrowserComponent). Compiled only when
// HOLY_SHIFTER_USE_WEBVIEW=1; otherwise PluginProcessor uses the Visage editor.
#if HOLY_SHIFTER_USE_WEBVIEW
#include <juce_gui_extra/juce_gui_extra.h>
#include "PluginProcessor.h"
#include <optional>

class WebViewEditor : public juce::AudioProcessorEditor {
public:
    explicit WebViewEditor(FrequencyShifterProcessor&);
    ~WebViewEditor() override;
    void resized() override;
private:
    std::optional<juce::WebBrowserComponent::Resource> getResource(const juce::String& url) const;
    FrequencyShifterProcessor& processorRef;

    // One relay per APVTS parameter; relay name == the parameter id.
    juce::WebSliderRelay shiftHzRelay { "shiftHz" };
    juce::WebSliderRelay quantizeStrengthRelay { "quantizeStrength" };
    juce::WebToggleButtonRelay scaleNote0Relay { "scaleNote0" };
    juce::WebToggleButtonRelay scaleNote1Relay { "scaleNote1" };
    juce::WebToggleButtonRelay scaleNote2Relay { "scaleNote2" };
    juce::WebToggleButtonRelay scaleNote3Relay { "scaleNote3" };
    juce::WebToggleButtonRelay scaleNote4Relay { "scaleNote4" };
    juce::WebToggleButtonRelay scaleNote5Relay { "scaleNote5" };
    juce::WebToggleButtonRelay scaleNote6Relay { "scaleNote6" };
    juce::WebToggleButtonRelay scaleNote7Relay { "scaleNote7" };
    juce::WebToggleButtonRelay scaleNote8Relay { "scaleNote8" };
    juce::WebToggleButtonRelay scaleNote9Relay { "scaleNote9" };
    juce::WebToggleButtonRelay scaleNote10Relay { "scaleNote10" };
    juce::WebToggleButtonRelay scaleNote11Relay { "scaleNote11" };
    juce::WebSliderRelay dryWetRelay { "dryWet" };
    juce::WebSliderRelay smearRelay { "smear" };
    juce::WebToggleButtonRelay logScaleRelay { "logScale" };
    juce::WebSliderRelay lfoDepthRelay { "lfoDepth" };
    juce::WebComboBoxRelay lfoDepthModeRelay { "lfoDepthMode" };
    juce::WebSliderRelay lfoRateRelay { "lfoRate" };
    juce::WebToggleButtonRelay lfoSyncRelay { "lfoSync" };
    juce::WebComboBoxRelay lfoDivisionRelay { "lfoDivision" };
    juce::WebComboBoxRelay lfoShapeRelay { "lfoShape" };
    juce::WebToggleButtonRelay lfoEnabledRelay { "lfoEnabled" };
    juce::WebSliderRelay dlyLfoDepthRelay { "dlyLfoDepth" };
    juce::WebSliderRelay dlyLfoRateRelay { "dlyLfoRate" };
    juce::WebToggleButtonRelay dlyLfoSyncRelay { "dlyLfoSync" };
    juce::WebComboBoxRelay dlyLfoDivisionRelay { "dlyLfoDivision" };
    juce::WebComboBoxRelay dlyLfoShapeRelay { "dlyLfoShape" };
    juce::WebToggleButtonRelay dlyLfoEnabledRelay { "dlyLfoEnabled" };
    juce::WebToggleButtonRelay maskEnabledRelay { "maskEnabled" };
    juce::WebComboBoxRelay maskModeRelay { "maskMode" };
    juce::WebSliderRelay maskLowFreqRelay { "maskLowFreq" };
    juce::WebSliderRelay maskHighFreqRelay { "maskHighFreq" };
    juce::WebSliderRelay maskTransitionRelay { "maskTransition" };
    juce::WebToggleButtonRelay delayEnabledRelay { "delayEnabled" };
    juce::WebSliderRelay delayTimeRelay { "delayTime" };
    juce::WebToggleButtonRelay delaySyncRelay { "delaySync" };
    juce::WebComboBoxRelay delayDivisionRelay { "delayDivision" };
    juce::WebSliderRelay delaySlopeRelay { "delaySlope" };
    juce::WebSliderRelay delayFeedbackRelay { "delayFeedback" };
    juce::WebSliderRelay delayDampingRelay { "delayDamping" };
    juce::WebSliderRelay delayGainRelay { "delayGain" };
    juce::WebSliderRelay preserveRelay { "preserve" };
    juce::WebSliderRelay transientsRelay { "transients" };
    juce::WebSliderRelay sensitivityRelay { "sensitivity" };
    juce::WebToggleButtonRelay peakSnapRelay { "peakSnap" };
    juce::WebSliderRelay noiseMixRelay { "noiseMix" };
    juce::WebSliderRelay peakSensRelay { "peakSens" };
    juce::WebComboBoxRelay processingModeRelay { "processingMode" };
    juce::WebToggleButtonRelay warmRelay { "warm" };

    juce::WebBrowserComponent webView;

    // Attachments wire each relay to its APVTS parameter (constructed after webView).
    juce::WebSliderParameterAttachment shiftHzAtt;
    juce::WebSliderParameterAttachment quantizeStrengthAtt;
    juce::WebToggleButtonParameterAttachment scaleNote0Att;
    juce::WebToggleButtonParameterAttachment scaleNote1Att;
    juce::WebToggleButtonParameterAttachment scaleNote2Att;
    juce::WebToggleButtonParameterAttachment scaleNote3Att;
    juce::WebToggleButtonParameterAttachment scaleNote4Att;
    juce::WebToggleButtonParameterAttachment scaleNote5Att;
    juce::WebToggleButtonParameterAttachment scaleNote6Att;
    juce::WebToggleButtonParameterAttachment scaleNote7Att;
    juce::WebToggleButtonParameterAttachment scaleNote8Att;
    juce::WebToggleButtonParameterAttachment scaleNote9Att;
    juce::WebToggleButtonParameterAttachment scaleNote10Att;
    juce::WebToggleButtonParameterAttachment scaleNote11Att;
    juce::WebSliderParameterAttachment dryWetAtt;
    juce::WebSliderParameterAttachment smearAtt;
    juce::WebToggleButtonParameterAttachment logScaleAtt;
    juce::WebSliderParameterAttachment lfoDepthAtt;
    juce::WebComboBoxParameterAttachment lfoDepthModeAtt;
    juce::WebSliderParameterAttachment lfoRateAtt;
    juce::WebToggleButtonParameterAttachment lfoSyncAtt;
    juce::WebComboBoxParameterAttachment lfoDivisionAtt;
    juce::WebComboBoxParameterAttachment lfoShapeAtt;
    juce::WebToggleButtonParameterAttachment lfoEnabledAtt;
    juce::WebSliderParameterAttachment dlyLfoDepthAtt;
    juce::WebSliderParameterAttachment dlyLfoRateAtt;
    juce::WebToggleButtonParameterAttachment dlyLfoSyncAtt;
    juce::WebComboBoxParameterAttachment dlyLfoDivisionAtt;
    juce::WebComboBoxParameterAttachment dlyLfoShapeAtt;
    juce::WebToggleButtonParameterAttachment dlyLfoEnabledAtt;
    juce::WebToggleButtonParameterAttachment maskEnabledAtt;
    juce::WebComboBoxParameterAttachment maskModeAtt;
    juce::WebSliderParameterAttachment maskLowFreqAtt;
    juce::WebSliderParameterAttachment maskHighFreqAtt;
    juce::WebSliderParameterAttachment maskTransitionAtt;
    juce::WebToggleButtonParameterAttachment delayEnabledAtt;
    juce::WebSliderParameterAttachment delayTimeAtt;
    juce::WebToggleButtonParameterAttachment delaySyncAtt;
    juce::WebComboBoxParameterAttachment delayDivisionAtt;
    juce::WebSliderParameterAttachment delaySlopeAtt;
    juce::WebSliderParameterAttachment delayFeedbackAtt;
    juce::WebSliderParameterAttachment delayDampingAtt;
    juce::WebSliderParameterAttachment delayGainAtt;
    juce::WebSliderParameterAttachment preserveAtt;
    juce::WebSliderParameterAttachment transientsAtt;
    juce::WebSliderParameterAttachment sensitivityAtt;
    juce::WebToggleButtonParameterAttachment peakSnapAtt;
    juce::WebSliderParameterAttachment noiseMixAtt;
    juce::WebSliderParameterAttachment peakSensAtt;
    juce::WebComboBoxParameterAttachment processingModeAtt;
    juce::WebToggleButtonParameterAttachment warmAtt;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WebViewEditor)
};
#endif // HOLY_SHIFTER_USE_WEBVIEW
