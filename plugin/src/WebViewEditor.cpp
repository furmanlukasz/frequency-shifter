#include "WebViewEditor.h"
#if HOLY_SHIFTER_USE_WEBVIEW
#include "BinaryData.h"
#include <cstring>
#include <vector>
#include <iostream>

namespace {
// Resolve a request path to a real filename; root -> index.html; strip query.
juce::String resolveFilename(const juce::String& url) {
    auto path = url.upToFirstOccurrenceOf("?", false, false);
    if (path.isEmpty() || path == "/") return "index.html";
    return path.fromFirstOccurrenceOf("/", false, false);
}
const char* mimeForExtension(const juce::String& ext) {
    if (ext == "html") return "text/html";
    if (ext == "js")   return "text/javascript";
    if (ext == "css")  return "text/css";
    if (ext == "json") return "application/json";
    if (ext == "svg")  return "image/svg+xml";
    if (ext == "png")  return "image/png";
    if (ext == "jpg" || ext == "jpeg") return "image/jpeg";
    if (ext == "woff2") return "font/woff2";
    if (ext == "woff")  return "font/woff";
    if (ext == "ttf")   return "font/ttf";
    return "application/octet-stream";
}
juce::String namesToJson(const juce::StringArray& a) {
    juce::Array<juce::var> v; for (auto& s : a) v.add(s);
    return juce::JSON::toString(juce::var(v));
}
} // namespace

WebViewEditor::WebViewEditor(FrequencyShifterProcessor& p)
    : juce::AudioProcessorEditor(p),
      processorRef(p),
      webView(juce::WebBrowserComponent::Options{}
          .withNativeIntegrationEnabled()
          .withOptionsFrom(shiftHzRelay)
          .withOptionsFrom(quantizeStrengthRelay)
          .withOptionsFrom(scaleNote0Relay)
          .withOptionsFrom(scaleNote1Relay)
          .withOptionsFrom(scaleNote2Relay)
          .withOptionsFrom(scaleNote3Relay)
          .withOptionsFrom(scaleNote4Relay)
          .withOptionsFrom(scaleNote5Relay)
          .withOptionsFrom(scaleNote6Relay)
          .withOptionsFrom(scaleNote7Relay)
          .withOptionsFrom(scaleNote8Relay)
          .withOptionsFrom(scaleNote9Relay)
          .withOptionsFrom(scaleNote10Relay)
          .withOptionsFrom(scaleNote11Relay)
          .withOptionsFrom(dryWetRelay)
          .withOptionsFrom(smearRelay)
          .withOptionsFrom(logScaleRelay)
          .withOptionsFrom(lfoDepthRelay)
          .withOptionsFrom(lfoDepthModeRelay)
          .withOptionsFrom(lfoRateRelay)
          .withOptionsFrom(lfoSyncRelay)
          .withOptionsFrom(lfoDivisionRelay)
          .withOptionsFrom(lfoShapeRelay)
          .withOptionsFrom(lfoEnabledRelay)
          .withOptionsFrom(dlyLfoDepthRelay)
          .withOptionsFrom(dlyLfoRateRelay)
          .withOptionsFrom(dlyLfoSyncRelay)
          .withOptionsFrom(dlyLfoDivisionRelay)
          .withOptionsFrom(dlyLfoShapeRelay)
          .withOptionsFrom(dlyLfoEnabledRelay)
          .withOptionsFrom(maskEnabledRelay)
          .withOptionsFrom(maskModeRelay)
          .withOptionsFrom(maskLowFreqRelay)
          .withOptionsFrom(maskHighFreqRelay)
          .withOptionsFrom(maskTransitionRelay)
          .withOptionsFrom(delayEnabledRelay)
          .withOptionsFrom(delayTimeRelay)
          .withOptionsFrom(delaySyncRelay)
          .withOptionsFrom(delayDivisionRelay)
          .withOptionsFrom(delaySlopeRelay)
          .withOptionsFrom(delayFeedbackRelay)
          .withOptionsFrom(delayDampingRelay)
          .withOptionsFrom(delayGainRelay)
          .withOptionsFrom(preserveRelay)
          .withOptionsFrom(transientsRelay)
          .withOptionsFrom(sensitivityRelay)
          .withOptionsFrom(peakSnapRelay)
          .withOptionsFrom(noiseMixRelay)
          .withOptionsFrom(peakSensRelay)
          .withOptionsFrom(processingModeRelay)
          .withOptionsFrom(warmRelay)
          // --- Preset bridge (PresetManager) ---
          .withNativeFunction(juce::Identifier("presetList"),
              [this](const juce::Array<juce::var>&, juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                  complete(juce::var(namesToJson(processorRef.getPresetManager().getAllPresetNames())));
              })
          .withNativeFunction(juce::Identifier("presetCurrent"),
              [this](const juce::Array<juce::var>&, juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                  complete(juce::var(processorRef.getPresetManager().getCurrentPresetName()));
              })
          .withNativeFunction(juce::Identifier("presetIsFactory"),
              [this](const juce::Array<juce::var>& a, juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                  complete(juce::var(a.size() > 0 && processorRef.getPresetManager().isFactoryPreset(a[0].toString())));
              })
          .withNativeFunction(juce::Identifier("presetLoad"),
              [this](const juce::Array<juce::var>& a, juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                  if (a.size() > 0) processorRef.getPresetManager().loadPreset(a[0].toString());
                  complete(juce::var(true));
              })
          .withNativeFunction(juce::Identifier("presetSave"),
              [this](const juce::Array<juce::var>& a, juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                  if (a.size() > 0) processorRef.getPresetManager().savePreset(a[0].toString());
                  complete(juce::var(true));
              })
          .withNativeFunction(juce::Identifier("presetDelete"),
              [this](const juce::Array<juce::var>& a, juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                  if (a.size() > 0) processorRef.getPresetManager().deletePreset(a[0].toString());
                  complete(juce::var(true));
              })
          // --- L/R Decorrelate (a processor flag, not an APVTS parameter) ---
          .withNativeFunction(juce::Identifier("decorrelateGet"),
              [this](const juce::Array<juce::var>&, juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                  complete(juce::var(processorRef.getStereoDecorrelate()));
              })
          .withNativeFunction(juce::Identifier("decorrelateSet"),
              [this](const juce::Array<juce::var>& a, juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                  if (a.size() > 0) processorRef.setStereoDecorrelate((bool) a[0]);
                  complete(juce::var(true));
              })
          // Surface JS console/errors to native stderr (headless diagnosis).
          .withNativeFunction(juce::Identifier("jsLog"),
              [](const juce::Array<juce::var>& a, juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                  juce::String m; for (auto& v : a) m += v.toString() + " ";
                  std::cerr << "[webui] " << m << std::endl;
                  complete(juce::var());
              })
          .withResourceProvider([this](const auto& url) { return getResource(url); })),
      shiftHzAtt (*processorRef.getValueTreeState().getParameter("shiftHz"), shiftHzRelay, nullptr),
      quantizeStrengthAtt (*processorRef.getValueTreeState().getParameter("quantizeStrength"), quantizeStrengthRelay, nullptr),
      scaleNote0Att (*processorRef.getValueTreeState().getParameter("scaleNote0"), scaleNote0Relay, nullptr),
      scaleNote1Att (*processorRef.getValueTreeState().getParameter("scaleNote1"), scaleNote1Relay, nullptr),
      scaleNote2Att (*processorRef.getValueTreeState().getParameter("scaleNote2"), scaleNote2Relay, nullptr),
      scaleNote3Att (*processorRef.getValueTreeState().getParameter("scaleNote3"), scaleNote3Relay, nullptr),
      scaleNote4Att (*processorRef.getValueTreeState().getParameter("scaleNote4"), scaleNote4Relay, nullptr),
      scaleNote5Att (*processorRef.getValueTreeState().getParameter("scaleNote5"), scaleNote5Relay, nullptr),
      scaleNote6Att (*processorRef.getValueTreeState().getParameter("scaleNote6"), scaleNote6Relay, nullptr),
      scaleNote7Att (*processorRef.getValueTreeState().getParameter("scaleNote7"), scaleNote7Relay, nullptr),
      scaleNote8Att (*processorRef.getValueTreeState().getParameter("scaleNote8"), scaleNote8Relay, nullptr),
      scaleNote9Att (*processorRef.getValueTreeState().getParameter("scaleNote9"), scaleNote9Relay, nullptr),
      scaleNote10Att (*processorRef.getValueTreeState().getParameter("scaleNote10"), scaleNote10Relay, nullptr),
      scaleNote11Att (*processorRef.getValueTreeState().getParameter("scaleNote11"), scaleNote11Relay, nullptr),
      dryWetAtt (*processorRef.getValueTreeState().getParameter("dryWet"), dryWetRelay, nullptr),
      smearAtt (*processorRef.getValueTreeState().getParameter("smear"), smearRelay, nullptr),
      logScaleAtt (*processorRef.getValueTreeState().getParameter("logScale"), logScaleRelay, nullptr),
      lfoDepthAtt (*processorRef.getValueTreeState().getParameter("lfoDepth"), lfoDepthRelay, nullptr),
      lfoDepthModeAtt (*processorRef.getValueTreeState().getParameter("lfoDepthMode"), lfoDepthModeRelay, nullptr),
      lfoRateAtt (*processorRef.getValueTreeState().getParameter("lfoRate"), lfoRateRelay, nullptr),
      lfoSyncAtt (*processorRef.getValueTreeState().getParameter("lfoSync"), lfoSyncRelay, nullptr),
      lfoDivisionAtt (*processorRef.getValueTreeState().getParameter("lfoDivision"), lfoDivisionRelay, nullptr),
      lfoShapeAtt (*processorRef.getValueTreeState().getParameter("lfoShape"), lfoShapeRelay, nullptr),
      lfoEnabledAtt (*processorRef.getValueTreeState().getParameter("lfoEnabled"), lfoEnabledRelay, nullptr),
      dlyLfoDepthAtt (*processorRef.getValueTreeState().getParameter("dlyLfoDepth"), dlyLfoDepthRelay, nullptr),
      dlyLfoRateAtt (*processorRef.getValueTreeState().getParameter("dlyLfoRate"), dlyLfoRateRelay, nullptr),
      dlyLfoSyncAtt (*processorRef.getValueTreeState().getParameter("dlyLfoSync"), dlyLfoSyncRelay, nullptr),
      dlyLfoDivisionAtt (*processorRef.getValueTreeState().getParameter("dlyLfoDivision"), dlyLfoDivisionRelay, nullptr),
      dlyLfoShapeAtt (*processorRef.getValueTreeState().getParameter("dlyLfoShape"), dlyLfoShapeRelay, nullptr),
      dlyLfoEnabledAtt (*processorRef.getValueTreeState().getParameter("dlyLfoEnabled"), dlyLfoEnabledRelay, nullptr),
      maskEnabledAtt (*processorRef.getValueTreeState().getParameter("maskEnabled"), maskEnabledRelay, nullptr),
      maskModeAtt (*processorRef.getValueTreeState().getParameter("maskMode"), maskModeRelay, nullptr),
      maskLowFreqAtt (*processorRef.getValueTreeState().getParameter("maskLowFreq"), maskLowFreqRelay, nullptr),
      maskHighFreqAtt (*processorRef.getValueTreeState().getParameter("maskHighFreq"), maskHighFreqRelay, nullptr),
      maskTransitionAtt (*processorRef.getValueTreeState().getParameter("maskTransition"), maskTransitionRelay, nullptr),
      delayEnabledAtt (*processorRef.getValueTreeState().getParameter("delayEnabled"), delayEnabledRelay, nullptr),
      delayTimeAtt (*processorRef.getValueTreeState().getParameter("delayTime"), delayTimeRelay, nullptr),
      delaySyncAtt (*processorRef.getValueTreeState().getParameter("delaySync"), delaySyncRelay, nullptr),
      delayDivisionAtt (*processorRef.getValueTreeState().getParameter("delayDivision"), delayDivisionRelay, nullptr),
      delaySlopeAtt (*processorRef.getValueTreeState().getParameter("delaySlope"), delaySlopeRelay, nullptr),
      delayFeedbackAtt (*processorRef.getValueTreeState().getParameter("delayFeedback"), delayFeedbackRelay, nullptr),
      delayDampingAtt (*processorRef.getValueTreeState().getParameter("delayDamping"), delayDampingRelay, nullptr),
      delayGainAtt (*processorRef.getValueTreeState().getParameter("delayGain"), delayGainRelay, nullptr),
      preserveAtt (*processorRef.getValueTreeState().getParameter("preserve"), preserveRelay, nullptr),
      transientsAtt (*processorRef.getValueTreeState().getParameter("transients"), transientsRelay, nullptr),
      sensitivityAtt (*processorRef.getValueTreeState().getParameter("sensitivity"), sensitivityRelay, nullptr),
      peakSnapAtt (*processorRef.getValueTreeState().getParameter("peakSnap"), peakSnapRelay, nullptr),
      noiseMixAtt (*processorRef.getValueTreeState().getParameter("noiseMix"), noiseMixRelay, nullptr),
      peakSensAtt (*processorRef.getValueTreeState().getParameter("peakSens"), peakSensRelay, nullptr),
      processingModeAtt (*processorRef.getValueTreeState().getParameter("processingMode"), processingModeRelay, nullptr),
      warmAtt (*processorRef.getValueTreeState().getParameter("warm"), warmRelay, nullptr) {
    addAndMakeVisible(webView);
    webView.goToURL(juce::WebBrowserComponent::getResourceProviderRoot());
    // The web UI replicates the Visage design at a fixed 700x928 canvas and scales to fit.
    // Lock the resize to that aspect so the layout stays pixel-faithful at any size.
    setResizable(true, true);
    setResizeLimits(420, 557, 1750, 2320);
    if (auto* c = getConstrainer())
        c->setFixedAspectRatio(700.0 / 928.0);
    setSize(700, 928);
}

WebViewEditor::~WebViewEditor() = default;

void WebViewEditor::resized() { webView.setBounds(getLocalBounds()); }

std::optional<juce::WebBrowserComponent::Resource>
WebViewEditor::getResource(const juce::String& url) const {
    const auto filename = resolveFilename(url);
    const auto ext      = filename.fromLastOccurrenceOf(".", false, false);
    // Match by ORIGINAL filename — juce_add_binary_data mangles names unpredictably
    // (e.g. "pagan-background.png" -> "paganbackground_png"), so don't guess the symbol.
    for (int i = 0; i < BinaryData::namedResourceListSize; ++i) {
        if (filename == BinaryData::originalFilenames[i]) {
            int size = 0;
            if (const char* data = BinaryData::getNamedResource(BinaryData::namedResourceList[i], size)) {
                std::vector<std::byte> bytes(static_cast<size_t>(size));
                std::memcpy(bytes.data(), data, static_cast<size_t>(size));
                return juce::WebBrowserComponent::Resource{ std::move(bytes), juce::String(mimeForExtension(ext)) };
            }
        }
    }
    std::cerr << "[webui] resource NOT FOUND: " << filename << std::endl;
    return std::nullopt;
}
#endif // HOLY_SHIFTER_USE_WEBVIEW
