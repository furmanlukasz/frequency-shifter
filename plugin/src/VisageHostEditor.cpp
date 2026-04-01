#include "VisageHostEditor.h"
#include "PluginProcessor.h"

VisageHostEditor::VisageHostEditor(FrequencyShifterProcessor& p)
    : juce::AudioProcessorEditor(p), processor_(p)
{
    setSize(700, 928);
    setResizable(false, false);
    setOpaque(true);
}

VisageHostEditor::~VisageHostEditor()
{
    stopTimer();
    if (visageWindow_)
    {
        visageWindow_->close();
        visageWindow_ = nullptr;
    }
    ui_ = nullptr;
}

void VisageHostEditor::parentHierarchyChanged()
{
    if (windowShown_)
        return;

    auto* peer = getPeer();
    if (!peer)
        return;

    void* nativeHandle = peer->getNativeHandle();
    if (!nativeHandle)
        return;

    visageWindow_ = std::make_unique<visage::ApplicationWindow>();
    visageWindow_->setWindowDimensions(visage::Dimension(700), visage::Dimension(928));

    ui_ = std::make_unique<HolyShifterUI>(processor_);
    visageWindow_->addChild(ui_.get());
    ui_->layout().setMargin(0);

    visageWindow_->show(nativeHandle);
    windowShown_ = true;

    // Poll at 30Hz for DAW automation and preset changes
    startTimerHz(30);
}

void VisageHostEditor::timerCallback()
{
    if (ui_)
        ui_->pollState();
}
