#pragma once

#include <JuceHeader.h>
#include "dsp/STFT.h"
#include "dsp/PhaseVocoder.h"
#include "dsp/FrequencyShifter.h"
#include "dsp/MusicalQuantizer.h"

/**
 * FrequencyShifterProcessor - Main audio processor for the Frequency Shifter plugin.
 *
 * This processor implements harmonic-preserving frequency shifting with:
 * - Enhanced phase vocoder for artifact reduction
 * - Musical scale quantization
 * - Stereo processing support
 */
class FrequencyShifterProcessor : public juce::AudioProcessor,
                                   public juce::AudioProcessorValueTreeState::Listener
{
public:
    FrequencyShifterProcessor();
    ~FrequencyShifterProcessor() override;

    // AudioProcessor interface
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }

    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override;

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // Parameter listener
    void parameterChanged(const juce::String& parameterID, float newValue) override;

    // Parameter tree
    juce::AudioProcessorValueTreeState& getValueTreeState() { return parameters; }

    // Parameter IDs
    static constexpr const char* PARAM_SHIFT_HZ = "shiftHz";
    static constexpr const char* PARAM_QUANTIZE_STRENGTH = "quantizeStrength";
    static constexpr const char* PARAM_ROOT_NOTE = "rootNote";
    static constexpr const char* PARAM_SCALE_TYPE = "scaleType";
    static constexpr const char* PARAM_DRY_WET = "dryWet";
    static constexpr const char* PARAM_PHASE_VOCODER = "phaseVocoder";

private:
    // Create parameter layout
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    // Process a single channel
    void processChannel(int channel, juce::AudioBuffer<float>& buffer);

    // Parameter tree state
    juce::AudioProcessorValueTreeState parameters;

    // DSP components (per channel for stereo)
    static constexpr int MAX_CHANNELS = 2;
    std::array<std::unique_ptr<fshift::STFT>, MAX_CHANNELS> stftProcessors;
    std::array<std::unique_ptr<fshift::PhaseVocoder>, MAX_CHANNELS> phaseVocoders;
    std::array<std::unique_ptr<fshift::FrequencyShifter>, MAX_CHANNELS> frequencyShifters;
    std::unique_ptr<fshift::MusicalQuantizer> quantizer;

    // Processing parameters (atomic for thread safety)
    std::atomic<float> shiftHz{ 0.0f };
    std::atomic<float> quantizeStrength{ 0.0f };
    std::atomic<float> dryWetMix{ 1.0f };
    std::atomic<bool> usePhaseVocoder{ true };
    std::atomic<int> rootNote{ 60 };  // C4
    std::atomic<int> scaleType{ 0 };  // Major

    // Processing state
    double currentSampleRate = 44100.0;
    int currentBlockSize = 512;

    // FFT settings
    static constexpr int FFT_SIZE = 4096;
    static constexpr int HOP_SIZE = 1024;

    // Input/output buffers for overlap-add
    std::array<std::vector<float>, MAX_CHANNELS> inputBuffers;
    std::array<std::vector<float>, MAX_CHANNELS> outputBuffers;
    std::array<int, MAX_CHANNELS> inputWritePos{};
    std::array<int, MAX_CHANNELS> outputReadPos{};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(FrequencyShifterProcessor)
};
