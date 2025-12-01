#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "dsp/Scales.h"

FrequencyShifterProcessor::FrequencyShifterProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, juce::Identifier("FrequencyShifter"), createParameterLayout())
{
    // Add parameter listeners
    parameters.addParameterListener(PARAM_SHIFT_HZ, this);
    parameters.addParameterListener(PARAM_QUANTIZE_STRENGTH, this);
    parameters.addParameterListener(PARAM_ROOT_NOTE, this);
    parameters.addParameterListener(PARAM_SCALE_TYPE, this);
    parameters.addParameterListener(PARAM_DRY_WET, this);
    parameters.addParameterListener(PARAM_PHASE_VOCODER, this);

    // Initialize quantizer with default scale (C Major)
    quantizer = std::make_unique<fshift::MusicalQuantizer>(60, fshift::ScaleType::Major);
}

FrequencyShifterProcessor::~FrequencyShifterProcessor()
{
    parameters.removeParameterListener(PARAM_SHIFT_HZ, this);
    parameters.removeParameterListener(PARAM_QUANTIZE_STRENGTH, this);
    parameters.removeParameterListener(PARAM_ROOT_NOTE, this);
    parameters.removeParameterListener(PARAM_SCALE_TYPE, this);
    parameters.removeParameterListener(PARAM_DRY_WET, this);
    parameters.removeParameterListener(PARAM_PHASE_VOCODER, this);
}

juce::AudioProcessorValueTreeState::ParameterLayout FrequencyShifterProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Frequency shift (-2000 to +2000 Hz)
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ PARAM_SHIFT_HZ, 1 },
        "Shift (Hz)",
        juce::NormalisableRange<float>(-2000.0f, 2000.0f, 0.1f),
        0.0f,
        juce::AudioParameterFloatAttributes().withLabel("Hz")));

    // Quantize strength (0-100%)
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ PARAM_QUANTIZE_STRENGTH, 1 },
        "Quantize",
        juce::NormalisableRange<float>(0.0f, 100.0f, 0.1f),
        0.0f,
        juce::AudioParameterFloatAttributes().withLabel("%")));

    // Root note (C0 to B8, MIDI 12-119)
    juce::StringArray noteNames;
    for (int octave = 0; octave <= 8; ++octave)
    {
        for (const auto& note : { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" })
        {
            noteNames.add(juce::String(note) + juce::String(octave));
        }
    }
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ PARAM_ROOT_NOTE, 1 },
        "Root Note",
        noteNames,
        48));  // Default to C4 (index 48 = MIDI 60)

    // Scale type
    juce::StringArray scaleNames;
    for (const auto& name : fshift::getScaleNames())
    {
        scaleNames.add(name);
    }
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{ PARAM_SCALE_TYPE, 1 },
        "Scale",
        scaleNames,
        0));  // Default to Major

    // Dry/Wet mix (0-100%)
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{ PARAM_DRY_WET, 1 },
        "Dry/Wet",
        juce::NormalisableRange<float>(0.0f, 100.0f, 0.1f),
        100.0f,
        juce::AudioParameterFloatAttributes().withLabel("%")));

    // Phase vocoder toggle
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{ PARAM_PHASE_VOCODER, 1 },
        "Enhanced Mode",
        true));

    return { params.begin(), params.end() };
}

void FrequencyShifterProcessor::parameterChanged(const juce::String& parameterID, float newValue)
{
    if (parameterID == PARAM_SHIFT_HZ)
    {
        shiftHz.store(newValue);
    }
    else if (parameterID == PARAM_QUANTIZE_STRENGTH)
    {
        quantizeStrength.store(newValue / 100.0f);
    }
    else if (parameterID == PARAM_ROOT_NOTE)
    {
        int midiNote = static_cast<int>(newValue) + 12;  // Convert index to MIDI (C0 = 12)
        rootNote.store(midiNote);
        if (quantizer)
        {
            quantizer->setRootNote(midiNote);
        }
    }
    else if (parameterID == PARAM_SCALE_TYPE)
    {
        int scale = static_cast<int>(newValue);
        scaleType.store(scale);
        if (quantizer)
        {
            quantizer->setScaleType(static_cast<fshift::ScaleType>(scale));
        }
    }
    else if (parameterID == PARAM_DRY_WET)
    {
        dryWetMix.store(newValue / 100.0f);
    }
    else if (parameterID == PARAM_PHASE_VOCODER)
    {
        usePhaseVocoder.store(newValue > 0.5f);
    }
}

void FrequencyShifterProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    currentBlockSize = samplesPerBlock;

    const int numChannels = getTotalNumInputChannels();

    // Initialize DSP components for each channel
    for (int ch = 0; ch < std::min(numChannels, MAX_CHANNELS); ++ch)
    {
        stftProcessors[ch] = std::make_unique<fshift::STFT>(FFT_SIZE, HOP_SIZE);
        stftProcessors[ch]->prepare(sampleRate);

        phaseVocoders[ch] = std::make_unique<fshift::PhaseVocoder>(FFT_SIZE, HOP_SIZE, sampleRate);

        frequencyShifters[ch] = std::make_unique<fshift::FrequencyShifter>(sampleRate, FFT_SIZE);

        // Initialize overlap-add buffers
        inputBuffers[ch].resize(FFT_SIZE * 2, 0.0f);
        outputBuffers[ch].resize(FFT_SIZE * 2, 0.0f);
        inputWritePos[ch] = 0;
        outputReadPos[ch] = 0;
    }
}

void FrequencyShifterProcessor::releaseResources()
{
    for (int ch = 0; ch < MAX_CHANNELS; ++ch)
    {
        stftProcessors[ch].reset();
        phaseVocoders[ch].reset();
        frequencyShifters[ch].reset();
        inputBuffers[ch].clear();
        outputBuffers[ch].clear();
    }
}

bool FrequencyShifterProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    // Support mono and stereo
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // Input must match output
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;

    return true;
}

void FrequencyShifterProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    // Get current parameter values
    const float currentShiftHz = shiftHz.load();
    const float currentQuantizeStrength = quantizeStrength.load();
    const float currentDryWet = dryWetMix.load();
    const bool currentUsePhaseVocoder = usePhaseVocoder.load();

    // If no processing needed, just pass through
    if (std::abs(currentShiftHz) < 0.01f && currentQuantizeStrength < 0.01f)
    {
        return;
    }

    // Process each channel
    for (int channel = 0; channel < std::min(numChannels, MAX_CHANNELS); ++channel)
    {
        auto* channelData = buffer.getWritePointer(channel);

        // Store dry signal for mixing
        std::vector<float> drySignal(channelData, channelData + numSamples);

        // Process through STFT pipeline
        for (int i = 0; i < numSamples; ++i)
        {
            // Write input sample to circular buffer
            inputBuffers[channel][inputWritePos[channel]] = channelData[i];
            inputWritePos[channel] = (inputWritePos[channel] + 1) % inputBuffers[channel].size();

            // Check if we have enough samples for an FFT frame
            if (inputWritePos[channel] % HOP_SIZE == 0)
            {
                // Get input frame
                std::vector<float> inputFrame(FFT_SIZE);
                int readPos = (inputWritePos[channel] - FFT_SIZE + inputBuffers[channel].size())
                              % inputBuffers[channel].size();
                for (int j = 0; j < FFT_SIZE; ++j)
                {
                    inputFrame[j] = inputBuffers[channel][(readPos + j) % inputBuffers[channel].size()];
                }

                // Perform STFT
                auto [magnitude, phase] = stftProcessors[channel]->forward(inputFrame);

                // Apply phase vocoder if enabled
                if (currentUsePhaseVocoder && std::abs(currentShiftHz) > 0.01f)
                {
                    phase = phaseVocoders[channel]->process(magnitude, phase, currentShiftHz);
                }

                // Apply frequency shifting
                if (std::abs(currentShiftHz) > 0.01f)
                {
                    std::tie(magnitude, phase) = frequencyShifters[channel]->shift(magnitude, phase, currentShiftHz);
                }

                // Apply musical quantization
                if (currentQuantizeStrength > 0.01f && quantizer)
                {
                    std::tie(magnitude, phase) = quantizer->quantizeSpectrum(
                        magnitude, phase, currentSampleRate, FFT_SIZE, currentQuantizeStrength);
                }

                // Perform inverse STFT
                auto outputFrame = stftProcessors[channel]->inverse(magnitude, phase);

                // Overlap-add to output buffer
                int writePos = (outputReadPos[channel] + i) % outputBuffers[channel].size();
                for (int j = 0; j < FFT_SIZE; ++j)
                {
                    int pos = (writePos + j) % outputBuffers[channel].size();
                    outputBuffers[channel][pos] += outputFrame[j];
                }
            }

            // Read from output buffer
            channelData[i] = outputBuffers[channel][outputReadPos[channel]];
            outputBuffers[channel][outputReadPos[channel]] = 0.0f;  // Clear for next overlap-add
            outputReadPos[channel] = (outputReadPos[channel] + 1) % outputBuffers[channel].size();
        }

        // Apply dry/wet mix
        if (currentDryWet < 0.99f)
        {
            for (int i = 0; i < numSamples; ++i)
            {
                channelData[i] = drySignal[i] * (1.0f - currentDryWet) + channelData[i] * currentDryWet;
            }
        }
    }
}

double FrequencyShifterProcessor::getTailLengthSeconds() const
{
    // Latency from FFT processing
    return static_cast<double>(FFT_SIZE + HOP_SIZE) / currentSampleRate;
}

juce::AudioProcessorEditor* FrequencyShifterProcessor::createEditor()
{
    return new FrequencyShifterEditor(*this);
}

void FrequencyShifterProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = parameters.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void FrequencyShifterProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState != nullptr && xmlState->hasTagName(parameters.state.getType()))
    {
        parameters.replaceState(juce::ValueTree::fromXml(*xmlState));
    }
}

// Plugin instantiation
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new FrequencyShifterProcessor();
}
