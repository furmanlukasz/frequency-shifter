#pragma once

#include <cmath>
#include <array>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fshift {

/**
 * HilbertShifter - Classic SSB/Hilbert frequency shifter with double-precision processing.
 *
 * Uses allpass filter networks to create quadrature signals (I/Q with 90° phase difference),
 * then applies single sideband modulation for frequency shifting.
 *
 * v104: Upgraded to 8-stage allpass chains with sample-rate adaptive coefficients.
 * Coefficients generated using the Weaver/Darlington phase-difference network design
 * with frequency prewarping for bilinear transform accuracy.
 * Sideband rejection: ~39 dB at 44.1kHz, ~46 dB at 48kHz, ~58+ dB at 88.2kHz+.
 * (Previous 6-stage design with wrong sign convention had ~0.2 dB rejection.)
 *
 * Uses double precision for allpass filter states and internal processing
 * to prevent quantization error accumulation in deep feedback loops.
 *
 * Advantages over spectral methods:
 * - Zero latency (aside from filter group delay ~10-20 samples)
 * - No smearing/time domain artifacts
 * - Simple and CPU efficient
 *
 * Limitations:
 * - Cannot preserve harmonic relationships (inharmonic shifting)
 * - Best for subtle shifts or creative effects
 */
class HilbertShifter
{
public:
    HilbertShifter() = default;

    void prepare(double sr)
    {
        sampleRate = sr;
        selectCoefficients(sr);
        reset();
    }

    void reset()
    {
        // Reset allpass filter states for all channels (double precision)
        for (int ch = 0; ch < MAX_CHANNELS; ++ch)
        {
            for (auto& state : allpassStatesI[ch])
                state = 0.0;
            for (auto& state : allpassStatesQ[ch])
                state = 0.0;
        }

        // Reset oscillator
        oscPhase = 0.0;
    }

    /**
     * Set the frequency shift amount in Hz.
     * Positive values shift up, negative values shift down.
     */
    void setShiftHz(float hz)
    {
        shiftHz = hz;
    }

    /**
     * Process a single sample and return the frequency-shifted output.
     * Uses default channel 0 for backwards compatibility.
     */
    float process(float input)
    {
        return process(input, 0);
    }

    /**
     * Process a single sample for a specific channel.
     * @param input The input sample
     * @param channel The channel index (0 = left, 1 = right)
     * @return The frequency-shifted output
     */
    float process(float input, int channel)
    {
        // Note: The channel parameter is ignored since each HilbertShifter instance
        // is dedicated to a single audio channel. We always use internal index 0.
        (void)channel;  // Suppress unused parameter warning

        // Generate quadrature signals using Hilbert transform (allpass networks)
        // Uses double precision internally for accurate feedback loops
        double I = processAllpassChainI(static_cast<double>(input), 0);
        double Q = processAllpassChainQ(static_cast<double>(input), 0);

        // Generate quadrature oscillator signals (already double precision)
        double cosOsc = std::cos(oscPhase);
        double sinOsc = std::sin(oscPhase);

        // Single sideband modulation (double precision).
        // Convention: Q lags I by 90° → I*cos - Q*sin shifts UP.
        //
        // A single formula handles BOTH shift directions because the oscillator
        // phase is advanced with the SIGNED shift (see below): for a downward
        // shift the phase runs backwards, and since sin() is odd this is exactly
        // equivalent to the old "+ Q*sin" lower-sideband branch. Crucially, it
        // stays continuous when the shift is modulated through 0 Hz. The previous
        // sign branch + abs() phase flipped the sign of the Q*sin term at the
        // crossing, producing a step of 2*Q*sin(phase) — an audible click on
        // sustained material under fast/synced LFO modulation.
        double output = I * cosOsc - Q * sinOsc;

        // Advance oscillator phase using the SIGNED shift frequency, so the
        // oscillator decelerates to a stop at 0 Hz and reverses direction rather
        // than jumping sidebands. The phase wrap below already handles the
        // negative case. (Each HilbertShifter instance handles one audio channel,
        // so we always advance the oscillator regardless of the channel param.)
        double phaseIncrement = 2.0 * M_PI * shiftHz / sampleRate;
        oscPhase += phaseIncrement;

        // Wrap phase to prevent numerical issues
        while (oscPhase >= 2.0 * M_PI)
            oscPhase -= 2.0 * M_PI;
        while (oscPhase < 0.0)
            oscPhase += 2.0 * M_PI;

        return static_cast<float>(output);
    }

    /**
     * Get the current oscillator phase (0 to 2π).
     * Useful for visualization.
     */
    double getOscillatorPhase() const { return oscPhase; }

private:
    double sampleRate = 44100.0;
    float shiftHz = 0.0f;
    double oscPhase = 0.0;

    // ===== 8-STAGE ALLPASS COEFFICIENTS =====
    // Generated using Weaver/Darlington phase-difference network design
    // with frequency prewarping for bilinear transform accuracy.
    // Passband: 20 Hz to 20000 Hz.
    //
    // NOTE: Coefficients are SWAPPED from generator output (A→Q, B→I)
    // so that Q lags I by 90°, matching the SSB formula convention.
    // Coefficients include negative values — this is correct for the
    // allpass form H(z) = (a + z^-1) / (1 + a*z^-1) with |a| < 1.

    static constexpr int HILBERT_ORDER = 8;

    // --- 44100 Hz (max phase error: 0.68°, rejection: ~39 dB) ---
    static constexpr std::array<double, HILBERT_ORDER> coeffsI_44100 = {{
        -0.9966789239045,
        -0.9876655104021,
        -0.9591616413744,
        -0.8656198275171,
        -0.5991725301098,
        -0.0784460758477,
         0.4852575252917,
         0.9012582586783
    }};
    static constexpr std::array<double, HILBERT_ORDER> coeffsQ_44100 = {{
        -0.9989943547547,
        -0.9933083026731,
        -0.9775825672554,
        -0.9255827529740,
        -0.7629203804023,
        -0.3654644516380,
         0.2190471481496,
         0.7069061720752
    }};

    // --- 48000 Hz (max phase error: 0.30°, rejection: ~46 dB) ---
    static constexpr std::array<double, HILBERT_ORDER> coeffsI_48000 = {{
        -0.9972486015686,
        -0.9899809538902,
        -0.9683227895730,
        -0.9010513844926,
        -0.7107743687347,
        -0.2963502523252,
         0.2776863462778,
         0.8425127789381
    }};
    static constexpr std::array<double, HILBERT_ORDER> coeffsQ_48000 = {{
        -0.9991652240544,
        -0.9944916665440,
        -0.9821601585862,
        -0.9438304725847,
        -0.8283911536319,
        -0.5342672052652,
        -0.0150704629855,
         0.5600736032347
    }};

    // --- 88200 Hz (max phase error: 0.07°, rejection: ~58 dB) ---
    static constexpr std::array<double, HILBERT_ORDER> coeffsI_88200 = {{
        -0.9987130319856,
        -0.9955359024489,
        -0.9869962083543,
        -0.9627146696895,
        -0.8953773209802,
        -0.7238033532775,
        -0.3457121711835,
         0.5148760283284
    }};
    static constexpr std::array<double, HILBERT_ORDER> coeffsQ_88200 = {{
        -0.9996060134519,
        -0.9974723006178,
        -0.9923493886848,
        -0.9779531191518,
        -0.9372566571353,
        -0.8281316811565,
        -0.5685849970816,
        -0.0227410040798
    }};

    // --- 96000 Hz (max phase error: 0.07°, rejection: ~58 dB) ---
    static constexpr std::array<double, HILBERT_ORDER> coeffsI_96000 = {{
        -0.9988241980115,
        -0.9959301373732,
        -0.9881821956105,
        -0.9662171519291,
        -0.9052879239636,
        -0.7486770844140,
        -0.3939080347309,
         0.4721174503023
    }};
    static constexpr std::array<double, HILBERT_ORDER> coeffsQ_96000 = {{
        -0.9996398894814,
        -0.9976926502140,
        -0.9930356311698,
        -0.9799963695316,
        -0.9432008187435,
        -0.8441837511359,
        -0.6047571112298,
        -0.0788954355798
    }};

    // --- 176400 Hz (max phase error: 0.06°, rejection: ~60 dB) ---
    static constexpr std::array<double, HILBERT_ORDER> coeffsI_176400 = {{
        -0.9993721967375,
        -0.9978425579323,
        -0.9938006601043,
        -0.9824377485686,
        -0.9507349481540,
        -0.8656266628239,
        -0.6452120016425,
         0.1583595547646
    }};
    static constexpr std::array<double, HILBERT_ORDER> coeffsQ_176400 = {{
        -0.9998074407682,
        -0.9987716868814,
        -0.9963268704857,
        -0.9895597891956,
        -0.9705222946350,
        -0.9182415709415,
        -0.7813881815732,
        -0.4064211465907
    }};

    // --- 192000 Hz (max phase error: 0.06°, rejection: ~60 dB) ---
    static constexpr std::array<double, HILBERT_ORDER> coeffsI_192000 = {{
        -0.9994238614054,
        -0.9980209306977,
        -0.9943164920362,
        -0.9839053209639,
        -0.9548306715211,
        -0.8764450969633,
        -0.6708172725981,
         0.1138815144184
    }};
    static constexpr std::array<double, HILBERT_ORDER> coeffsQ_192000 = {{
        -0.9998232724182,
        -0.9988729659728,
        -0.9966315219795,
        -0.9904308684346,
        -0.9729839352838,
        -0.9249655861138,
        -0.7983187099069,
        -0.4435097678397
    }};

    // Active coefficients (set by selectCoefficients)
    std::array<double, HILBERT_ORDER> coeffsI = {};
    std::array<double, HILBERT_ORDER> coeffsQ = {};

    // Allpass filter states - DOUBLE PRECISION for accurate feedback loops
    static constexpr int MAX_CHANNELS = 2;
    std::array<std::array<double, HILBERT_ORDER>, MAX_CHANNELS> allpassStatesI = {};
    std::array<std::array<double, HILBERT_ORDER>, MAX_CHANNELS> allpassStatesQ = {};

    /**
     * Select coefficient tables based on sample rate.
     */
    void selectCoefficients(double sr)
    {
        if (sr <= 44200.0) {
            std::copy(coeffsI_44100.begin(), coeffsI_44100.end(), coeffsI.begin());
            std::copy(coeffsQ_44100.begin(), coeffsQ_44100.end(), coeffsQ.begin());
        } else if (sr <= 48100.0) {
            std::copy(coeffsI_48000.begin(), coeffsI_48000.end(), coeffsI.begin());
            std::copy(coeffsQ_48000.begin(), coeffsQ_48000.end(), coeffsQ.begin());
        } else if (sr <= 88300.0) {
            std::copy(coeffsI_88200.begin(), coeffsI_88200.end(), coeffsI.begin());
            std::copy(coeffsQ_88200.begin(), coeffsQ_88200.end(), coeffsQ.begin());
        } else if (sr <= 96100.0) {
            std::copy(coeffsI_96000.begin(), coeffsI_96000.end(), coeffsI.begin());
            std::copy(coeffsQ_96000.begin(), coeffsQ_96000.end(), coeffsQ.begin());
        } else if (sr <= 176500.0) {
            std::copy(coeffsI_176400.begin(), coeffsI_176400.end(), coeffsI.begin());
            std::copy(coeffsQ_176400.begin(), coeffsQ_176400.end(), coeffsQ.begin());
        } else {
            std::copy(coeffsI_192000.begin(), coeffsI_192000.end(), coeffsI.begin());
            std::copy(coeffsQ_192000.begin(), coeffsQ_192000.end(), coeffsQ.begin());
        }
    }

    /**
     * Process through the I-channel allpass chain for a specific channel.
     * First-order allpass transfer function: H(z) = (a + z^-1) / (1 + a*z^-1)
     * Direct form: y[n] = a * x[n] + state; state = x[n] - a * y[n]
     *
     * Uses double precision throughout to prevent quantization errors
     * from accumulating in deep feedback loops.
     */
    double processAllpassChainI(double input, int channel)
    {
        double x = input;
        for (size_t i = 0; i < coeffsI.size(); ++i)
        {
            double a = coeffsI[i];
            double output = a * x + allpassStatesI[channel][i];
            allpassStatesI[channel][i] = x - a * output;
            x = output;
        }
        return x;
    }

    /**
     * Process through the Q-channel allpass chain for a specific channel.
     * Uses double precision throughout.
     */
    double processAllpassChainQ(double input, int channel)
    {
        double x = input;
        for (size_t i = 0; i < coeffsQ.size(); ++i)
        {
            double a = coeffsQ[i];
            double output = a * x + allpassStatesQ[channel][i];
            allpassStatesQ[channel][i] = x - a * output;
            x = output;
        }
        return x;
    }
};

} // namespace fshift
