#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "dsp/HilbertShifter.h"

#include <algorithm>
#include <cmath>

using Catch::Approx;

namespace {

constexpr double kTwoPi = 6.283185307179586476925287;

// Bit-for-bit replica of the PRE-FIX oscillator + SSB mixing: the oscillator
// phase advances with abs(shift) and the sideband is selected by a sign branch.
// This is exactly what the shipping HilbertShifter::process used to do.
struct LegacyMixer
{
    double sampleRate;
    double phase = 0.0;

    double process(double I, double Q, double shiftHz)
    {
        const double c = std::cos(phase);
        const double s = std::sin(phase);
        const double out = (shiftHz >= 0.0) ? (I * c - Q * s)    // upper sideband
                                            : (I * c + Q * s);   // lower sideband
        phase += kTwoPi * std::abs(shiftHz) / sampleRate;
        while (phase >= kTwoPi) phase -= kTwoPi;
        while (phase < 0.0)     phase += kTwoPi;
        return out;
    }
};

// Replica of the CURRENT (fixed) mixing now used by HilbertShifter::process:
// one formula, with a SIGNED phase increment.
struct SignedMixer
{
    double sampleRate;
    double phase = 0.0;

    double process(double I, double Q, double shiftHz)
    {
        const double c = std::cos(phase);
        const double s = std::sin(phase);
        const double out = I * c - Q * s;
        phase += kTwoPi * shiftHz / sampleRate;   // signed: no abs()
        while (phase >= kTwoPi) phase -= kTwoPi;
        while (phase < 0.0)     phase += kTwoPi;
        return out;
    }
};

}  // namespace

TEST_CASE("HilbertShifter: signed-phase refactor preserves output for a constant shift",
          "[hilbert][equivalence]")
{
    // For a constant shift, the new single-formula / signed-phase mixing is
    // mathematically identical to the old branch / abs-phase mixing, because
    // cos is even and sin is odd. The I/Q (Hilbert) generation path is
    // untouched, so identical mixing for a fixed shift => identical audio =>
    // the shifter's steady-state tone is unchanged. Both shift directions are
    // checked.
    const double sr = 48000.0;

    for (double shift : {+137.0, -137.0, +1000.0, -1000.0})
    {
        LegacyMixer oldM{sr};
        SignedMixer newM{sr};

        double maxDiff = 0.0;
        for (int n = 0; n < 20000; ++n)
        {
            // Arbitrary, non-degenerate I/Q content; the identity holds for any.
            const double t = static_cast<double>(n) / sr;
            const double I = 0.7 * std::sin(kTwoPi * 213.0 * t) + 0.3 * std::sin(kTwoPi * 517.0 * t);
            const double Q = 0.7 * std::cos(kTwoPi * 213.0 * t) + 0.3 * std::cos(kTwoPi * 517.0 * t);

            const double a = oldM.process(I, Q, shift);
            const double b = newM.process(I, Q, shift);
            maxDiff = std::max(maxDiff, std::abs(a - b));
        }

        INFO("shift = " << shift << " Hz");
        REQUIRE(maxDiff < 1e-7);
    }
}

TEST_CASE("HilbertShifter: legacy mixing clicks at a shift sign flip; signed phase does not",
          "[hilbert][click]")
{
    // Cross 0 Hz as an instantaneous sign flip of the shift (what a block-rate
    // LFO does at a block boundary). The legacy mixing flips the sign of the
    // Q*sin term and jumps by ~2*Q*sin(phase); the signed-phase mixing keeps the
    // oscillator phase continuous, so the output only moves by its natural slew.
    const double sr = 48000.0;
    const double shiftMag = 100.0;
    const int    flipAt   = 120;   // osc phase ~ pi/2 here => sin ~ 1 => large jump
    const int    N        = 400;

    LegacyMixer oldM{sr};
    SignedMixer newM{sr};

    double prevOld = 0.0, prevNew = 0.0;
    double maxDeltaOld = 0.0, maxDeltaNew = 0.0;

    for (int n = 0; n < N; ++n)
    {
        const double t = static_cast<double>(n) / sr;
        const double I = std::sin(kTwoPi * 50.0 * t);
        const double Q = std::cos(kTwoPi * 50.0 * t);
        const double shift = (n < flipAt) ? +shiftMag : -shiftMag;

        const double a = oldM.process(I, Q, shift);
        const double b = newM.process(I, Q, shift);
        if (n > 0)
        {
            maxDeltaOld = std::max(maxDeltaOld, std::abs(a - prevOld));
            maxDeltaNew = std::max(maxDeltaNew, std::abs(b - prevNew));
        }
        prevOld = a;
        prevNew = b;
    }

    INFO("maxDeltaOld = " << maxDeltaOld << ", maxDeltaNew = " << maxDeltaNew);
    REQUIRE(maxDeltaOld > 0.5);    // legacy: audible discontinuity (the click)
    REQUIRE(maxDeltaNew < 0.05);   // fixed: only the signal's natural slew
}

TEST_CASE("HilbertShifter: real shifter sweeps through 0 Hz without a click",
          "[hilbert][click][integration]")
{
    // Drive the actual class through its public API: a sustained tone while the
    // shift ramps +120 -> -120 Hz (through zero). A 200 Hz tone shifted by
    // <= 120 Hz stays below ~320 Hz, so at 0.5 amplitude its per-sample slew is
    // under ~0.03. The pre-fix zero-crossing click was order ~1.0.
    const double sr = 48000.0;
    fshift::HilbertShifter sh;
    sh.prepare(sr);
    sh.reset();

    double inPhase = 0.0;
    auto nextInput = [&]() -> float {
        const float v = 0.5f * static_cast<float>(std::sin(inPhase));
        inPhase += kTwoPi * 200.0 / sr;
        if (inPhase >= kTwoPi) inPhase -= kTwoPi;
        return v;
    };

    // Let the allpass Hilbert network settle at a constant shift first.
    for (int n = 0; n < 8192; ++n)
    {
        sh.setShiftHz(120.0f);
        (void) sh.process(nextInput(), 0);
    }

    const int N = 48000;
    double prev = 0.0;
    double maxDelta = 0.0;
    bool allFinite = true;
    for (int n = 0; n < N; ++n)
    {
        const float shift = 120.0f - 240.0f * (static_cast<float>(n) / static_cast<float>(N));
        sh.setShiftHz(shift);
        const float out = sh.process(nextInput(), 0);
        allFinite = allFinite && std::isfinite(out);
        if (n > 0)
            maxDelta = std::max(maxDelta, std::abs(static_cast<double>(out) - prev));
        prev = out;
    }

    INFO("maxDelta across the through-zero sweep = " << maxDelta);
    REQUIRE(allFinite);
    REQUIRE(maxDelta < 0.1);
}

TEST_CASE("HilbertShifter: real shifter is smooth and non-trivial at a constant shift",
          "[hilbert][integration]")
{
    // Sanity that the edited class still shifts cleanly in steady state, both up
    // and down: output is finite, carries signal (not silence), and is smooth.
    const double sr = 48000.0;

    for (float shift : {+150.0f, -150.0f})
    {
        fshift::HilbertShifter sh;
        sh.prepare(sr);
        sh.reset();
        sh.setShiftHz(shift);

        double inPhase = 0.0;
        double prev = 0.0;
        double maxDelta = 0.0;
        double sumSq = 0.0;
        bool allFinite = true;
        const int N = 20000;
        const int warmup = 4096;

        for (int n = 0; n < N; ++n)
        {
            const float in = 0.5f * static_cast<float>(std::sin(inPhase));
            inPhase += kTwoPi * 440.0 / sr;
            if (inPhase >= kTwoPi) inPhase -= kTwoPi;

            const float out = sh.process(in, 0);
            allFinite = allFinite && std::isfinite(out);
            if (n >= warmup)
            {
                if (n > warmup)
                    maxDelta = std::max(maxDelta, std::abs(static_cast<double>(out) - prev));
                sumSq += static_cast<double>(out) * out;
                prev = out;
            }
        }

        const double rms = std::sqrt(sumSq / static_cast<double>(N - warmup));
        INFO("shift = " << shift << " Hz, rms = " << rms << ", maxDelta = " << maxDelta);
        REQUIRE(allFinite);
        REQUIRE(rms > 0.1);        // passes signal, not silence
        REQUIRE(maxDelta < 0.1);   // constant-shift output is smooth (no clicks)
    }
}
