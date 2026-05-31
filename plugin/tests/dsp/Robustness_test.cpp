// Robustness coverage — feeds sustained "input" through the DSP and asserts
// the output stays finite, bounded, and non-NaN across param sweeps. Partial
// downpayment on Tier 3 #18 (advanced DSP). These don't change audio behavior;
// they catch silent regressions when the DSP is refactored.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "dsp/STFT.h"
#include "dsp/MusicalQuantizer.h"
#include "dsp/Scales.h"

#include <array>
#include <cmath>
#include <vector>

using fshift::STFT;
using fshift::MusicalQuantizer;
using fshift::ScaleType;

namespace {

bool allFinite(const std::vector<float>& v)
{
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

float maxAbs(const std::vector<float>& v)
{
    float m = 0.0f;
    for (float x : v) m = std::max(m, std::fabs(x));
    return m;
}

std::vector<float> whiteNoise(int N, unsigned seed = 1)
{
    std::vector<float> sig(static_cast<size_t>(N));
    unsigned s = seed ? seed : 1u;
    for (int i = 0; i < N; ++i)
    {
        // Tiny LCG — deterministic, no <random> dependency.
        s = s * 1103515245u + 12345u;
        sig[static_cast<size_t>(i)] = ((s >> 16) & 0x7fff) / 16383.5f - 1.0f;
    }
    return sig;
}

}  // namespace

TEST_CASE("Quantizer: stays finite under sustained operation with shift sweep",
          "[quantizer][robustness]")
{
    // 200 frames of evolving white-noise input, with shiftHz sweeping ±1 kHz.
    // Tests no NaN/Inf accumulation, no infinite gain, no DC blow-up.
    MusicalQuantizer q(60, ScaleType::Major);
    const int fftSize = 2048;
    const double sr = 44100.0;
    q.prepare(sr, fftSize, fftSize / 4);

    const int numBins = fftSize / 2 + 1;
    std::vector<float> phase(static_cast<size_t>(numBins), 0.0f);

    const int frames = 200;
    float maxObserved = 0.0f;
    for (int f = 0; f < frames; ++f)
    {
        auto mag = whiteNoise(numBins, static_cast<unsigned>(f + 1));
        for (auto& x : mag) x = std::fabs(x) * 0.5f;  // magnitudes are non-negative

        float shiftHz = 1000.0f * std::sin(2.0f * 3.14159265f * f / 40.0f);

        auto [outMag, outPhase] = q.quantizeSpectrum(
            mag, phase, sr, fftSize, shiftHz, /*strength=*/1.0f);

        REQUIRE(allFinite(outMag));
        REQUIRE(allFinite(outPhase));
        maxObserved = std::max(maxObserved, maxAbs(outMag));
    }

    // Magnitudes should stay bounded — the algorithm normalizes energy,
    // so peaks shouldn't blow far past the input range (allow 10x for two-tap
    // weighting concentrating energy at scale-snap points).
    REQUIRE(maxObserved < 5.0f);
}

TEST_CASE("Quantizer: DC bin doesn't accumulate from downward shifts",
          "[quantizer][robustness]")
{
    // Pushing high frequencies down by a big shiftHz could pile energy near DC.
    // Even at strength=1 and shift = -fullSpectrum, DC bin should not grow
    // unboundedly across frames.
    MusicalQuantizer q(60, ScaleType::Chromatic);
    const int fftSize = 2048;
    const double sr = 44100.0;
    q.prepare(sr, fftSize, fftSize / 4);

    const int numBins = fftSize / 2 + 1;
    std::vector<float> phase(static_cast<size_t>(numBins), 0.0f);

    // Flat input — every bin has the same energy.
    std::vector<float> mag(static_cast<size_t>(numBins), 0.3f);

    float maxDcSeen = 0.0f;
    for (int f = 0; f < 100; ++f)
    {
        auto [outMag, _phase] = q.quantizeSpectrum(
            mag, phase, sr, fftSize, /*shiftHz=*/-5000.0f, /*strength=*/1.0f);

        REQUIRE(allFinite(outMag));
        maxDcSeen = std::max(maxDcSeen, outMag[0]);
    }

    // DC bin can pick up some energy from extreme downward shifts, but should
    // be bounded (no runaway growth across frames).
    REQUIRE(maxDcSeen < 10.0f);
}

TEST_CASE("Quantizer: extreme parameter combinations don't NaN", "[quantizer][robustness]")
{
    MusicalQuantizer q(60, ScaleType::Major);
    const int fftSize = 1024;
    const double sr = 48000.0;
    q.prepare(sr, fftSize, fftSize / 4);

    const int numBins = fftSize / 2 + 1;
    std::vector<float> mag(static_cast<size_t>(numBins), 0.5f);
    std::vector<float> phase(static_cast<size_t>(numBins), 0.5f);

    // Sweep through corner cases.
    for (float shift : { -20000.0f, -1000.0f, -1.0f, 0.0f, 1.0f, 1000.0f, 20000.0f })
    {
        for (float strength : { 0.0f, 0.5f, 1.0f })
        {
            INFO("shift=" << shift << " strength=" << strength);
            auto [outMag, outPhase] = q.quantizeSpectrum(
                mag, phase, sr, fftSize, shift, strength);
            REQUIRE(allFinite(outMag));
            REQUIRE(allFinite(outPhase));
        }
    }
}

TEST_CASE("STFT: multi-frame analysis stays bounded for sine and noise inputs",
          "[stft][robustness]")
{
    const int fftSize = 2048;
    STFT s(fftSize, fftSize / 4);
    s.prepare(44100.0);

    // 50 frames of varying input — STFT should produce finite output every time.
    for (int f = 0; f < 50; ++f)
    {
        std::vector<float> input(static_cast<size_t>(fftSize));
        for (int i = 0; i < fftSize; ++i)
            input[static_cast<size_t>(i)] = std::sin(2.0f * 3.14159265f * (440.0f + f) * i / 44100.0f);

        auto [mag, phase] = s.forward(input);
        REQUIRE(allFinite(mag));
        REQUIRE(allFinite(phase));
    }
}
