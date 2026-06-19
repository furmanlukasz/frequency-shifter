#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "dsp/STFT.h"

#include <cmath>
#include <numeric>
#include <vector>

using fshift::STFT;
using Catch::Approx;

namespace {

constexpr float kPi = 3.14159265358979323846f;

std::vector<float> sineWave(int N, float freq, float sr, float amp = 1.0f)
{
    std::vector<float> sig(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        sig[static_cast<size_t>(i)] = amp * std::sin(2.0f * kPi * freq * static_cast<float>(i) / sr);
    return sig;
}

int findPeakBin(const std::vector<float>& mag, int startBin = 1)
{
    int peak = startBin;
    for (size_t i = static_cast<size_t>(startBin) + 1; i < mag.size(); ++i)
        if (mag[i] > mag[static_cast<size_t>(peak)])
            peak = static_cast<int>(i);
    return peak;
}

}  // namespace

TEST_CASE("STFT: numBins is fftSize/2 + 1 for all valid FFT sizes", "[stft]")
{
    for (int fftSize : {256, 512, 1024, 2048, 4096})
    {
        STFT s(fftSize, fftSize / 4);
        s.prepare(44100.0);
        REQUIRE(s.getNumBins() == fftSize / 2 + 1);
    }
}

TEST_CASE("STFT: frequency bins span [0, Nyquist] with correct spacing", "[stft]")
{
    const int fftSize = 2048;
    const double sr = 48000.0;

    STFT s(fftSize, fftSize / 4);
    s.prepare(sr);

    auto bins = s.getFrequencyBins();
    REQUIRE(bins.size() == static_cast<size_t>(fftSize / 2 + 1));
    REQUIRE(bins.front() == Approx(0.0f));
    REQUIRE(bins[1] == Approx(static_cast<float>(sr) / fftSize));
    REQUIRE(bins.back() == Approx(static_cast<float>(sr) / 2.0));
}

TEST_CASE("STFT: sine wave produces a magnitude peak at the expected bin", "[stft]")
{
    // 1000 Hz sine at 48 kHz with 2048-point FFT → bin index ≈ 42.67.
    // The discrete peak will land at bin 42 or 43.
    const int fftSize = 2048;
    const float sr = 48000.0f;
    const float freq = 1000.0f;

    STFT s(fftSize, fftSize / 4);
    s.prepare(static_cast<double>(sr));

    auto input = sineWave(fftSize, freq, sr);
    auto [mag, phase] = s.forward(input);

    int peak = findPeakBin(mag);
    int expected = static_cast<int>(std::round(freq * fftSize / sr));
    REQUIRE(std::abs(peak - expected) <= 1);
}

TEST_CASE("STFT: peak magnitude scales linearly with input amplitude", "[stft]")
{
    const int fftSize = 2048;
    const float sr = 44100.0f;
    const float freq = 1000.0f;

    STFT s(fftSize, fftSize / 4);
    s.prepare(static_cast<double>(sr));

    auto [mag1, _p1] = s.forward(sineWave(fftSize, freq, sr, 1.0f));
    auto [mag2, _p2] = s.forward(sineWave(fftSize, freq, sr, 0.5f));

    int peakBin = findPeakBin(mag1);
    REQUIRE(peakBin == findPeakBin(mag2));

    // Doubling input amplitude doubles peak magnitude.
    REQUIRE(mag1[static_cast<size_t>(peakBin)]
            == Approx(2.0f * mag2[static_cast<size_t>(peakBin)]).margin(1e-4f));
}

TEST_CASE("STFT: DC input concentrates energy in bin 0", "[stft]")
{
    const int fftSize = 1024;
    STFT s(fftSize, fftSize / 4);
    s.prepare(44100.0);

    std::vector<float> dc(static_cast<size_t>(fftSize), 0.5f);
    auto [mag, _phase] = s.forward(dc);

    REQUIRE(mag[0] > 0.0f);
    // DC bin should dominate over any non-DC bin
    for (size_t i = 1; i < mag.size(); ++i)
        REQUIRE(mag[i] < mag[0]);
}

TEST_CASE("STFT: behaves consistently across sample rates", "[stft][sr-sweep]")
{
    // A 1 kHz tone should always land at bin = round(freq * fftSize / sr),
    // and the relative peak energy should stay above the noise floor regardless of SR.
    const int fftSize = 4096;

    for (double sr : {44100.0, 48000.0, 88200.0, 96000.0, 192000.0})
    {
        STFT s(fftSize, fftSize / 4);
        s.prepare(sr);

        auto input = sineWave(fftSize, 1000.0f, static_cast<float>(sr));
        auto [mag, _phase] = s.forward(input);

        int expected = static_cast<int>(std::round(1000.0 * fftSize / sr));
        int peak = findPeakBin(mag);
        INFO("Sample rate: " << sr << " Hz");
        REQUIRE(std::abs(peak - expected) <= 1);

        // Peak should be at least 20 dB above the average of bins outside ±5 of the peak.
        double bgEnergy = 0.0;
        int bgCount = 0;
        for (size_t i = 0; i < mag.size(); ++i)
        {
            if (std::abs(static_cast<int>(i) - peak) > 5)
            {
                bgEnergy += mag[i] * mag[i];
                ++bgCount;
            }
        }
        double bgRms = std::sqrt(bgEnergy / std::max(1, bgCount));
        double peakAmp = static_cast<double>(mag[static_cast<size_t>(peak)]);
        REQUIRE(peakAmp > bgRms * 10.0);  // 20 dB SNR floor
    }
}
