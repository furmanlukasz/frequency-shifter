#pragma once

#include <cmath>
#include <numbers>
#include <random>
#include <array>
#include <vector>

namespace fshift
{

/**
 * DriftModulator - Generates smooth modulation for pitch drift effect.
 *
 * Supports three modes:
 * - LFO: Sine wave oscillation (continuous, predictable)
 * - Perlin: Smooth pseudo-random noise (organic, continuous drift)
 * - Stochastic: Event-based modulation (bubbles, pops, organic events)
 *
 * Stochastic mode has three sub-types:
 * - Poisson: Random events with smooth envelopes (bubbles)
 * - RandomWalk: Direction changes with momentum (wandering)
 * - JumpDiffusion: Small continuous drift + occasional jumps
 *
 * Each frequency bin can have independent modulation for natural movement.
 */
class DriftModulator
{
public:
    enum class Mode
    {
        LFO = 0,
        Perlin = 1,
        Stochastic = 2
    };

    enum class StochasticType
    {
        Poisson = 0,      // Random events with smooth attack/decay envelopes
        RandomWalk = 1,   // Direction changes with inertia/momentum
        JumpDiffusion = 2 // Small Brownian motion + occasional larger jumps
    };

    enum class LFOShape
    {
        Sine = 0,
        Triangle = 1
    };

    DriftModulator() = default;

    /**
     * Prepare the modulator for processing.
     *
     * @param sampleRate Audio sample rate
     * @param numBins Number of frequency bins to modulate
     */
    void prepare(double newSampleRate, int newNumBins)
    {
        sampleRate = newSampleRate;
        numBins = newNumBins;

        // Initialize per-bin phase offsets for variety
        binPhaseOffsets.resize(static_cast<size_t>(numBins));
        std::random_device rd;
        rng.seed(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (size_t i = 0; i < binPhaseOffsets.size(); ++i)
        {
            binPhaseOffsets[i] = dist(rng);
        }

        // Initialize stochastic state per bin
        stochasticState.resize(static_cast<size_t>(numBins));
        for (auto& state : stochasticState)
        {
            state = StochasticBinState{};
        }

        // Initialize Perlin noise state
        initPerlin();
    }

    /**
     * Reset modulator state.
     */
    void reset()
    {
        phase = 0.0;
        perlinTime = 0.0;

        for (auto& state : stochasticState)
        {
            state = StochasticBinState{};
        }
    }

    /**
     * Advance the modulator by one hop (FFT frame).
     *
     * @param hopSize Number of samples per hop
     */
    void advanceFrame(int hopSize)
    {
        double timeIncrement = static_cast<double>(hopSize) / sampleRate;

        // Advance LFO phase
        phase += rate * timeIncrement;
        if (phase >= 1.0)
            phase -= 1.0;

        // Advance Perlin time
        perlinTime += rate * timeIncrement;

        // Advance stochastic state for all bins
        if (mode == Mode::Stochastic)
        {
            advanceStochastic(timeIncrement);
        }
    }

    /**
     * Get drift amount for a specific frequency bin.
     * Returns value in cents scaled by depth.
     *
     * @param binIndex Frequency bin index
     * @return Drift amount in cents (scaled by depth parameter)
     */
    float getDrift(int binIndex) const
    {
        if (depth <= 0.0f || binIndex < 0 || binIndex >= numBins)
            return 0.0f;

        float modValue = 0.0f;
        float binPhase = binPhaseOffsets[static_cast<size_t>(binIndex)];

        switch (mode)
        {
            case Mode::LFO:
                modValue = computeLFO(phase + binPhase * phaseSpread);
                break;
            case Mode::Perlin:
                modValue = computePerlin(binIndex, binPhase);
                break;
            case Mode::Stochastic:
                modValue = getStochasticValue(binIndex);
                break;
        }

        // Scale by depth (in cents, max ±50 cents = half semitone)
        return modValue * depth * 50.0f;
    }

    // Mode setters
    void setMode(Mode newMode) { mode = newMode; }
    void setStochasticType(StochasticType type) { stochasticType = type; }
    void setLFOShape(LFOShape newShape) { lfoShape = newShape; }

    // Common parameters
    void setRate(float newRate) { rate = std::clamp(newRate, 0.01f, 20.0f); }
    void setDepth(float newDepth) { depth = std::clamp(newDepth, 0.0f, 1.0f); }
    void setPhaseSpread(float newSpread) { phaseSpread = std::clamp(newSpread, 0.0f, 1.0f); }

    // Stochastic parameters
    void setDensity(float newDensity) { density = std::clamp(newDensity, 0.0f, 1.0f); }
    void setSmoothness(float newSmoothness) { smoothness = std::clamp(newSmoothness, 0.0f, 1.0f); }

    // Perlin-specific parameters
    void setPerlinOctaves(int octaves) { perlinOctaves = std::clamp(octaves, 1, 4); }
    void setPerlinLacunarity(float lac) { perlinLacunarity = std::clamp(lac, 1.0f, 4.0f); }
    void setPerlinPersistence(float pers) { perlinPersistence = std::clamp(pers, 0.0f, 1.0f); }

    // Getters
    Mode getMode() const { return mode; }
    StochasticType getStochasticType() const { return stochasticType; }
    LFOShape getLFOShape() const { return lfoShape; }
    float getRate() const { return rate; }
    float getDepth() const { return depth; }
    float getDensity() const { return density; }
    float getSmoothness() const { return smoothness; }

private:
    // Per-bin state for stochastic modes
    struct StochasticBinState
    {
        // Poisson mode: envelope state
        float envelopeValue = 0.0f;     // Current envelope level [0, 1]
        float envelopeTarget = 0.0f;    // Target value for current event
        float envelopePhase = 1.0f;     // 0 = attack, 1 = idle/decay complete
        float eventDirection = 1.0f;    // +1 or -1 for random direction

        // RandomWalk mode: position and velocity
        float position = 0.0f;          // Current drift position [-1, 1]
        float velocity = 0.0f;          // Current velocity

        // JumpDiffusion mode: base position + brownian component
        float basePosition = 0.0f;      // Position from jumps
        float brownianOffset = 0.0f;    // Small continuous drift

        // Timing
        float timeSinceLastEvent = 0.0f;
        float nextEventTime = 1.0f;     // When next event should occur
    };

    // LFO computation
    float computeLFO(double p) const
    {
        // Wrap phase to [0, 1)
        p = p - std::floor(p);

        if (lfoShape == LFOShape::Sine)
        {
            return static_cast<float>(std::sin(2.0 * std::numbers::pi * p));
        }
        else // Triangle
        {
            if (p < 0.25)
                return static_cast<float>(p * 4.0);
            else if (p < 0.75)
                return static_cast<float>(1.0 - (p - 0.25) * 4.0);
            else
                return static_cast<float>(-1.0 + (p - 0.75) * 4.0);
        }
    }

    // Perlin noise implementation
    void initPerlin()
    {
        std::mt19937 gen(42); // Fixed seed for reproducibility

        for (int i = 0; i < 256; ++i)
        {
            perm[static_cast<size_t>(i)] = i;
        }

        for (int i = 255; i > 0; --i)
        {
            std::uniform_int_distribution<int> dist(0, i);
            int j = dist(gen);
            std::swap(perm[static_cast<size_t>(i)], perm[static_cast<size_t>(j)]);
        }

        for (int i = 0; i < 256; ++i)
        {
            perm[static_cast<size_t>(256 + i)] = perm[static_cast<size_t>(i)];
        }
    }

    float fade(float t) const
    {
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    }

    float grad(int hash, float x) const
    {
        return (hash & 1) ? x : -x;
    }

    float noise1D(float x) const
    {
        int xi = static_cast<int>(std::floor(x)) & 255;
        float xf = x - std::floor(x);
        float u = fade(xf);

        int aa = perm[static_cast<size_t>(xi)];
        int ab = perm[static_cast<size_t>(xi + 1)];

        float g1 = grad(perm[static_cast<size_t>(aa)], xf);
        float g2 = grad(perm[static_cast<size_t>(ab)], xf - 1.0f);

        return g1 + u * (g2 - g1);
    }

    float computePerlin(int binIndex, float binPhase) const
    {
        float x = static_cast<float>(perlinTime) + binPhase * 10.0f + static_cast<float>(binIndex) * 0.1f;

        float total = 0.0f;
        float amplitude = 1.0f;
        float frequency = 1.0f;
        float maxValue = 0.0f;

        for (int i = 0; i < perlinOctaves; ++i)
        {
            total += noise1D(x * frequency) * amplitude;
            maxValue += amplitude;
            amplitude *= perlinPersistence;
            frequency *= perlinLacunarity;
        }

        return total / maxValue;
    }

    // Stochastic mode implementation
    void advanceStochastic(double timeIncrement)
    {
        float dt = static_cast<float>(timeIncrement);

        // Event rate scales with density and rate parameters
        // density 0 = very sparse events, density 1 = frequent events
        float eventRate = 0.1f + density * rate * 2.0f; // Events per second

        // Smoothness affects envelope times
        // smoothness 0 = sharp pops, smoothness 1 = very slow swells
        float attackTime = 0.01f + smoothness * 0.5f;   // 10ms to 510ms
        float decayTime = 0.05f + smoothness * 2.0f;    // 50ms to 2050ms

        std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);
        std::normal_distribution<float> normalDist(0.0f, 1.0f);

        for (size_t i = 0; i < stochasticState.size(); ++i)
        {
            auto& state = stochasticState[i];
            state.timeSinceLastEvent += dt;

            switch (stochasticType)
            {
                case StochasticType::Poisson:
                    advancePoissonBin(state, dt, eventRate, attackTime, decayTime, uniformDist);
                    break;
                case StochasticType::RandomWalk:
                    advanceRandomWalkBin(state, dt, eventRate, uniformDist, normalDist);
                    break;
                case StochasticType::JumpDiffusion:
                    advanceJumpDiffusionBin(state, dt, eventRate, uniformDist, normalDist);
                    break;
            }
        }
    }

    void advancePoissonBin(StochasticBinState& state, float dt, float eventRate,
                           float attackTime, float decayTime,
                           std::uniform_real_distribution<float>& uniformDist)
    {
        // Check if it's time for a new event (Poisson process)
        if (state.timeSinceLastEvent >= state.nextEventTime)
        {
            // Trigger new event
            state.envelopePhase = 0.0f;
            state.envelopeTarget = 0.3f + uniformDist(rng) * 0.7f; // Random magnitude 0.3-1.0
            state.eventDirection = (uniformDist(rng) > 0.5f) ? 1.0f : -1.0f;
            state.timeSinceLastEvent = 0.0f;

            // Schedule next event (exponential distribution for Poisson process)
            float lambda = eventRate;
            state.nextEventTime = -std::log(1.0f - uniformDist(rng) + 1e-10f) / lambda;
        }

        // Update envelope
        if (state.envelopePhase < 1.0f)
        {
            // Attack phase
            float attackRate = 1.0f / attackTime;
            state.envelopeValue += (state.envelopeTarget - state.envelopeValue) * attackRate * dt * 10.0f;

            // Check if we've reached target (within 95%)
            if (state.envelopeValue >= state.envelopeTarget * 0.95f)
            {
                state.envelopePhase = 1.0f; // Switch to decay
            }
        }

        // Always decay (even during attack, creates more organic shape)
        float decayRate = 1.0f / decayTime;
        state.envelopeValue *= (1.0f - decayRate * dt);

        // Clamp to valid range
        state.envelopeValue = std::clamp(state.envelopeValue, 0.0f, 1.0f);
    }

    void advanceRandomWalkBin(StochasticBinState& state, float dt, float eventRate,
                              std::uniform_real_distribution<float>& uniformDist,
                              std::normal_distribution<float>& normalDist)
    {
        // Momentum-based random walk with occasional direction changes

        // Friction/damping (smoothness affects how much momentum is preserved)
        float friction = 0.5f + (1.0f - smoothness) * 4.0f; // Higher smoothness = less friction
        state.velocity *= std::exp(-friction * dt);

        // Random impulses based on event rate
        float impulseProb = eventRate * dt;
        if (uniformDist(rng) < impulseProb)
        {
            // Apply random impulse
            float impulseMagnitude = 0.5f + uniformDist(rng) * 1.5f;
            state.velocity += normalDist(rng) * impulseMagnitude;
        }

        // Small continuous random force (Brownian-like)
        state.velocity += normalDist(rng) * dt * 0.5f;

        // Clamp velocity
        state.velocity = std::clamp(state.velocity, -5.0f, 5.0f);

        // Update position
        state.position += state.velocity * dt;

        // Soft boundary: spring force pulling back to center
        float boundaryForce = -state.position * 2.0f;
        state.velocity += boundaryForce * dt;

        // Hard clamp position
        state.position = std::clamp(state.position, -1.0f, 1.0f);
    }

    void advanceJumpDiffusionBin(StochasticBinState& state, float dt, float eventRate,
                                  std::uniform_real_distribution<float>& uniformDist,
                                  std::normal_distribution<float>& normalDist)
    {
        // Small continuous Brownian motion
        float brownianScale = 0.1f * (1.0f - smoothness * 0.8f); // Smoothness reduces brownian
        state.brownianOffset += normalDist(rng) * std::sqrt(dt) * brownianScale;

        // Mean reversion for brownian component
        state.brownianOffset *= (1.0f - dt * 2.0f);
        state.brownianOffset = std::clamp(state.brownianOffset, -0.3f, 0.3f);

        // Occasional jumps (Poisson-like)
        float jumpProb = eventRate * dt * 0.5f; // Jumps are less frequent than Poisson events
        if (uniformDist(rng) < jumpProb)
        {
            // Jump to new position
            float jumpMagnitude = 0.3f + uniformDist(rng) * 0.7f;
            float jumpDirection = (uniformDist(rng) > 0.5f) ? 1.0f : -1.0f;

            // Smooth transition: set target and interpolate
            float newTarget = jumpDirection * jumpMagnitude;

            // Apply jump with some smoothing (not instant)
            float jumpSpeed = 1.0f - smoothness * 0.9f; // Smoothness slows jumps
            state.basePosition += (newTarget - state.basePosition) * jumpSpeed;
        }

        // Slow drift back to center (mean reversion)
        state.basePosition *= (1.0f - dt * 0.5f);

        // Clamp total position
        state.basePosition = std::clamp(state.basePosition, -1.0f, 1.0f);
    }

    float getStochasticValue(int binIndex) const
    {
        if (binIndex < 0 || static_cast<size_t>(binIndex) >= stochasticState.size())
            return 0.0f;

        const auto& state = stochasticState[static_cast<size_t>(binIndex)];

        switch (stochasticType)
        {
            case StochasticType::Poisson:
                return state.envelopeValue * state.eventDirection;

            case StochasticType::RandomWalk:
                return state.position;

            case StochasticType::JumpDiffusion:
                return std::clamp(state.basePosition + state.brownianOffset, -1.0f, 1.0f);
        }

        return 0.0f;
    }

    // Parameters
    Mode mode = Mode::LFO;
    StochasticType stochasticType = StochasticType::Poisson;
    LFOShape lfoShape = LFOShape::Sine;
    float rate = 1.0f;           // Hz (cycles per second) / event rate multiplier
    float depth = 0.0f;          // 0-1 (0 = no drift, 1 = max ±50 cents)
    float phaseSpread = 0.5f;    // How much bins differ in phase (0 = sync, 1 = random)

    // Stochastic parameters
    float density = 0.5f;        // 0-1: how frequently events occur
    float smoothness = 0.5f;     // 0-1: sharp pops (0) to slow swells (1)

    // Perlin parameters
    int perlinOctaves = 2;
    float perlinLacunarity = 2.0f;
    float perlinPersistence = 0.5f;

    // State
    double sampleRate = 44100.0;
    int numBins = 2048;
    double phase = 0.0;
    double perlinTime = 0.0;

    // Per-bin state
    std::vector<float> binPhaseOffsets;
    mutable std::vector<StochasticBinState> stochasticState;

    // Random number generator (mutable for const methods that need randomness)
    mutable std::mt19937 rng;

    // Perlin permutation table
    std::array<int, 512> perm{};
};

} // namespace fshift
