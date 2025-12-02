---
layout: default
title: Harmonic Frequency Shifter
---

# Harmonic Frequency Shifter

A VST3/AU audio plugin that combines frequency shifting with musical scale quantization for creative sound design.

![Plugin Screenshot](plugin.png)

## What It Does

This plugin performs **frequency shifting** - moving all frequencies in your audio by a fixed Hz amount - while keeping the output **musical** through intelligent scale quantization.

Unlike pitch shifting (which preserves harmonic relationships), frequency shifting creates unique, often metallic or otherworldly tones. By adding scale quantization, you get the best of both worlds: the character of frequency shifting with musical coherence.

## Features

- **Frequency Shift**: ±20,000 Hz range with linear or logarithmic control
- **Musical Quantization**: Snap frequencies to any musical scale
- **22 Scale Types**: Major, minor, modes, pentatonic, blues, chromatic, world scales
- **Phase Vocoder**: High-quality processing with minimal artifacts
- **Quality Modes**: Low Latency, Balanced, or Quality presets
- **Real-time Spectrum Analyzer**: Visualize your frequency content
- **Dry/Wet Mix**: Blend processed and original signals

## Quick Start

1. **Download** the plugin from the [Releases page](https://github.com/ludzeller/frequency-shifter/releases)
2. **Install** the VST3 or AU to your plugin folder
3. **Load** in your DAW and start experimenting!

### Basic Usage

| Parameter | Description |
|-----------|-------------|
| **Shift (Hz)** | Amount to shift frequencies. Positive = up, negative = down |
| **Quantize** | How strongly to snap to scale notes (0% = pure shift, 100% = fully quantized) |
| **Root Note** | The root of your scale (C, C#, D, etc.) |
| **Scale** | Choose from Major, Minor, Dorian, Pentatonic, Blues, and more |
| **Dry/Wet** | Mix between original and processed audio |

### Creative Tips

- **Metallic vocals**: Shift by 50-200 Hz with 0% quantization
- **Re-harmonize**: Use 100% quantization to force audio into a new scale
- **Subtle detuning**: Small shifts (5-20 Hz) with 50% quantization for chorus-like effects
- **Robotic sounds**: Large shifts with the Chromatic scale

## Downloads

Get the latest release for your platform:

| Platform | Format | Download |
|----------|--------|----------|
| macOS | VST3 | [Download](https://github.com/ludzeller/frequency-shifter/releases/latest) |
| macOS | AU | [Download](https://github.com/ludzeller/frequency-shifter/releases/latest) |
| Windows | VST3 | [Download](https://github.com/ludzeller/frequency-shifter/releases/latest) |

## How It Works

The plugin uses a sophisticated DSP pipeline:

```
Audio Input
    ↓
[STFT Analysis] → Convert to frequency domain
    ↓
[Frequency Shift] → Move all bins by Hz offset
    ↓
[Scale Quantization] → Snap to musical notes
    ↓
[Phase Vocoder] → Maintain phase coherence
    ↓
[ISTFT Synthesis] → Convert back to audio
```

### The Algorithm

1. **STFT (Short-Time Fourier Transform)**: Break audio into overlapping frames, apply window function, transform to frequency domain

2. **Frequency Shifting**: Reassign each frequency bin by adding the shift amount in Hz:
   ```
   f_new = f_original + shift_hz
   ```

3. **Musical Quantization**: For each frequency, find the nearest note in the selected scale:
   ```
   midi = 69 + 12 × log₂(f / 440)
   quantized_midi = nearest_scale_note(midi)
   f_quantized = 440 × 2^((quantized_midi - 69) / 12)
   ```

4. **Phase Vocoder**: Use identity phase locking (Laroche & Dolson) to maintain phase coherence between frames

5. **ISTFT**: Overlap-add synthesis to reconstruct the time-domain signal

For complete mathematical details, see [Algorithm Documentation](algorithm.html).

## Documentation

- [Algorithm Details](algorithm.html) - Full technical documentation
- [Phase Vocoder](phase-vocoder.html) - How phase coherence is maintained
- [Mathematical Foundation](math.html) - The DSP math behind the plugin

## System Requirements

- **macOS**: 10.13+ (Intel or Apple Silicon)
- **Windows**: Windows 10+
- **DAW**: Any VST3 or AU compatible host

## Source Code

This project is open source! Check out the [GitHub repository](https://github.com/ludzeller/frequency-shifter) to:

- Browse the source code
- Report issues
- Contribute improvements
- Build from source

## License

MIT License - Free to use, modify, and distribute.

## Acknowledgments

Based on established DSP techniques:
- Phase vocoder (Laroche & Dolson, 1999)
- STFT overlap-add (Allen & Rabiner, 1977)
- Identity phase locking for improved quality

---

**Questions?** Open an issue on [GitHub](https://github.com/ludzeller/frequency-shifter/issues).
