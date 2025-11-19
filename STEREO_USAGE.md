# Stereo Audio Processing Guide

## Quick Start

### Stereo Processing (Recommended)

```python
from harmonic_shifter import HarmonicShifter, load_audio, save_audio

# Initialize processor
processor = HarmonicShifter(sample_rate=44100, fft_size=4096, hop_size=1024)
processor.set_scale(root_midi=57, scale_type='minor')  # A minor

# Load stereo audio (set mono=False to preserve L/R channels)
audio, sr = load_audio('input.wav', mono=False)  # Shape: (n_samples, 2)

# Process - automatic loudness compensation enabled by default
output = processor.process(
    audio=audio,
    shift_hz=150,
    quantize_strength=1.0,
    preserve_loudness=True  # This is the DEFAULT
)

# Save stereo output
save_audio('output.wav', output, sample_rate=sr)
```

## Key Parameters

### `load_audio()` - Loading Audio

```python
audio, sr = load_audio(
    filepath='input.wav',
    sample_rate=44100,    # Target sample rate (None to keep original)
    mono=False            # SET TO FALSE FOR STEREO!
)
```

- **`mono=True`** (default): Converts to mono by averaging L+R channels
- **`mono=False`**: Preserves stereo - each channel processed independently

### `processor.process()` - Processing Audio

```python
output = processor.process(
    audio=audio,              # Accepts 1D (mono) or 2D (stereo)
    shift_hz=150,             # Frequency shift in Hz
    quantize_strength=1.0,    # 0.0=no quantization, 1.0=full quantization
    preserve_loudness=True    # Automatic gain compensation (DEFAULT)
)
```

- **`preserve_loudness=True`** (default): Automatically compensates for energy loss during shifting
- **`preserve_loudness=False`**: Disables loudness compensation (may result in quieter output)

## Output Shapes

| Input Shape | Output Shape | Description |
|-------------|--------------|-------------|
| `(n_samples,)` | `(n_samples,)` | Mono in → Mono out |
| `(n_samples, 2)` | `(n_samples, 2)` | Stereo in → Stereo out |
| `(n_samples, 6)` | `(n_samples, 6)` | 5.1 surround → 5.1 surround |

**Output always matches input shape!**

## Benefits of Stereo Processing

### Before (Mono Only)
- ❌ L/R channels averaged together
- ❌ Spatial information lost
- ❌ Panning/width destroyed
- ❌ Sounds "flat" and centered

### After (Stereo Support)
- ✅ Each channel processed independently
- ✅ Spatial information preserved
- ✅ Panning/width maintained
- ✅ Sounds natural and spacious

## Benefits of Loudness Compensation

### Before (No Compensation)
- ❌ Output ~40% quieter than input
- ❌ Energy lost when bins shift out of range
- ❌ Inconsistent loudness with different shift amounts

### After (With Compensation)
- ✅ Output loudness matches input
- ✅ Automatic gain adjustment
- ✅ Consistent loudness regardless of shift amount
- ✅ No manual adjustment needed

## Complete Example

```python
from harmonic_shifter import HarmonicShifter, load_audio, save_audio
import numpy as np

# Setup
processor = HarmonicShifter(
    sample_rate=44100,
    fft_size=4096,
    hop_size=1024,
    use_enhanced_phase_vocoder=True  # Reduces metallic artifacts
)
processor.set_scale(root_midi=60, scale_type='major')  # C major

# Load stereo audio
audio, sr = load_audio('input.wav', mono=False)
print(f"Input shape: {audio.shape}")  # (n_samples, 2)

# Measure input loudness
if len(audio.shape) == 2:
    left_rms = np.sqrt(np.mean(audio[:, 0]**2))
    right_rms = np.sqrt(np.mean(audio[:, 1]**2))
    print(f"Input RMS - Left: {left_rms:.4f}, Right: {right_rms:.4f}")

# Process with all features enabled
output = processor.process(
    audio=audio,
    shift_hz=150,              # Shift up by 150 Hz
    quantize_strength=1.0,     # Full scale quantization
    preserve_loudness=True     # Loudness compensation enabled
)

# Measure output loudness
if len(output.shape) == 2:
    left_rms_out = np.sqrt(np.mean(output[:, 0]**2))
    right_rms_out = np.sqrt(np.mean(output[:, 1]**2))
    print(f"Output RMS - Left: {left_rms_out:.4f}, Right: {right_rms_out:.4f}")
    print(f"Loudness preserved: {abs(left_rms_out - left_rms) / left_rms * 100:.1f}% difference")

# Save stereo output
save_audio('output_stereo.wav', output, sample_rate=sr)
print(f"Output shape: {output.shape}")  # (n_samples, 2)
```

## Troubleshooting

### Output is quieter than input
- **Solution**: Ensure `preserve_loudness=True` (it's the default)
- Check that you're using the latest version with loudness compensation

### Stereo sounds mono/flat
- **Solution**: Use `mono=False` when loading audio
- Verify input file is actually stereo (some files claim stereo but are dual-mono)

### Processing is slow
- **Solution**: Increase `hop_size` (e.g., from 1024 to 2048)
- Decrease `fft_size` (e.g., from 4096 to 2048)
- Set `use_enhanced_phase_vocoder=False` for fastest processing (but more metallic)

### Too metallic/robotic sound
- **Solution**: Ensure `use_enhanced_phase_vocoder=True`
- Try reducing `shift_hz` amount
- Try lower `quantize_strength` (e.g., 0.5 instead of 1.0)

## Performance Notes

- **Stereo processing time**: ~2x mono (each channel processed independently)
- **Loudness compensation overhead**: Negligible (~0.1% - just 2 RMS calculations)
- **Enhanced phase vocoder**: 3-5x slower than basic, but much better quality

## Migration from Mono to Stereo

### Old code (mono only)
```python
audio, sr = load_audio('input.wav')  # mono=True by default
output = processor.process(audio, shift_hz=150)
save_audio('output.wav', output, sr)
```

### New code (stereo)
```python
audio, sr = load_audio('input.wav', mono=False)  # ← Add this!
output = processor.process(audio, shift_hz=150)   # Same API
save_audio('output.wav', output, sr)              # Already supports stereo
```

**Only one character change needed: add `mono=False`!**
