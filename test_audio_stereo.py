"""
Comprehensive test script for stereo audio processing with loudness compensation.

This script demonstrates:
- Stereo audio loading (preserving L/R channels)
- Automatic loudness compensation (preserve_loudness=True by default)
- Enhanced phase vocoder for reduced metallic artifacts
- Musical scale quantization

Usage:
    python test_audio_stereo.py
"""

import numpy as np
from harmonic_shifter import HarmonicShifter, load_audio, save_audio

print("="*70)
print("Harmonic Frequency Shifter - Stereo Test")
print("="*70)

# Initialize processor with enhanced phase vocoder
processor = HarmonicShifter(
    sample_rate=44100,
    fft_size=4096,
    hop_size=1024,
    use_enhanced_phase_vocoder=True  # Reduces metallic artifacts
)

# Set musical scale (A minor)
processor.set_scale(root_midi=57, scale_type='minor')
print("\n✓ Processor initialized")
print(f"  - Sample rate: 44100 Hz")
print(f"  - FFT size: 4096")
print(f"  - Hop size: 1024")
print(f"  - Scale: A minor")
print(f"  - Enhanced phase vocoder: ENABLED")

# Load audio file in STEREO (mono=False preserves L/R channels)
print("\n" + "="*70)
print("Loading audio file...")
print("="*70)

audio_file = 'AURORA - When The Dark Dresses Lightly.wav'
audio_stereo, sr = load_audio(audio_file, sample_rate=44100, mono=False)

# Display input information
is_stereo = len(audio_stereo.shape) == 2
n_channels = audio_stereo.shape[1] if is_stereo else 1
duration = len(audio_stereo) / sr

print(f"\n✓ Audio loaded: {audio_file}")
print(f"  - Shape: {audio_stereo.shape}")
print(f"  - Channels: {n_channels} ({'STEREO' if n_channels == 2 else 'MONO'})")
print(f"  - Duration: {duration:.2f} seconds")
print(f"  - Sample rate: {sr} Hz")

if is_stereo:
    left_rms = np.sqrt(np.mean(audio_stereo[:, 0]**2))
    right_rms = np.sqrt(np.mean(audio_stereo[:, 1]**2))
    print(f"  - Input RMS (Left): {left_rms:.4f}")
    print(f"  - Input RMS (Right): {right_rms:.4f}")
    print(f"  - Input Peak (Left): {np.max(np.abs(audio_stereo[:, 0])):.4f}")
    print(f"  - Input Peak (Right): {np.max(np.abs(audio_stereo[:, 1])):.4f}")
else:
    input_rms = np.sqrt(np.mean(audio_stereo**2))
    print(f"  - Input RMS: {input_rms:.4f}")
    print(f"  - Input Peak: {np.max(np.abs(audio_stereo)):.4f}")

# Process with stereo + loudness compensation
print("\n" + "="*70)
print("Processing audio...")
print("="*70)
print("\nSettings:")
print(f"  - Frequency shift: +150 Hz")
print(f"  - Quantize strength: 1.0 (full scale quantization)")
print(f"  - Preserve loudness: TRUE (automatic gain compensation)")
print(f"  - Stereo processing: {'YES - each channel processed independently' if is_stereo else 'N/A - mono input'}")

print("\n⏳ Processing... (this may take a moment)")

output_stereo = processor.process(
    audio=audio_stereo,
    shift_hz=150,              # Shift up by 150 Hz
    quantize_strength=1.0,     # Full quantization to A minor scale
    preserve_loudness=True     # Enable automatic gain compensation (DEFAULT)
)

print("✓ Processing complete!")

# Display output information
print("\n" + "="*70)
print("Output Information")
print("="*70)
print(f"\n✓ Output shape: {output_stereo.shape}")

if len(output_stereo.shape) == 2:
    left_rms_out = np.sqrt(np.mean(output_stereo[:, 0]**2))
    right_rms_out = np.sqrt(np.mean(output_stereo[:, 1]**2))
    print(f"  - Output RMS (Left): {left_rms_out:.4f}")
    print(f"  - Output RMS (Right): {right_rms_out:.4f}")
    print(f"  - Output Peak (Left): {np.max(np.abs(output_stereo[:, 0])):.4f}")
    print(f"  - Output Peak (Right): {np.max(np.abs(output_stereo[:, 1])):.4f}")

    # Show loudness preservation effectiveness
    left_gain = left_rms_out / left_rms if left_rms > 0 else 1.0
    right_gain = right_rms_out / right_rms if right_rms > 0 else 1.0
    print(f"\n  Loudness preservation:")
    print(f"  - Left channel gain: {left_gain:.2f}x ({(left_gain-1)*100:+.1f}%)")
    print(f"  - Right channel gain: {right_gain:.2f}x ({(right_gain-1)*100:+.1f}%)")
else:
    output_rms = np.sqrt(np.mean(output_stereo**2))
    output_peak = np.max(np.abs(output_stereo))
    print(f"  - Output RMS: {output_rms:.4f}")
    print(f"  - Output Peak: {output_peak:.4f}")

    gain = output_rms / input_rms if input_rms > 0 else 1.0
    print(f"\n  Loudness preservation:")
    print(f"  - Gain: {gain:.2f}x ({(gain-1)*100:+.1f}%)")

# Save the stereo output
output_filename = 'output_stereo_loudness_corrected.wav'
save_audio(output_filename, output_stereo, sample_rate=sr)
print(f"\n✓ Saved: {output_filename}")

# Also create a mono version for comparison (if input was stereo)
if is_stereo:
    print("\n" + "="*70)
    print("Creating mono comparison...")
    print("="*70)

    # Convert to mono by averaging channels
    audio_mono = np.mean(audio_stereo, axis=1)

    output_mono = processor.process(
        audio=audio_mono,
        shift_hz=150,
        quantize_strength=1.0,
        preserve_loudness=True
    )

    mono_filename = 'output_mono_loudness_corrected.wav'
    save_audio(mono_filename, output_mono, sample_rate=sr)
    print(f"✓ Saved mono version: {mono_filename}")

print("\n" + "="*70)
print("Summary")
print("="*70)
print(f"\n✓ All processing complete!")
print(f"\nFiles created:")
print(f"  1. {output_filename} - Stereo with loudness compensation")
if is_stereo:
    print(f"  2. {mono_filename} - Mono version for comparison")

print(f"\nKey features used:")
print(f"  ✓ Stereo processing (L/R channels independent)")
print(f"  ✓ Automatic loudness compensation (preserve_loudness=True)")
print(f"  ✓ Enhanced phase vocoder (reduced metallic artifacts)")
print(f"  ✓ Musical scale quantization (A minor)")

print(f"\nTo disable loudness compensation, use:")
print(f"  processor.process(audio, shift_hz=150, preserve_loudness=False)")

print("\n" + "="*70)
