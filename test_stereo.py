"""
Test script to verify stereo/multi-channel audio processing.
"""

import numpy as np
from src.harmonic_shifter.processing.processor import HarmonicShifter
from src.harmonic_shifter.audio.io import save_audio

# Create stereo test signal
sample_rate = 44100
duration = 2.0
t = np.linspace(0, duration, int(sample_rate * duration))

# Left channel: 440 Hz (A4)
left_channel = 0.5 * np.sin(2 * np.pi * 440 * t)

# Right channel: 554.37 Hz (C#5) - different from left to verify independence
right_channel = 0.5 * np.sin(2 * np.pi * 554.37 * t)

# Combine into stereo (n_samples, 2)
stereo_audio = np.column_stack([left_channel, right_channel])

print("Testing stereo audio processing...")
print(f"Input shape: {stereo_audio.shape}")
print(f"Input channels: {stereo_audio.shape[1]}")
print(f"Left channel RMS: {np.sqrt(np.mean(left_channel**2)):.4f}")
print(f"Right channel RMS: {np.sqrt(np.mean(right_channel**2)):.4f}")

# Initialize processor
processor = HarmonicShifter(
    sample_rate=sample_rate,
    fft_size=4096,
    hop_size=1024,
    use_enhanced_phase_vocoder=True
)

# Process stereo audio
output_stereo = processor.process(
    stereo_audio,
    shift_hz=100,
    quantize_strength=0.0,
    preserve_loudness=True
)

print(f"\nOutput shape: {output_stereo.shape}")
print(f"Output channels: {output_stereo.shape[1]}")
print(f"Left channel RMS: {np.sqrt(np.mean(output_stereo[:, 0]**2)):.4f}")
print(f"Right channel RMS: {np.sqrt(np.mean(output_stereo[:, 1]**2)):.4f}")

# Verify channels are different (preserved independence)
correlation = np.corrcoef(output_stereo[:, 0], output_stereo[:, 1])[0, 1]
print(f"\nChannel correlation: {correlation:.4f}")
print("(Should be low if channels are processed independently)")

# Save stereo output
save_audio('test_output_stereo.wav', output_stereo, sample_rate=sample_rate)
print("\n✓ Stereo test complete! Saved to test_output_stereo.wav")

# Also test mono for comparison
print("\n" + "="*50)
print("Testing mono audio processing for comparison...")
mono_audio = left_channel  # Just use left channel

output_mono = processor.process(
    mono_audio,
    shift_hz=100,
    quantize_strength=0.0,
    preserve_loudness=True
)

print(f"Mono input shape: {mono_audio.shape}")
print(f"Mono output shape: {output_mono.shape}")
print(f"Mono output RMS: {np.sqrt(np.mean(output_mono**2)):.4f}")

save_audio('test_output_mono.wav', output_mono, sample_rate=sample_rate)
print("✓ Mono test complete! Saved to test_output_mono.wav")
