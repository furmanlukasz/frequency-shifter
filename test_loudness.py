"""
Test script to verify loudness preservation with gain compensation.
"""

import numpy as np
from src.harmonic_shifter.processing.processor import HarmonicShifter

# Create test signal (1 second of 440Hz sine wave)
sample_rate = 44100
duration = 1.0
t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 0.5 amplitude

print("Testing loudness preservation...")
print(f"Input RMS: {np.sqrt(np.mean(audio**2)):.4f}")
print(f"Input peak: {np.max(np.abs(audio)):.4f}")

# Initialize processor
processor = HarmonicShifter(
    sample_rate=sample_rate,
    fft_size=4096,
    hop_size=1024,
    use_enhanced_phase_vocoder=True
)

# Test WITHOUT loudness preservation
output_no_comp = processor.process(
    audio,
    shift_hz=150,
    quantize_strength=0.0,
    preserve_loudness=False
)

print(f"\nWithout compensation:")
print(f"  Output RMS: {np.sqrt(np.mean(output_no_comp**2)):.4f}")
print(f"  Output peak: {np.max(np.abs(output_no_comp)):.4f}")
print(f"  RMS loss: {(1 - np.sqrt(np.mean(output_no_comp**2)) / np.sqrt(np.mean(audio**2))) * 100:.1f}%")

# Test WITH loudness preservation
output_with_comp = processor.process(
    audio,
    shift_hz=150,
    quantize_strength=0.0,
    preserve_loudness=True
)

print(f"\nWith compensation:")
print(f"  Output RMS: {np.sqrt(np.mean(output_with_comp**2)):.4f}")
print(f"  Output peak: {np.max(np.abs(output_with_comp)):.4f}")
print(f"  RMS difference: {abs(1 - np.sqrt(np.mean(output_with_comp**2)) / np.sqrt(np.mean(audio**2))) * 100:.1f}%")

print("\n✓ Loudness preservation test complete!")
