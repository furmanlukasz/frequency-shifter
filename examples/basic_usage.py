"""
Basic usage example for HarmonicShifter.

This example demonstrates how to:
1. Initialize the processor
2. Configure a musical scale
3. Process audio with frequency shifting and quantization
4. Save the output
"""

import numpy as np
from harmonic_shifter import HarmonicShifter, load_audio, save_audio


def generate_test_signal(sample_rate=44100, duration=2.0):
    """Generate a simple test signal (440 Hz sine wave)."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t)
    return signal


def main():
    print("Harmonic-Preserving Frequency Shifter - Basic Usage Example")
    print("=" * 60)

    # Initialize processor
    print("\n1. Initializing processor...")
    processor = HarmonicShifter(
        sample_rate=44100,
        fft_size=4096,
        hop_size=1024
    )

    # Show processor info
    info = processor.get_info()
    print(f"   Sample rate: {info['sample_rate']} Hz")
    print(f"   FFT size: {info['fft_size']}")
    print(f"   Hop size: {info['hop_size']}")
    print(f"   Latency: {info['latency_ms']:.1f} ms")

    # Configure musical scale
    print("\n2. Configuring scale...")
    processor.set_scale(
        root_midi=60,  # C4
        scale_type='major'
    )
    print("   Scale: C Major")

    # Generate or load audio
    print("\n3. Generating test audio...")
    audio = generate_test_signal(sample_rate=44100, duration=2.0)
    print(f"   Generated 440 Hz sine wave, {len(audio)} samples")

    # Example 1: Pure frequency shift (inharmonic)
    print("\n4. Processing with pure frequency shift...")
    output_shift = processor.process(
        audio=audio,
        shift_hz=100,  # Shift up by 100 Hz
        quantize_strength=0.0  # No quantization
    )
    print("   Result: 540 Hz (inharmonic)")

    # Example 2: Frequency shift + full quantization (harmonic)
    print("\n5. Processing with quantization to C Major...")
    output_quantized = processor.process(
        audio=audio,
        shift_hz=100,  # Shift up by 100 Hz
        quantize_strength=1.0  # Full quantization
    )
    print("   Result: Quantized to nearest C Major note")

    # Example 3: Partial quantization (blend)
    print("\n6. Processing with partial quantization...")
    output_blend = processor.process(
        audio=audio,
        shift_hz=100,
        quantize_strength=0.5  # 50% blend
    )
    print("   Result: 50% blend between shifted and quantized")

    # Save outputs (optional - uncomment to save)
    # print("\n7. Saving outputs...")
    # save_audio('output_shift.wav', output_shift, sample_rate=44100)
    # save_audio('output_quantized.wav', output_quantized, sample_rate=44100)
    # save_audio('output_blend.wav', output_blend, sample_rate=44100)
    # print("   Saved: output_shift.wav, output_quantized.wav, output_blend.wav")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nTo process your own audio files:")
    print("  audio, sr = load_audio('input.wav')")
    print("  output = processor.process(audio, shift_hz=100, quantize_strength=1.0)")
    print("  save_audio('output.wav', output, sample_rate=sr)")


if __name__ == '__main__':
    main()
