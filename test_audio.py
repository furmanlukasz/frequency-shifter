from harmonic_shifter import HarmonicShifter, load_audio, save_audio

# Initialize processor
processor = HarmonicShifter(
    sample_rate=44100,
    fft_size=4096,
    hop_size=1024
)

# Set musical scale
processor.set_scale(root_midi=57, scale_type='minor')  # A minor

# Load your audio file
audio, sr = load_audio('AURORA - When The Dark Dresses Lightly.wav')

# Process with frequency shift
output = processor.process(
    audio=audio,
    shift_hz=150,  # Shift down by 150 Hz
    quantize_strength=1.0  # Full quantization to scale
)

# Save the result
save_audio('output.wav', output, sample_rate=sr)
print("Done! Saved to output.wav")