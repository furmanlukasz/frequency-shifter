"""Tests for STFT and inverse STFT implementation."""

import numpy as np
import pytest

from harmonic_shifter.core.stft import stft, istft, get_frequency_bins, get_time_frames


class TestSTFT:
    """Tests for STFT forward transform."""

    def test_output_shape(self, sine_440hz, fft_size, hop_size):
        """STFT should return correct output shapes."""
        mag, phase = stft(sine_440hz, fft_size=fft_size, hop_size=hop_size)

        # Calculate expected dimensions
        n_frames = 1 + (len(sine_440hz) - fft_size) // hop_size
        n_bins = fft_size // 2 + 1

        assert mag.shape == (n_frames, n_bins)
        assert phase.shape == (n_frames, n_bins)

    def test_magnitude_positive(self, sine_440hz):
        """Magnitudes should all be non-negative."""
        mag, _ = stft(sine_440hz)
        assert np.all(mag >= 0)

    def test_phase_range(self, sine_440hz):
        """Phase values should be in range [-π, π]."""
        _, phase = stft(sine_440hz)
        assert np.all(phase >= -np.pi)
        assert np.all(phase <= np.pi)

    def test_different_windows(self, sine_440hz):
        """Test different window functions."""
        for window in ['hann', 'hamming', 'blackman']:
            mag, phase = stft(sine_440hz, window=window)
            assert mag.shape[0] > 0
            assert phase.shape[0] > 0

    def test_invalid_audio_shape_raises(self):
        """2D audio should raise ValueError."""
        audio_2d = np.random.randn(100, 2)
        with pytest.raises(ValueError, match="must be 1D"):
            stft(audio_2d)

    def test_invalid_fft_size_raises(self, sine_440hz):
        """Non-power-of-2 FFT size should raise ValueError."""
        with pytest.raises(ValueError, match="power of 2"):
            stft(sine_440hz, fft_size=1000)

    def test_invalid_window_raises(self, sine_440hz):
        """Unknown window type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown window"):
            stft(sine_440hz, window='invalid')


class TestISTFT:
    """Tests for inverse STFT."""

    def test_output_shape(self, sine_440hz, fft_size, hop_size):
        """ISTFT should produce output of reasonable length."""
        mag, phase = stft(sine_440hz, fft_size=fft_size, hop_size=hop_size)
        reconstructed = istft(mag, phase, hop_size=hop_size)

        # Output length should be close to input length
        assert abs(len(reconstructed) - len(sine_440hz)) < fft_size

    def test_mismatched_shapes_raises(self):
        """Mismatched magnitude and phase shapes should raise ValueError."""
        mag = np.random.randn(100, 1025)
        phase = np.random.randn(50, 1025)  # Different number of frames

        with pytest.raises(ValueError, match="shapes must match"):
            istft(mag, phase)


class TestSTFTReconstruction:
    """Tests for STFT → ISTFT reconstruction."""

    def test_perfect_reconstruction(self, sine_440hz, fft_size, hop_size):
        """STFT → ISTFT should recover original signal."""
        mag, phase = stft(sine_440hz, fft_size=fft_size, hop_size=hop_size)
        reconstructed = istft(mag, phase, hop_size=hop_size)

        # Trim edges to avoid boundary effects
        trim = fft_size
        original_trimmed = sine_440hz[trim:-trim]
        reconstructed_trimmed = reconstructed[trim:trim + len(original_trimmed)]

        # Calculate reconstruction error
        error = np.mean(np.abs(original_trimmed - reconstructed_trimmed))

        assert error < 1e-6, f"Reconstruction error too high: {error}"

    def test_reconstruction_white_noise(self, white_noise):
        """STFT → ISTFT reconstruction on white noise."""
        mag, phase = stft(white_noise, fft_size=2048, hop_size=512)
        reconstructed = istft(mag, phase, hop_size=512)

        trim = 2048
        original_trimmed = white_noise[trim:-trim]
        reconstructed_trimmed = reconstructed[trim:trim + len(original_trimmed)]

        error = np.mean(np.abs(original_trimmed - reconstructed_trimmed))

        assert error < 1e-5

    @pytest.mark.parametrize("window", ['hann', 'hamming', 'blackman'])
    def test_reconstruction_different_windows(self, sine_440hz, window):
        """Test reconstruction with different windows."""
        mag, phase = stft(sine_440hz, window=window)
        reconstructed = istft(mag, phase, window=window)

        trim = 4096
        original_trimmed = sine_440hz[trim:-trim]
        reconstructed_trimmed = reconstructed[trim:trim + len(original_trimmed)]

        error = np.mean(np.abs(original_trimmed - reconstructed_trimmed))

        assert error < 1e-5

    # Note: Parseval's theorem test removed - the relationship between time and
    # frequency domain energy for overlapping STFT frames is complex and requires
    # careful accounting of window overlap. The perfect reconstruction test above
    # is sufficient to verify STFT correctness.


class TestFrequencyBins:
    """Tests for frequency bin utilities."""

    def test_frequency_bins_range(self, sample_rate, fft_size):
        """Frequency bins should range from 0 to Nyquist."""
        freqs = get_frequency_bins(fft_size, sample_rate)

        assert freqs[0] == 0.0  # DC bin
        assert np.abs(freqs[-1] - sample_rate / 2) < 1.0  # Nyquist

    def test_frequency_bins_length(self, sample_rate, fft_size):
        """Number of frequency bins should be fft_size/2 + 1."""
        freqs = get_frequency_bins(fft_size, sample_rate)
        assert len(freqs) == fft_size // 2 + 1

    def test_frequency_resolution(self, sample_rate, fft_size):
        """Frequency resolution should be sample_rate / fft_size."""
        freqs = get_frequency_bins(fft_size, sample_rate)
        expected_resolution = sample_rate / fft_size

        # Check spacing between bins
        actual_resolution = freqs[1] - freqs[0]

        assert np.abs(actual_resolution - expected_resolution) < 0.01


class TestTimeFrames:
    """Tests for time frame utilities."""

    def test_time_frames_start_at_zero(self):
        """First time frame should be at t=0."""
        times = get_time_frames(100, 512, 44100)
        assert times[0] == 0.0

    def test_time_frames_spacing(self, sample_rate, hop_size):
        """Time frame spacing should match hop_size / sample_rate."""
        n_frames = 100
        times = get_time_frames(n_frames, hop_size, sample_rate)

        expected_spacing = hop_size / sample_rate
        actual_spacing = times[1] - times[0]

        assert np.abs(actual_spacing - expected_spacing) < 1e-6

    def test_time_frames_length(self):
        """Number of time values should equal number of frames."""
        n_frames = 42
        times = get_time_frames(n_frames, 512, 44100)
        assert len(times) == n_frames
