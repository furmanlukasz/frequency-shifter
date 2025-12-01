"""
Frequency shifting in the spectral domain.

This module implements linear frequency shifting by reassigning FFT bins.
"""

from typing import Tuple

import numpy as np


class FrequencyShifter:
    """
    Frequency shifter using spectral bin reassignment.

    Shifts all frequencies by a fixed Hz amount in the frequency domain.
    """

    def __init__(self, sample_rate: int, fft_size: int):
        """
        Initialize frequency shifter.

        Args:
            sample_rate: Audio sample rate in Hz
            fft_size: FFT size used for analysis

        Example:
            >>> shifter = FrequencyShifter(44100, 4096)
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.bin_resolution = sample_rate / fft_size
        self.n_bins = fft_size // 2 + 1

    def shift(
        self,
        magnitude: np.ndarray,
        phase: np.ndarray,
        shift_hz: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shift all frequencies by shift_hz in the spectral domain.

        Args:
            magnitude: (n_frames, n_bins) magnitude spectrum
            phase: (n_frames, n_bins) phase spectrum in radians
            shift_hz: Amount to shift in Hz (can be negative)

        Returns:
            Tuple of (shifted_magnitude, shifted_phase)

        Example:
            >>> mag, phase = stft(signal)
            >>> shifted_mag, shifted_phase = shifter.shift(mag, phase, 100)
        """
        n_frames, n_bins = magnitude.shape

        # Calculate bin shift
        bin_shift = int(round(shift_hz / self.bin_resolution))

        # Initialize output arrays
        shifted_magnitude = np.zeros_like(magnitude)
        shifted_phase = np.zeros_like(phase)

        # Shift each bin
        for k in range(n_bins):
            # Calculate target bin
            k_new = k + bin_shift

            # Check if target bin is within valid range
            if 0 <= k_new < n_bins:
                # Copy magnitude and phase to new bin
                shifted_magnitude[:, k_new] = magnitude[:, k]
                shifted_phase[:, k_new] = phase[:, k]

        return shifted_magnitude, shifted_phase

    def get_shifted_frequencies(self, shift_hz: float) -> np.ndarray:
        """
        Get shifted frequency values for each bin.

        Args:
            shift_hz: Frequency shift amount in Hz

        Returns:
            Array of shifted frequencies for each bin

        Example:
            >>> shifted_freqs = shifter.get_shifted_frequencies(100)
        """
        original_freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        return original_freqs + shift_hz
