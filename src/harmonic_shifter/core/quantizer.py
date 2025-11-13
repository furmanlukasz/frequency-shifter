"""
Musical scale quantization for frequency spectra.

This module provides quantization of frequencies to musical scales,
allowing harmonic-preserving frequency shifting.
"""

from typing import List, Tuple

import numpy as np

from ..theory.scales import SCALES
from ..theory.tuning import freq_to_midi, midi_to_freq, quantize_to_scale


class MusicalQuantizer:
    """
    Quantizer for mapping frequencies to musical scales.

    Takes a frequency spectrum and quantizes frequencies to the nearest
    notes in a specified musical scale.
    """

    def __init__(self, root_midi: int, scale_type: str):
        """
        Initialize musical quantizer.

        Args:
            root_midi: MIDI note number for scale root (0-127)
            scale_type: Scale name from SCALES dict

        Raises:
            ValueError: If scale_type not found or root_midi out of range

        Example:
            >>> quantizer = MusicalQuantizer(60, 'major')  # C major
        """
        if not 0 <= root_midi <= 127:
            raise ValueError(f"Root MIDI must be 0-127, got {root_midi}")

        if scale_type not in SCALES:
            raise ValueError(
                f"Unknown scale type '{scale_type}'. "
                f"Available: {', '.join(sorted(SCALES.keys()))}"
            )

        self.root_midi = root_midi
        self.scale_type = scale_type
        self.scale_degrees = SCALES[scale_type]

    def quantize_frequencies(
        self,
        frequencies: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Quantize frequencies to nearest scale notes.

        Args:
            frequencies: Array of frequencies in Hz
            strength: Quantization strength (0.0 = no quantization, 1.0 = full)

        Returns:
            Quantized frequencies in Hz

        Example:
            >>> freqs = np.array([440.0, 550.0, 660.0])
            >>> quantized = quantizer.quantize_frequencies(freqs, strength=1.0)
        """
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Strength must be 0-1, got {strength}")

        if strength == 0.0:
            return frequencies.copy()

        quantized = np.zeros_like(frequencies)

        for i, freq in enumerate(frequencies):
            if freq <= 0:
                quantized[i] = 0
                continue

            # Convert to MIDI
            midi_note = freq_to_midi(freq)

            # Quantize to scale
            quantized_midi = quantize_to_scale(
                midi_note,
                self.root_midi,
                self.scale_degrees
            )

            # Convert back to frequency
            quantized_freq = midi_to_freq(quantized_midi)

            # Interpolate based on strength
            quantized[i] = (1 - strength) * freq + strength * quantized_freq

        return quantized

    def quantize_spectrum(
        self,
        magnitude: np.ndarray,
        phase: np.ndarray,
        sample_rate: int,
        fft_size: int,
        strength: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize entire spectrum to scale.

        Maps energy from each frequency bin to the nearest scale frequency.

        Args:
            magnitude: (n_frames, n_bins) magnitude spectrum
            phase: (n_frames, n_bins) phase spectrum
            sample_rate: Sample rate in Hz
            fft_size: FFT size
            strength: Quantization strength (0-1)

        Returns:
            Tuple of (quantized_magnitude, quantized_phase)

        Example:
            >>> q_mag, q_phase = quantizer.quantize_spectrum(
            ...     mag, phase, 44100, 4096, strength=1.0
            ... )
        """
        if strength == 0.0:
            return magnitude.copy(), phase.copy()

        n_frames, n_bins = magnitude.shape

        # Get frequency values for each bin
        bin_freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

        # Quantize bin frequencies
        quantized_freqs = self.quantize_frequencies(bin_freqs, strength)

        # Calculate target bin indices
        bin_resolution = sample_rate / fft_size
        target_bins = np.round(quantized_freqs / bin_resolution).astype(int)

        # Clip to valid range
        target_bins = np.clip(target_bins, 0, n_bins - 1)

        # Initialize output arrays
        quantized_magnitude = np.zeros_like(magnitude)
        quantized_phase = np.zeros_like(phase)

        # Redistribute energy to quantized bins
        for k in range(n_bins):
            k_target = target_bins[k]

            # Accumulate magnitude (energy conservation)
            quantized_magnitude[:, k_target] += magnitude[:, k]

            # Use phase from strongest contributor (could be improved)
            # Simple approach: use phase of first contributor
            if magnitude[:, k].max() > quantized_magnitude[:, k_target].max() * 0.1:
                quantized_phase[:, k_target] = phase[:, k]

        return quantized_magnitude, quantized_phase

    def get_scale_frequencies(self, freq_range: Tuple[float, float] = (20, 20000)) -> List[float]:
        """
        Get all scale frequencies in a given range.

        Args:
            freq_range: (min_freq, max_freq) in Hz

        Returns:
            List of frequencies in scale within range

        Example:
            >>> scale_freqs = quantizer.get_scale_frequencies((100, 1000))
        """
        min_freq, max_freq = freq_range
        frequencies = []

        # Convert frequency range to MIDI range
        min_midi = int(np.floor(freq_to_midi(min_freq)))
        max_midi = int(np.ceil(freq_to_midi(max_freq)))

        # Generate all scale notes in MIDI range
        for midi in range(min_midi, max_midi + 1):
            # Check if this MIDI note is in the scale
            relative = (midi - self.root_midi) % 12
            if relative in self.scale_degrees:
                freq = midi_to_freq(midi)
                if min_freq <= freq <= max_freq:
                    frequencies.append(freq)

        return frequencies
