"""
Main processing pipeline for harmonic-preserving frequency shifting.

This module orchestrates the STFT, frequency shifting, quantization,
and phase vocoder to create the final effect.
"""

from typing import Optional

import numpy as np

from ..core.stft import stft, istft
from ..core.frequency_shifter import FrequencyShifter
from ..core.quantizer import MusicalQuantizer


class HarmonicShifter:
    """
    Main processor for harmonic-preserving frequency shifting.

    Combines frequency shifting with musical scale quantization to create
    shifted audio that remains musical.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        fft_size: int = 4096,
        hop_size: int = 1024,
        window: str = 'hann'
    ):
        """
        Initialize harmonic shifter.

        Args:
            sample_rate: Audio sample rate in Hz
            fft_size: FFT window size (power of 2)
            hop_size: Hop size between frames in samples
            window: Window function ('hann', 'hamming', 'blackman')

        Example:
            >>> processor = HarmonicShifter(sample_rate=44100, fft_size=4096)
        """
        if fft_size <= 0 or (fft_size & (fft_size - 1)) != 0:
            raise ValueError(f"FFT size must be power of 2, got {fft_size}")

        if hop_size <= 0 or hop_size > fft_size:
            raise ValueError(f"Hop size must be positive and <= fft_size")

        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")

        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window = window

        # Initialize frequency shifter
        self.shifter = FrequencyShifter(sample_rate, fft_size)

        # Quantizer will be set when scale is configured
        self.quantizer: Optional[MusicalQuantizer] = None

    def set_scale(self, root_midi: int, scale_type: str):
        """
        Configure musical scale for quantization.

        Args:
            root_midi: MIDI note number for scale root (0-127)
            scale_type: Scale name from SCALES dict

        Raises:
            ValueError: If invalid root_midi or scale_type

        Example:
            >>> processor.set_scale(60, 'major')  # C major
        """
        self.quantizer = MusicalQuantizer(root_midi, scale_type)

    def process(
        self,
        audio: np.ndarray,
        shift_hz: float,
        quantize_strength: float = 1.0
    ) -> np.ndarray:
        """
        Process audio with frequency shifting and scale quantization.

        Args:
            audio: Input audio (mono, 1D array, normalized to [-1, 1])
            shift_hz: Frequency shift in Hz (positive or negative)
            quantize_strength: 0-1, amount of scale quantization
                             0.0 = pure frequency shift (inharmonic)
                             1.0 = fully quantized to scale (harmonic)

        Returns:
            Processed audio (same shape as input)

        Raises:
            ValueError: If scale not set when quantize_strength > 0
            ValueError: If audio format is invalid

        Example:
            >>> output = processor.process(audio, shift_hz=100, quantize_strength=1.0)
        """
        # Validate inputs
        if len(audio.shape) != 1:
            raise ValueError(f"Audio must be 1D, got shape {audio.shape}")

        if quantize_strength < 0 or quantize_strength > 1:
            raise ValueError(f"Quantize strength must be 0-1, got {quantize_strength}")

        if quantize_strength > 0 and self.quantizer is None:
            raise ValueError(
                "Scale not set. Call set_scale() before processing with quantization."
            )

        # Handle empty or very short audio
        if len(audio) < self.fft_size:
            # Pad with zeros
            padded = np.zeros(self.fft_size)
            padded[:len(audio)] = audio
            audio = padded

        # Step 1: STFT analysis
        magnitude, phase = stft(
            audio,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            window=self.window
        )

        # Step 2: Frequency shifting
        if shift_hz != 0:
            magnitude, phase = self.shifter.shift(magnitude, phase, shift_hz)

        # Step 3: Musical quantization (if requested)
        if quantize_strength > 0 and self.quantizer is not None:
            magnitude, phase = self.quantizer.quantize_spectrum(
                magnitude,
                phase,
                self.sample_rate,
                self.fft_size,
                strength=quantize_strength
            )

        # Step 4: ISTFT synthesis
        output = istft(
            magnitude,
            phase,
            hop_size=self.hop_size,
            window=self.window
        )

        # Trim to original length
        output = output[:len(audio)]

        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val

        return output

    def process_batch(
        self,
        audio_list: list,
        shift_hz: float,
        quantize_strength: float = 1.0
    ) -> list:
        """
        Process multiple audio signals with same parameters.

        Args:
            audio_list: List of audio arrays
            shift_hz: Frequency shift in Hz
            quantize_strength: Quantization strength (0-1)

        Returns:
            List of processed audio arrays

        Example:
            >>> outputs = processor.process_batch([audio1, audio2], shift_hz=100)
        """
        return [
            self.process(audio, shift_hz, quantize_strength)
            for audio in audio_list
        ]

    def get_latency_ms(self) -> float:
        """
        Calculate processing latency in milliseconds.

        Returns:
            Latency in ms

        Example:
            >>> latency = processor.get_latency_ms()
            >>> print(f"Latency: {latency:.1f} ms")
        """
        latency_samples = self.fft_size + self.hop_size
        return (latency_samples / self.sample_rate) * 1000

    def get_info(self) -> dict:
        """
        Get processor configuration information.

        Returns:
            Dictionary with configuration details

        Example:
            >>> info = processor.get_info()
            >>> print(info['latency_ms'])
        """
        info = {
            'sample_rate': self.sample_rate,
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'window': self.window,
            'latency_ms': self.get_latency_ms(),
            'frequency_resolution_hz': self.sample_rate / self.fft_size,
        }

        if self.quantizer is not None:
            info['scale_root_midi'] = self.quantizer.root_midi
            info['scale_type'] = self.quantizer.scale_type

        return info
