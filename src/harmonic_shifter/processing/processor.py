"""
Main processing pipeline for harmonic-preserving frequency shifting.

This module orchestrates the STFT, frequency shifting, quantization,
and enhanced phase vocoder to create high-quality frequency-shifted audio.
"""

from typing import Optional

import numpy as np

from ..core.stft import stft, istft
from ..core.frequency_shifter import FrequencyShifter
from ..core.quantizer import MusicalQuantizer
from ..core.phase_vocoder import propagate_phase_enhanced


class HarmonicShifter:
    """
    Main processor for harmonic-preserving frequency shifting.

    Combines frequency shifting with musical scale quantization and
    enhanced phase vocoder processing to create shifted audio that
    remains musical with minimal metallic artifacts.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        fft_size: int = 4096,
        hop_size: int = 1024,
        window: str = 'hann',
        use_enhanced_phase_vocoder: bool = True
    ):
        """
        Initialize harmonic shifter.

        Args:
            sample_rate: Audio sample rate in Hz
            fft_size: FFT window size (power of 2)
            hop_size: Hop size between frames in samples
            window: Window function ('hann', 'hamming', 'blackman')
            use_enhanced_phase_vocoder: Use enhanced phase vocoder (recommended)

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
        self.use_enhanced_phase_vocoder = use_enhanced_phase_vocoder

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
        quantize_strength: float = 1.0,
        preserve_loudness: bool = True
    ) -> np.ndarray:
        """
        Process audio with frequency shifting and scale quantization.

        This method uses an enhanced phase vocoder to maintain phase coherence
        across frames, reducing metallic artifacts.

        Supports both mono (1D) and stereo/multi-channel (2D) audio. For stereo,
        each channel is processed independently to preserve spatial information.

        Args:
            audio: Input audio, normalized to [-1, 1]
                  - Mono: (n_samples,) 1D array
                  - Stereo/Multi-channel: (n_samples, n_channels) 2D array
            shift_hz: Frequency shift in Hz (positive or negative)
            quantize_strength: 0-1, amount of scale quantization
                             0.0 = pure frequency shift (inharmonic)
                             1.0 = fully quantized to scale (harmonic)
            preserve_loudness: Apply automatic gain compensation (recommended)

        Returns:
            Processed audio (same shape as input)

        Raises:
            ValueError: If scale not set when quantize_strength > 0
            ValueError: If audio format is invalid

        Example:
            >>> # Mono
            >>> output = processor.process(mono_audio, shift_hz=100)
            >>> # Stereo
            >>> output = processor.process(stereo_audio, shift_hz=100)
        """
        # Handle multi-channel audio by processing each channel independently
        if len(audio.shape) == 2:
            # Multi-channel audio: process each channel separately
            n_samples, n_channels = audio.shape
            output_channels = []

            for ch in range(n_channels):
                channel_output = self._process_mono(
                    audio[:, ch],
                    shift_hz,
                    quantize_strength,
                    preserve_loudness
                )
                output_channels.append(channel_output)

            # Stack channels back together
            output = np.column_stack(output_channels)
            return output

        elif len(audio.shape) == 1:
            # Mono audio: process directly
            return self._process_mono(audio, shift_hz, quantize_strength, preserve_loudness)

        else:
            raise ValueError(f"Audio must be 1D (mono) or 2D (multi-channel), got shape {audio.shape}")

    def _process_mono(
        self,
        audio: np.ndarray,
        shift_hz: float,
        quantize_strength: float,
        preserve_loudness: bool
    ) -> np.ndarray:
        """
        Process a single channel of audio.

        This is the core processing function called by process() for each channel.

        Args:
            audio: Mono audio (1D array, normalized to [-1, 1])
            shift_hz: Frequency shift in Hz
            quantize_strength: Quantization strength (0-1)
            preserve_loudness: Apply gain compensation

        Returns:
            Processed mono audio (1D array)
        """
        # Validate inputs
        if len(audio.shape) != 1:
            raise ValueError(f"_process_mono expects 1D audio, got shape {audio.shape}")

        if quantize_strength < 0 or quantize_strength > 1:
            raise ValueError(f"Quantize strength must be 0-1, got {quantize_strength}")

        if quantize_strength > 0 and self.quantizer is None:
            raise ValueError(
                "Scale not set. Call set_scale() before processing with quantization."
            )

        # Measure input RMS for loudness preservation
        input_rms = np.sqrt(np.mean(audio ** 2)) if preserve_loudness else None

        # Handle empty or very short audio
        original_length = len(audio)
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

        # If using enhanced phase vocoder, process frame by frame
        if self.use_enhanced_phase_vocoder and shift_hz != 0:
            magnitude, phase = self._process_with_phase_vocoder(
                magnitude, phase, shift_hz, quantize_strength
            )
        else:
            # Simple processing without phase vocoder (legacy)
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
        output = output[:original_length]

        # Apply loudness preservation (automatic gain compensation)
        if preserve_loudness and input_rms is not None and input_rms > 1e-8:
            output_rms = np.sqrt(np.mean(output ** 2))
            if output_rms > 1e-8:
                # Apply gain to match input loudness
                gain = input_rms / output_rms
                # Limit gain to prevent extreme amplification
                gain = np.clip(gain, 0.1, 10.0)
                output = output * gain

        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val

        return output

    def _process_with_phase_vocoder(
        self,
        magnitude: np.ndarray,
        phase: np.ndarray,
        shift_hz: float,
        quantize_strength: float
    ) -> tuple:
        """
        Process spectrum with enhanced phase vocoder.

        This maintains phase coherence across frames to reduce artifacts.

        Args:
            magnitude: (n_frames, n_bins) magnitude spectrum
            phase: (n_frames, n_bins) phase spectrum
            shift_hz: Frequency shift in Hz
            quantize_strength: Quantization strength (0-1)

        Returns:
            Tuple of (processed_magnitude, processed_phase)
        """
        n_frames, n_bins = magnitude.shape

        # Initialize output arrays
        output_magnitude = np.zeros_like(magnitude)
        output_phase = np.zeros_like(phase)

        # Initialize synthesis phase for first frame
        synth_phase_prev = phase[0].copy()

        for frame_idx in range(n_frames):
            # Current frame
            mag_curr = magnitude[frame_idx]
            phase_curr = phase[frame_idx]

            if frame_idx == 0:
                # First frame: just copy (no previous frame for analysis)
                mag_shifted = mag_curr.copy()
                phase_shifted = phase_curr.copy()
            else:
                # Previous frame
                mag_prev = magnitude[frame_idx - 1]
                phase_prev = phase[frame_idx - 1]

                # Apply enhanced phase vocoder with peak detection
                inst_freq, locked_phase = propagate_phase_enhanced(
                    mag_prev, phase_prev,
                    mag_curr, phase_curr,
                    self.hop_size, self.sample_rate,
                    use_phase_locking=True
                )

                # Apply frequency shift to instantaneous frequencies
                shifted_freq = inst_freq + shift_hz

                # Synthesize phase for shifted frequencies
                phase_advance = 2 * np.pi * shifted_freq * self.hop_size / self.sample_rate
                synth_phase_curr = synth_phase_prev + phase_advance
                synth_phase_curr = np.angle(np.exp(1j * synth_phase_curr))

                mag_shifted = mag_curr.copy()
                phase_shifted = synth_phase_curr

                # Update previous synthesis phase
                synth_phase_prev = synth_phase_curr

            # Shift frequency bins (magnitude reassignment)
            if shift_hz != 0:
                bin_shift = int(round(shift_hz / (self.sample_rate / self.fft_size)))
                mag_shifted_bins = np.zeros_like(mag_shifted)
                phase_shifted_bins = np.zeros_like(phase_shifted)

                for k in range(n_bins):
                    k_new = k + bin_shift
                    if 0 <= k_new < n_bins:
                        mag_shifted_bins[k_new] = mag_shifted[k]
                        phase_shifted_bins[k_new] = phase_shifted[k]

                mag_shifted = mag_shifted_bins
                phase_shifted = phase_shifted_bins

            # Apply quantization if requested
            if quantize_strength > 0 and self.quantizer is not None:
                # Quantize to scale
                mag_reshaped = mag_shifted.reshape(1, -1)
                phase_reshaped = phase_shifted.reshape(1, -1)

                mag_quantized, phase_quantized = self.quantizer.quantize_spectrum(
                    mag_reshaped,
                    phase_reshaped,
                    self.sample_rate,
                    self.fft_size,
                    strength=quantize_strength
                )

                mag_shifted = mag_quantized[0]
                phase_shifted = phase_quantized[0]

            output_magnitude[frame_idx] = mag_shifted
            output_phase[frame_idx] = phase_shifted

        return output_magnitude, output_phase

    def process_batch(
        self,
        audio_list: list,
        shift_hz: float,
        quantize_strength: float = 1.0,
        preserve_loudness: bool = True
    ) -> list:
        """
        Process multiple audio signals with same parameters.

        Args:
            audio_list: List of audio arrays
            shift_hz: Frequency shift in Hz
            quantize_strength: Quantization strength (0-1)
            preserve_loudness: Apply automatic gain compensation (recommended)

        Returns:
            List of processed audio arrays

        Example:
            >>> outputs = processor.process_batch([audio1, audio2], shift_hz=100)
        """
        return [
            self.process(audio, shift_hz, quantize_strength, preserve_loudness)
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
            'enhanced_phase_vocoder': self.use_enhanced_phase_vocoder,
        }

        if self.quantizer is not None:
            info['scale_root_midi'] = self.quantizer.root_midi
            info['scale_type'] = self.quantizer.scale_type

        return info
