"""
Short-Time Fourier Transform (STFT) and inverse STFT implementation.

This module provides windowed FFT analysis and overlap-add synthesis
for time-frequency processing of audio signals.
"""

from typing import Tuple

import numpy as np
import scipy.signal as signal


def stft(
    audio: np.ndarray,
    fft_size: int = 4096,
    hop_size: int = 1024,
    window: str = 'hann'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Short-Time Fourier Transform.

    Analyzes audio signal using windowed FFT, returning magnitude and
    phase spectra.

    Args:
        audio: Input audio signal (1D array)
        fft_size: FFT window size (should be power of 2)
        hop_size: Hop size between frames in samples
        window: Window function type ('hann', 'hamming', 'blackman')

    Returns:
        Tuple of (magnitude, phase):
            - magnitude: (n_frames, n_bins) magnitude spectrum
            - phase: (n_frames, n_bins) phase spectrum in radians

    Example:
        >>> import numpy as np
        >>> signal = np.random.randn(44100)
        >>> mag, phase = stft(signal, fft_size=2048, hop_size=512)
        >>> mag.shape
        (86, 1025)
    """
    # Validate inputs
    if len(audio.shape) != 1:
        raise ValueError(f"Audio must be 1D, got shape {audio.shape}")

    if fft_size <= 0 or (fft_size & (fft_size - 1)) != 0:
        raise ValueError(f"FFT size must be power of 2, got {fft_size}")

    if hop_size <= 0:
        raise ValueError(f"Hop size must be positive, got {hop_size}")

    # Get window function
    if window == 'hann':
        win = np.hanning(fft_size)
    elif window == 'hamming':
        win = np.hamming(fft_size)
    elif window == 'blackman':
        win = np.blackman(fft_size)
    else:
        raise ValueError(f"Unknown window type: {window}")

    # Calculate number of frames
    n_frames = 1 + (len(audio) - fft_size) // hop_size

    # Initialize output arrays
    n_bins = fft_size // 2 + 1
    magnitude = np.zeros((n_frames, n_bins))
    phase = np.zeros((n_frames, n_bins))

    # Process each frame
    for frame_idx in range(n_frames):
        # Extract frame
        start = frame_idx * hop_size
        end = start + fft_size

        if end > len(audio):
            # Zero-pad if necessary
            frame = np.zeros(fft_size)
            frame[:len(audio) - start] = audio[start:]
        else:
            frame = audio[start:end]

        # Apply window
        windowed_frame = frame * win

        # Compute FFT (only positive frequencies)
        spectrum = np.fft.rfft(windowed_frame)

        # Extract magnitude and phase
        magnitude[frame_idx] = np.abs(spectrum)
        phase[frame_idx] = np.angle(spectrum)

    return magnitude, phase


def istft(
    magnitude: np.ndarray,
    phase: np.ndarray,
    hop_size: int = 1024,
    window: str = 'hann'
) -> np.ndarray:
    """
    Inverse Short-Time Fourier Transform with overlap-add.

    Synthesizes audio signal from magnitude and phase spectra using
    overlap-add method.

    Args:
        magnitude: (n_frames, n_bins) magnitude spectrum
        phase: (n_frames, n_bins) phase spectrum in radians
        hop_size: Hop size between frames in samples
        window: Window function type (must match STFT)

    Returns:
        Reconstructed audio signal (1D array)

    Example:
        >>> mag, phase = stft(signal)
        >>> reconstructed = istft(mag, phase)
    """
    # Validate inputs
    if magnitude.shape != phase.shape:
        raise ValueError(
            f"Magnitude and phase shapes must match, "
            f"got {magnitude.shape} and {phase.shape}"
        )

    n_frames, n_bins = magnitude.shape
    fft_size = (n_bins - 1) * 2

    # Get window function
    if window == 'hann':
        win = np.hanning(fft_size)
    elif window == 'hamming':
        win = np.hamming(fft_size)
    elif window == 'blackman':
        win = np.blackman(fft_size)
    else:
        raise ValueError(f"Unknown window type: {window}")

    # Calculate output length
    output_length = fft_size + (n_frames - 1) * hop_size

    # Initialize output and window normalization arrays
    output = np.zeros(output_length)
    window_sum = np.zeros(output_length)

    # Process each frame
    for frame_idx in range(n_frames):
        # Reconstruct complex spectrum
        spectrum = magnitude[frame_idx] * np.exp(1j * phase[frame_idx])

        # Inverse FFT
        frame = np.fft.irfft(spectrum)

        # Apply window
        windowed_frame = frame * win

        # Overlap-add
        start = frame_idx * hop_size
        end = start + fft_size

        output[start:end] += windowed_frame
        window_sum[start:end] += win ** 2

    # Normalize by window
    # Avoid division by zero
    window_sum = np.maximum(window_sum, 1e-8)
    output /= window_sum

    return output


def get_frequency_bins(fft_size: int, sample_rate: int) -> np.ndarray:
    """
    Get frequency values for each FFT bin.

    Args:
        fft_size: FFT size
        sample_rate: Sample rate in Hz

    Returns:
        Array of frequency values in Hz for each bin

    Example:
        >>> freqs = get_frequency_bins(4096, 44100)
        >>> freqs[0]  # DC bin
        0.0
        >>> freqs[-1]  # Nyquist bin
        22050.0
    """
    return np.fft.rfftfreq(fft_size, 1.0 / sample_rate)


def get_time_frames(n_frames: int, hop_size: int, sample_rate: int) -> np.ndarray:
    """
    Get time values for each STFT frame.

    Args:
        n_frames: Number of frames
        hop_size: Hop size in samples
        sample_rate: Sample rate in Hz

    Returns:
        Array of time values in seconds for each frame

    Example:
        >>> times = get_time_frames(100, 512, 44100)
        >>> times[0]
        0.0
        >>> times[1]
        0.0116...
    """
    return np.arange(n_frames) * hop_size / sample_rate


def get_spectrogram(audio: np.ndarray, **stft_params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute magnitude spectrogram with time and frequency axes.

    Args:
        audio: Input audio signal
        **stft_params: Parameters passed to stft()

    Returns:
        Tuple of (magnitude, frequencies, times):
            - magnitude: (n_frames, n_bins) magnitude spectrogram
            - frequencies: (n_bins,) frequency values in Hz
            - times: (n_frames,) time values in seconds

    Example:
        >>> mag, freqs, times = get_spectrogram(signal, fft_size=2048)
    """
    # Get defaults
    fft_size = stft_params.get('fft_size', 4096)
    hop_size = stft_params.get('hop_size', 1024)
    sample_rate = stft_params.get('sample_rate', 44100)

    # Compute STFT
    magnitude, _ = stft(audio, **stft_params)

    # Get axes
    frequencies = get_frequency_bins(fft_size, sample_rate)
    times = get_time_frames(magnitude.shape[0], hop_size, sample_rate)

    return magnitude, frequencies, times
