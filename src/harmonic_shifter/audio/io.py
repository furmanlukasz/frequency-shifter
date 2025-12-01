"""
Audio file I/O utilities.

This module provides functions for loading and saving audio files
with proper error handling and format validation.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


def load_audio(
    filepath: str,
    sample_rate: int = 44100,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file, resample if needed, convert to mono if requested.

    Args:
        filepath: Path to audio file
        sample_rate: Target sample rate (None to keep original)
        mono: Convert to mono if True

    Returns:
        Tuple of (audio, sr):
            - audio: Audio data normalized to [-1, 1]
            - sr: Sample rate

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not supported

    Example:
        >>> audio, sr = load_audio('input.wav', sample_rate=44100, mono=True)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    try:
        # Load audio file
        audio, sr = sf.read(str(filepath))

    except Exception as e:
        raise ValueError(f"Error loading audio file: {e}")

    # Convert to mono if requested and audio is stereo/multi-channel
    if mono and len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Ensure 1D array (only if mono was requested)
    if mono and len(audio.shape) > 1:
        raise ValueError(
            f"Expected mono audio after conversion, got shape {audio.shape}"
        )

    # Resample if needed
    if sample_rate is not None and sr != sample_rate:
        try:
            import scipy.signal as signal
            # Calculate resampling ratio
            num_samples = int(len(audio) * sample_rate / sr)
            audio = signal.resample(audio, num_samples)
            sr = sample_rate
        except ImportError:
            import warnings
            warnings.warn(
                f"scipy not available for resampling. "
                f"Returning audio at {sr} Hz instead of {sample_rate} Hz"
            )

    # Normalize to [-1, 1]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Validate
    validate_audio(audio)

    return audio, sr


def save_audio(
    filepath: str,
    audio: np.ndarray,
    sample_rate: int = 44100,
    subtype: str = 'PCM_24'
):
    """
    Save audio to file.

    Supports both mono and stereo/multi-channel audio.

    Args:
        filepath: Output file path
        audio: Audio data (will be clipped to [-1, 1])
              - Mono: (n_samples,) 1D array
              - Stereo/Multi-channel: (n_samples, n_channels) 2D array
        sample_rate: Sample rate in Hz
        subtype: Bit depth ('PCM_16', 'PCM_24', 'FLOAT')

    Raises:
        ValueError: If audio format is invalid

    Example:
        >>> # Mono
        >>> save_audio('output_mono.wav', mono_audio, sample_rate=44100)
        >>> # Stereo
        >>> save_audio('output_stereo.wav', stereo_audio, sample_rate=44100)
    """
    # Validate audio (allow multi-channel)
    validate_audio(audio, allow_multichannel=True)

    # Clip to [-1, 1] to prevent clipping artifacts
    audio = np.clip(audio, -1.0, 1.0)

    # Create output directory if it doesn't exist
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        sf.write(str(filepath), audio, sample_rate, subtype=subtype)
    except Exception as e:
        raise ValueError(f"Error saving audio file: {e}")


def validate_audio(audio: np.ndarray, allow_multichannel: bool = True) -> None:
    """
    Validate audio array format.

    Args:
        audio: Audio array to validate
        allow_multichannel: Allow 2D multi-channel audio (default True)

    Raises:
        ValueError: If audio format is invalid

    Example:
        >>> validate_audio(audio)
    """
    if not isinstance(audio, np.ndarray):
        raise ValueError(f"Audio must be numpy array, got {type(audio)}")

    if len(audio.shape) == 1:
        # Mono audio
        if len(audio) == 0:
            raise ValueError("Audio array is empty")
    elif len(audio.shape) == 2:
        # Multi-channel audio
        if not allow_multichannel:
            raise ValueError(f"Multi-channel audio not allowed, got shape {audio.shape}")
        if audio.shape[0] == 0:
            raise ValueError("Audio array is empty")
    else:
        raise ValueError(f"Audio must be 1D (mono) or 2D (multi-channel), got shape {audio.shape}")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or Inf values")


def get_audio_info(filepath: str) -> dict:
    """
    Get information about an audio file without loading it.

    Args:
        filepath: Path to audio file

    Returns:
        Dictionary with file information

    Example:
        >>> info = get_audio_info('audio.wav')
        >>> print(f"Duration: {info['duration']:.2f} seconds")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    info = sf.info(str(filepath))

    return {
        'sample_rate': info.samplerate,
        'channels': info.channels,
        'duration': info.duration,
        'frames': info.frames,
        'format': info.format,
        'subtype': info.subtype,
    }
