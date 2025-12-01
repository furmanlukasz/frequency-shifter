"""
Audio quality validation and metrics.

This module provides functions for measuring audio quality metrics
such as SNR, THD, and scale conformance.
"""

from typing import Dict

import numpy as np


def compute_snr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio in dB.

    Args:
        original: Reference signal
        processed: Processed signal

    Returns:
        SNR in dB (higher is better, >60 dB is excellent)

    Example:
        >>> snr = compute_snr(original, processed)
        >>> print(f"SNR: {snr:.1f} dB")
    """
    # Ensure same length
    min_len = min(len(original), len(processed))
    original = original[:min_len]
    processed = processed[:min_len]

    # Calculate noise (difference)
    noise = processed - original

    # Calculate powers
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)

    # Avoid division by zero
    if noise_power < 1e-10:
        return 100.0  # Effectively perfect

    # SNR in dB
    snr_db = 10 * np.log10(signal_power / noise_power)

    return snr_db


def compute_rms(audio: np.ndarray) -> float:
    """
    Calculate RMS (Root Mean Square) level.

    Args:
        audio: Audio signal

    Returns:
        RMS level

    Example:
        >>> rms = compute_rms(audio)
    """
    return np.sqrt(np.mean(audio ** 2))


def measure_latency(fft_size: int, hop_size: int, sample_rate: int) -> float:
    """
    Calculate processing latency in milliseconds.

    Args:
        fft_size: FFT size in samples
        hop_size: Hop size in samples
        sample_rate: Sample rate in Hz

    Returns:
        Latency in ms

    Example:
        >>> latency = measure_latency(4096, 1024, 44100)
        >>> print(f"Latency: {latency:.1f} ms")
    """
    latency_samples = fft_size + hop_size
    latency_ms = (latency_samples / sample_rate) * 1000
    return latency_ms


def compute_spectral_centroid(magnitude: np.ndarray, frequencies: np.ndarray) -> float:
    """
    Calculate spectral centroid (brightness measure).

    Args:
        magnitude: Magnitude spectrum
        frequencies: Frequency values for each bin

    Returns:
        Spectral centroid in Hz

    Example:
        >>> centroid = compute_spectral_centroid(mag, freqs)
    """
    # Normalize magnitude
    magnitude = magnitude / (np.sum(magnitude) + 1e-10)

    # Weighted average of frequencies
    centroid = np.sum(frequencies * magnitude)

    return centroid


def check_for_clipping(audio: np.ndarray, threshold: float = 0.99) -> Dict[str, any]:
    """
    Check if audio is clipping.

    Args:
        audio: Audio signal
        threshold: Threshold for clipping detection (default 0.99)

    Returns:
        Dictionary with clipping information

    Example:
        >>> clip_info = check_for_clipping(audio)
        >>> if clip_info['is_clipping']:
        ...     print(f"Warning: {clip_info['percent_clipped']:.1f}% samples clipping")
    """
    max_val = np.max(np.abs(audio))
    clipped_samples = np.sum(np.abs(audio) >= threshold)
    total_samples = len(audio)

    return {
        'is_clipping': max_val >= 1.0,
        'max_value': max_val,
        'num_clipped_samples': clipped_samples,
        'percent_clipped': (clipped_samples / total_samples) * 100,
    }


def compute_thd(
    audio: np.ndarray,
    sample_rate: int,
    fundamental_freq: float,
    n_harmonics: int = 5
) -> float:
    """
    Calculate Total Harmonic Distortion percentage.

    Args:
        audio: Audio signal
        sample_rate: Sample rate
        fundamental_freq: Fundamental frequency in Hz
        n_harmonics: Number of harmonics to analyze

    Returns:
        THD as percentage (0-100, lower is better, <1% is excellent)

    Example:
        >>> thd = compute_thd(audio, 44100, 440.0, n_harmonics=5)
        >>> print(f"THD: {thd:.2f}%")
    """
    # Compute FFT
    spectrum = np.fft.rfft(audio)
    magnitude = np.abs(spectrum)
    frequencies = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)

    # Find fundamental
    fund_bin = np.argmin(np.abs(frequencies - fundamental_freq))
    fund_power = magnitude[fund_bin] ** 2

    # Find harmonics
    harmonic_power = 0
    for i in range(2, n_harmonics + 2):  # Start from 2nd harmonic
        harmonic_freq = fundamental_freq * i
        if harmonic_freq > sample_rate / 2:
            break

        harm_bin = np.argmin(np.abs(frequencies - harmonic_freq))
        harmonic_power += magnitude[harm_bin] ** 2

    # Calculate THD
    if fund_power < 1e-10:
        return 0.0

    thd = np.sqrt(harmonic_power / fund_power) * 100

    return thd


def check_scale_conformance(
    audio: np.ndarray,
    sample_rate: int,
    root_midi: int,
    scale_type: str,
    tolerance_cents: float = 50.0
) -> Dict[str, float]:
    """
    Measure how well audio conforms to musical scale.

    This is a simplified version that checks spectral peaks against scale notes.

    Args:
        audio: Audio signal
        sample_rate: Sample rate
        root_midi: Scale root note
        scale_type: Scale name
        tolerance_cents: Tolerance window in cents

    Returns:
        Dictionary with:
            - 'conformance': % of energy within tolerance
            - 'mean_deviation': Average cents from nearest scale note (simplified)

    Example:
        >>> metrics = check_scale_conformance(audio, 44100, 60, 'major')
        >>> print(f"Conformance: {metrics['conformance']*100:.1f}%")
    """
    from ..theory.scales import SCALES
    from ..theory.tuning import freq_to_midi, cents_difference, midi_to_freq

    # Get scale frequencies
    scale_degrees = SCALES[scale_type]

    # Compute spectrum
    spectrum = np.fft.rfft(audio)
    magnitude = np.abs(spectrum)
    frequencies = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)

    # Find spectral peaks (simplified)
    threshold = np.max(magnitude) * 0.1
    peaks = magnitude > threshold

    # Check conformance
    total_energy = np.sum(magnitude[peaks] ** 2)
    conforming_energy = 0

    deviations = []

    for i, is_peak in enumerate(peaks):
        if is_peak and frequencies[i] > 20:  # Ignore very low frequencies
            freq = frequencies[i]

            # Convert to MIDI
            midi = freq_to_midi(freq)

            # Find nearest scale note
            relative = (midi - root_midi) % 12
            distances = [abs(relative - deg) for deg in scale_degrees]

            # Handle wraparound
            distances.append(abs(relative - 12))

            min_dist_semitones = min(distances)
            min_dist_cents = min_dist_semitones * 100

            if min_dist_cents <= tolerance_cents:
                conforming_energy += magnitude[i] ** 2

            deviations.append(min_dist_cents)

    # Calculate metrics
    conformance = conforming_energy / (total_energy + 1e-10)
    mean_deviation = np.mean(deviations) if deviations else 0.0

    return {
        'conformance': conformance,
        'mean_deviation': mean_deviation,
        'max_deviation': max(deviations) if deviations else 0.0,
    }
