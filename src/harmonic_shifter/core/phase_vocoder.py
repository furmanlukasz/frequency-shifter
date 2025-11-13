"""
Phase vocoder for maintaining phase coherence.

This module provides phase propagation and transfer functions to maintain
phase relationships when modifying spectral content.
"""

import numpy as np


def propagate_phase(
    phase_prev: np.ndarray,
    phase_curr: np.ndarray,
    hop_size: int,
    fft_size: int
) -> np.ndarray:
    """
    Calculate instantaneous phase for smooth transitions.

    Computes the phase advance expected for each bin and unwraps phase
    to maintain continuity across frames.

    Args:
        phase_prev: Phase from previous frame (n_bins,)
        phase_curr: Phase from current frame (n_bins,)
        hop_size: Hop size in samples
        fft_size: FFT size

    Returns:
        Instantaneous phase (n_bins,)

    Example:
        >>> inst_phase = propagate_phase(phase_prev, phase_curr, 1024, 4096)
    """
    n_bins = len(phase_curr)

    # Expected phase advance for each bin
    expected_phase_advance = 2 * np.pi * hop_size * np.arange(n_bins) / fft_size

    # Actual phase advance
    phase_diff = phase_curr - phase_prev

    # Unwrap phase difference
    phase_diff_unwrapped = np.angle(np.exp(1j * phase_diff))

    # Deviation from expected phase advance
    phase_deviation = phase_diff_unwrapped - expected_phase_advance

    # Wrap to [-π, π]
    phase_deviation = np.angle(np.exp(1j * phase_deviation))

    # Instantaneous phase
    instantaneous_phase = expected_phase_advance + phase_deviation

    return instantaneous_phase


def transfer_phase(
    phase: np.ndarray,
    source_bins: np.ndarray,
    target_bins: np.ndarray,
    instantaneous_phase: np.ndarray
) -> np.ndarray:
    """
    Transfer phase from source to target bins while maintaining coherence.

    When moving energy between bins (e.g., for frequency shifting or
    quantization), this function properly scales the phase to maintain
    time-frequency relationships.

    Args:
        phase: Original phase values (n_bins,)
        source_bins: Source bin indices
        target_bins: Target bin indices
        instantaneous_phase: Instantaneous phase from propagate_phase

    Returns:
        Transferred phase values at target bins (n_bins,)

    Example:
        >>> new_phase = transfer_phase(phase, src, tgt, inst_phase)
    """
    n_bins = len(phase)
    transferred_phase = np.zeros(n_bins)

    for src_bin, tgt_bin in zip(source_bins, target_bins):
        if 0 <= src_bin < n_bins and 0 <= tgt_bin < n_bins:
            # Scale phase based on frequency ratio
            if src_bin > 0:
                freq_ratio = tgt_bin / src_bin
            else:
                freq_ratio = 1.0

            # Transfer and scale instantaneous phase
            transferred_phase[tgt_bin] = instantaneous_phase[src_bin] * freq_ratio

    return transferred_phase


def compute_phase_advance(
    hop_size: int,
    fft_size: int,
    n_bins: int
) -> np.ndarray:
    """
    Compute expected phase advance for each frequency bin.

    Args:
        hop_size: Hop size in samples
        fft_size: FFT size
        n_bins: Number of frequency bins

    Returns:
        Expected phase advance for each bin (radians)

    Example:
        >>> phase_advance = compute_phase_advance(1024, 4096, 2049)
    """
    return 2 * np.pi * hop_size * np.arange(n_bins) / fft_size


def unwrap_phase(phase: np.ndarray) -> np.ndarray:
    """
    Unwrap phase to remove discontinuities.

    Args:
        phase: Phase values in radians

    Returns:
        Unwrapped phase values

    Example:
        >>> unwrapped = unwrap_phase(phase)
    """
    return np.unwrap(phase)


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """
    Wrap phase to [-π, π] range.

    Args:
        phase: Phase values in radians

    Returns:
        Wrapped phase values

    Example:
        >>> wrapped = wrap_phase(phase)
    """
    return np.angle(np.exp(1j * phase))
