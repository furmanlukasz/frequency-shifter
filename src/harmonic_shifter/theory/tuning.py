"""
Musical tuning and MIDI conversion utilities.

This module provides functions for converting between frequencies and MIDI notes,
quantizing to musical scales, and calculating pitch differences in cents.
"""

from typing import List

import numpy as np


def freq_to_midi(freq: float, a4_freq: float = 440.0) -> float:
    """
    Convert frequency to MIDI note number.

    Uses the standard equal temperament formula:
    midi = 69 + 12 * log2(freq / 440)

    Args:
        freq: Frequency in Hz
        a4_freq: Reference frequency for A4 (default 440 Hz)

    Returns:
        MIDI note number (can be fractional for microtonal pitches)

    Example:
        >>> freq_to_midi(440.0)
        69.0
        >>> freq_to_midi(880.0)
        81.0
    """
    if freq <= 0:
        raise ValueError(f"Frequency must be positive, got {freq}")

    return 69 + 12 * np.log2(freq / a4_freq)


def midi_to_freq(midi: float, a4_freq: float = 440.0) -> float:
    """
    Convert MIDI note number to frequency.

    Uses the standard equal temperament formula:
    freq = 440 * 2^((midi - 69) / 12)

    Args:
        midi: MIDI note number (can be fractional)
        a4_freq: Reference frequency for A4 (default 440 Hz)

    Returns:
        Frequency in Hz

    Example:
        >>> midi_to_freq(69)
        440.0
        >>> midi_to_freq(81)
        880.0
    """
    return a4_freq * (2 ** ((midi - 69) / 12))


def quantize_to_scale(
    midi_note: float,
    root_midi: int,
    scale_degrees: List[int]
) -> int:
    """
    Quantize MIDI note to nearest scale degree.

    Takes a potentially fractional MIDI note and snaps it to the nearest
    note in the specified musical scale.

    Args:
        midi_note: Input MIDI note (can be fractional)
        root_midi: Root note of scale (MIDI number)
        scale_degrees: List of semitones from root (e.g., [0, 2, 4, 5, 7, 9, 11] for major)

    Returns:
        Quantized MIDI note number (integer)

    Example:
        >>> # Quantize to C major scale (C=60)
        >>> quantize_to_scale(61.5, 60, [0, 2, 4, 5, 7, 9, 11])
        62  # Snaps to D (62)
    """
    # Calculate relative note within octave
    relative_note = (midi_note - root_midi) % 12

    # Find closest scale degree
    scale_degrees_array = np.array(scale_degrees)
    differences = np.abs(scale_degrees_array - relative_note)

    # Handle wraparound (e.g., 11.5 might be closer to 0 than to 11)
    wraparound_diff = np.abs(scale_degrees_array - (relative_note - 12))
    differences = np.minimum(differences, wraparound_diff)

    closest_idx = np.argmin(differences)
    closest_degree = scale_degrees[closest_idx]

    # Calculate which octave we're in
    octave = int(np.floor((midi_note - root_midi) / 12))

    # Handle edge case where we wrapped around to lower octave
    if relative_note < closest_degree and closest_degree > 6:
        octave -= 1

    return root_midi + octave * 12 + closest_degree


def cents_difference(freq1: float, freq2: float) -> float:
    """
    Calculate difference between two frequencies in cents.

    A cent is 1/100 of a semitone. This function returns positive values
    when freq2 > freq1.

    Args:
        freq1: First frequency in Hz
        freq2: Second frequency in Hz

    Returns:
        Difference in cents (1 semitone = 100 cents, 1 octave = 1200 cents)

    Example:
        >>> cents_difference(440, 880)  # Octave
        1200.0
        >>> cents_difference(440, 466.16)  # Semitone
        100.0
    """
    if freq1 <= 0 or freq2 <= 0:
        raise ValueError("Frequencies must be positive")

    return 1200 * np.log2(freq2 / freq1)


def note_name_to_midi(note_name: str) -> int:
    """
    Convert note name to MIDI number.

    Args:
        note_name: Note name like 'C4', 'A#5', 'Bb3'

    Returns:
        MIDI note number

    Example:
        >>> note_name_to_midi('A4')
        69
        >>> note_name_to_midi('C4')
        60
    """
    note_map = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }

    # Parse note name
    if len(note_name) < 2:
        raise ValueError(f"Invalid note name: {note_name}")

    # Extract octave
    octave = int(note_name[-1])

    # Extract note
    note = note_name[:-1].upper()

    if note not in note_map:
        raise ValueError(f"Invalid note: {note}")

    # C4 (middle C) is MIDI 60
    # MIDI = (octave + 1) * 12 + note_offset
    return (octave + 1) * 12 + note_map[note]


def midi_to_note_name(midi: int) -> str:
    """
    Convert MIDI number to note name.

    Args:
        midi: MIDI note number (0-127)

    Returns:
        Note name (e.g., 'C4', 'A4')

    Example:
        >>> midi_to_note_name(69)
        'A4'
        >>> midi_to_note_name(60)
        'C4'
    """
    if not 0 <= midi <= 127:
        raise ValueError(f"MIDI note must be 0-127, got {midi}")

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    octave = (midi // 12) - 1
    note_idx = midi % 12

    return f"{note_names[note_idx]}{octave}"
