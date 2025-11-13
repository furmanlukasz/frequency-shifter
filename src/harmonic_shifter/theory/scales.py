"""
Musical scale definitions and utilities.

This module provides scale definitions for various musical scales and
utilities for working with them.
"""

from typing import Dict, List, Tuple

from .tuning import midi_to_freq


# Scale definitions: semitones from root
SCALES: Dict[str, List[int]] = {
    # Western scales
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'natural_minor': [0, 2, 3, 5, 7, 8, 10],  # Alias for minor
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],

    # Modes
    'ionian': [0, 2, 4, 5, 7, 9, 11],  # Same as major
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'aeolian': [0, 2, 3, 5, 7, 8, 10],  # Same as natural minor
    'locrian': [0, 1, 3, 5, 6, 8, 10],

    # Pentatonic
    'pentatonic_major': [0, 2, 4, 7, 9],
    'pentatonic_minor': [0, 3, 5, 7, 10],

    # Blues
    'blues': [0, 3, 5, 6, 7, 10],

    # Chromatic
    'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],

    # Whole tone
    'whole_tone': [0, 2, 4, 6, 8, 10],

    # Diminished
    'diminished': [0, 2, 3, 5, 6, 8, 9, 11],
    'half_whole_diminished': [0, 1, 3, 4, 6, 7, 9, 10],

    # Exotic scales
    'arabic': [0, 1, 4, 5, 7, 8, 11],
    'japanese': [0, 1, 5, 7, 8],
    'spanish': [0, 1, 3, 4, 5, 6, 8, 10],
}


# Human-readable scale names
SCALE_NAMES: Dict[str, str] = {
    'major': 'Major',
    'minor': 'Minor',
    'natural_minor': 'Natural Minor',
    'harmonic_minor': 'Harmonic Minor',
    'melodic_minor': 'Melodic Minor',
    'ionian': 'Ionian',
    'dorian': 'Dorian',
    'phrygian': 'Phrygian',
    'lydian': 'Lydian',
    'mixolydian': 'Mixolydian',
    'aeolian': 'Aeolian',
    'locrian': 'Locrian',
    'pentatonic_major': 'Pentatonic Major',
    'pentatonic_minor': 'Pentatonic Minor',
    'blues': 'Blues',
    'chromatic': 'Chromatic',
    'whole_tone': 'Whole Tone',
    'diminished': 'Diminished',
    'half_whole_diminished': 'Half-Whole Diminished',
    'arabic': 'Arabic',
    'japanese': 'Japanese',
    'spanish': 'Spanish (Phrygian Dominant)',
}


def get_scale_frequencies(
    root_midi: int,
    scale_type: str,
    octave_range: Tuple[int, int] = (0, 10)
) -> List[float]:
    """
    Return all frequencies in scale across MIDI range.

    Args:
        root_midi: Root note MIDI number
        scale_type: Scale name from SCALES dict
        octave_range: (min_octave, max_octave) relative to root

    Returns:
        List of frequencies in Hz, sorted ascending

    Raises:
        ValueError: If scale_type not found

    Example:
        >>> # Get C major scale across 3 octaves
        >>> freqs = get_scale_frequencies(60, 'major', octave_range=(0, 3))
        >>> len(freqs)  # 7 notes * 3 octaves + root of next octave
        22
    """
    if scale_type not in SCALES:
        raise ValueError(
            f"Unknown scale type '{scale_type}'. "
            f"Available scales: {', '.join(sorted(SCALES.keys()))}"
        )

    scale_degrees = SCALES[scale_type]
    frequencies = []

    min_octave, max_octave = octave_range

    for octave in range(min_octave, max_octave):
        for degree in scale_degrees:
            midi_note = root_midi + octave * 12 + degree
            # Stay within MIDI range (0-127)
            if 0 <= midi_note <= 127:
                freq = midi_to_freq(midi_note)
                frequencies.append(freq)

    return sorted(frequencies)


def get_scale_name(scale_type: str) -> str:
    """
    Get human-readable scale name.

    Args:
        scale_type: Scale type key

    Returns:
        Human-readable name

    Example:
        >>> get_scale_name('pentatonic_major')
        'Pentatonic Major'
    """
    return SCALE_NAMES.get(scale_type, scale_type.replace('_', ' ').title())


def get_available_scales() -> List[str]:
    """
    Get list of all available scale types.

    Returns:
        Sorted list of scale type keys

    Example:
        >>> scales = get_available_scales()
        >>> 'major' in scales
        True
    """
    return sorted(SCALES.keys())


def validate_scale_type(scale_type: str) -> bool:
    """
    Check if scale type is valid.

    Args:
        scale_type: Scale type to validate

    Returns:
        True if valid, False otherwise

    Example:
        >>> validate_scale_type('major')
        True
        >>> validate_scale_type('invalid')
        False
    """
    return scale_type in SCALES


def get_scale_info(scale_type: str) -> Dict[str, any]:
    """
    Get detailed information about a scale.

    Args:
        scale_type: Scale type key

    Returns:
        Dictionary with scale information

    Raises:
        ValueError: If scale_type not found

    Example:
        >>> info = get_scale_info('major')
        >>> info['name']
        'Major'
        >>> info['degrees']
        [0, 2, 4, 5, 7, 9, 11]
    """
    if scale_type not in SCALES:
        raise ValueError(f"Unknown scale type '{scale_type}'")

    return {
        'type': scale_type,
        'name': get_scale_name(scale_type),
        'degrees': SCALES[scale_type],
        'num_notes': len(SCALES[scale_type]),
    }
