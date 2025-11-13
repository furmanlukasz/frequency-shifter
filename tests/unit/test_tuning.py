"""Tests for musical tuning and MIDI conversion utilities."""

import numpy as np
import pytest

from harmonic_shifter.theory.tuning import (
    freq_to_midi,
    midi_to_freq,
    quantize_to_scale,
    cents_difference,
    note_name_to_midi,
    midi_to_note_name,
)


class TestFreqToMidi:
    """Tests for freq_to_midi conversion."""

    def test_a4_reference(self):
        """MIDI 69 should be 440 Hz."""
        assert np.abs(freq_to_midi(440.0) - 69) < 0.01

    def test_octaves(self):
        """Test octave relationships."""
        a3 = freq_to_midi(220.0)  # A3
        a4 = freq_to_midi(440.0)  # A4
        a5 = freq_to_midi(880.0)  # A5

        assert np.abs(a4 - a3 - 12) < 0.01  # One octave = 12 semitones
        assert np.abs(a5 - a4 - 12) < 0.01

    def test_c4_middle_c(self):
        """Middle C (C4) should be MIDI 60."""
        c4_freq = 261.63  # Approximate C4 frequency
        midi = freq_to_midi(c4_freq)
        assert np.abs(midi - 60) < 0.1

    def test_negative_frequency_raises(self):
        """Negative frequency should raise ValueError."""
        with pytest.raises(ValueError):
            freq_to_midi(-100)

    def test_zero_frequency_raises(self):
        """Zero frequency should raise ValueError."""
        with pytest.raises(ValueError):
            freq_to_midi(0)


class TestMidiToFreq:
    """Tests for midi_to_freq conversion."""

    def test_a4_reference(self):
        """MIDI 69 should be 440 Hz."""
        assert np.abs(midi_to_freq(69) - 440.0) < 0.01

    def test_octave_doubling(self):
        """Each octave should double frequency."""
        c4 = midi_to_freq(60)  # C4
        c5 = midi_to_freq(72)  # C5
        c6 = midi_to_freq(84)  # C6

        assert np.abs(c5 / c4 - 2.0) < 0.001
        assert np.abs(c6 / c5 - 2.0) < 0.001

    def test_semitone_ratio(self):
        """Semitone ratio should be 2^(1/12)."""
        c4 = midi_to_freq(60)
        c_sharp_4 = midi_to_freq(61)

        ratio = c_sharp_4 / c4
        expected_ratio = 2 ** (1/12)

        assert np.abs(ratio - expected_ratio) < 0.001


class TestFreqMidiRoundtrip:
    """Tests for roundtrip conversions."""

    def test_freq_midi_roundtrip(self):
        """freq → MIDI → freq should be identical."""
        test_freqs = [220.0, 440.0, 880.0, 1000.0, 100.0, 5000.0]
        for freq in test_freqs:
            midi = freq_to_midi(freq)
            freq_back = midi_to_freq(midi)
            assert np.abs(freq - freq_back) < 0.01, \
                f"Roundtrip failed for {freq} Hz"

    def test_midi_freq_roundtrip(self):
        """MIDI → freq → MIDI should be identical."""
        test_midis = [21, 60, 69, 108, 127]  # Range of piano
        for midi in test_midis:
            freq = midi_to_freq(midi)
            midi_back = freq_to_midi(freq)
            assert np.abs(midi - midi_back) < 0.01, \
                f"Roundtrip failed for MIDI {midi}"


class TestQuantizeToScale:
    """Tests for scale quantization."""

    def test_exact_scale_note_unchanged(self):
        """Note already on scale should not change."""
        major_scale = [0, 2, 4, 5, 7, 9, 11]

        # C4 (60) in C major
        result = quantize_to_scale(60.0, 60, major_scale)
        assert result == 60

        # E4 (64) in C major
        result = quantize_to_scale(64.0, 60, major_scale)
        assert result == 64

    def test_quantize_between_notes(self):
        """Note between scale degrees should snap to nearest."""
        major_scale = [0, 2, 4, 5, 7, 9, 11]

        # 61 (C#) should snap to either 60 (C) or 62 (D)
        result = quantize_to_scale(61.0, 60, major_scale)
        assert result in [60, 62]

        # 61.8 should snap to 62 (D) - closer to D
        result = quantize_to_scale(61.8, 60, major_scale)
        assert result == 62

        # 61.2 should snap to 60 (C) - closer to C
        result = quantize_to_scale(61.2, 60, major_scale)
        assert result == 60

    def test_quantize_across_octaves(self):
        """Quantization should work across octaves."""
        major_scale = [0, 2, 4, 5, 7, 9, 11]

        # Test in different octaves
        result = quantize_to_scale(73.0, 60, major_scale)  # C#5
        assert result in [72, 74]  # Should snap to C5 or D5

        result = quantize_to_scale(49.0, 60, major_scale)  # C#3
        assert result in [48, 50]  # Should snap to C3 or D3

    def test_minor_scale(self):
        """Test quantization to minor scale."""
        minor_scale = [0, 2, 3, 5, 7, 8, 10]

        # Eb (63) is in C minor, E (64) is not
        result = quantize_to_scale(63.0, 60, minor_scale)
        assert result == 63

        result = quantize_to_scale(64.0, 60, minor_scale)
        assert result in [63, 65]  # Should snap to Eb or F

    def test_pentatonic_scale(self):
        """Test quantization to pentatonic scale."""
        pentatonic = [0, 2, 4, 7, 9]

        # F (65) not in C pentatonic major
        result = quantize_to_scale(65.0, 60, pentatonic)
        assert result in [64, 67]  # Should snap to E or G


class TestCentsDifference:
    """Tests for cents calculation."""

    def test_octave_1200_cents(self):
        """Octave should be 1200 cents."""
        cents = cents_difference(440, 880)
        assert np.abs(cents - 1200) < 0.1

    def test_semitone_100_cents(self):
        """Semitone should be 100 cents."""
        c4 = midi_to_freq(60)
        c_sharp_4 = midi_to_freq(61)
        cents = cents_difference(c4, c_sharp_4)
        assert np.abs(cents - 100) < 0.1

    def test_negative_cents(self):
        """Lower frequency should give negative cents."""
        cents = cents_difference(880, 440)
        assert np.abs(cents + 1200) < 0.1  # Should be -1200

    def test_same_frequency_zero_cents(self):
        """Same frequency should give 0 cents."""
        cents = cents_difference(440, 440)
        assert np.abs(cents) < 0.001

    def test_negative_frequency_raises(self):
        """Negative frequencies should raise ValueError."""
        with pytest.raises(ValueError):
            cents_difference(-440, 440)

        with pytest.raises(ValueError):
            cents_difference(440, -440)


class TestNoteNameConversions:
    """Tests for note name conversions."""

    def test_note_name_to_midi(self):
        """Test note name to MIDI conversion."""
        assert note_name_to_midi('A4') == 69
        assert note_name_to_midi('C4') == 60
        assert note_name_to_midi('C0') == 12
        assert note_name_to_midi('G9') == 127

    def test_sharps_and_flats(self):
        """Test sharps and flats."""
        assert note_name_to_midi('C#4') == 61
        assert note_name_to_midi('Db4') == 61  # Enharmonic
        assert note_name_to_midi('F#4') == 66
        assert note_name_to_midi('Gb4') == 66  # Enharmonic

    def test_midi_to_note_name(self):
        """Test MIDI to note name conversion."""
        assert midi_to_note_name(69) == 'A4'
        assert midi_to_note_name(60) == 'C4'
        assert midi_to_note_name(61) == 'C#4'

    def test_note_name_roundtrip(self):
        """Test roundtrip conversion."""
        notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
        for note in notes:
            midi = note_name_to_midi(note)
            note_back = midi_to_note_name(midi)
            # Note: sharps convert to sharps, so enharmonic equivalents won't match
            assert note_name_to_midi(note_back) == note_name_to_midi(note)

    def test_invalid_note_name_raises(self):
        """Invalid note names should raise ValueError."""
        with pytest.raises(ValueError):
            note_name_to_midi('X4')

        with pytest.raises(ValueError):
            note_name_to_midi('C')  # Missing octave

    def test_invalid_midi_raises(self):
        """Invalid MIDI numbers should raise ValueError."""
        with pytest.raises(ValueError):
            midi_to_note_name(-1)

        with pytest.raises(ValueError):
            midi_to_note_name(128)
