"""
Harmonic-Preserving Frequency Shifter

A Python library for frequency shifting audio while maintaining musical
harmonic relationships through intelligent scale quantization.
"""

__version__ = "0.1.0"

from .processing.processor import HarmonicShifter
from .theory.scales import SCALES
from .audio.io import load_audio, save_audio

__all__ = [
    "__version__",
    "HarmonicShifter",
    "SCALES",
    "load_audio",
    "save_audio",
]
