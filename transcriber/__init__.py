# -*- coding: utf-8 -*-
"""Local transcription with speaker separation and speech measurement.

the public surface is small on purpose: most work goes through the cli, and the json is the
real interface for anything downstream.
"""
from .config import SAMPLE_RATE
from .emotion import DEVIATION_GUIDE, EMOTION_DIMENSION_GUIDE, EMOTION_DIMENSIONS
from .pipeline import Transcriber
from .state import SCHEMA_VERSION, load_state, save_state

__version__ = "0.6.0"

__all__ = [
    "Transcriber",
    "SCHEMA_VERSION",
    "SAMPLE_RATE",
    "EMOTION_DIMENSIONS",
    "DEVIATION_GUIDE",
    "EMOTION_DIMENSION_GUIDE",
    "load_state",
    "save_state",
    "__version__",
]
