# -*- coding: utf-8 -*-
"""Entry point. the code lives in the transcriber/ package.

kept as a file so that `python transcribe.py ...` keeps working, and so do transcribe.sh
and transcribe.bat.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transcriber.cli import run

if __name__ == "__main__":
    sys.exit(run())
