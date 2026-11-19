"""Launch the bundled DiariZen worker from a source checkout."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    worker = Path(__file__).resolve().parents[1] / "transcriber/_workers/diarize_diarizen.py"
    runpy.run_path(str(worker), run_name="__main__")
