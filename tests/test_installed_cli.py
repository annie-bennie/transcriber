"""Exercise both text formats with the installed transcribe command outside the checkout."""
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    state = {
        "source": {"name": "example.wav"},
        "stages": {"transcribe": "done"},
        "speakers": ["A", "B"],
        "segments": [
            {"id": 0, "start": 0, "end": 1, "speaker": "A", "text": "First."},
            {"id": 1, "start": 1, "end": 2, "speaker": "A", "text": "Second."},
            {"id": 2, "start": 2, "end": 3, "speaker": "B", "text": "Third."},
        ],
        "emotion_units": [{"segments": [0], "arousal_z": 3.0}],
    }
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "example.json").write_text(json.dumps(state), encoding="utf-8")
        subprocess.run(["transcribe", ".", "--rebuild-txt"], cwd=folder, check=True)
        report = (folder / "example_speakers.txt").read_text(encoding="utf-8")
        assert "[A 00:00-00:01] (aro +3.0σ): First." in report, report
        subprocess.run(["transcribe", "example.json", "--format", "lecture"],
                       cwd=folder, check=True)
        lecture = (folder / "example_lecture.txt").read_text(encoding="utf-8")
        assert lecture == "A: First. Second.\n\nB: Third.\n", lecture


if __name__ == "__main__":
    main()
