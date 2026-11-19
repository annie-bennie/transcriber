# Project structure

## Application

`transcriber/` contains the application code. `cli.py` parses commands, `pipeline.py`
coordinates processing, `state.py` persists results, and `render.py` produces text.
Recognition, speaker separation, measurements, and sources have separate modules.

The installed command is `transcribe`. Source checkouts can use `transcribe.py`,
`transcribe.bat`, or `transcribe.sh`. See [usage](usage.md) for options and
[installation](install.md) for environments and optional dependencies.

`docker/` contains container packaging. `pyproject.toml`, `requirements.txt`, and
`requirements-pyannote.txt` describe the package and runtime environments.

## Evaluation tools

These scripts are developer tools, not required steps in ordinary transcription:

| Script | Purpose |
|---|---|
| `tools/benchmark.py` | Evaluate against public datasets; see [benchmarks](benchmarks.md) |
| `tools/eval_speakers.py` | Compare speaker labels with a manually corrected reference |

`tools/diarize_diarizen.py` is a subprocess helper for a separate DiariZen environment;
it is part of runtime integration, not a benchmark.

## Checks and data

`tests/test_pipeline.py` exercises logic without model downloads or a GPU.
On pushes to main and pull requests, CI installs the CPU package in a clean environment
on Linux (Python 3.10) and Windows (Python 3.12). It runs `pip check`, the logic tests,
Ruff and the installed CLI. `tests/test_installed_cli.py` exports a synthetic transcript
in both text formats without audio or model weights.

Recordings, transcripts, model weights, credentials, generated reports, and local
working notes do not belong in commits. Keep private working material under `.local/`.
Before committing, inspect `git diff --cached` and the staged file list; ignore rules
do not protect files that Git already tracks.
