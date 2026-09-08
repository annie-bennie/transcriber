# installation

[← back to readme](../README.md)

## the main environment

```bash
git clone <repo> && cd transcriber
python -m venv .venv
# windows: .venv\Scripts\activate
source .venv/bin/activate
```

torch comes first, matched to the hardware. do not let pip pick it — it will happily install
a build with no support for the card in the machine.

```bash
# nvidia, cuda 13 (modern cards up to blackwell / rtx 50xx)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
# nvidia, cuda 12.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# cpu only
pip install torch torchaudio
```

then everything else:

```bash
pip install -r requirements.txt
# or, as a package with a `transcribe` command:
pip install -e .
```

optional extras:

```bash
pip install -e ".[voice]"   # jitter, shimmer, HNR
pip install -e ".[bench]"   # public-dataset benchmarks, tools/benchmark.py
pip install -e ".[url]"     # a youtube link or playlist as the input
```

ffmpeg has to be on the PATH:

```bash
brew install ffmpeg          # macos
sudo apt install ffmpeg      # ubuntu
winget install ffmpeg        # windows
```

## which interpreter the launchers use

`transcribe.sh` and `transcribe.bat` look for one in this order: `TRANSCRIBE_PYTHON`, a
`.venv` or `venv` inside the project, an activated environment, the system python. if the one
they land on has no dependencies installed they say so and stop, rather than failing ten
frames into an import.

so an environment kept somewhere else is named once:

```bash
export TRANSCRIBE_PYTHON=/path/to/env/bin/python      # macos, linux
set TRANSCRIBE_PYTHON=C:\path\to\env\Scripts\python.exe   # windows
```

`python transcribe.py` and the installed `transcribe` command ignore all of this — they run
under whatever interpreter invoked them.

### two interpreters, two variables

with diarizen installed there are two environments in play, and they do not replace each
other:

| variable | which interpreter | what it runs |
|---|---|---|
| `TRANSCRIBE_PYTHON` | the main environment | the whole pipeline: whisper, pyannote, emotion, acoustics |
| `DIARIZEN_PYTHON` | the diarizen environment, python 3.10 | one subprocess per file, nothing else |

the pipeline never switches between them. it starts under the first and stays there; diarizen
is launched as a separate process, hands back a json list of turns, and exits. that is the
whole reason the two can coexist while their pyannote versions cannot.

so on a machine with both, set both:

```bash
setx TRANSCRIBE_PYTHON "C:\path\to\whisper-env\Scripts\python.exe"
setx DIARIZEN_PYTHON "C:\path\to\diarizen-env\Scripts\python.exe"
```

`setx` writes to the registry and takes effect in a **new** terminal. on macos and linux the
same two lines with `export`, in a shell profile.

## model access

the diarization models are gated on hugging face. without accepting their terms the speaker
stage will not start.

1. accept the conditions on both pages:
   - https://hf.co/pyannote/speaker-diarization-3.1
   - https://hf.co/pyannote/segmentation-3.0
2. optionally also https://hf.co/pyannote/speaker-diarization-community-1 — the pipeline
   takes plda from there, which improves speaker refinement on short turns.
3. log in:

```bash
huggingface-cli login
# or
export HF_TOKEN=hf_...
```

if a model does not load, the log says which one and links the page to accept.

## diarizen — optional, better on some recordings

diarizen brings its own fork of `pyannote-audio` 3.1.1, and the pipeline runs on
`pyannote.audio` 4.x. that is one namespace package — only one of them fits in an
environment. they can be combined, but then the pyannote fallback, speaker refinement and
plda all disappear, and the environment gets pinned to python ≤3.12 and numpy 1.26.4.

so diarizen lives in its own venv and is called as a subprocess. full instructions are in the
header of [`requirements-diarizen.txt`](../requirements-diarizen.txt); briefly:

```bash
git clone https://github.com/BUTSpeechFIT/DiariZen.git
uv venv --python 3.10 /path/to/diarizen-venv
PY=/path/to/diarizen-venv/bin/python

uv pip install --python $PY torch torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install --python $PY -r requirements-diarizen.txt
uv pip install --python $PY --no-deps -e ./DiariZen/pyannote-audio -e ./DiariZen
```

there is no `diarizen` package on pypi — it installs from source only, and `--no-deps` is
required or pip drags in old pins.

nothing from that environment ever needs to be run by hand. point the pipeline at its
interpreter, in any of three ways:

```bash
# a flag
./transcribe.sh "meeting.mp4" --diarizen-python /path/to/diarizen-venv/bin/python

# an environment variable
export DIARIZEN_PYTHON=/path/to/diarizen-venv/bin/python
setx DIARIZEN_PYTHON "C:\path\to\diarizen-env\Scripts\python.exe"    # windows, permanently

# or put the environment next to the project as .venv-diarizen/ or diarizen-venv/
```

this is a *second* variable, not a replacement for `TRANSCRIBE_PYTHON` — see
[two interpreters, two variables](#two-interpreters-two-variables).

if diarizen is not found the pipeline uses pyannote and says so once.

## checking it works

```bash
python tests/test_pipeline.py     # logic only, no models, a couple of seconds
./transcribe.sh "some.mp4"
```
