# installation

[← back to readme](../README.md)

## the environment

one environment holds everything: whisper, diarizen for speaker separation, the emotion scales
and a static ffmpeg. it is the environment the docker image is built from, so the two behave
alike.

python 3.10, 3.11 or 3.12 — diarizen pins numpy 1.26.4, which has no build for 3.13.

the short way — one command that does everything below and checks the result:

```bash
git clone <repo> && cd transcriber
./install.sh                  # windows: .\install.ps1
```

it finds a python between 3.10 and 3.12 — or uses `uv` to fetch one, if `uv` is installed — picks
the torch build from whether an nvidia card is present, installs the rest, and finishes by running
`--check`. `--cpu` forces the build without cuda, `--venv <path>` puts the environment elsewhere.

the same thing by hand, which is also what the script does:

```bash
git clone <repo> && cd transcriber
python -m venv .venv
# windows: .venv\Scripts\activate
source .venv/bin/activate
```

torch first: matched to the hardware, and at 2.8.0 exactly. left to pip it arrives as whatever
build pypi offers, which on windows has no cuda at all.

```bash
# nvidia, rtx 50xx (blackwell) included
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# no gpu
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

then everything else, in one step:

```bash
pip install -r requirements.txt
# or, as a package with a `transcribe` command:
pip install -e .
```

diarizen and its fork of pyannote are not on pypi, so they come from an archive of a pinned
commit on github — this needs the network, not git. ffmpeg comes with the rest; one already on
the PATH is used first.

then check the card is seen — the one line that tells a gpu run from a slow cpu one before an
hour of audio does:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

followed exactly as written on windows 11 with an rtx 5080: about 7 minutes for torch, 3 for
the rest, 7.9 GB on disk, and a two-minute recording transcribed in 43 s.

optional extras:

```bash
pip install -e ".[voice]"   # jitter, shimmer, HNR
pip install -e ".[bench]"   # public-dataset benchmarks, tools/benchmark.py
pip install -e ".[url]"     # a youtube link or playlist as the input
```

## docker instead

one image with everything the pipeline needs — whisper, diarizen and the emotion scales in a
single environment, and a static ffmpeg. from the repository root:

```bash
docker build -f docker/Dockerfile -t transcriber:cpu .
docker build -f docker/Dockerfile -t transcriber:gpu --build-arg TORCH=cu128 .
```

| image | size | build | two-minute recording | an hour |
|---|---:|---:|---:|---:|
| `transcriber:cpu` | 2.5 GB | ~3 min | 5.4 min | not measured |
| `transcriber:gpu` | 12.3 GB | ~13 min | 38 s | 5.3 min |

measured on the machine described in the readme, with the models already downloaded. the gpu
image also runs on a machine without a card, as slowly as the cpu one.

run it through compose, naming the folder with the recordings and the folder for results:

```bash
INPUT_DIR=/path/to/recordings OUTPUT_DIR=/path/to/results \
  docker compose -f docker/compose.yaml run --rm transcriber "meeting.mp4" --language ru
```

`transcriber-gpu` instead of `transcriber` uses the card. in powershell, set the two folders
first with `$env:INPUT_DIR="C:\recordings"` and `$env:OUTPUT_DIR="C:\results"`. plain docker
does the same:

```bash
docker run --rm --gpus all \
  -v /path/to/recordings:/in:ro -v /path/to/results:/out \
  -v transcriber-models:/models \
  transcriber:gpu "meeting.mp4" --language ru
```

what to know about it:

- **the recordings folder is read-only** inside the container, and results go to the second
  folder. both are required: a default would be a folder inside this repository.
- **the first run needs the network, and nothing else** — no hugging face token. 2.5 GB of
  weights go into the `transcriber-models` volume.
- **after that, no network at all.** checked with `--network none` on both images: the
  pipeline sees that the hub is unreachable and reads the cache straight away.
- **diarizen is the only speaker engine in the image.** one environment means diarizen's own
  fork of pyannote 3.1.1, so the pyannote 4 pipelines are not there, and neither is plda. on an
  hour of dialogue scored against hand-corrected labels, plda changed nothing — 27 of 28 lines
  and 22 of 23 boundaries right with or without it — but that reference is small. it also makes
  the image non-commercial as built: diarizen's weights are CC-BY-NC, see the licence table in
  the readme.
- **the build sees only the code.** `.dockerignore` denies everything else, so recordings lying
  in the project folder never reach the docker daemon.
- **gpu on windows** needs docker desktop with the wsl2 backend; on linux, the nvidia container
  toolkit. check with `docker run --rm --gpus all debian:bookworm-slim nvidia-smi`.
- the gpu image prints *failed to launch triton kernels*: triton needs a compiler the image
  deliberately does not carry, so whisper uses its slower word-timing code. results are the same.

## which interpreter the launchers use

`transcribe.sh` and `transcribe.bat` look for one in this order: `TRANSCRIBE_PYTHON`, a
`.venv` or `venv` inside the project, an activated environment, the system python. if the one
they land on has no dependencies installed they say so and stop, rather than failing ten
frames into an import.

so an environment kept somewhere else is named once:

```bash
export TRANSCRIBE_PYTHON=/path/to/env/bin/python             # macos, linux
setx TRANSCRIBE_PYTHON "C:\path\to\env\Scripts\python.exe"   # windows, permanently
```

`setx` writes to the registry and takes effect in a **new** terminal.

`python transcribe.py` and the installed `transcribe` command ignore all of this — they run
under whatever interpreter invoked them.

## model access

the default environment needs no hugging face account at all: diarizen, the voice embedding
model and the emotion scales are openly downloadable. checked by running the docker image on an
empty model cache with no token set.

a token is needed only for the
[optional pyannote 4 environment](#optional-pyannote-4-with-plda), whose models are gated:

1. accept the conditions on both pages:
   - https://hf.co/pyannote/speaker-diarization-3.1
   - https://hf.co/pyannote/segmentation-3.0
2. optionally also https://hf.co/pyannote/speaker-diarization-community-1 — plda comes from
   there.
3. log in:

```bash
huggingface-cli login
# or
export HF_TOKEN=hf_...
```

if a model does not load, the log says which one and links the page to accept.

## optional: pyannote 4 with plda

the default environment runs diarizen's own fork of pyannote 3.1.1. pyannote 4 adds two
things: plda, which makes speaker refinement more decisive on short turns, and pyannote's own
pipelines (3.1, community-1) as speaker engines. on an hour of dialogue scored against
hand-corrected labels plda changed nothing — 27 of 28 lines and 22 of 23 boundaries right either
way — so most setups do not need it.

pyannote 4 and diarizen's fork install under one package name, so this is a second
environment, and diarizen keeps running from the first one as a subprocess:

```bash
python -m venv .venv-pyannote
# activate it, install torch 2.8.0 exactly as above, then:
pip install -r requirements-pyannote.txt
```

run the pipeline from this environment and point it at the default one, in any of three ways:

```bash
# a flag
python transcribe.py "meeting.mp4" --diarizen-python .venv/bin/python

# an environment variable
export DIARIZEN_PYTHON=/path/to/.venv/bin/python
setx DIARIZEN_PYTHON "C:\path\to\.venv\Scripts\python.exe"    # windows, permanently

# or name the default environment .venv-diarizen/ next to the project
```

the pyannote models are gated — see [model access](#model-access).

## checking it works

```bash
python transcribe.py --check      # the machine: torch, the card, ffmpeg, models, disk
python tests/test_pipeline.py     # the logic: no models, a couple of seconds
./transcribe.sh "some.mp4"
```

`--check` exits with 1 when something essential is missing, so it works as a smoke test in a
script. it names the command to run for each missing piece.

in the log, `pyannote.audio: 3.1.1` and `diarization (diarizen, …)` mean the default
environment is the one running.
