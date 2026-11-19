# transcriber

local transcription of meetings, lectures and interviews — with who spoke, what they said,
and how it sounded.

the recording stays on your machine. the first run downloads the models.

## first transcript

you need python 3.10–3.12 or `uv`, which can fetch python for you.
[download the project](https://github.com/annie-bennie/transcriber/archive/refs/heads/main.zip),
unzip it and open a terminal in the folder with the install scripts.
the example uses a russian recording. replace `class.mp4` with your file and
`ru` with its language code.

**windows (powershell)**

```powershell
.\install.ps1
.\transcribe.bat "class.mp4" --language ru
```

if powershell says running scripts is disabled, follow the
[manual installation](docs/install.md#manual-installation).

**linux**

```sh
sh install.sh
sh transcribe.sh "class.mp4" --language ru
```

the installer runs once and sets up whisper, diarizen and the emotion models in one
environment. look for `ready to run` at the end. the torch line says `device: cuda`
when a supported nvidia gpu is available, or `device: cpu` otherwise. to check again,
run `.\transcribe.bat --check` or `sh transcribe.sh --check`. to use the cpu even
with a card installed, add `--device cpu` to the transcription command.

when the run finishes, open the `transcripts` folder next to your recording.
`class_speakers.txt` is the transcript (`class.txt` if no speakers were found).
`class.json` is the saved result you can draw other text formats from.

## what it produces

### lecture text

for a lecture or an interview you just want to read, add `--format lecture`.
consecutive lines from one speaker become a paragraph. when another person speaks,
a new paragraph starts. the words stay in their original order.

```text
SPEAKER_01: so what happened next

SPEAKER_00: i honestly do not know
```

if you already have the json, you can write `transcripts/class_lecture.txt` straight
from it. this also works after the recording has been moved or deleted.

```powershell
.\transcribe.bat "transcripts/class.json" --format lecture
```

on linux, use `sh transcribe.sh` with the same arguments. for a fresh recording,
you can also skip measuring emotions when all you need is the text and speakers:

```sh
sh transcribe.sh "class.mp4" --language ru --format lecture --no-emotion
```

### detailed transcript

the default txt keeps the time of each line, the speaker and the measurements:

```
[SPEAKER_01 05:12-05:14] (aro +2.3σ | -19dB ↑7dB | pitch 254Hz ↑6st): so what happened next
[SPEAKER_00 05:15-05:17] (-27dB | pitch 138Hz): i honestly do not know

Dialogue
  duration 42:10, speech 36:24, silence 14%
  floor changes 204, of them over someone else's speech 11
  response latency: median +0.35 s, latches without a pause 48
  SPEAKER_00: spoke 19:50 (54% of speech), turns 103, median 3.2 s
      rate 4.2 syl/s, pause within a turn 0.31 s, interrupted 5, backchannelled 9
      scales: arousal 0.61, valence 0.53, dominance 0.62 (over 228 utterances)
```

every mark is a deviation from that speaker's own norm in this recording, each in the unit
the measure has: decibels for loudness, semitones for pitch, σ where there is no unit. the
baselines they are measured against sit at the top of the same file.

the json holds everything the txt shows and a good deal more — word timestamps, per-speaker
baselines, confidence flags, and a block explaining how to read every number. it is the
interface for anything downstream.

## what it does

| stage | tool |
|---|---|
| decoding | [ffmpeg](https://ffmpeg.org/) |
| speech recognition | [whisper](https://github.com/openai/whisper), word-level timestamps |
| speaker separation | [DiariZen](https://github.com/BUTSpeechFIT/DiariZen) in the standard installation; see [speaker benchmarks](docs/diarization.md) for the optional pyannote engine |
| emotion | [audeering MSP-Dim](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) — arousal, dominance, valence |
| loudness, pitch, dialogue statistics | arithmetic over the waveform, no models |
| voice quality (optional) | [praat](https://github.com/YannickJadoul/Parselmouth) — jitter, shimmer, HNR |
| input from a url (optional) | [yt-dlp](https://github.com/yt-dlp/yt-dlp) — one video or a playlist, audio only |

emotion is three numbers rather than a label. a label answers "what was that"; the numbers
answer "how did this person change", and only the second question survives being asked
across recordings and languages. the reasoning and the measurements are in
[docs/measurements.md](docs/measurements.md).

## documentation

- [docs/install.md](docs/install.md) — installation, model access, Docker and optional environments
- [docs/usage.md](docs/usage.md) — every option, common combinations, resuming after a crash
- [docs/output.md](docs/output.md) — json structure, txt marks, units of analysis
- [docs/measurements.md](docs/measurements.md) — what the numbers mean and how far to trust them
- [docs/diarization.md](docs/diarization.md) — choosing an engine, benchmark results
- [docs/benchmarks.md](docs/benchmarks.md) — reproducing the numbers on your own data
- [docs/development.md](docs/development.md) — source layout, evaluation tools, and checks

## requirements

allow about 2.5 GB for the model weights in addition to the python environment.
subsequent runs reuse the downloaded weights. a supported nvidia gpu speeds up all
three model stages. the same command also runs on cpu.

for a container install, the [docker instructions](docs/install.md#docker-instead)
cover both the cpu and cuda builds and the volume that keeps the models between runs.

## speed

measured on the machine this was built on — ryzen 7 7800X3D, rtx 5080 (16 GB), 63 GB ram,
windows 11, python 3.10, torch 2.8 on cuda 12.8. one hour of audio, full pass from scratch,
4 minutes 44 seconds in total:

| stage | time |
|---|---|
| decoding the audio | 17 s |
| recognition (whisper turbo) | 2 min 11 s |
| speaker separation (diarizen) | 1 min 54 s |
| emotion, loudness, pitch | 18 s |
| voice quality, if enabled | +15 s |

models load only when a stage actually needs them. a finished file is redrawn in four
seconds, and `--rediarize` never touches recognition.

## language

recognition defaults to english; pass `--language ru` or any other whisper language code.
that setting is about the audio only — logs, field names and reports stay in english
whatever the recording is.

## what it is not

emotion recognition from voice is unreliable and is not a diagnosis. the numbers here are a
hint for a person who listens to the recording themselves, not grounds for decisions about
people.

record people only with their consent. in most places that is the law rather than manners.

whisper gets names, terms and numbers wrong, and invents text in silence — the invented
parts are flagged, but a flag is not a guarantee. check anything that matters against the
audio.

## licence

the code is MIT — see `LICENSE`. the models are not, and two of them are non-commercial:

| model | licence | commercial use |
|---|---|---|
| whisper | Apache-2.0 | yes |
| pyannote 3.1, segmentation-3.0 | MIT | yes |
| pyannote community-1, wespeaker | CC-BY-4.0 | yes, with attribution |
| audeering MSP-Dim — the emotion scales | **CC-BY-NC-SA-4.0** | **no** |
| diarizen wavlm-large | **CC-BY-NC-4.0** | **no** |

so the pipeline **as configured by default is non-commercial**, whatever the code licence
says — the restriction comes from the weights, not from here. running it without the
emotion scales (`--no-emotion`) and selecting `--diarize-model pyannote/speaker-diarization-3.1`
in the optional pyannote 4 environment leaves only permissive models; pyannote's weights are gated on hugging face and
need their conditions accepted first.

## acknowledgements

built on [whisper](https://github.com/openai/whisper),
[pyannote.audio](https://github.com/pyannote/pyannote-audio),
[DiariZen](https://github.com/BUTSpeechFIT/DiariZen),
[audeering's dimensional emotion model](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)
and [wespeaker](https://github.com/wenet-e2e/wespeaker). thanks to their authors.
