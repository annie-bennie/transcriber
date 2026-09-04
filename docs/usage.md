# usage

[← back to readme](../README.md)

the first argument is a file or a folder. a folder is walked recursively and every media file
inside is taken.

```bash
./transcribe.sh "lecture.mp4"                  # macos, linux
transcribe.bat "lecture.mp4"                   # windows
python transcribe.py "lecture.mp4"             # any platform
transcribe "lecture.mp4"                       # after pip install -e .
```

## common combinations

```bash
# fast: text only, no speakers, no measurements
./transcribe.sh "lecture.mp4" --no-diarize --no-emotion

# a russian recording with four known participants
./transcribe.sh "meeting.mp4" --language ru --num-speakers 4

# more accurate, if the time is worth it
./transcribe.sh "meeting.mp4" --model large-v3

# redraw the txt after changing output settings, computing nothing
./transcribe.sh "meeting.mp4" --rebuild-txt --drop-suspect

# a folder of recordings, two people in each
./transcribe.sh ~/records --num-speakers 2
```

## output

| option | default | what for |
|---|---|---|
| `--output-dir PATH` | `<input>/transcripts` | where json and txt go |
| `--keep-audio` | off | save the extracted audio next to the results |
| `--no-combined` | off | do not build a folder-wide `full_transcript.txt` |

## speech recognition

| option | default | what for |
|---|---|---|
| `--model NAME` | `turbo` | whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`. bigger is more accurate and slower |
| `--language CODE` | `en` | language of the recording. this is the only setting about the audio |
| `--device cuda\|cpu` | autodetect | where to compute |
| `--no-word-timestamps` | off | skip word timestamps. about 40% faster, but speakers are assigned per whole segment and boundaries suffer |

## speaker separation

| option | default | what for |
|---|---|---|
| `--diarize-model` | `auto` | `auto`, `diarizen`, or a pyannote model name on huggingface |
| `--diarizen-python PATH` | — | interpreter of the diarizen environment |
| `--diarizen-model NAME` | `diarizen-wavlm-large-s80-md-v2` | a different diarizen model |
| `--num-speakers N` | — | exact number of participants, when known. helps pyannote noticeably |
| `--min-speakers N`, `--max-speakers N` | — | bounds when the exact number is unknown |
| `--no-speaker-refine` | off | do not refine speakers against voice references |
| `--rediarize` | off | label speakers again without touching recognition |
| `--no-diarize` | off | do not separate speakers at all |
| `--diarize-batch N` | 32 | pyannote batch size, lower it if vram runs short |
| `--hf-token TOKEN` | from `HF_TOKEN` or the login cache | hugging face token |

which engine to pick is in [diarization.md](diarization.md).

## measurement

| option | default | what for |
|---|---|---|
| `--no-emotion` | off | skip emotion scales, loudness and pitch entirely |
| `--no-emotion-dimensions` | off | skip arousal, dominance and valence but keep loudness and pitch |
| `--emotion-dim-model NAME` | audeering MSP-Dim | a different scales model |
| `--voice-quality` | off | compute jitter, shimmer and HNR. needs `praat-parselmouth`, about a minute per hour |
| `--emotion-unit-target SEC` | `5` | measurement window. shorter is more precise in time but noisier |
| `--emotion-batch N` | 4 | batch size for the measurement models |

## output and rebuilding

| option | default | what for |
|---|---|---|
| `--rebuild-txt` | off | rebuild everything derived from existing json. needs no models and no audio |
| `--drop-suspect` | off | leave flagged segments out of the txt. they stay in the json either way |
| `--overwrite` | off | recompute everything, ignoring saved progress |
| `--checkpoint-every N` | 200 | how often to flush measurement progress, in utterances |
| `--debug` | off | print every recognised line with its timecode as it goes |

`transcribe.py --help` prints the same list.

## resuming after a crash

the json is written after whisper, after diarization, and as measurement proceeds. running
the same command again continues from where it stopped:

```
continuing from the last run — whisper: done (1847 segments) | diarization: done | measurement: partial (200/431 utterances)
whisper skipped: meeting.json already has 1847 segments
to measure: 231 of 431 utterances
```

writes are atomic, through a temporary file and a rename, so a crash mid-write leaves the
previous whole version rather than a stump.

changing a setting invalidates exactly what depends on it. a different `--num-speakers`
re-cuts the recording without re-recognising it; a different measurement window recomputes
only the measurements. recomputing everything from scratch is `--overwrite`.

a finished file is skipped, and its txt is redrawn anyway — that is cheap.
