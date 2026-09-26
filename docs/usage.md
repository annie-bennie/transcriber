# usage

[← back to readme](../README.md)

the first argument is a file, a folder, or a url. a folder is walked recursively and every
media file inside is taken; a url is fetched first and then treated like any other input.

```bash
./transcribe.sh "lecture.mp4"                  # macos, linux
transcribe.bat "lecture.mp4"                   # windows
python transcribe.py "lecture.mp4"             # any platform
transcribe "lecture.mp4"                       # after pip install -e .
```

## is this machine ready

```bash
python transcribe.py --check
```

prints what is installed and what is not: the interpreter, torch and whether it sees the card,
which ffmpeg was found, diarizen and pyannote, which model weights are already downloaded, free
disk space, and the optional packages. every line that is not a tick comes with the command that
fixes it. it loads no models — a second or two — and exits with 1 only if a run could not start
at all, so it can be used as a smoke test after an install.

## live, from a microphone

```bash
python transcribe.py --live --language ru
python transcribe.py --live-devices            # which inputs can record
python transcribe.py --live --live-device 1     # listen to a particular one
python transcribe.py --live --no-diarize        # text only, no speaker labels
```

it prints speech as it is recognised and writes it to a txt named by the date — by default in
the documents folder, as the system reports it rather than as `~/Documents` guesses:

```
C:\Users\you\OneDrive\Documents\2026-09-27.txt
```

`--output-dir` puts it elsewhere, and on windows that is worth knowing: the documents folder is
usually synced to onedrive, so a recording of a private conversation leaves the machine. for
anything sensitive, give a local folder.

several sessions on the same day go into the same file, each under its own `--- live, 14:32 ---`
header, and nothing already in the file is touched. the file is rewritten as the session goes on,
because a speaker label can be corrected after a line was written — what is on disk keeps matching
what the screen last said. the audio itself is never kept.

the offline run is still what produces a result to read afterwards: it sees the whole recording.

what it costs, measured before it was built (the numbers are in PLAN, stage 9):

| | |
|---|---|
| a word stops changing | about 1.8 s after it is spoken |
| a line is finished | when the speaker pauses — 3.8 s after its last word, typically |
| text against a normal run of the same model | 14% of words differ on a close microphone, 16–23% in a room |
| speaker labels, two people | agree with a normal run on 95–100% of words |
| speaker labels, six people | 87%, and about one line in six has its label corrected after it was shown |

so three things are drawn differently, because they are not equally certain: the rough tail is
redrawn in place, a finished line is printed with its speaker, and a correction to a speaker
already printed appears as a separate note — a terminal cannot go back and rewrite a line.

use it to see that something is being heard, to follow a conversation, to check a microphone.
do not use it for a transcript anyone will read: for that, record the audio and run the normal
way, which sees the whole recording and is measurably better.

## from a url

a single video or a whole playlist, anything yt-dlp can reach. only the audio is fetched —
the pipeline downmixes to mono 16 kHz anyway, so pulling video would cost bandwidth for
nothing.

```bash
./transcribe.sh "https://www.youtube.com/watch?v=..." --output-dir ~/results
./transcribe.sh "https://www.youtube.com/playlist?list=..." --output-dir ~/results
```

needs `pip install yt-dlp`. without `--output-dir` the results land in `./transcripts`,
since a url has no folder of its own to sit next to. yt-dlp's own search prefixes work too:
`"ytsearch3:interview with a designer"`.

**a playlist is handled one item at a time**: it is listed first, without downloading
anything, and then each item is fetched, transcribed and deleted before the next one starts.
so peak disk usage is one episode rather than the whole playlist, and transcripts appear as
they are finished instead of after the last byte lands.

```bash
./transcribe.sh "https://www.youtube.com/playlist?list=..." --output-dir ~/results
./transcribe.sh "https://www.youtube.com/playlist?list=..." --playlist-limit 5
```

resuming works off the video id, which yt-dlp writes into the filename as `[id]`. the audio
is gone by then but the json is not, so re-running the same playlist skips what is already
finished **without downloading it again** — an interrupted run of forty episodes picks up at
episode thirty-one.

the downloaded audio is deleted right after each file is transcribed. `--keep-audio` keeps
it. a file you supplied yourself is never deleted, whatever the flags say.

one blocked video does not cost the rest: the run reports it and moves on. what you download
is somebody's work under somebody's terms, and fetching it may be against the terms of the
site it sits on — that call is yours.

`--rebuild-txt` does not take a url: it works from json files, and the url is not where they
are. point it at the output folder instead — it rebuilds from the json even though the audio
is long gone:

```bash
./transcribe.sh ~/results --rebuild-txt
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
| `--hf-token TOKEN` | from `HF_TOKEN` or the login cache | hugging face token — only the optional pyannote models need one |

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
| `--playlist-limit N` | all | take only the first N items of a playlist or search |
| `--rebuild-txt` | off | rebuild everything derived from existing json. needs no models and no audio |
| `--drop-suspect` | off | leave flagged segments out of the txt. they stay in the json either way |
| `--overwrite` | off | recompute everything, ignoring saved progress |
| `--checkpoint-every N` | 200 | how often to flush measurement progress, in utterances |
| `--debug` | off | print every recognised line with its timecode as it goes |

`transcribe.py --help` prints the same list.

## progress

each long stage draws a bar while it runs — downloading, whisper, diarization, measurement.
it is the same tqdm bar whisper itself uses, so every stage looks alike:

```
  measuring:  57%|█████▋      | 384/672 [00:12<00:09, 31.2utterances/s, block 2/4]
```

a stage with nothing to count against — diarizen runs as a subprocess and reports no
progress — shows the elapsed time instead of a fake percentage:

```
  diarization (diarizen): 00:41
```

the bar is a screen thing only. the log file gets one line per stage instead, because a bar
rewritten in place turns into a thousand near-identical lines in a file, and the log is what
you read after a crash. when output is piped rather than shown on a terminal, the bar turns
itself off and a line every 10% takes its place.

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
