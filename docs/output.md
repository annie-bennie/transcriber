# what comes out

[← back to readme](../README.md)

each file produces `<name>.json` and a txt drawn from it. the txt is a view: everything in
it can be redrawn from the json with `--rebuild-txt`, so nothing is stored only in that
format.

the txt is named `<name>_speakers.txt` when speaker separation found anyone, and `<name>.txt`
when it did not — a transcript with speaker labels and one without are different documents,
and the name says which you have without opening it. the json keeps the plain name either
way: it is the artefact, and its name should not depend on which stages ran.

changing that (turning `--no-diarize` on or off) leaves the other file behind from the
earlier run. nothing is deleted for you — the txt is yours and cheap to redraw — but the log
says which file has gone stale.

## units: segment, utterance, turn

three words appear everywhere and mean different things. the same minute of recording is
divided into them differently, because different values are meaningful at different scales.

| unit | what it is | typical length | what is measured on it |
|---|---|---|---|
| **segment** | a transcript line: a whisper chunk, cut where the speaker changes | ~0.8 s | text, loudness, pitch |
| **utterance** | consecutive segments of one person, merged to a usable length | ~5 s | emotion scales, voice quality |
| **turn** | everything a person said while holding the floor, until someone else started | ~3–20 s | speaking shares, pauses, latency, interruptions |

why the split. a whisper segment is short, and on it the emotion model guesses: chunks of
0.3–0.5 s produced 8% junk labels, from two seconds on it was zero. but dropping the short
ones is not an option either — that is 45% of the recording — so they are merged into an
utterance. dialogue statistics, in turn, are only meaningful on turns: interruptions cannot
be detected from segments at all, since segments never overlap by construction.

in the transcript you see segments (lines) and turns (separated by a blank line). an
utterance shows indirectly: neighbouring lines of one utterance carry the same emotion
deviation.

## marks in the transcript

- **`⚠`** — looks invented. whisper produces caption boilerplate in silence. flagged, never
  deleted, with the reason in brackets. hide with `--drop-suspect`.
- **`?`** — the speaker is in doubt. diarization found no confident answer and the label was
  left as it was.
- **`↑7dB` / `↓5st` / `+2.4σ`** — how far this line moved from that person's own norm
  in this recording, each measure in its own unit: decibels for loudness, semitones for
  pitch, σ where the quantity has no unit. small moves are not printed; see
  [measurements.md](measurements.md).

the txt also opens with the reading key for the emotion scales — the same block that sits in
the json — so that a reader who has only the txt can interpret the numbers.

## json structure

two layers. the raw one never changes after its stage; the derived one is rebuilt from it on
demand.

```
schema_version          format version. a mismatch rebuilds the derived layer, keeping
                        recognition
source                  name, path, size, duration
params                  every setting this result was computed with
stages                  transcribe / diarize / analyze: done, partial, failed or null

── raw layer ──────────────────────────────────────────────────────────
text                    whisper's continuous text, as it came
asr_segments[]          whisper segments with words and confidence metrics
  .words[]              text, start, end, prob
  .no_speech_prob       whisper's own estimate that there is no speech here
  .avg_logprob          mean likelihood
  .compression_ratio    high values mean the text repeats itself
diarization[]           [start, end, speaker, uncertain]

── derived layer ──────────────────────────────────────────────────────
segments[]              transcript lines
  .asr_id               which whisper segment it came from
  .speaker              label, or null without diarization
  .speaker_conf         share of words the diarizer covered confidently
  .speaker_uncertain    refinement found no footing for a decision
  .word_prob            mean word confidence
  .unit                 which utterance it belongs to
  .loudness_db          rms in dBFS
  .pitch_mean/_std      fundamental frequency and its spread
  .loudness_z, .pitch_z, .pitch_var_z    deviations in units of the speaker's spread
  .loudness_delta_db, .pitch_delta_st    the same distance in decibels and semitones
  .suspect[]            reasons it looks invented, if any
turns[]                 speaker, start, end, segment ids
emotion_units[]         utterances
  .arousal/.dominance/.valence           0 to 1
  .arousal_z/.dominance_z/.valence_z     deviations from the speaker's baseline
  .jitter/.shimmer/.hnr                  voice quality, with --voice-quality
  .analyzed             measured already — this is what resuming reads
speakers[]              labels found
speaker_stats{}         per speaker: median and spread of every value
dialogue{}              speaking shares, silence, latency, interruptions, speech rate
emotion_guide{}         how to read the scales: what each means, and the quadrant table
deviation_guide{}       what each deviation is measured against, in which unit, from
                        which threshold, and why that threshold
```

`emotion_guide` and `deviation_guide` are written into every file on purpose. anything picking the json up later —
a person, a script, a language model — gets the reading key together with the numbers rather
than having to find this page.

## why two layers

recognising an hour of audio takes two and a half minutes. rebuilding the derived layer takes
four seconds:

```bash
./transcribe.sh "meeting.mp4" --rebuild-txt
```

so cutting rules, filter thresholds and output settings can be changed freely, and a
recording can be re-cut under a different diarization without ever re-running whisper.

## a folder

processing a folder also produces `full_transcript.txt` — continuous text from every file,
with hallucinations removed. files this run did not touch are read from their json, so the
combined file does not lose earlier work. disable with `--no-combined`.
