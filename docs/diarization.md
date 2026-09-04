# separating speakers

[← back to readme](../README.md)

two engines. **pyannote** ships in the main environment and works out of the box.
**diarizen** installs separately and is noticeably better on some recordings. there is no
universal winner — it depends on how the audio was captured.

| recording | take | why |
|---|---|---|
| a meeting from one microphone in the room | diarizen | DER 14.3% against 20.5% |
| lapel mics, a channel per participant | pyannote | DER 18.7% against 29.6% |
| video call, podcast, a video | roughly equal | 6.9% against 7.7% |
| fast exchange of short turns, similar voices | diarizen | speaker confusion 2–5× lower |

diarizen is steadier at deciding *who* is speaking (confusion 1–3% against 2–5%) but worse at
deciding *where* speech is: on lapel recordings it misses 27.7% of it.

## choosing

```bash
./transcribe.sh "meeting.mp4"                                  # auto: diarizen, else pyannote
./transcribe.sh "meeting.mp4" --diarize-model diarizen          # diarizen only
./transcribe.sh "meeting.mp4" --diarize-model pyannote/speaker-diarization-3.1
./transcribe.sh "meeting.mp4" --diarize-model pyannote/speaker-diarization-community-1
./transcribe.sh "meeting.mp4" --no-diarize                      # not at all
```

changing the engine on an already processed file costs one diarization, not a full rerun:

```bash
./transcribe.sh "meeting.mp4" --rediarize --diarize-model pyannote/speaker-diarization-3.1
```

`community-1` is newer than 3.1, but on short recordings it collapsed two speakers into one,
so 3.1 is the pyannote default. plda is taken from community-1 separately and improves
refinement for both.

installing diarizen: [install.md](install.md#diarizen--optional-better-on-some-recordings).

## benchmark

corpora with human annotation, 5 recordings each, the first 10 minutes of every one. DER with
no forgiveness collar, overlaps counted.

| corpus | engine | DER | confusion | missed | false alarm |
|---|---|---:|---:|---:|---:|
| VoxConverse — youtube videos, many languages | pyannote 3.1 | **6.9%** | 2.0 | 2.1 | 2.8 |
| | pyannote + refinement | 7.7% | 2.8 | 2.1 | 2.8 |
| | diarizen | 7.7% | 3.1 | 2.3 | 2.4 |
| AMI — meetings, distant microphone | pyannote 3.1 | 20.5% | 4.9 | 11.2 | 4.3 |
| | pyannote + refinement | 20.8% | 5.3 | 11.2 | 4.3 |
| | diarizen | **14.3%** | 2.0 | 7.3 | 5.0 |
| AMI — meetings, lapel mics | pyannote 3.1 | **18.7%** | 5.0 | 9.7 | 3.9 |
| | pyannote + refinement | 19.0% | 5.4 | 9.7 | 3.9 |
| | diarizen | 29.6% | 1.1 | 27.7 | 0.9 |

*confusion* is the share of time given to the wrong speaker. *missed* and *false alarm* are
errors about whether there is speech at all. these are computed on a subset, so they are not
directly comparable to published figures.

```bash
python tools/benchmark.py real --datasets voxconverse ami-sdm ami-ihm --files 5
```

## refining against voice references

after diarization, each turn is compared to an averaged voice built from that speaker's long
turns, and relabelled when the margin is decisive.

the reason is that clustering compares short turns against each other, and their embeddings
are close to noise. measured on an hour of audio: on three-second chunks the same-speaker
distance is 0.31 against 0.53 for different speakers; on one-second chunks it collapses to
0.82 against 0.94. a greeting exchange made of one-second turns therefore lands entirely on
one person, and no clustering setting fixes it — `--num-speakers 2` and a tuned clustering
threshold both changed nothing.

references change the question: a short chunk is compared not to an equally noisy neighbour
but to a reliable average. on a hand-labelled exchange that moved 4 correct labels out of 8
to 6, and both remaining errors landed on margins of 0.000 and 0.010 — the method shows where
it does not know, and those turns are marked `?` in the transcript.

on the benchmark corpora refinement makes things very slightly worse (+0.3…0.8 DER), because
there it second-guesses labels that were already right. it stays on by default: the cost when
it is unnecessary is small, and it rescues the case it was built for. turn it off with
`--no-speaker-refine`.

## why speakers are labelled after recognition, not before

it would seem natural to cut the audio by speaker first and recognise each turn separately.
measured, and it is worse — the reasoning is worth keeping because the intuition is strong.

speed is not the argument. on a gpu, cutting first is actually *faster* (4.1–4.7 s against
5.2 s on a two-minute excerpt). on cpu it is the other way round: each chunk is padded to
thirty seconds internally, which costs about 5 seconds regardless of the chunk's real length.

the real argument is which class of error you get. recognising the whole file once means
whisper decides the words with full context, and diarization only attaches labels; a boundary
error mislabels a correctly recognised word. cutting first means pyannote decides the
boundaries and whisper recognises inside them; a boundary error damages the text itself.
without padding a word gets clipped mid-way, with ±0.25 s padding the neighbour's word lands
in both chunks — 5 of 19 boundaries showed a repeat on a dense exchange.

and label errors are recoverable. a different diarization costs one re-cut of the stored
whisper output, a few seconds. with cutting first, any change to diarization means
re-recognising the whole file.

for streaming the same ordering holds even more firmly: cutting first means no text can be
emitted until the speaker stops, so latency equals turn length. online diarization also
revises its labels as it learns, which needs labels to be re-assignable — exactly what
attaching them to immutable words gives.
