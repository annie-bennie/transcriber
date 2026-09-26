# benchmarks

[← back to readme](../README.md)

`tools/benchmark.py` runs the pipeline against public datasets. install the extras first:

```bash
pip install -e ".[bench]"
```

three modes, and each measures something different.

## speaker separation on real corpora

corpora with human annotation — VoxConverse and AMI, both ungated on hugging face.

```bash
python tools/benchmark.py real --datasets voxconverse ami-sdm ami-ihm --files 5
```

results and how to read them: [diarization.md](diarization.md#benchmark).

## speech recognition

standard russian test sets, whisper `turbo`, up to 40 clips from each (podlodka's test split
yields 20), with the normalisation usually applied for russian. re-measured in the environment
that ships now — python 3.10, torch 2.8 — and the numbers did not move:

| set | what it is | WER | CER |
|---|---|---:|---:|
| [RuLibriSpeech](https://huggingface.co/datasets/bond005/rulibrispeech) | read aloud | 9.7% | 2.3% |
| [Podlodka](https://huggingface.co/datasets/bond005/podlodka_speech) | podcast, spontaneous speech | 7.6% | 4.1% |
| [SOVA RuDevices](https://huggingface.co/datasets/bond005/sova_rudevices) | short household commands | 15.5% | 6.7% |

```bash
python tools/benchmark.py wer --clips 40
```

## emotion models

accuracy on annotated sets in two languages, four-class core. RESD is chosen deliberately: it
does not overlap the training data of the russian categorical model, so it is fair to
everyone.

```bash
python tools/benchmark.py emotion --clips 200
```

this is what settled the question of whether english needs its own emotion model. the answer
was no — and then the dimensional model replaced the categorical one anyway, for reasons in
[measurements.md](measurements.md#how-much-to-trust-them).

## synthetic russian dialogues

there is no standard russian corpus for diarization — every known one (VoxConverse, AMI,
DIHARD, RAMC) is in another language. so this mode assembles dialogues from single-voice
RuLibriSpeech clips, where who speaks when is known by construction, and checks that the
voices really are different using embeddings.

```bash
python tools/benchmark.py der --dialogues 10
```

what it can and cannot tell you, measured rather than assumed:

- **comparing engines by the confusion column works** — conditions are identical, and diarizen
  comes out ahead there too, independently confirming the result on live recordings.
- **absolute DER is not trustworthy.** the ground truth marks the whole clip as speech,
  including pauses inside the reading, so 15–22% of "missed speech" is a property of the
  labelling rather than an error of the systems. do not compare these numbers to published
  DER.
- **thresholds cannot be tuned on it.** on clean, dissimilar audiobook voices confusion came
  out at 0.9% for pyannote and 0.0% for diarizen — there is nothing to move, and every
  threshold gives the same result. thresholds are tuned on live recordings with hand
  labelling, which is what `tools/eval_speakers.py` is for.
- synthetic dialogues reproduce neither interruptions, nor room reverberation, nor natural
  turn-taking dynamics.

## your own recordings

`tools/eval_speakers.py` scores speaker labelling against a hand-corrected transcript. take
the txt the pipeline produced, fix the speaker labels by hand in a few places, separate the
pieces with `---`, and run:

```bash
python tools/eval_speakers.py reference.txt "transcripts/meeting.json"
```

it reports correct labels, correct boundaries, and how many of the errors were already marked
uncertain. labels do not have to match the pipeline's names — the comparison is about the
division into speakers, not the names.

this is the only surface that is actually useful for tuning thresholds. a couple of dozen
hand-labelled lines from a real recording beat any amount of synthetic data.
