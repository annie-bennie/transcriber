# -*- coding: utf-8 -*-
"""Scoring the pipeline on open datasets.

two independent checks:

  wer  — recognition quality. clips from the standard russian test sets are run through
         whisper and WER is computed. the numbers are comparable to published ones.

  der  — speaker labelling quality. no standard russian set for diarization exists (every
         known one is english, mandarin or german), so dialogues are assembled out of
         single-voice clips, where who speaks when is known by construction. voice identity
         is verified with embeddings so that "two speakers" do not turn out to be one reader.

examples:
    python tools/benchmark.py wer --clips 100
    python tools/benchmark.py der --dialogues 8 --backend pyannote diarizen
    python tools/benchmark.py der --dialogues 8 --sweep-margin 0.03 0.05 0.08 0.12

what this benchmark measures and what it does not — measured, not assumed:

  * comparing variants by the "confusion" column works: the conditions are identical.
  * absolute DER is not trustworthy. the ground truth marks the whole clip as speech,
    pauses inside the reading included, so 15-22% of "missed speech" is a property of the
    labelling rather than an error of the systems. do not compare these to published DER.
  * thresholds cannot be tuned on it. on clean, dissimilar audiobook voices confusion came
    out at 0.9% for pyannote and 0.0% for diarizen — there is nothing to move, and every
    threshold gives the same result. thresholds are tuned on live recordings with hand
    labelling (see tools/eval_speakers.py); synthetic audio is far too easy for that.
  * synthetic dialogues reproduce neither interruptions, nor room reverberation, nor the
    natural dynamics of turn-taking.
"""
import argparse
import io
import itertools
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_RATE = 16000

# standard russian test sets. the first is reading aloud, the other two are closer to our
# case: spontaneous speech and short household commands.
WER_DATASETS = {
    "rulibrispeech": ("bond005/rulibrispeech", None, "test"),
    "podlodka": ("bond005/podlodka_speech", None, "test"),
    "sova": ("bond005/sova_rudevices", None, "test"),
}
SPEAKER_SOURCE = ("bond005/rulibrispeech", None, "test")

# real diarization corpora with human annotation. none of them is russian — none exists —
# but diarization leans on voice rather than language, so they serve for comparing engines
# against each other, and the numbers are comparable to published ones as a bonus.
REAL_DATASETS = {
    "voxconverse": ("diarizers-community/voxconverse", None, "test",
                    "youtube videos, many languages"),
    "ami-sdm": ("diarizers-community/ami", "sdm", "test",
                "meetings, distant microphone"),
    "ami-ihm": ("diarizers-community/ami", "ihm", "test",
                "meetings, lapel microphones"),
    "simsamu": ("diarizers-community/simsamu", None, "train",
                "phone conversations, french"),
}


# sets with annotated emotions. RESD is chosen deliberately: it does not overlap DUSHA,
# which the russian categorical model was trained on, so the check is fair to everyone.
EMOTION_DATASETS = {
    "resd": ("Aniemore/resd", None, "test", "speech", "emotion", "russian"),
    "ravdess": ("xbgoose/ravdess", None, "train", "audio", "emotion", "english"),
    "dusha": ("xbgoose/dusha", None, "test", "audio", "emotion", "russian, training set"),
}

# models with different label sets are compared on a shared core of four emotions.
# everything outside it (fear, disgust, surprise, calm) is excluded from the check:
# matching those across sets only works with a stretch.
EMOTION_CORE = ("neutral", "happy", "sad", "angry")
EMOTION_SYNONYMS = {
    "neutral": "neutral", "neu": "neutral", "нейтрально": "neutral",
    "happy": "happy", "happiness": "happy", "hap": "happy", "joy": "happy",
    "positive": "happy", "радость": "happy",
    "sad": "sad", "sadness": "sad", "sadn": "sad", "грусть": "sad",
    "angry": "angry", "anger": "angry", "ang": "angry", "злость": "angry",
}

EMOTION_MODELS = {
    "dusha": "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned",
    "superb": "superb/wav2vec2-base-superb-er",
    "ehcalabres": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    "whisper-ser": "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
    "dpngtm": "Dpngtm/wav2vec2-emotion-recognition",
}


def to_core_emotion(label):
    """a model or dataset label -> one of the four shared ones. None if outside the core."""
    if label is None:
        return None
    return EMOTION_SYNONYMS.get(str(label).strip().lower())


def run_emotion(args):
    """accuracy of emotion models on annotated sets in several languages."""
    import torch
    from datasets import Audio, load_dataset
    from transformers import pipeline as hf_pipeline

    device = 0 if torch.cuda.is_available() else -1
    prepared = {}
    for name in args.datasets:
        repo, config, split, audio_key, label_key, language = EMOTION_DATASETS[name]
        stream = load_dataset(repo, config, split=split, streaming=True)
        stream = stream.cast_column(audio_key, Audio(decode=False))
        clips = []
        for row in stream:
            gold = to_core_emotion(row[label_key])
            if gold is None:
                continue
            try:
                audio = decode_clip(row[audio_key])
            except Exception:
                continue
            clips.append((audio, gold))
            if len(clips) >= args.clips:
                break
        prepared[name] = (clips, language)
        counts = {}
        for _audio, gold in clips:
            counts[gold] = counts.get(gold, 0) + 1
        print(f"{name} ({language}): {len(clips)} clips, {counts}")

    results = []
    for short in args.models:
        model_id = EMOTION_MODELS[short]
        try:
            classifier = hf_pipeline("audio-classification", model=model_id, device=device)
        except Exception as e:
            print(f"\n{short}: failed to load — {type(e).__name__}")
            continue
        labels = list(getattr(classifier.model.config, "id2label", {}).values())
        covered = sorted({to_core_emotion(label) for label in labels} - {None})
        print(f"\n=== {short}\n    model labels: {labels}")
        print(f"    of those, inside the shared core: {covered}")
        if len(covered) < len(EMOTION_CORE):
            missing = sorted(set(EMOTION_CORE) - set(covered))
            print(f"    ⚠ the model cannot output {missing} — it has no way of getting "
                  f"clips of those classes right")

        for name, (clips, language) in prepared.items():
            inputs = [{"array": audio, "sampling_rate": SAMPLE_RATE} for audio, _g in clips]
            started = time.perf_counter()
            try:
                predictions = classifier(inputs, top_k=1, batch_size=8)
            except Exception as e:
                print(f"    {name}: failed — {type(e).__name__}: {str(e)[:70]}")
                continue
            elapsed = time.perf_counter() - started
            hits = usable = reachable = 0
            for (_audio, gold), prediction in zip(clips, predictions):
                top = prediction[0] if isinstance(prediction, list) else prediction
                got = to_core_emotion(top["label"])
                usable += 1
                # ceiling: a clip whose class the model cannot output at all must not
                # count for it, and comparing overall accuracy against a model that does
                # have that class would be unfair.
                reachable += int(gold in covered)
                hits += int(got == gold)
            accuracy = hits / max(usable, 1) * 100
            ceiling = reachable / max(usable, 1) * 100
            results.append((short, name, language, accuracy, ceiling, usable, elapsed))
            print(f"    {name:9} ({language:18}): accuracy {accuracy:5.1f}% "
                  f"against a ceiling of {ceiling:5.1f}% on {usable} clips, {elapsed:.0f} s")
        del classifier
        gc_cuda()

    print("\n" + "=" * 72)
    header = f"{'model':14}"
    for name in args.datasets:
        header += f"{name:>22}"
    print(header)
    for short in args.models:
        row = f"{short:14}"
        for name in args.datasets:
            found = [r for r in results if r[0] == short and r[1] == name]
            if found:
                row += f"{found[0][3]:14.1f}% of {found[0][4]:.0f}%"
            else:
                row += f"{'—':>22}"
        print(row)
    baseline = 100.0 / len(EMOTION_CORE)
    print(f"\nrandom guessing would give {baseline:.0f}%. the comparison runs on the "
          "shared core "
          f"of four emotions; clips labelled outside it are excluded.")
    print("\"of N%\" is the ceiling: the share of clips whose class the model can "
          "output at all.")
    return 0


def gc_cuda():
    import gc

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def decode_clip(payload):
    """a dataset clip -> mono float32 at 16 kHz."""
    import soundfile as sf

    data, rate = sf.read(io.BytesIO(payload["bytes"]), dtype="float32", always_2d=True)
    audio = data.mean(axis=1)
    if rate != SAMPLE_RATE:

        target = int(round(len(audio) * SAMPLE_RATE / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, target),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def trim_silence(audio, threshold_db=-45.0, window=0.02):
    """trim the silence off the edges of a clip.

    audiobook clips start and end with a pause. glued together as they are, the ground
    truth would declare those pauses speech too, and DER would measure not the speaker
    labelling but how willing the detector is to call silence speech. measured, and it
    holds: without trimming, all of DER consisted of "missed speech" while speaker
    confusion sat near zero, so there was nothing to tune on such a benchmark.
    """
    step = max(1, int(window * SAMPLE_RATE))
    frames = len(audio) // step
    if frames < 2:
        return audio
    energy = np.array([np.sqrt(np.mean(np.square(audio[i * step:(i + 1) * step],
                                                 dtype=np.float64)))
                       for i in range(frames)])
    loud = 20 * np.log10(np.maximum(energy, 1e-9))
    voiced = np.where(loud > threshold_db)[0]
    if voiced.size == 0:
        return audio
    start = max(0, voiced[0] - 1) * step
    end = min(len(audio), (voiced[-1] + 2) * step)
    return audio[start:end]


def normalize_ru(text: str) -> str:
    """normalise to the form WER is usually computed on for russian."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- WER --------------
def run_wer(args):
    import jiwer
    import whisper
    from datasets import Audio, load_dataset

    model = whisper.load_model(args.model, device=args.device)
    print(f"model: {args.model} on {args.device}\n")

    rows = []
    for name in args.datasets:
        repo, config, split = WER_DATASETS[name]
        stream = load_dataset(repo, config, split=split, streaming=True)
        stream = stream.cast_column("audio", Audio(decode=False))

        refs, hyps, seconds = [], [], 0.0
        started = time.perf_counter()
        for row in itertools.islice(stream, args.clips):
            try:
                audio = decode_clip(row["audio"])
            except Exception:
                continue
            seconds += len(audio) / SAMPLE_RATE
            result = model.transcribe(audio, language="ru", verbose=False,
                                      word_timestamps=False)
            refs.append(normalize_ru(row["transcription"]))
            hyps.append(normalize_ru(result["text"]))
        elapsed = time.perf_counter() - started

        pairs = [(r, h) for r, h in zip(refs, hyps) if r]
        wer = jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]) * 100
        cer = jiwer.cer([r for r, _ in pairs], [h for _, h in pairs]) * 100
        rows.append((name, len(pairs), seconds / 60, wer, cer, seconds / elapsed))
        print(f"{name}: {len(pairs)} clips, {seconds / 60:.1f} min — "
              f"WER {wer:.1f}%, CER {cer:.1f}%")

    print()
    print(f"{'set':16}{'clips':>8}{'min':>8}{'WER %':>9}{'CER %':>9}{'× realtime':>13}")
    for name, n, minutes, wer, cer, speed in rows:
        print(f"{name:16}{n:8d}{minutes:8.1f}{wer:9.1f}{cer:9.1f}{speed:13.1f}")
    return 0


# ---------------------------------------------------------------- DER --------------
def collect_speakers(count, per_speaker, seed=0):
    """clips grouped by reader, with a check that the readers really are different."""
    from datasets import Audio, load_dataset

    repo, config, split = SPEAKER_SOURCE
    stream = load_dataset(repo, config, split=split, streaming=True)
    stream = stream.cast_column("audio", Audio(decode=False))

    # in RuLibriSpeech the file name is <book>_<chapter>_<author>_<number>. one book is
    # read by one person, so the prefix works as a reader label. verified below.
    groups = {}
    for row in itertools.islice(stream, count * per_speaker * 12):
        key = re.sub(r"_\d+\.wav$", "", row["audio"]["path"])
        bucket = groups.setdefault(key, [])
        if len(bucket) < per_speaker:
            try:
                bucket.append((decode_clip(row["audio"]), row["transcription"]))
            except Exception:
                pass
        if len([k for k, v in groups.items() if len(v) >= per_speaker]) >= count * 2:
            break

    ready = {k: v for k, v in groups.items() if len(v) >= per_speaker}
    print(f"groups collected by file prefix: {len(ready)}")
    return ready


def verify_distinct(groups, min_distance=0.45):
    """drop groups that, to the model's ear, turned out to be one person.

    the file-name prefix is only a guess about the reader. if one voice read two
    "different" books, the synthetic dialogue would be single-voiced and DER would be
    computed against a wrong ground truth. hence the embedding check.
    """
    import torch
    import torchaudio as _ta
    if not hasattr(_ta, "list_audio_backends"):
        _ta.list_audio_backends = lambda: ["soundfile"]
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

    embedder = PretrainedSpeakerEmbedding(
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    centroids = {}
    for key, clips in groups.items():
        vectors = []
        for audio, _text in clips[:3]:
            if len(audio) < SAMPLE_RATE:
                continue
            with torch.no_grad():
                vector = embedder(torch.from_numpy(audio.copy()).unsqueeze(0).unsqueeze(0))
            vector = np.asarray(vector).reshape(-1)
            if not np.isnan(vector).any():
                vectors.append(vector / (np.linalg.norm(vector) + 1e-9))
        if vectors:
            centre = np.mean(vectors, axis=0)
            centroids[key] = centre / (np.linalg.norm(centre) + 1e-9)

    keys = sorted(centroids)
    kept = []
    for key in keys:
        if all(1 - float(centroids[key] @ centroids[other]) >= min_distance
               for other in kept):
            kept.append(key)
    print(f"of those, distinguishable as different voices: {len(kept)}")
    return {k: groups[k] for k in kept}


def build_dialogue(groups, speakers=2, turns=24, seed=0, rng=None):
    """assemble a dialogue out of clips and return (audio, reference turns).

    turn lengths are taken as they come in a live conversation: plenty of short ones.
    those are exactly where the pipeline errs, so synthetic audio without short turns
    would be uselessly easy.
    """
    rng = rng or random.Random(seed)
    chosen = rng.sample(sorted(groups), speakers)
    pools = {name: list(groups[name]) for name in chosen}
    for pool in pools.values():
        rng.shuffle(pool)

    pieces, reference, cursor = [], [], 0.0
    for index in range(turns):
        name = chosen[index % speakers]
        if not pools[name]:
            break
        audio, _text = pools[name].pop()
        audio = trim_silence(audio)
        # a third of the turns are made short — the hardest case for diarization.
        if rng.random() < 0.35:
            length = int(rng.uniform(0.5, 1.6) * SAMPLE_RATE)
            if len(audio) > length:
                start = rng.randint(0, len(audio) - length)
                audio = audio[start:start + length]
        duration = len(audio) / SAMPLE_RATE
        if duration < 0.3:
            continue
        pieces.append(audio)
        reference.append((cursor, cursor + duration, f"SPK{chosen.index(name)}"))
        cursor += duration
        gap = rng.uniform(0.05, 0.45)
        pieces.append(np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32))
        cursor += gap
    return np.concatenate(pieces), reference


def to_annotation(turns):
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    # refined turns carry a fourth element, the uncertainty flag; take the first three.
    for turn in turns:
        start, end, speaker = turn[0], turn[1], turn[2]
        if end > start:
            annotation[Segment(start, end)] = speaker
    return annotation


def diarize_pyannote(audio, refine, margin=None):
    import transcribe as pipeline

    transcriber = diarize_pyannote.transcriber
    turns = pipeline.diarization_turns(transcriber._diarize(audio))
    if refine:
        if margin is not None:
            original = pipeline.SPEAKER_REFINE_MIN_MARGIN
            pipeline.SPEAKER_REFINE_MIN_MARGIN = margin
            pipeline.refine_speaker_turns.__defaults__ = (
                pipeline.SPEAKER_REFERENCE_MIN, margin,
                pipeline.SPEAKER_REFINE_MIN_SEPARATION, pipeline.SPEAKER_REFINE_ROUNDS)
        turns = transcriber._refine_speakers(turns, audio)
        if margin is not None:
            pipeline.SPEAKER_REFINE_MIN_MARGIN = original
    return turns


def diarize_diarizen(audio, python_path, model=None):
    import soundfile as sf

    helper = Path(__file__).resolve().parent / "diarize_diarizen.py"
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        out = Path(tmp) / "t.json"
        sf.write(str(wav), audio, SAMPLE_RATE)
        command = [str(python_path), str(helper), "--input", str(wav), "--output", str(out)]
        if model:
            command += ["--model", model]
        result = subprocess.run(command, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", "replace")[-400:])
        return json.loads(out.read_text(encoding="utf-8"))["turns"]


def run_der(args):
    from pyannote.metrics.diarization import DiarizationErrorRate

    import transcribe as pipeline

    groups = verify_distinct(collect_speakers(args.speakers_pool, args.clips_per_speaker))
    if len(groups) < 2:
        print("could not gather two distinguishable voices — raise --speakers-pool")
        return 1

    rng = random.Random(args.seed)
    dialogues = [build_dialogue(groups, args.speakers, args.turns, rng=rng)
                 for _ in range(args.dialogues)]
    total = sum(len(a) for a, _ in dialogues) / SAMPLE_RATE
    print(f"dialogues: {len(dialogues)}, {total / 60:.1f} min in total, "
          f"turns {sum(len(r) for _, r in dialogues)}\n")

    variants = []
    if "pyannote" in args.backend:
        variants.append(("pyannote 3.1", lambda a: diarize_pyannote(a, refine=False)))
        variants.append(("pyannote + refinement", lambda a: diarize_pyannote(a, refine=True)))
    if args.sweep_margin:
        for margin in args.sweep_margin:
            variants.append((f"refinement, margin {margin}",
                             lambda a, m=margin: diarize_pyannote(a, True, m)))
    if "diarizen" in args.backend:
        variants.append(("DiariZen", lambda a: diarize_diarizen(a, args.diarizen_python,
                                                                args.diarizen_model)))

    if any(name.startswith(("pyannote", "refinement")) for name, _ in variants):
        diarize_pyannote.transcriber = pipeline.Transcriber(
            model_name="tiny", diarize=True, emotion=False,
            diarize_model=args.diarize_model)

    results = []
    for name, run in variants:
        metric = DiarizationErrorRate(collar=args.collar)
        started = time.perf_counter()
        for audio, reference in dialogues:
            try:
                hypothesis = run(audio)
            except Exception as e:
                print(f"  {name}: failed — {type(e).__name__}: {e}")
                hypothesis = []
            metric(to_annotation(reference), to_annotation(hypothesis))
        der = abs(metric) * 100
        # the breakdown matters more than the total: speaker confusion is fixed by
        # refining labels, while missed and extra speech is the detector's business alone.
        # without splitting the metric it is easy to tune thresholds against a number they
        # do not affect at all.
        parts = metric[:]
        total = parts.get("total", 0) or 1
        confusion = parts.get("confusion", 0) / total * 100
        missed = parts.get("missed detection", 0) / total * 100
        false_alarm = parts.get("false alarm", 0) / total * 100
        results.append((name, der, confusion, missed, false_alarm,
                        time.perf_counter() - started))
        print(f"{name}: DER {der:.1f}% (confusion {confusion:.1f}, "
              f"missed {missed:.1f}, false alarm {false_alarm:.1f})")

    print()
    print(f"{'variant':28}{'DER %':>8}{'confusion':>10}{'missed':>9}"
          f"{'false al.':>9}{'sec':>7}")
    for name, der, confusion, missed, false_alarm, seconds in sorted(
            results, key=lambda r: r[1]):
        print(f"{name:28}{der:8.1f}{confusion:10.1f}{missed:9.1f}"
              f"{false_alarm:9.1f}{seconds:7.0f}")
    print()
    print(f"DER with a {args.collar} s collar on boundaries. confusion is the share of time "
          f"given to the wrong")
    print("speaker; missed and false alarm are errors about whether there is speech at all.")
    return 0


def run_real(args):
    """DER on corpora with human annotation."""
    from datasets import Audio, load_dataset
    from pyannote.metrics.diarization import DiarizationErrorRate

    import transcribe as pipeline

    variants = []
    if "pyannote" in args.backend:
        variants.append(("pyannote 3.1", lambda a: diarize_pyannote(a, refine=False)))
        variants.append(("pyannote + refinement", lambda a: diarize_pyannote(a, refine=True)))
    if "diarizen" in args.backend:
        variants.append(("DiariZen", lambda a: diarize_diarizen(a, args.diarizen_python,
                                                                args.diarizen_model)))
    if any(name.startswith(("pyannote", "refinement")) for name, _ in variants):
        diarize_pyannote.transcriber = pipeline.Transcriber(
            model_name="tiny", diarize=True, emotion=False,
            diarize_model=args.diarize_model)

    table = []
    for name in args.datasets:
        repo, config, split, description = REAL_DATASETS[name]
        stream = load_dataset(repo, config, split=split, streaming=True)
        stream = stream.cast_column("audio", Audio(decode=False))

        recordings = []
        for row in itertools.islice(stream, args.files):
            try:
                audio = decode_clip(row["audio"])
            except Exception as e:
                print(f"  clip skipped: {type(e).__name__}")
                continue
            limit = args.limit_minutes * 60
            if limit and len(audio) / SAMPLE_RATE > limit:
                audio = audio[:int(limit * SAMPLE_RATE)]
            end_of_audio = len(audio) / SAMPLE_RATE
            reference = [(start, min(end, end_of_audio), speaker)
                         for start, end, speaker in zip(row["timestamps_start"],
                                                        row["timestamps_end"],
                                                        row["speakers"])
                         if start < end_of_audio]
            recordings.append((audio, reference))

        if not recordings:
            print(f"{name}: could not read a single recording")
            continue
        minutes = sum(len(a) for a, _ in recordings) / SAMPLE_RATE / 60
        speakers = {s for _, ref in recordings for _, _, s in ref}
        print(f"\n=== {name} — {description}")
        print(f"    recordings {len(recordings)}, {minutes:.0f} min, "
              f"speakers in total {len(speakers)}")

        for label, run in variants:
            metric = DiarizationErrorRate(collar=args.collar,
                                          skip_overlap=args.skip_overlap)
            started = time.perf_counter()
            for audio, reference in recordings:
                try:
                    hypothesis = run(audio)
                except Exception as e:
                    print(f"    {label}: failed — {type(e).__name__}: {e}")
                    hypothesis = []
                metric(to_annotation(reference), to_annotation(hypothesis))
            parts = metric[:]
            total = parts.get("total", 0) or 1
            row = (name, label, abs(metric) * 100,
                   parts.get("confusion", 0) / total * 100,
                   parts.get("missed detection", 0) / total * 100,
                   parts.get("false alarm", 0) / total * 100,
                   time.perf_counter() - started, minutes)
            table.append(row)
            print(f"    {label}: DER {row[2]:.1f}% (confusion {row[3]:.1f}, "
                  f"missed {row[4]:.1f}, false alarm {row[5]:.1f})")

    print()
    print(f"{'corpus':14}{'variant':24}{'DER %':>8}{'confusion':>10}"
          f"{'missed':>9}{'false al.':>9}{'× realtime':>13}")
    for name, label, der, confusion, missed, false_alarm, seconds, minutes in table:
        speed = minutes * 60 / max(seconds, 1e-6)
        print(f"{name:14}{label:24}{der:8.1f}{confusion:10.1f}{missed:9.1f}"
              f"{false_alarm:9.1f}{speed:13.1f}")
    collar_note = ("with a %.2f s collar" % args.collar) if args.collar else "with no collar"
    overlap_note = "overlaps skipped" if args.skip_overlap else "overlaps counted"
    print(f"\nDER {collar_note} on boundaries, {overlap_note}.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    wer = sub.add_parser("wer", help="WER on russian test sets")
    wer.add_argument("--clips", type=int, default=100, help="clips from each set")
    wer.add_argument("--datasets", nargs="+", default=list(WER_DATASETS),
                     choices=list(WER_DATASETS))
    wer.add_argument("--model", default="turbo")
    wer.add_argument("--device", default="cuda")
    wer.set_defaults(func=run_wer)

    der = sub.add_parser("der", help="DER on synthetic russian dialogues")
    der.add_argument("--dialogues", type=int, default=8)
    der.add_argument("--turns", type=int, default=24, help="turns per dialogue")
    der.add_argument("--speakers", type=int, default=2)
    der.add_argument("--speakers-pool", type=int, default=6, help="how many readers to gather")
    der.add_argument("--clips-per-speaker", type=int, default=14)
    der.add_argument("--collar", type=float, default=0.25, help="collar on boundaries, s")
    der.add_argument("--seed", type=int, default=0)
    der.add_argument("--backend", nargs="+", default=["pyannote", "diarizen"],
                     choices=["pyannote", "diarizen"])
    der.add_argument("--diarize-model", default="pyannote/speaker-diarization-3.1")
    der.add_argument("--diarizen-python", default=None)
    der.add_argument("--diarizen-model", default=None)
    der.add_argument("--sweep-margin", nargs="+", type=float, default=None,
                     help="sweep the margin thresholds of speaker refinement")
    der.set_defaults(func=run_der)

    real = sub.add_parser("real", help="DER on corpora with human annotation")
    real.add_argument("--datasets", nargs="+", default=["voxconverse", "ami-sdm"],
                      choices=list(REAL_DATASETS))
    real.add_argument("--files", type=int, default=4, help="recordings from each corpus")
    real.add_argument("--limit-minutes", type=float, default=10.0,
                      help="take only the first N minutes of a recording (0 for all of it)")
    real.add_argument("--collar", type=float, default=0.0,
                      help="collar on boundaries. 0 is what pyannote publications use")
    real.add_argument("--skip-overlap", action="store_true",
                      help="do not penalise stretches where people speak at once")
    real.add_argument("--backend", nargs="+", default=["pyannote", "diarizen"],
                      choices=["pyannote", "diarizen"])
    real.add_argument("--diarize-model", default="pyannote/speaker-diarization-3.1")
    real.add_argument("--diarizen-python", default=None)
    real.add_argument("--diarizen-model", default=None)
    real.set_defaults(func=run_real)

    emotion = sub.add_parser("emotion", help="accuracy of emotion models across languages")
    emotion.add_argument("--clips", type=int, default=150, help="clips from each set")
    emotion.add_argument("--datasets", nargs="+", default=["resd", "ravdess"],
                         choices=list(EMOTION_DATASETS))
    emotion.add_argument("--models", nargs="+", default=list(EMOTION_MODELS),
                         choices=list(EMOTION_MODELS))
    emotion.set_defaults(func=run_emotion)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
