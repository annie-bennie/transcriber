# -*- coding: utf-8 -*-
"""Does the live mode hold up — measured on a recording, without a microphone.

this drives transcriber/live.py, the same engine the live mode uses, and replays a file into it
as if it were arriving from a microphone. it answers the three questions that decide whether a
live mode is worth having, and keeps answering them as the engine changes:

  * how long after a word is spoken does it appear;
  * how much of what was shown at once later changes;
  * how much is lost against the same whisper over the whole file — and, with --speakers,
    against the offline diarizen labels of the same audio.

nothing is slept through: the stream clock is simulated, and each pass adds its own compute
time to it. so the latency is what this machine would produce, and if a pass takes longer
than the step, the lag accumulates exactly as it would live.

    python tools/streaming_probe.py "sample/clip.mp4" --language ru --speakers

it prints one report and writes nothing.
"""
import argparse
import itertools
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transcriber.audio import decode_audio  # noqa: E402
from transcriber.config import SAMPLE_RATE  # noqa: E402
from transcriber.diarizen import helper, interpreter  # noqa: E402
from transcriber.live import Live, Speakers, voice_embedder  # noqa: E402

PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalise(text):
    """lower case, no punctuation, single spaces — the usual comparison for russian."""
    return " ".join(PUNCT.sub(" ", text.lower().replace("ё", "е")).split())


def reference_turns(audio):
    """Offline diarizen on the same audio — what the live labels are compared against."""
    import soundfile as sf

    python = interpreter()
    if python is None:
        print("no environment with diarizen found — speaker labels cannot be compared")
        return None
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        out = Path(tmp) / "turns.json"
        sf.write(str(wav), audio, SAMPLE_RATE)
        result = subprocess.run([str(python), str(helper()), "--input", str(wav),
                                 "--output", str(out)], capture_output=True)
        if result.returncode != 0:
            print("diarizen failed:", result.stderr.decode("utf-8", "replace")[-200:])
            return None
        return json.loads(out.read_text(encoding="utf-8"))["turns"]


def speaker_at(turns, start, end):
    """The reference speaker whose turn overlaps this span most."""
    best, best_overlap = None, 0.0
    for turn_start, turn_end, label in turns:
        overlap = min(end, turn_end) - max(start, turn_start)
        if overlap > best_overlap:
            best, best_overlap = label, overlap
    return best


def agreement(pairs):
    """Share of words where live and offline agree, under the best mapping of label names.

    the names are arbitrary on both sides, so what is compared is the division into speakers.
    """
    live_labels = sorted({live for live, _ in pairs if live})
    ref_labels = sorted({ref for _, ref in pairs if ref})
    if not live_labels or not ref_labels:
        return None
    best = 0
    for mapping in itertools.permutations(ref_labels, min(len(live_labels), len(ref_labels))):
        table = dict(zip(live_labels, mapping))
        best = max(best, sum(1 for live, ref in pairs if table.get(live) == ref))
    return best / len(pairs)


def run(args):
    import whisper

    audio = decode_audio(Path(args.media))
    if args.skip_minutes:
        audio = audio[int(args.skip_minutes * 60 * SAMPLE_RATE):]
    if args.minutes:
        audio = audio[:int(args.minutes * 60 * SAMPLE_RATE)]
    duration = len(audio) / SAMPLE_RATE
    print(f"{Path(args.media).name}: {duration / 60:.1f} min, "
          f"from {args.skip_minutes:g} min in")

    print(f"loading whisper ({args.model})...")
    model = whisper.load_model(args.model, device=args.device)

    # the reference for the text: the same model, the same audio, one pass. so what is measured
    # is the streaming, not the model.
    started = time.perf_counter()
    offline = model.transcribe(audio, language=args.language, verbose=False)
    offline_spent = time.perf_counter() - started
    offline_text = normalise(offline["text"])
    print(f"offline: {len(offline_text.split())} words in {offline_spent:.1f} s "
          f"({duration / offline_spent:.1f}× realtime)")

    speakers = None
    reference = None
    if args.speakers:
        speakers = Speakers(voice_embedder(args.device), distance=args.speaker_distance)
        reference = reference_turns(audio)

    live = Live(model, language=args.language, speakers=speakers, window=args.window,
                step=args.step, settle=args.commit_delay, gap=args.gap,
                recluster=args.recluster)

    settled = []
    final_latencies = []
    rewritten = set()
    rewrite_events = 0
    revisions = 0
    tail_words = {}         # word start -> the text last shown for it
    latencies = []
    now = 0.0               # the simulated stream clock
    lag_max = 0.0
    step_samples = int(args.step * SAMPLE_RATE)

    def take(events):
        nonlocal rewrite_events, revisions
        for event in events:
            if event["kind"] == "settled":
                settled.append(event)
                latencies.append(now - event["end"])
            elif event["kind"] == "final":
                for word in event["words"]:
                    final_latencies.append(now - word["end"])
            elif event["kind"] == "partial":
                # matched by the word's own time, not by position: the tail shifts as the
                # window slides, and comparing by position would count that as a change
                for word in event["words"]:
                    key = round(word["start"], 1)
                    was = tail_words.get(key)
                    if was is not None and normalise(was) != normalise(word["text"]):
                        revisions += 1
                    tail_words[key] = word["text"]
            elif event["kind"] == "relabel":
                rewrite_events += len(event["pieces"])
                rewritten.update(piece["start"] for piece in event["pieces"])

    for begin in range(0, len(audio), step_samples):
        slice_of_audio = audio[begin:begin + step_samples]
        arrived = (begin + len(slice_of_audio)) / SAMPLE_RATE
        wall = time.perf_counter()
        events = live.push(slice_of_audio)
        spent = time.perf_counter() - wall
        # the clock cannot go back: a pass starts when its audio has arrived and finishes
        # `spent` later. if that is after the next audio is due, the stream is behind.
        now = max(now, arrived) + spent
        lag_max = max(lag_max, now - arrived)
        take(events)

    take(live.finish())

    stream_text = normalise(" ".join(event["text"] for event in settled))
    print()
    print(f"window {args.window:g} s, step {args.step:g} s, settle after "
          f"{args.commit_delay:g} s")
    print(f"pieces settled            {len(settled)}, {len(stream_text.split())} words "
          f"(offline: {len(offline_text.split())})")
    if final_latencies:
        final_latencies.sort()
        print(f"a word stops changing     {final_latencies[len(final_latencies) // 2]:.1f} s "
              f"after it is spoken (median), "
              f"{final_latencies[int(len(final_latencies) * 0.9)]:.1f} s at the 90th")
    if latencies:
        latencies.sort()
        print(f"a line is finished        {latencies[len(latencies) // 2]:.1f} s after its "
              f"last word (median), {latencies[-1]:.1f} s at worst")
    print(f"lag behind the stream     {lag_max:.1f} s at worst")
    print(f"tail shown then changed   {revisions} words")

    if args.speakers:
        voices = sorted({piece.speaker for piece in live.pieces if piece.speaker})
        counts = {label: sum(piece.words for piece in live.pieces if piece.speaker == label)
                  for label in voices}
        print(f"voices found              {len(voices)} — {counts}")
        if args.recluster:
            print(f"labels rewritten          {len(rewritten)} of {len(live.pieces)} "
                  f"pieces at least once, {rewrite_events} times in total "
                  f"(re-clustered every {args.recluster:g} s)")
        if reference:
            pairs = []
            for piece in live.pieces:
                ref = speaker_at(reference, piece.start, piece.end)
                if piece.speaker and ref:
                    pairs.extend([(piece.speaker, ref)] * max(1, piece.words))
            share = agreement(pairs)
            print(f"offline, same audio       "
                  f"{len({label for _, _, label in reference})} voices, "
                  f"{len(reference)} turns")
            if share is not None:
                print(f"speaker agreement         {share * 100:.0f}% of words "
                      f"(best mapping of names)")
    try:
        import jiwer
        print(f"WER against the same model offline  "
              f"{jiwer.wer(offline_text, stream_text) * 100:.1f}%")
    except ImportError:
        print("WER not computed — pip install jiwer")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("media")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--device", default=None)
    parser.add_argument("--window", type=float, default=8.0,
                        help="seconds of audio each pass sees (default: 8)")
    parser.add_argument("--step", type=float, default=1.0,
                        help="how often a pass runs, in seconds of audio (default: 1)")
    parser.add_argument("--commit-delay", type=float, default=1.0,
                        help="how far back from the edge a word is taken as final "
                             "(default: 1)")
    parser.add_argument("--gap", type=float, default=0.6,
                        help="pause that ends a piece of speech, in seconds (default: 0.6)")
    parser.add_argument("--skip-minutes", type=float, default=0,
                        help="start N minutes in — a lecture becomes a conversation "
                             "later, and that is where several voices are")
    parser.add_argument("--minutes", type=float, default=0,
                        help="use only N minutes of the recording (default: all)")
    parser.add_argument("--speakers", action="store_true",
                        help="also label speakers, and compare with offline diarizen")
    parser.add_argument("--speaker-distance", type=float, default=0.75,
                        help="cosine distance at which a voice counts as someone new")
    parser.add_argument("--recluster", type=float, default=15.0,
                        help="cluster everything heard so far every N seconds, letting shown "
                             "labels change; 0 turns it off (default: 15)")
    args = parser.parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return run(args) or 0


if __name__ == "__main__":
    sys.exit(main())
