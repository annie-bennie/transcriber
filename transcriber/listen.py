# -*- coding: utf-8 -*-
"""The microphone and the terminal: a live mode around the engine in live.py.

what is here is the part that touches the world — an input device, a screen, a Ctrl+C — and it
deliberately holds no logic about recognition or speakers. the engine is measured by
tools/streaming_probe.py, which drives the same module without a microphone, so the numbers in
the plan keep describing what runs here.

three things are drawn differently because they are not equally certain:

  * the rough tail is redrawn in place and will change;
  * a finished line is printed with its speaker;
  * a correction to a speaker already printed is a separate note. a terminal cannot go back and
    rewrite a line, and pretending otherwise would hide that a label moved.
"""
import sys

from .config import SAMPLE_RATE
from .live import (
    RECLUSTER_SEC,
    SETTLE_SEC,
    SPEAKER_DISTANCE,
    STEP_SEC,
    WINDOW_SEC,
    Live,
    Speakers,
    voice_embedder,
)
from .logging import log
from .transcript import LiveTranscript, live_path


def audio_library():
    """sounddevice, or a message a person can act on.

    it is imported through here rather than at the top of the module because the offline
    pipeline must not need it. inside the generator below it was worse still: a generator body
    runs on the first read, which is outside any try around creating it, so a missing package
    arrived as a traceback from ten frames down.
    """
    try:
        import sounddevice as sd
    except Exception as e:
        raise RuntimeError("the live mode needs sounddevice: pip install sounddevice") from e
    return sd


def microphone(device=None, chunk_seconds=STEP_SEC):
    """Yield (chunk, overflowed) — mono float32 at SAMPLE_RATE — as audio is captured.

    the device is opened at its own rate and resampled here: asking portaudio for 16 kHz works
    on some devices and fails on others, and that failure looks like a broken microphone rather
    than a wrong number.
    """
    from math import gcd

    import numpy as np
    from scipy.signal import resample_poly

    sd = audio_library()
    info = sd.query_devices(device, "input")
    native = int(info["default_samplerate"])
    frames = max(1, int(chunk_seconds * native))
    factor = gcd(SAMPLE_RATE, native)
    up, down = SAMPLE_RATE // factor, native // factor

    with sd.InputStream(device=device, channels=1, samplerate=native, dtype="float32",
                        blocksize=frames) as stream:
        while True:
            data, overflowed = stream.read(frames)
            mono = np.asarray(data, dtype="float32").reshape(-1)
            if native != SAMPLE_RATE:
                mono = resample_poly(mono, up, down).astype("float32")
            yield mono, overflowed


def input_devices():
    """(index, name, channels) for every device that can record."""
    sd = audio_library()

    return [(index, device["name"], device["max_input_channels"])
            for index, device in enumerate(sd.query_devices())
            if device["max_input_channels"] > 0]


def clock(seconds):
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


class Screen:
    """Where the three kinds of certainty are drawn."""

    def __init__(self, out=None):
        self.out = out or sys.stdout
        self.drawn = ""
        self.words = []

    def redraw(self, tail=""):
        line = " ".join(self.words + ([tail] if tail else [])).strip()
        padding = " " * max(0, len(self.drawn) - len(line))
        self.out.write("\r  " + line + padding)
        self.out.flush()
        self.drawn = line

    def erase(self):
        self.out.write("\r" + " " * (len(self.drawn) + 2) + "\r")
        self.drawn = ""

    def line(self, event):
        self.erase()
        who = event["speaker"] or "?"
        self.out.write(f"[{who} {clock(event['start'])}] {event['text']}\n")
        self.out.flush()
        self.words = []

    def note(self, text):
        self.erase()
        self.out.write(f"  ({text})\n")
        self.out.flush()

    def show(self, event):
        kind = event["kind"]
        if kind == "final":
            self.words.extend(word["text"] for word in event["words"])
            self.redraw()
        elif kind == "partial":
            self.redraw(event["text"])
        elif kind == "settled":
            self.line(event)
        elif kind == "relabel":
            pieces = event["pieces"]
            shown = ", ".join(f"{clock(piece['start'])} → {piece['speaker']}"
                              for piece in pieces[:4])
            more = "" if len(pieces) <= 4 else f" and {len(pieces) - 4} more"
            self.note(f"speakers corrected: {shown}{more}")


def run_live(model_name="turbo", language="en", device=None, speakers=True, input_device=None,
             recluster=RECLUSTER_SEC, window=WINDOW_SEC, step=STEP_SEC, settle=SETTLE_SEC,
             distance=SPEAKER_DISTANCE, out=None, output_dir=None):
    """Listen, print and write until interrupted. 0, or 1 if the input cannot be opened."""
    import whisper

    screen = Screen(out)
    try:
        audio_library().query_devices(input_device, "input")
    except Exception as e:
        log(f"cannot listen: {type(e).__name__}: {e}")
        try:
            log("inputs that can record: "
                + ", ".join(f"{index}: {name}" for index, name, _ in input_devices()))
        except Exception:
            pass
        return 1

    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    transcript = LiveTranscript(live_path(output_dir), language=language)
    log(f"writing to {transcript.path}")

    log(f"loading whisper ({model_name}) on {device}...")
    recogniser = whisper.load_model(model_name, device=device)
    voices = None
    if speakers:
        log("loading the voice model...")
        voices = Speakers(voice_embedder(device), distance=distance)
    live = Live(recogniser, language=language, speakers=voices, window=window, step=step,
                settle=settle, recluster=recluster if speakers else 0)

    try:
        source = microphone(input_device, step)
    except Exception as e:
        log(f"cannot open the input: {type(e).__name__}: {e}")
        return 1

    log("listening — ctrl+c to stop")
    log("text appears about two seconds late and differs from a normal run; the recording "
        "itself is not kept")
    try:
        for chunk, overflowed in source:
            if overflowed:
                log("  ⚠ audio was dropped — this machine is not keeping up")
            for event in live.push(chunk):
                screen.show(event)
                if event["kind"] == "settled":
                    transcript.add(event["start"], event["speaker"], event["text"])
                    transcript.save()
                elif event["kind"] == "relabel":
                    # a corrected speaker has to reach the file too, or what is on disk stops
                    # matching what the screen said last
                    transcript.relabel(event["pieces"])
                    transcript.save()
    except KeyboardInterrupt:
        screen.erase()
        for event in live.finish():
            if event["kind"] == "settled":
                screen.line(event)
                transcript.add(event["start"], event["speaker"], event["text"])
            elif event["kind"] == "relabel":
                transcript.relabel(event["pieces"])
        voices_found = len({piece.speaker for piece in live.pieces if piece.speaker})
        log(f"stopped. {len(live.pieces)} lines, {voices_found} voices, "
            f"{live.heard / 60:.1f} min heard")
        log(f"written to {transcript.save()}")
    return 0
