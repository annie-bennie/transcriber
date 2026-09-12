# -*- coding: utf-8 -*-
"""Turning audio into words — the recogniser behind one narrow contract.

The pipeline needs four things from a recogniser and does not care how they are produced:

  * segments with start, end and text;
  * a word inside each segment with its own start, end and probability — the speaker
    labelling attaches people to words, not to segments, and without them a whisper chunk
    gets smeared across two speakers;
  * the three confidence numbers the hallucination filter reads (no_speech_prob,
    avg_logprob, compression_ratio);
  * the continuous text.

Everything downstream — cutting, diarization, measurement, statistics — works from the json
these produce, so the recogniser can be replaced without touching another stage. Whisper is
the only one here; the seam is kept so that swapping it stays a change to this file alone.
"""
from .config import MAX_EMOTION_DURATION
from .logging import log, mem_snapshot

WHISPER = "whisper"
BACKENDS = (WHISPER,)
DEFAULT_BACKEND = WHISPER


def _round(value, digits):
    return round(value, digits) if isinstance(value, (int, float)) else None


class Transcription:
    """What a backend returns, in one shape.

    `segments` are dicts ready for the json: the pipeline stores them as-is, so a backend
    that shifts a field name breaks the hallucination filter rather than the transcript,
    which is the sort of failure that goes unnoticed. building them here keeps that in one
    place.
    """

    def __init__(self, segments, text):
        self.segments = segments
        self.text = text

    @property
    def words(self):
        return sum(len(s["words"]) for s in self.segments)


def _segment(index, start, end, text, no_speech, avg_logprob, compression, words):
    return {
        "id": index,
        "start": round(start, 3),
        "end": round(end, 3),
        "text": text.strip(),
        "no_speech_prob": _round(no_speech, 4),
        "avg_logprob": _round(avg_logprob, 4),
        "compression_ratio": _round(compression, 3),
        "words": words,
    }


def split_long_segments(segments, ceiling=MAX_EMOTION_DURATION):
    """No segment longer than the emotion measurement window.

    a segment becomes an utterance, the emotion model reads only the first
    MAX_EMOTION_DURATION seconds of one, and the resulting label is then attached to all of
    it — so the rest of an over-long segment would carry a number nobody measured, and
    nothing anywhere would say so. whisper keeps most segments well under the ceiling (the
    longest on an hour of dialogue was 8.1 s), but nothing in it guarantees that.

    splitting here rather than in the emotion stage keeps it a property of what the
    recogniser delivers: every stage after this one can rely on it.

    the cut goes on a word boundary, so word timings — which the speaker labelling depends
    on — stay intact. a segment with no word timings cannot be split safely and is left as
    it is.
    """
    out = []
    for seg in segments:
        out.extend(_split_one(seg, ceiling))
    for index, seg in enumerate(out):
        seg["id"] = index
    return out


def _split_one(seg, ceiling):
    if seg["end"] - seg["start"] <= ceiling or len(seg.get("words") or []) < 2:
        return [seg]

    words = seg["words"]
    # the first word that would take the piece past the ceiling starts the next one
    cut = next((i for i, w in enumerate(words)
                if w["end"] - seg["start"] > ceiling), len(words) - 1)
    cut = max(1, min(cut, len(words) - 1))
    head_words, tail_words = words[:cut], words[cut:]

    head = dict(seg)
    head["words"] = head_words
    head["end"] = head_words[-1]["end"]
    head["text"] = "".join(w["text"] for w in head_words).strip()

    tail = dict(seg)
    tail["words"] = tail_words
    tail["start"] = tail_words[0]["start"]
    tail["text"] = "".join(w["text"] for w in tail_words).strip()

    return [head] + _split_one(tail, ceiling)


class WhisperBackend:
    name = WHISPER

    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model = None

    def load(self):
        if self.model is None:
            log(f"loading whisper ({self.model_name})...")
            import whisper
            self.model = whisper.load_model(self.model_name, device=self.device)
            log(f"whisper loaded. {mem_snapshot()}")
        return self.model

    def to(self, device):
        """Move the weights. named like torch's own so the pipeline can park a backend
        exactly as it parks pyannote or the scales model."""
        import torch
        if self.model is not None:
            self.model.to(torch.device(device))

    def transcribe(self, audio, language, word_timestamps=True, verbose=False):
        model = self.load()
        # verbose=False gives a progress bar, verbose=True prints every line with its
        # timecode — that shows which second of the recording things break on.
        result = model.transcribe(audio, language=language, verbose=verbose,
                                  word_timestamps=word_timestamps)
        segments = [
            _segment(i, seg["start"], seg["end"], seg["text"],
                     seg.get("no_speech_prob"), seg.get("avg_logprob"),
                     seg.get("compression_ratio"),
                     [{"text": w["word"], "start": round(w["start"], 3),
                       "end": round(w["end"], 3), "prob": _round(w.get("probability"), 3)}
                      for w in (seg.get("words") or [])])
            for i, seg in enumerate(result["segments"])
        ]
        return Transcription(split_long_segments(segments),
                             (result.get("text") or "").strip())


def make_backend(name, model_name, device):
    """A backend by name. an unknown name is a typo, not a reason to fall back silently."""
    name = (name or DEFAULT_BACKEND).lower()
    if name == WHISPER:
        return WhisperBackend(model_name, device)
    raise ValueError(f"unknown asr backend {name!r}; have: {', '.join(BACKENDS)}")
