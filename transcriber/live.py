# -*- coding: utf-8 -*-
"""Recognition while the person is still speaking.

audio goes in as it arrives, events come out:

  * `final` — words that will not change any more, as soon as that is true of them;
  * `settled` — a finished line, closed by a pause, with a speaker on it;
  * `partial` — the rough tail that is still being decided, words and their times;
  * `relabel` — speaker labels of lines already shown, corrected.

they are separate because they happen at different moments: a word stops changing about a
second after it is spoken, while the line it belongs to is only finished when the speaker
pauses. an
interface can show the first immediately and draw the second as a completed line.

what this costs, measured on this machine before any of it was built (see PLAN, stage 9):

  * text appears about 1.8 s after it is spoken;
  * it differs from the same whisper over the whole recording by 14% of words on a close
    microphone and 16–23% in a room. the offline pipeline stays the source of truth;
  * speaker labels on a two-person recording agree with the offline pipeline on 95–100% of
    words and never need correcting. in a six-person discussion, agreement is 59% if each piece
    is labelled once and never revisited, and 87% if everything heard so far is clustered again
    every 15 s — at the price of two thirds of the pieces having their label rewritten at least
    once. hence `relabel` events, and hence the labels being drawn as provisional.

the engine is source-agnostic on purpose: a microphone and a file replayed at its own speed are
two callers of the same code. that is what keeps the numbers above about the thing that
actually runs — the harness in tools/streaming_probe.py drives this module, not a copy
of it.
"""
from .config import SAMPLE_RATE

# how much audio each pass sees. 12 s was measured to be no better than 8.
WINDOW_SEC = 8.0
# how often a pass runs. every second keeps up with a 0.3 s pass and leaves room for the rest.
STEP_SEC = 1.0
# how far back from the edge of the audio a word is taken as final. 1 s beat 2 s on both
# latency and how much text had to be rewritten.
SETTLE_SEC = 1.0
# a pause this long ends a piece of speech — the same figure the offline segmentation uses.
GAP_SEC = 0.6
# cosine distance at which a voice counts as someone new. 0.75 found the right number of people
# in a six-voice discussion; smaller values invented extra ones.
SPEAKER_DISTANCE = 0.75
# how often everything heard so far is clustered again. 0 turns it off, which is right for two
# people: they are already at 95–100% without it and nothing has to be rewritten.
RECLUSTER_SEC = 15.0
# when a line is finished, its audio is recognised once more as a whole, with this much
# context on each side. the words of a live pass come from the edge of a sliding window, where
# the model cannot hear how the phrase ends; a closed line is a whole phrase, and the same
# model does better on it. the cost is one more pass over a few seconds, on a line that is
# already three seconds old.
POLISH_CONTEXT_SEC = 0.5

# only pieces at least this long decide where the clusters are: a shorter one gives a noisy
# embedding, and clustering on those invents people.
MIN_CLUSTER_SEC = 1.5


class Piece:
    """A stretch of speech that has settled: the text will not change, the speaker may."""

    __slots__ = ("start", "end", "text", "words", "speaker", "vector", "rough")

    def __init__(self, start, end, text, words):
        self.start = start
        self.end = end
        self.text = text
        self.words = words
        self.speaker = None
        self.vector = None
        self.rough = text

    def as_event(self, kind="settled"):
        return {"kind": kind, "start": round(self.start, 3), "end": round(self.end, 3),
                "text": self.text, "speaker": self.speaker, "words": self.words,
                "rough": self.rough}


class Live:
    """Push audio, take events. Nothing here sleeps, blocks or prints.

    `recogniser` is anything with whisper's `transcribe(audio, language=..., ...)`, and
    `speakers` is an optional transcriber.live.Speakers. both are passed in rather than built
    here so that a caller can decide what to load — the measuring harness loads whisper only.
    """

    def __init__(self, recogniser, language="en", speakers=None, window=WINDOW_SEC,
                 step=STEP_SEC, settle=SETTLE_SEC, gap=GAP_SEC, recluster=RECLUSTER_SEC,
                 polish=True, polish_context=POLISH_CONTEXT_SEC, polish_prompt=True):
        self.recogniser = recogniser
        self.language = language
        self.speakers = speakers
        self.window = window
        self.step = step
        self.settle = settle
        self.gap = gap
        self.recluster = recluster
        self.polish = polish
        self.polish_context = polish_context
        # whisper takes the text before as a hint; a line rarely starts a new subject, and the
        # previous line is exactly the context the sliding window could not give it
        self.polish_prompt = polish_prompt

        self.buffer = None          # numpy array of everything pushed, at SAMPLE_RATE
        self.heard = 0.0            # seconds of audio pushed so far
        self.last_pass = 0.0
        self.last_recluster = 0.0
        self.pieces = []            # settled speech, in order
        self.pending = []           # settled words not yet closed into a piece
        self.last_settled_end = 0.0
        self.partial = ""
        self.partial_words = []

    # ------------------------------------------------------------------ pushing audio in
    def push(self, samples):
        """Hand over the newest audio. returns the events it produced, possibly none."""
        import numpy as np

        samples = np.asarray(samples, dtype="float32").reshape(-1)
        if self.buffer is None:
            self.buffer = samples
        else:
            self.buffer = np.concatenate([self.buffer, samples])
        self.heard = len(self.buffer) / SAMPLE_RATE
        if self.heard - self.last_pass < self.step:
            return []
        self.last_pass = self.heard
        return self._pass()

    def finish(self):
        """The stream has ended: whatever is still pending becomes final."""
        events = []
        if self.buffer is not None and self.heard > self.last_pass:
            self.last_pass = self.heard
            events.extend(self._pass())
        for word in self.partial_words:
            if word["start"] < self.last_settled_end - 0.01:
                continue
            if self.pending and word["start"] - self.pending[-1][2] > self.gap:
                events.append(self._close_piece())
            self.pending.append((word["text"], word["start"], word["end"]))
            self.last_settled_end = word["end"]
        self.partial_words = []
        if self.pending:
            events.append(self._close_piece())
        if self.partial:
            self.partial = ""
        events.extend(self._relabel_now())
        return [event for event in events if event]

    # ------------------------------------------------------------------ one pass
    def _pass(self):
        begin = max(0.0, self.heard - self.window)
        chunk = self.buffer[int(begin * SAMPLE_RATE):]
        if len(chunk) < SAMPLE_RATE // 2:
            return []

        # temperature=0: whisper's default is to re-decode with randomness when it dislikes
        # a result, so the same audio gives different text from one pass to the next. live that
        # shows up as a tail that flickers between readings, and in measurement it moves the
        # baseline — a difference of a point or two then says nothing about anything.
        result = self.recogniser.transcribe(chunk, language=self.language, verbose=False,
                                            word_timestamps=True, temperature=0,
                                            condition_on_previous_text=False)
        words = []
        for segment in result.get("segments") or []:
            for word in segment.get("words") or []:
                text = (word.get("word") or "").strip()
                if text:
                    words.append((text, word["start"] + begin, word["end"] + begin))

        events = []
        horizon = self.heard - self.settle
        became_final = []
        for text, start, end in words:
            if end > horizon or start < self.last_settled_end - 0.01:
                continue
            if self.pending and start - self.pending[-1][2] > self.gap:
                events.append(self._close_piece())
            self.pending.append((text, start, end))
            self.last_settled_end = end
            became_final.append({"text": text, "start": round(start, 3),
                                 "end": round(end, 3)})
        if became_final:
            events.append({"kind": "final", "words": became_final})

        # the tail is what a screen can show at once, knowing it may change
        rough = [{"text": text, "start": round(start, 3), "end": round(end, 3)}
                 for text, start, end in words if end > horizon]
        tail = " ".join(word["text"] for word in rough).strip()
        self.partial_words = rough
        if tail != self.partial:
            self.partial = tail
            events.append({"kind": "partial", "text": tail, "words": rough})

        if self.recluster and self.heard - self.last_recluster >= self.recluster:
            self.last_recluster = self.heard
            events.extend(self._relabel_now())
        return [event for event in events if event]

    def _polish(self, start, end):
        """The whole line, recognised again. returns text, or None if that failed."""
        begin = max(0.0, start - self.polish_context)
        stop = min(self.heard, end + self.polish_context)
        chunk = self.buffer[int(begin * SAMPLE_RATE):int(stop * SAMPLE_RATE)]
        if len(chunk) < SAMPLE_RATE // 2:
            return None
        prompt = self.pieces[-1].text if (self.polish_prompt and self.pieces) else None
        try:
            result = self.recogniser.transcribe(
                chunk, language=self.language, verbose=False, word_timestamps=True,
                temperature=0, condition_on_previous_text=False, initial_prompt=prompt)
        except TypeError:
            # a recogniser that does not take a prompt is still a recogniser
            try:
                result = self.recogniser.transcribe(
                    chunk, language=self.language, verbose=False, word_timestamps=True,
                    temperature=0, condition_on_previous_text=False)
            except Exception:
                return None
        except Exception:
            return None
        words = []
        for segment in result.get("segments") or []:
            for word in segment.get("words") or []:
                text = (word.get("word") or "").strip()
                # the context is there to help the model, not to be quoted: only what falls
                # inside the line itself is kept
                middle = begin + (word["start"] + word["end"]) / 2
                if text and start - 0.05 <= middle <= end + 0.05:
                    words.append(text)
        return " ".join(words).strip() or None

    def _close_piece(self):
        words = self.pending
        self.pending = []
        rough = " ".join(text for text, _, _ in words).strip()
        text = rough
        if self.polish:
            text = self._polish(words[0][1], words[-1][2]) or rough
        piece = Piece(words[0][1], words[-1][2], text, len(words))
        piece.rough = rough
        if self.speakers is not None:
            audio = self.buffer[int(piece.start * SAMPLE_RATE):int(piece.end * SAMPLE_RATE)]
            piece.speaker, piece.vector = self.speakers.assign(audio)
        self.pieces.append(piece)
        return piece.as_event()

    def _relabel_now(self):
        """Cluster everything heard so far; report only the labels that actually changed."""
        if self.speakers is None or not self.recluster:
            return []
        changed = self.speakers.recluster(self.pieces)
        if not changed:
            return []
        return [{"kind": "relabel",
                 "pieces": [{"start": round(piece.start, 3), "speaker": piece.speaker}
                            for piece in changed]}]


class Speakers:
    """Who is speaking, decided as pieces settle and reconsidered as more voices are heard.

    two decisions, not one. the first is immediate and cannot wait: nearest of the voices heard
    so far, or a new person. the second, optional, is the one that makes a group work at all —
    cluster every piece again, letting already-shown labels change.
    """

    def __init__(self, embed, distance=SPEAKER_DISTANCE, min_cluster=MIN_CLUSTER_SEC):
        self.embed = embed
        self.distance = distance
        self.min_cluster = min_cluster
        self.centroids = {}

    def assign(self, audio):
        """(label, vector) for a piece of speech, or (None, None) if it cannot be embedded."""
        import numpy as np

        vector = self.embed(audio)
        if vector is None:
            return None, None
        best, best_distance = None, 2.0
        for label, (total, count) in self.centroids.items():
            centroid = total / count
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            gap = 1.0 - float(np.dot(centroid, vector))
            if gap < best_distance:
                best, best_distance = label, gap
        if best is None or best_distance > self.distance:
            best = f"SPEAKER_{len(self.centroids):02d}"
            self.centroids[best] = (vector.copy(), 1)
            return best, vector
        total, count = self.centroids[best]
        self.centroids[best] = (total + vector, count + 1)
        return best, vector

    def recluster(self, pieces):
        """Decide all the speakers again. returns the pieces whose label changed."""
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering

        usable = [piece for piece in pieces if piece.vector is not None]
        anchors = [piece for piece in usable
                   if piece.end - piece.start >= self.min_cluster]
        if len(anchors) < 2:
            return []

        clusters = AgglomerativeClustering(
            n_clusters=None, distance_threshold=self.distance,
            metric="cosine", linkage="average",
        ).fit_predict(np.vstack([piece.vector for piece in anchors]))

        sums = {}
        for piece, cluster in zip(anchors, clusters):
            total, count = sums.get(cluster, (0, 0))
            sums[cluster] = (total + piece.vector, count + 1)
        centroids = {}
        for cluster, (total, count) in sums.items():
            centroid = total / count
            centroids[cluster] = centroid / (np.linalg.norm(centroid) + 1e-9)

        # cluster numbers are arbitrary and change from one run to the next. naming a cluster
        # after the speaker most of its pieces already carry keeps the labels on screen still:
        # without this every re-clustering renamed everybody, and each piece looked reassigned.
        assignment = {}
        for piece in usable:
            nearest = min(centroids,
                          key=lambda c: 1.0 - float(np.dot(centroids[c], piece.vector)))
            assignment[id(piece)] = nearest

        names = {}
        taken = set()
        order = sorted(centroids, key=lambda c: -sum(1 for piece in usable
                                                     if assignment[id(piece)] == c))
        for cluster in order:
            votes = {}
            for piece in usable:
                if assignment[id(piece)] == cluster and piece.speaker:
                    votes[piece.speaker] = votes.get(piece.speaker, 0) + 1
            for candidate in sorted(votes, key=lambda name: -votes[name]):
                if candidate not in taken:
                    names[cluster] = candidate
                    taken.add(candidate)
                    break
        spare = 0
        for cluster in order:
            if cluster in names:
                continue
            while f"SPEAKER_{spare:02d}" in taken:
                spare += 1
            names[cluster] = f"SPEAKER_{spare:02d}"
            taken.add(names[cluster])

        changed = []
        for piece in usable:
            name = names[assignment[id(piece)]]
            if piece.speaker != name:
                piece.speaker = name
                changed.append(piece)
        return changed


def voice_embedder(device="cpu"):
    """embed(audio) -> unit vector, using the same voice model the offline refinement uses."""
    import numpy as np
    import torch

    from .compat import allow_pyannote_checkpoints, patch_torchaudio

    patch_torchaudio()
    allow_pyannote_checkpoints()
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

    from .config import DEFAULT_EMBEDDING_MODEL

    model = PretrainedSpeakerEmbedding(DEFAULT_EMBEDDING_MODEL, device=torch.device(device))

    def embed(audio):
        if audio.size < int(0.3 * SAMPLE_RATE):
            return None
        with torch.no_grad():
            vector = model(torch.from_numpy(audio.copy()).unsqueeze(0).unsqueeze(0))
        vector = np.asarray(vector).reshape(-1)
        if np.isnan(vector).any():
            return None
        return vector / (np.linalg.norm(vector) + 1e-9)

    return embed
