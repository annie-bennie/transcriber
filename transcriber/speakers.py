# -*- coding: utf-8 -*-
"""Who said what: diarization turns, word attribution, and the units built from them.

the central idea of this module is that a speaker label is an attribute of already
recognised words, not a way of cutting the audio before recognition. that ordering is
deliberate — see docs/diarization.md for the measurements behind it.
"""
import bisect
import re

from .config import (BACKCHANNEL_MAX_SEC, EMOTION_UNIT_MIN, EMOTION_UNIT_TARGET,
                     HALLUCINATION_PHRASES, MAX_EMOTION_DURATION,
                     SPEAKER_REFERENCE_MIN, SPEAKER_REFINE_MIN_MARGIN,
                     SPEAKER_REFINE_MIN_SEPARATION, SPEAKER_REFINE_ROUNDS,
                     SUSPECT_AVG_LOGPROB, SUSPECT_COMPRESSION_RATIO,
                     SUSPECT_NO_SPEECH_PROB, SUSPECT_REPEAT_RUN, SUSPECT_WORD_PROB,
                     TURN_MAX_GAP, WORD_MIN_OVERLAP, WORD_SPEAKER_GAP)


# ---------------------------------------------------------------- turns from pyannote
def diarization_turns(diarization):
    """Pull turns out of a pyannote object into a plain [start, end, speaker] list.

    no pyannote object appears below this line: everything works on the list that ends up
    in the json. that is what makes it possible to re-cut a recording or re-flag
    hallucinations straight from a finished file, without models or a gpu — and any other
    diarization engine only has to produce this list.
    """
    return [[round(turn.start, 3), round(turn.end, 3), spk]
            for turn, _, spk in diarization.itertracks(yield_label=True)]


def prepare_turns(turns):
    """Sort turns and prepare a list of starts for binary search.

    a fourth element, when present, marks a turn whose speaker refinement found no footing
    and left the diarizer's label as it was. everything below reads the first three.
    """
    ordered = sorted((float(t[0]), float(t[1]), t[2],
                      bool(t[3]) if len(t) > 3 else False) for t in turns)
    return ordered, [t[0] for t in ordered]


def _speaker_overlaps(turns, starts, start, end):
    """How many seconds each speaker overlaps the interval [start, end].

    turns are sorted and found by binary search: scanning all of them for every word would
    be tens of millions of iterations on a three-hour recording.
    """
    overlap_by_speaker = {}
    # first turn that can intersect: search by start, then step back while previous turns
    # still end after our start.
    i = bisect.bisect_left(starts, start)
    while i > 0 and turns[i - 1][1] > start:
        i -= 1
    while i < len(turns) and turns[i][0] < end:
        overlap = min(end, turns[i][1]) - max(start, turns[i][0])
        if overlap > 0:
            spk = turns[i][2]
            overlap_by_speaker[spk] = overlap_by_speaker.get(spk, 0.0) + overlap
        i += 1
    return overlap_by_speaker


def uncertain_span(turns, starts, start, end) -> bool:
    """Was the turn that covers this span most marked as uncertain?"""
    best_overlap, uncertain = 0.0, False
    i = bisect.bisect_left(starts, start)
    while i > 0 and turns[i - 1][1] > start:
        i -= 1
    while i < len(turns) and turns[i][0] < end:
        overlap = min(end, turns[i][1]) - max(start, turns[i][0])
        if overlap > best_overlap:
            best_overlap, uncertain = overlap, turns[i][3]
        i += 1
    return uncertain


def _nearest_speaker(turns, starts, word):
    """Nearest turn in time — a last resort for a word sitting in silence."""
    if not turns:
        return "UNKNOWN"
    i = bisect.bisect_left(starts, word["start"])
    best, best_gap = None, None
    for j in (i - 1, i):
        if 0 <= j < len(turns):
            gap = max(turns[j][0] - word["end"], word["start"] - turns[j][1], 0.0)
            if best_gap is None or gap < best_gap:
                best, best_gap = turns[j][2], gap
    return best if best is not None and best_gap <= WORD_SPEAKER_GAP else "UNKNOWN"


def attribute_words(words, turns, starts):
    """Speaker for every word. returns (speakers, confidence flags).

    a word covered confidently picks its own speaker — including a lone "uh-huh" in the
    middle of someone else's monologue, which is a backchannel and must not be lost. a word
    covered weakly or not at all does not trust its own fragmentary evidence and takes the
    speaker from the nearest confident neighbours. that way a turn boundary follows meaning
    rather than a random detector error.
    """
    n = len(words)
    speakers = [None] * n
    strong = [False] * n
    for i, word in enumerate(words):
        duration = max(word["end"] - word["start"], 1e-6)
        overlaps = _speaker_overlaps(turns, starts, word["start"], word["end"])
        if not overlaps:
            continue
        best = max(overlaps, key=overlaps.get)
        if overlaps[best] / duration >= WORD_MIN_OVERLAP:
            speakers[i] = best
            strong[i] = True

    prev_strong = [-1] * n
    last = -1
    for i in range(n):
        prev_strong[i] = last
        if strong[i]:
            last = i
    next_strong = [-1] * n
    following = -1
    for i in range(n - 1, -1, -1):
        next_strong[i] = following
        if strong[i]:
            following = i

    for i in range(n):
        if strong[i]:
            continue
        left, right = prev_strong[i], next_strong[i]
        if left >= 0 and right >= 0:
            if speakers[left] == speakers[right]:
                speakers[i] = speakers[left]
            else:
                # a word right on the boundary goes to whichever side is closer in time.
                gap_left = words[i]["start"] - words[left]["end"]
                gap_right = words[right]["start"] - words[i]["end"]
                speakers[i] = speakers[left] if gap_left <= gap_right else speakers[right]
        elif left >= 0:
            speakers[i] = speakers[left]
        elif right >= 0:
            speakers[i] = speakers[right]
        else:
            speakers[i] = _nearest_speaker(turns, starts, words[i])
    return speakers, strong


def assign_speakers(segments, turns, starts):
    """Whole-segment speaker — whoever overlaps it longest.

    the fallback path for recordings without word timestamps: coarser than word-level
    attribution, because a segment spanning a speaker change goes entirely to one of them.
    """
    out = []
    for seg in segments:
        overlaps = _speaker_overlaps(turns, starts, seg["start"], seg["end"])
        speaker = max(overlaps, key=overlaps.get) if overlaps else "UNKNOWN"
        out.append({**seg, "speaker": speaker})
    return out


# ---------------------------------------------------------------- refinement
def refine_speaker_turns(turns, embed, min_reference=SPEAKER_REFERENCE_MIN,
                         min_margin=SPEAKER_REFINE_MIN_MARGIN,
                         min_separation=SPEAKER_REFINE_MIN_SEPARATION,
                         max_rounds=SPEAKER_REFINE_ROUNDS):
    """Reassign speakers by proximity to voice references.

    embed(start, end) -> normalised vector or None. returns (turns, report); the input list
    is left alone.

    references are built from long turns: the diarizer labels those reliably, because at
    that length the embedding works. every short turn is then compared not to an equally
    noisy neighbour but to that averaged voice.

    several rounds, because the first pass also fixes long turns and the next one therefore
    gets cleaner references. embeddings are computed once — only the references change
    between rounds, so repeating costs nothing.
    """
    import numpy as np

    report = {"changed": 0, "abstained": 0, "checked": 0, "separation": None,
              "references": {}, "rounds": 0, "skipped": None, "uncertain": 0}

    vectors = [embed(start, end) for start, end, _ in turns]
    labels = [speaker for _, _, speaker in turns]
    original = list(labels)
    durations = [end - start for start, end, _ in turns]
    unsure = [False] * len(turns)

    def centroids_from(current):
        out = {}
        picked = {}
        for i, (vector, label) in enumerate(zip(vectors, current)):
            if vector is None or durations[i] < min_reference:
                continue
            picked.setdefault(label, []).append(i)
        for label, idx in picked.items():
            centre = np.mean([vectors[i] for i in idx], axis=0)
            out[label] = centre / (np.linalg.norm(centre) + 1e-9)
            report["references"][label] = (len(idx), sum(durations[i] for i in idx))
        return out

    for round_number in range(1, max_rounds + 1):
        centroids = centroids_from(labels)
        if len(centroids) < 2:
            report["skipped"] = "fewer than two speakers with long turns"
            return list(turns), report

        names = sorted(centroids)
        separation = min(1 - float(centroids[a] @ centroids[b])
                         for i, a in enumerate(names) for b in names[i + 1:])
        report["separation"] = round(separation, 3)
        if separation < min_separation:
            report["skipped"] = f"voice references too similar ({separation:.2f})"
            return list(turns), report

        updated, moved = [], 0
        checked = abstained = 0
        unsure = []
        for vector, label in zip(vectors, labels):
            if vector is None:
                # embedding did not compute (chunk too short) — no footing for a decision.
                updated.append(label)
                unsure.append(True)
                continue
            checked += 1
            distances = {name: 1 - float(vector @ centroids[name]) for name in names}
            order = sorted(distances, key=distances.get)
            if distances[order[1]] - distances[order[0]] < min_margin:
                # no margin means no evidence. keep the diarizer's label rather than flip a
                # coin: both errors on the reference exchange landed on exactly these cases.
                # but stay honest about it — the turn reaches the transcript marked.
                abstained += 1
                updated.append(label)
                unsure.append(True)
                continue
            if order[0] != label:
                moved += 1
            updated.append(order[0])
            unsure.append(False)

        report.update({"checked": checked, "abstained": abstained, "rounds": round_number})
        labels = updated
        if not moved:
            break

    report["changed"] = sum(1 for a, b in zip(original, labels) if a != b)
    report["uncertain"] = sum(unsure)
    return ([[start, end, label, flag]
             for (start, end, _), label, flag in zip(turns, labels, unsure)], report)


# ---------------------------------------------------------------- hallucination filter
def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _script_of(char: str) -> str:
    code = ord(char.lower())
    if 0x0400 <= code <= 0x04FF:
        return "cyrillic"
    if code < 0x0250:
        return "latin"
    if 0x0370 <= code <= 0x03FF:
        return "greek"
    if code >= 0x2E80:
        return "cjk"
    return "other"


def has_mixed_script_word(text: str) -> bool:
    """Is there a word with mixed scripts inside it?

    at the end of a one-hour recording whisper produced "иerna." and "HampиEM化." — debris
    with cyrillic, latin and cjk inside one word. real text does not look like that: a
    loanword written entirely in latin does not mix scripts within a word. the signal does
    not depend on the recording language, so it works outside russian too.
    """
    for token in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE):
        if len({_script_of(c) for c in token}) > 1:
            return True
    return False


def flag_suspect_segments(segments, asr_segments, turns=None, starts=None):
    """Mark segments that look invented. deletes nothing.

    rules were picked from a measurement on a one-hour recording, and on that recording
    every firing was checked to be a real hallucination. the expensive lesson was the first
    version: "outside diarized speech" fired 89 times and almost all of them were ordinary
    speech. pyannote's diarization is not a VAD — its turns cover 86% of the recording, and
    words legitimately fall into the gaps. that is no longer evidence on its own, only
    together with near-zero word confidence.
    """
    metrics = {a["id"]: a for a in asr_segments}
    counts = {}

    for seg in segments:
        reasons = []
        normalized = _normalize_for_match(seg["text"])
        if not normalized:
            reasons.append("empty text")
        elif any(phrase in normalized for phrase in HALLUCINATION_PHRASES):
            reasons.append("caption artefact")
        if has_mixed_script_word(seg["text"]):
            reasons.append("mixed scripts in a word")

        source = metrics.get(seg.get("asr_id"))
        if source is not None:
            ratio = source.get("compression_ratio")
            if ratio is not None and ratio > SUSPECT_COMPRESSION_RATIO:
                reasons.append("text repeats itself")
            no_speech, logprob = source.get("no_speech_prob"), source.get("avg_logprob")
            if (no_speech is not None and no_speech > SUSPECT_NO_SPEECH_PROB
                    and logprob is not None and logprob < SUSPECT_AVG_LOGPROB):
                reasons.append("whisper unsure there is speech here")

        word_prob = seg.get("word_prob")
        if (turns is not None and word_prob is not None and word_prob < SUSPECT_WORD_PROB
                and not _speaker_overlaps(turns, starts, seg["start"], seg["end"])):
            # neither signal works alone: plenty of real speech falls outside diarized
            # turns, and near-zero confidence happens on live fillers. together they point
            # at an empty spot where whisper talked over the silence.
            reasons.append("near-zero confidence outside diarized speech")

        if reasons:
            seg["suspect"] = reasons
        else:
            seg.pop("suspect", None)

    # decoder loop: the same line repeated many times in a row.
    run_start = 0
    for i in range(1, len(segments) + 1):
        current = _normalize_for_match(segments[i]["text"]) if i < len(segments) else None
        same = bool(current) and current == _normalize_for_match(segments[run_start]["text"])
        if not same:
            if i - run_start >= SUSPECT_REPEAT_RUN:
                for seg in segments[run_start:i]:
                    seg.setdefault("suspect", []).append("stuck on one phrase")
            run_start = i

    for seg in segments:
        for reason in seg.get("suspect", ()):
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def suspect_counts(segments):
    counts = {}
    for seg in segments:
        for reason in seg.get("suspect", ()):
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def describe_suspects(segments) -> str:
    counts = suspect_counts(segments)
    if not counts:
        return "none"
    flagged = sum(1 for s in segments if s.get("suspect"))
    detail = ", ".join(f"{reason}: {n}" for reason, n in
                       sorted(counts.items(), key=lambda kv: -kv[1]))
    return f"{flagged} of {len(segments)} ({detail})"


# ---------------------------------------------------------------- units of analysis
def build_segments(asr_segments, turns=None):
    """Working segments: whisper's own segments, cut where the speaker changes.

    the speaker used to be chosen for a whole segment by largest overlap. in dialogue a
    whisper segment regularly spans a speaker change, and then the tail of someone else's
    sentence was attributed to the wrong person — and the emotion model received a chunk
    with two voices in it. now the speaker is decided per word and the segment is cut at
    the change: both the text and the audio inside a segment belong to one person.

    the result is rebuilt from asr_segments on every recomputation, so a recording can be
    re-cut under a different diarization without running speech recognition again.

    turns is a [start, end, speaker] list from diarization_turns(); None means there is
    nobody to cut by.
    """
    prepared = prepare_turns(turns) if turns else (None, None)
    turns, starts = prepared
    flat = [(a["id"], w) for a in asr_segments for w in (a.get("words") or [])]

    if not flat:
        # no word timestamps (--no-word-timestamps, or the model did not return them) —
        # fall back to whole-segment assignment.
        segments = [
            {"id": i, "asr_id": a["id"], "start": a["start"], "end": a["end"],
             "text": a["text"], "word_count": 0}
            for i, a in enumerate(asr_segments)
        ]
        if turns is not None:
            segments = assign_speakers(segments, turns, starts)
        flag_suspect_segments(segments, asr_segments, turns, starts)
        return segments

    if turns is None:
        speakers, strong = [None] * len(flat), [False] * len(flat)
    else:
        speakers, strong = attribute_words([w for _, w in flat], turns, starts)

    by_asr = {}
    offset = 0
    for a in asr_segments:
        count = len(a.get("words") or [])
        by_asr[a["id"]] = (speakers[offset:offset + count], strong[offset:offset + count])
        offset += count

    out = []
    for a in asr_segments:
        asr_words = a.get("words") or []
        if not asr_words:
            # whisper occasionally emits a segment with no word marks. usually an empty
            # zero-length one at the very end, but if it carries text that text must reach
            # the result, even with a coarse whole-segment speaker.
            if not a["text"]:
                continue
            speaker = None
            if turns is not None:
                overlaps = _speaker_overlaps(turns, starts, a["start"], a["end"])
                speaker = max(overlaps, key=overlaps.get) if overlaps else "UNKNOWN"
            out.append({"id": len(out), "asr_id": a["id"], "speaker": speaker,
                        "start": a["start"], "end": a["end"], "text": a["text"],
                        "word_count": 0,
                        **({"speaker_conf": 0.0} if turns is not None else {}),
                        **({"speaker_uncertain": True}
                           if turns is not None
                           and uncertain_span(turns, starts, a["start"], a["end"])
                           else {})})
            continue

        word_speakers, word_strong = by_asr[a["id"]]
        current = None
        for word, speaker, is_strong in zip(asr_words, word_speakers, word_strong):
            # cut on a speaker change and on a whisper segment boundary: its own splitting
            # follows punctuation and intonation, and there is no reason to break it.
            if current is None or current["speaker"] != speaker:
                current = {"id": len(out), "asr_id": a["id"], "speaker": speaker,
                           "start": word["start"], "end": word["end"],
                           "_words": [word], "_strong": 0}
                out.append(current)
            else:
                current["end"] = word["end"]
                current["_words"].append(word)
            current["_strong"] += 1 if is_strong else 0

    for seg in out:
        words = seg.pop("_words", None)
        strong_count = seg.pop("_strong", 0)
        if words is None:
            continue
        seg["text"] = "".join(w["text"] for w in words).strip()
        seg["word_count"] = len(words)
        seg["start"] = round(seg["start"], 3)
        seg["end"] = round(seg["end"], 3)
        probs = [w["prob"] for w in words if w.get("prob") is not None]
        if probs:
            seg["word_prob"] = round(sum(probs) / len(probs), 3)
        if turns is not None:
            # share of words the diarizer touched confidently. a low value means the segment
            # sits on a boundary or in noise, and its speaker deserves suspicion.
            seg["speaker_conf"] = round(strong_count / len(words), 2)
            if uncertain_span(turns, starts, seg["start"], seg["end"]):
                seg["speaker_uncertain"] = True

    # flagging happens here rather than as a separate stage: it depends on exactly the same
    # two inputs and needs neither models nor audio. split them apart and they drift, the
    # way the readiness check and the compatibility check drifted three times.
    flag_suspect_segments(out, asr_segments, turns, starts)
    return out


def build_turns(segments, max_gap=TURN_MAX_GAP):
    """Turns: consecutive segments by one speaker.

    a turn is holding the floor as a whole, not a single sentence, so a pause inside one's
    own monologue does not break it. only a pause longer than max_gap does: after three
    seconds of silence it is a new utterance even from the same person. all the dialogue
    statistics — pauses, response latency, share of speaking, interruptions — are counted
    from turns.
    """
    turns = []
    for seg in segments:
        if (turns and turns[-1]["speaker"] == seg.get("speaker")
                and seg["start"] - turns[-1]["end"] <= max_gap):
            turns[-1]["end"] = seg["end"]
            turns[-1]["segments"].append(seg["id"])
        else:
            turns.append({"id": len(turns), "speaker": seg.get("speaker"),
                          "start": seg["start"], "end": seg["end"],
                          "segments": [seg["id"]]})
    return turns


def build_emotion_units(segments, turns, target=EMOTION_UNIT_TARGET):
    """Units for emotion measurement: consecutive segments of one speaker, merged.

    after word-level cutting a whisper segment is short: median 0.78 s on an hour of audio,
    62% under a second. on a chunk like that the model guesses. but dropping the short ones
    is not an option either — that is 45% of the recording, and a "yeah" in the middle of a
    story is said with the same intonation as everything around it. so they are merged with
    neighbours from the same turn until the target length is reached.

    merging never crosses a turn boundary, otherwise one chunk would hold two voices. the
    ceiling is MAX_EMOTION_DURATION: beyond that the model truncates the input anyway.
    """
    by_id = {seg["id"]: seg for seg in segments}
    units = []
    for turn in turns:
        current = None
        for seg_id in turn["segments"]:
            seg = by_id.get(seg_id)
            if seg is None:
                continue
            fits = (current is not None
                    and current["end"] - current["start"] < target
                    and seg["end"] - current["start"] <= MAX_EMOTION_DURATION)
            if fits:
                current["end"] = seg["end"]
                current["segments"].append(seg_id)
            else:
                current = {"id": len(units), "turn": turn["id"],
                           "speaker": turn.get("speaker"),
                           "start": seg["start"], "end": seg["end"],
                           "segments": [seg_id]}
                units.append(current)
            seg["unit"] = current["id"]
    return units


def units_by_segment(units):
    """segment id -> the unit it belongs to."""
    return {seg_id: unit for unit in units for seg_id in unit["segments"]}
