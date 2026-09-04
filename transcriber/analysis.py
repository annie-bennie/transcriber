# -*- coding: utf-8 -*-
"""Turning measurements into numbers that mean something.

two jobs live here. first, per-speaker baselines: every acoustic value is expressed as a
deviation from that person's own habit, because absolute values are not comparable between
people or recordings. second, dialogue statistics — who spoke how much, who cut in, how
fast people answered.
"""
import math

from .audio import analyze_acoustics, voice_quality
from .config import (BACKCHANNEL_MAX_SEC, EMOTION_BUCKET_SECONDS, EMOTION_UNIT_MIN,
                     EN_VOWELS, FAST_RESPONSE_SEC, MAD_TO_SIGMA, MAX_EMOTION_DURATION,
                     LOUDNESS_NOTABLE_DB, PITCH_NOTABLE_SEMITONES, RU_VOWELS,
                     SILENCE_FLOOR_DB, SPEAKER_BASELINE_MIN_DURATION,
                     SPEAKER_BASELINE_MIN_SEGMENTS, TURN_MAX_GAP, Z_NOTABLE)
from .emotion import EMOTION_DIMENSIONS
from .logging import log

VOICE_FIELDS = ("jitter", "shimmer", "hnr")


# ---------------------------------------------------------------- robust statistics
def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def robust_baseline(values):
    """(median, spread), resistant to outliers. spread of 0 becomes None — cannot divide."""
    if len(values) < 2:
        return (_median(values) if values else None), None
    centre = _median(values)
    spread = _median([abs(v - centre) for v in values]) * MAD_TO_SIGMA
    if spread <= 1e-9:
        # the whole sample is nearly identical and mad degenerated. fall back to the plain
        # standard deviation: returning None would leave the speaker without deviations at
        # all, and an outlier would go unnoticed.
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        spread = variance ** 0.5
    return centre, (spread if spread > 1e-9 else None)


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


# ---------------------------------------------------------------- speaker baselines
def build_speaker_stats(segments):
    """Loudness and pitch baseline per speaker.

    only segments at least SPEAKER_BASELINE_MIN_DURATION long take part: on chunks of a few
    tenths of a second both loudness and pitch are computed from a handful of samples and
    describe noise rather than a person.
    """
    fields = ("loudness_db", "pitch_mean", "pitch_std")
    collected = {}
    for seg in segments:
        speaker = seg.get("speaker")
        if not speaker or seg["end"] - seg["start"] < SPEAKER_BASELINE_MIN_DURATION:
            continue
        loudness = seg.get("loudness_db")
        if isinstance(loudness, (int, float)) and loudness < SILENCE_FLOOR_DB:
            continue
        bucket = collected.setdefault(speaker, {name: [] for name in fields})
        for name in fields:
            value = seg.get(name)
            if isinstance(value, (int, float)):
                bucket[name].append(float(value))

    stats = {}
    for speaker, values in collected.items():
        entry = {"segments": len(values["loudness_db"])}
        if entry["segments"] < SPEAKER_BASELINE_MIN_SEGMENTS:
            entry["enough"] = False
            stats[speaker] = entry
            continue
        entry["enough"] = True
        for name in fields:
            centre, spread = robust_baseline(values[name])
            entry[name] = {"median": round(centre, 2) if centre is not None else None,
                           "spread": round(spread, 3) if spread is not None else None}
        stats[speaker] = entry
    return stats


def apply_speaker_z(segments, stats):
    """Attach deviations from the speaker's baseline.

    two forms of the same thing. the z fields are distances in units of that speaker's own
    spread, comparable between measures; the delta fields are the same distance in the unit
    the measure actually has — decibels for loudness, semitones for pitch. the transcript
    prints the delta where one exists, because "7 dB quieter than usual" is a statement about
    the world and "-2.1σ" is a statement about a distribution.
    """
    targets = (("loudness_db", "loudness_z"), ("pitch_mean", "pitch_z"),
               ("pitch_std", "pitch_var_z"))
    deltas = ("loudness_delta_db", "pitch_delta_st")
    marked = 0
    for seg in segments:
        entry = stats.get(seg.get("speaker") or "")
        for _source, field in targets:
            seg.pop(field, None)
        for field in deltas:
            seg.pop(field, None)
        if not entry or not entry.get("enough"):
            continue
        loudness = seg.get("loudness_db")
        if isinstance(loudness, (int, float)) and loudness < SILENCE_FLOOR_DB:
            continue
        for source, field in targets:
            value = seg.get(source)
            baseline = entry.get(source) or {}
            centre, spread = baseline.get("median"), baseline.get("spread")
            if isinstance(value, (int, float)) and centre is not None and spread:
                seg[field] = round((value - centre) / spread, 2)
                marked += 1
        _apply_natural_deltas(seg, entry)
    return marked


def _apply_natural_deltas(seg, entry):
    """Distance from the speaker's median in decibels and semitones."""
    centre = (entry.get("loudness_db") or {}).get("median")
    value = seg.get("loudness_db")
    if isinstance(value, (int, float)) and centre is not None and "loudness_z" in seg:
        seg["loudness_delta_db"] = round(value - centre, 1)
    centre = (entry.get("pitch_mean") or {}).get("median")
    value = seg.get("pitch_mean")
    if isinstance(value, (int, float)) and centre and value > 0 and "pitch_z" in seg:
        # semitones: a ratio, so the same number means the same interval for any voice.
        seg["pitch_delta_st"] = round(12.0 * math.log2(value / centre), 1)


def build_unit_stats(units, stats, fields, counter_key):
    """Baseline for values that live on utterances.

    one mechanism for voice quality and for the emotion scales: both are measured over a
    whole utterance and mean something only when a person is compared with themselves.
    """
    collected = {}
    for unit in units:
        speaker = unit.get("speaker")
        if not speaker or unit.get(fields[0]) is None:
            continue
        bucket = collected.setdefault(speaker, {name: [] for name in fields})
        for name in fields:
            value = unit.get(name)
            if isinstance(value, (int, float)):
                bucket[name].append(float(value))

    for speaker, values in collected.items():
        entry = stats.setdefault(speaker, {"segments": 0, "enough": False})
        entry[counter_key] = len(values[fields[0]])
        if entry[counter_key] < SPEAKER_BASELINE_MIN_SEGMENTS:
            continue
        for name in fields:
            centre, spread = robust_baseline(values[name])
            entry[name] = {"median": round(centre, 4) if centre is not None else None,
                           "spread": round(spread, 5) if spread is not None else None}
    return stats


def apply_unit_z(units, stats, fields):
    marked = 0
    for unit in units:
        entry = stats.get(unit.get("speaker") or "") or {}
        for name in fields:
            unit.pop(name + "_z", None)
            value = unit.get(name)
            baseline = entry.get(name) or {}
            centre, spread = baseline.get("median"), baseline.get("spread")
            if isinstance(value, (int, float)) and centre is not None and spread:
                unit[name + "_z"] = round((value - centre) / spread, 2)
                marked += 1
    return marked


def build_voice_stats(units, stats):
    return build_unit_stats(units, stats, VOICE_FIELDS, "voice_units")


def apply_voice_z(units, stats):
    return apply_unit_z(units, stats, VOICE_FIELDS)


def build_dimension_stats(units, stats):
    return build_unit_stats(units, stats, EMOTION_DIMENSIONS, "dimension_units")


def apply_dimension_z(units, stats):
    return apply_unit_z(units, stats, EMOTION_DIMENSIONS)


def format_z(value):
    """Deviation for the transcript. only notable ones are shown, to keep lines readable."""
    if value is None or abs(value) < Z_NOTABLE:
        return ""
    return f"{value:+.1f}σ"


def _arrow(value, threshold, unit, digits=0):
    if not isinstance(value, (int, float)) or abs(value) < threshold:
        return ""
    return f"{'↑' if value > 0 else '↓'}{abs(value):.{digits}f}{unit}"


def format_delta_db(value):
    """How much louder or quieter than this speaker's usual level."""
    return _arrow(value, LOUDNESS_NOTABLE_DB, "dB")


def format_delta_semitones(value):
    """How far above or below this speaker's usual pitch, as a musical interval."""
    return _arrow(value, PITCH_NOTABLE_SEMITONES, "st")


# ---------------------------------------------------------------- measurement passes
def _bucket_pad(chunk, sample_rate):
    """Pad a clip with zeros to a multiple length — see EMOTION_BUCKET_SECONDS.

    padding does not disturb the model: the feature extractor produces an attention mask
    and statistics are taken over the real part only.
    """
    import numpy as np

    step = int(EMOTION_BUCKET_SECONDS * sample_rate)
    target = min(int(MAX_EMOTION_DURATION * sample_rate), -(-chunk.size // step) * step)
    return np.pad(chunk, (0, target - chunk.size)) if target > chunk.size else chunk


def _analyze_unit_block(units, idx, by_id, audio, sample_rate, dimensions,
                        voice=False, pitch_track=None):
    """One block: emotion scales per utterance, loudness and pitch per segment."""
    total = audio.shape[-1]

    def chunk_of(start, end):
        return audio[max(0, int(start * sample_rate)):min(total, int(end * sample_rate))]

    if dimensions is not None:
        long_enough = [i for i in idx
                       if units[i]["end"] - units[i]["start"] >= EMOTION_UNIT_MIN]
        if long_enough:
            clips = [chunk_of(units[i]["start"],
                              min(units[i]["end"],
                                  units[i]["start"] + MAX_EMOTION_DURATION))
                     for i in long_enough]
            for i, values in zip(long_enough, dimensions.measure(clips, sample_rate)):
                if values:
                    units[i].update(values)

    for i in idx:
        unit = units[i]
        unit["analyzed"] = True
        if voice:
            quality = voice_quality(chunk_of(unit["start"], unit["end"]), sample_rate)
            if quality:
                unit["jitter"], unit["shimmer"], unit["hnr"] = (
                    round(quality[0], 3), round(quality[1], 3), round(quality[2], 2))
        # acoustics stays per segment: loudness and pitch are meaningful on a short chunk,
        # unlike emotion, and per line they are more useful in the transcript.
        for seg_id in unit["segments"]:
            seg = by_id.get(seg_id)
            if seg is None:
                continue
            loudness_db, pitch_mean, pitch_std = analyze_acoustics(
                chunk_of(seg["start"], seg["end"]), sample_rate,
                pitch_track, seg["start"], seg["end"])
            seg["loudness_db"] = round(loudness_db, 2) if loudness_db is not None else None
            seg["pitch_mean"] = round(pitch_mean, 1) if pitch_mean is not None else None
            seg["pitch_std"] = round(pitch_std, 1) if pitch_std is not None else None


def analyze_units(units, segments, audio, sample_rate, dimensions=None, block_size=200,
                  on_block=None, voice=False, pitch_track=None):
    """Measurement in blocks, with a checkpoint after each one.

    units are modified in place and already measured ones are skipped, so after a crash the
    stage continues where it stopped. this is the longest and most fragile stage — vram runs
    out here — so writing partial results to disk as it goes is worth the trouble.
    """
    from .audio import PitchTrack

    pending = [i for i, unit in enumerate(units) if not unit.get("analyzed")]
    if not pending:
        return
    by_id = {seg["id"]: seg for seg in segments}
    if pitch_track is None:
        import time
        started = time.monotonic()
        pitch_track = PitchTrack(audio, sample_rate)
        if pitch_track.ready():
            log(f"    pitch track built in {time.monotonic() - started:.1f} s")
    block_size = max(1, block_size)
    total_blocks = -(-len(pending) // block_size)
    log(f"    to measure: {len(pending)} of {len(units)} utterances "
        f"({total_blocks} blocks of {block_size})")
    for b in range(total_blocks):
        idx = pending[b * block_size:(b + 1) * block_size]
        _analyze_unit_block(units, idx, by_id, audio, sample_rate, dimensions,
                            voice=voice, pitch_track=pitch_track)
        if on_block is not None:
            on_block()
        done = sum(1 for unit in units if unit.get("analyzed"))
        log(f"    block {b + 1}/{total_blocks}: measured {done}/{len(units)}")


# ---------------------------------------------------------------- dialogue statistics
#
# who spoke how much, who cut in on whom, how fast people answered, how fast they spoke.
# all arithmetic over data that already exists — no models, no audio.
#
# one subtlety about the source of timing. turns in state["turns"] are assembled from
# whisper segments and by construction follow one another without overlap: an interruption
# cannot be detected from them at all. so timing values — share of speaking, overlaps,
# response latency — are computed from diarization turns, where two people can sound at
# once. anything about text (speech rate, word counts) comes from segments.
def raw_speaker_spans(turns):
    """Raw speech intervals per speaker, with no merging."""
    by_speaker = {}
    for turn in turns:
        start, end, speaker = float(turn[0]), float(turn[1]), turn[2]
        if end > start:
            by_speaker.setdefault(speaker, []).append([start, end])
    for spans in by_speaker.values():
        spans.sort()
    return by_speaker


def _anyone_speaking(spans, starts, start, end) -> bool:
    """Does any of the intervals sound inside (start, end)?"""
    import bisect

    if end <= start or not spans:
        return False
    index = bisect.bisect_left(starts, start)
    while index > 0 and spans[index - 1][1] > start:
        index -= 1
    while index < len(spans) and spans[index][0] < end:
        if min(end, spans[index][1]) - max(start, spans[index][0]) > 0:
            return True
        index += 1
    return False


def merge_speaker_intervals(turns, max_gap=None):
    """Floor-holding intervals per speaker.

    a short pause inside one's own story does not break a turn — but only if nobody else
    spoke during that pause. without that caveat merging jumped over the other person's
    speech and it landed in "overlap": on an hour of conversation that produced 24% of the
    recording with simultaneous speech and speaking shares summing to 102%, which cannot be.
    somebody else started talking means the floor moved to them.
    """
    if max_gap is None:
        max_gap = TURN_MAX_GAP
    by_speaker = raw_speaker_spans(turns)

    merged = {}
    for speaker, spans in by_speaker.items():
        others = sorted(span for name, group in by_speaker.items() if name != speaker
                        for span in group)
        other_starts = [span[0] for span in others]
        out = []
        for start, end in spans:
            if out and start - out[-1][1] <= max_gap and not _anyone_speaking(
                    others, other_starts, out[-1][1], start):
                out[-1][1] = max(out[-1][1], end)
            else:
                out.append([start, end])
        merged[speaker] = out
    return merged


def union_seconds(spans):
    """Total length of the union of intervals — overlaps not counted twice."""
    ordered = sorted(spans)
    total, current_end, current_start = 0.0, None, None
    for start, end in ordered:
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def overlap_seconds(first, second):
    """How long two sets of intervals sound at the same time."""
    total, i, j = 0.0, 0, 0
    a, b = sorted(first), sorted(second)
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def count_syllables(text: str) -> int:
    """Syllables, approximated by groups of vowel letters.

    counting vowel letters one by one works for russian, where spelling is close to one
    vowel per syllable, and badly overcounts english: "you" is three vowel letters and one
    syllable. grouping adjacent vowels fixes that, and dropping a trailing silent "e"
    handles the other common english case — "are" becomes one syllable rather than two.

    still an approximation. it is used for speech rate, where what matters is the change
    within one speaker rather than the absolute value.
    """
    import re

    total = 0
    for word in re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE):
        groups = 0
        previous_was_vowel = False
        for ch in word:
            is_vowel = ch in RU_VOWELS or ch in EN_VOWELS
            if is_vowel and not previous_was_vowel:
                groups += 1
            previous_was_vowel = is_vowel
        # silent final "e": only in latin words, and never down to zero.
        if groups > 1 and word.endswith("e") and word[-1] in EN_VOWELS:
            groups -= 1
        total += groups
    return total


def analyze_dialogue(state) -> dict:
    """Dialogue statistics. returns the block for state["dialogue"]."""
    segments = state.get("segments") or []
    duration = (state.get("source") or {}).get("duration_sec")
    if not duration and segments:
        duration = max(seg["end"] for seg in segments)
    if not duration:
        return {}

    diarization = state.get("diarization") or []
    floors = merge_speaker_intervals(diarization) if diarization else {}
    if not floors:
        return {"duration_sec": round(duration, 2), "speakers": {},
                "note": "no diarization"}

    # speech time and overlaps come from raw intervals: merging pulls pauses inside a turn,
    # and the share of speech would come out inflated.
    raw = raw_speaker_spans(diarization)
    speech = union_seconds([span for spans in raw.values() for span in spans])

    timeline = sorted((start, end, speaker)
                      for speaker, spans in floors.items() for start, end in spans)

    # three different events that are easy to confuse:
    #   * a floor change — the previous speaker finished and the next continued past them;
    #   * a cut-in — the second started while the first was still speaking;
    #   * a nested turn — the second fitted entirely inside someone else's and took nothing.
    # the first version counted a nested turn as a floor change, and latency came out as
    # "minus the whole monologue": one "uh-huh" inside fifty seconds gave fifty seconds of
    # overlap. that is where the impossible 24% came from.
    latencies, interruptions, backchannels = [], {}, {}
    transitions = overlapping_transitions = 0
    for (start, end, speaker), (next_start, next_end, next_speaker) in zip(timeline,
                                                                          timeline[1:]):
        if next_speaker == speaker:
            continue
        cut_in = next_start < end
        if cut_in:
            target = (backchannels if next_end - next_start <= BACKCHANNEL_MAX_SEC
                      else interruptions)
            target[next_speaker] = target.get(next_speaker, 0) + 1
        if next_end <= end:
            continue
        transitions += 1
        latencies.append(next_start - end)
        if cut_in:
            overlapping_transitions += 1

    by_speaker_text = {}
    for seg in segments:
        speaker = seg.get("speaker")
        if not speaker:
            continue
        entry = by_speaker_text.setdefault(speaker, {"syllables": 0, "words": 0,
                                                     "voiced": 0.0})
        entry["syllables"] += count_syllables(seg.get("text") or "")
        entry["words"] += seg.get("word_count") or 0
        entry["voiced"] += max(0.0, seg["end"] - seg["start"])

    by_id = {seg["id"]: seg for seg in segments}
    pauses = {}
    for turn in state.get("turns") or []:
        members = [by_id[i] for i in turn.get("segments", []) if i in by_id]
        for previous, following in zip(members, members[1:]):
            gap = following["start"] - previous["end"]
            if gap > 0:
                pauses.setdefault(turn.get("speaker") or "", []).append(gap)

    speakers = {}
    for speaker, spans in sorted(floors.items()):
        talk = union_seconds(raw.get(speaker) or [])
        others = [span for name, group in raw.items() if name != speaker
                  for span in group]
        text = by_speaker_text.get(speaker, {"syllables": 0, "words": 0, "voiced": 0.0})
        own_pauses = pauses.get(speaker, [])
        lengths = [end - start for start, end in spans]
        speakers[speaker] = {
            "talk_sec": round(talk, 2),
            "talk_share": round(talk / speech, 3) if speech else None,
            "turns": len(spans),
            "median_turn_sec": round(_percentile(lengths, 0.5), 2) if lengths else None,
            "words": text["words"],
            "voiced_sec": round(text["voiced"], 2),
            # speech rate is measured against the time a person actually produced words,
            # not against floor time: otherwise a pause inside a story would deflate it.
            "syllables_per_sec": (round(text["syllables"] / text["voiced"], 2)
                                  if text["voiced"] > 0 else None),
            "words_per_sec": (round(text["words"] / text["voiced"], 2)
                              if text["voiced"] > 0 else None),
            "median_pause_within_sec": (round(_percentile(own_pauses, 0.5), 2)
                                        if own_pauses else None),
            "overlapped_by_others_sec": round(overlap_seconds(spans, others), 2),
            "interruptions_made": interruptions.get(speaker, 0),
            "backchannels_made": backchannels.get(speaker, 0),
        }

    uncertain = sum(seg["end"] - seg["start"] for seg in segments
                    if seg.get("speaker_uncertain"))
    voiced_total = sum(seg["end"] - seg["start"] for seg in segments)

    # simultaneous speech is computed by union arithmetic rather than by summing negative
    # latencies: each speaker's time minus the union is exactly the time two voices sounded.
    # that agrees with the speaking shares, which the earlier version did not.
    talk_total = sum(entry["talk_sec"] for entry in speakers.values())
    simultaneous = max(0.0, talk_total - speech)

    return {
        "duration_sec": round(duration, 2),
        "speech_sec": round(speech, 2),
        "silence_share": round(max(0.0, duration - speech) / duration, 3),
        "transitions": transitions,
        "overlapping_transitions": overlapping_transitions,
        "simultaneous_sec": round(simultaneous, 2),
        "simultaneous_share": round(simultaneous / speech, 3) if speech else None,
        "response_latency_median": (round(_percentile(latencies, 0.5), 2)
                                    if latencies else None),
        "response_latency_p10": (round(_percentile(latencies, 0.1), 2)
                                 if latencies else None),
        "fast_responses": sum(1 for value in latencies if 0 <= value < FAST_RESPONSE_SEC),
        # share of time where the speaker is in doubt. without this number the statistics
        # would look more precise than they are.
        "uncertain_share": (round(uncertain / voiced_total, 3) if voiced_total else None),
        "speakers": speakers,
    }
