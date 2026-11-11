# -*- coding: utf-8 -*-
"""Pure-function tests: speaker labelling, the hallucination filter, baselines, dialogue.

no models, no gpu, no audio — only logic over structures. run with:
    python tests/test_pipeline.py

the package is found relative to this file. an absolute path here once meant that after the
project moved the tests kept checking a copy at the old address and passed quietly.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from transcriber.analysis import (
    analyze_dialogue,
    apply_dimension_z,
    apply_speaker_z,
    apply_voice_z,
    build_dimension_stats,
    build_speaker_stats,
    build_voice_stats,
    count_syllables,
    format_delta_db,
    format_delta_semitones,
    format_z,
    merge_speaker_intervals,
    overlap_seconds,
    robust_baseline,
    union_seconds,
)
from transcriber.asr import make_backend, split_long_segments
from transcriber.check import FAIL, OK, WARN, line, ready, render
from transcriber.compat import from_pretrained
from transcriber.config import LOUDNESS_NOTABLE_DB, PITCH_NOTABLE_SEMITONES, SILENCE_FLOOR_DB
from transcriber.diarizen import interpreter
from transcriber.emotion import DEVIATION_GUIDE, EMOTION_DIMENSION_GUIDE, EMOTION_DIMENSIONS
from transcriber.listen import Screen, clock
from transcriber.live import Live
from transcriber.logging import software_versions
from transcriber.network import hub_reachable, prefer_local_models
from transcriber.render import transcript_path
from transcriber.sources import is_url
from transcriber.speakers import (
    build_emotion_units,
    build_segments,
    build_turns,
    flag_suspect_segments,
    refine_speaker_turns,
)
from transcriber.transcript import LiveTranscript, live_path

ok = True


def _raises(call, kind):
    try:
        call()
    except kind:
        return True
    except Exception:
        return False
    return False


def check(name, got, want):
    global ok
    good = got == want
    ok = ok and good
    print(f"{'OK  ' if good else 'FAIL'} {name}")
    if not good:
        print(f"       got:      {got}")
        print(f"       expected: {want}")


def words(*triples):
    return [{"text": t, "start": s, "end": e, "prob": 0.9} for t, s, e in triples]


def asr(*segs):
    return [{"id": i, "start": w[0]["start"], "end": w[-1]["end"],
             "text": "".join(x["text"] for x in w).strip(), "words": w}
            for i, w in enumerate(segs)]


# =====================================================================
#  cutting segments by speaker change
# =====================================================================
segs = asr(
    words((" well", 0.00, 0.30), (" it bothers", 0.30, 0.90), (" me", 0.90, 1.10)),
    words((" that", 1.20, 1.40), (" yes", 1.55, 1.75), (" exactly", 1.80, 2.40)),
)
dia = [(0.0, 1.45, "A"), (1.45, 2.50, "B")]
out = build_segments(segs, dia)
check("a segment spanning a speaker change is split in two",
      [(s["speaker"], s["text"]) for s in out],
      [("A", "well it bothers me"), ("A", "that"), ("B", "yes exactly")])
check("segment ids are renumbered", [s["id"] for s in out], [0, 1, 2])
check("the source whisper segment is kept", [s["asr_id"] for s in out], [0, 1, 1])
check("turns glue two consecutive pieces of A",
      [(t["speaker"], t["segments"]) for t in build_turns(out)],
      [("A", [0, 1]), ("B", [2])])

# a backchannel entirely inside someone else's turn must survive
segs = asr(words((" i", 0.0, 0.3), (" think", 0.3, 0.9), (" mhm", 1.0, 1.3),
                 (" that", 1.5, 1.8), (" next", 1.8, 2.4)))
dia = [(0.0, 0.95, "A"), (0.95, 1.35, "B"), (1.35, 2.5, "A")]
out = build_segments(segs, dia)
check("a lone backchannel stays with the other speaker",
      [(s["speaker"], s["text"]) for s in out],
      [("A", "i think"), ("B", "mhm"), ("A", "that next")])

# a word clipped by a neighbouring turn should not split a phrase
segs = asr(words((" on", 0.0, 0.5), (" the border", 1.0, 1.6), (" word", 1.7, 2.2)))
dia = [(0.0, 1.45, "A"), (1.45, 3.0, "B")]
out = build_segments(segs, dia)
check("a word on the boundary does not cut a phrase in half",
      [(s["speaker"], s["text"]) for s in out],
      [("A", "on the border"), ("B", "word")])

out = build_segments(segs)
check("without diarization segments are not cut",
      [(s["speaker"], s["text"]) for s in out], [(None, "on the border word")])

plain = [{"id": 0, "start": 0.0, "end": 2.2, "text": "on the border word", "words": []}]
out = build_segments(plain, [(0.0, 1.45, "A"), (1.45, 3.0, "B")])
check("fallback to whole-segment assignment when there are no words",
      [(s["speaker"], s["text"], s["word_count"]) for s in out],
      [("A", "on the border word", 0)])

segments = [
    {"id": 0, "speaker": "A", "start": 0.0, "end": 2.0},
    {"id": 1, "speaker": "A", "start": 2.5, "end": 4.0},
    {"id": 2, "speaker": "A", "start": 12.0, "end": 13.0},
]
check("a long pause starts a new turn",
      [(t["speaker"], t["segments"]) for t in build_turns(segments)],
      [("A", [0, 1]), ("A", [2])])

segs = asr(words((" first", 0.0, 0.6)), words((" third", 3.0, 3.6)))
segs.insert(1, {"id": 9, "start": 1.0, "end": 2.0, "text": "second without words",
                "words": []})
out = build_segments(segs, [(0.0, 4.0, "A")])
check("a segment without word marks is kept and gets a speaker",
      [(s["speaker"], s["text"], s["word_count"]) for s in out],
      [("A", "first", 1), ("A", "second without words", 0), ("A", "third", 1)])

segs = asr(words((" word", 0.0, 0.6)))
segs.append({"id": 9, "start": 120.0, "end": 120.0, "text": "", "words": []})
check("an empty zero-length segment is dropped",
      [s["text"] for s in build_segments(segs)], ["word"])


# =====================================================================
#  hallucination filter
# =====================================================================
def seg(id_, text, start=0.0, end=1.0, asr_id=None):
    return {"id": id_, "asr_id": id_ if asr_id is None else asr_id,
            "start": start, "end": end, "text": text}


def metrics(id_, no_speech=0.0, logprob=-0.3, compression=1.5):
    return {"id": id_, "no_speech_prob": no_speech, "avg_logprob": logprob,
            "compression_ratio": compression}


def reasons_for(segments, asr_meta, turns=None, starts=None):
    flag_suspect_segments(segments, asr_meta, turns, starts)
    return [s.get("suspect") for s in segments]


check("a caption template is flagged",
      reasons_for([seg(0, "Subtitles by DimaTorzok")], [metrics(0)]),
      [["caption artefact"]])
check("ordinary speech is not flagged",
      reasons_for([seg(0, "well it bothers me that i cannot do this")], [metrics(0)]),
      [None])
check("self-repeating text is flagged",
      reasons_for([seg(0, "yes yes yes yes yes")], [metrics(0, compression=3.1)]),
      [["text repeats itself"]])
check("high silence probability alone is not evidence",
      reasons_for([seg(0, "quiet but real speech")], [metrics(0, no_speech=0.9)]),
      [None])
check("silence together with low likelihood is evidence",
      reasons_for([seg(0, "something")], [metrics(0, no_speech=0.9, logprob=-1.4)]),
      [["whisper unsure there is speech here"]])

dia_turns, dia_starts = [(0.0, 5.0, "A", False)], [0.0]


def seg_p(id_, text, start, end, prob):
    d = seg(id_, text, start, end)
    d["word_prob"] = prob
    return d


check("outside diarized speech but confident — not evidence",
      reasons_for([seg_p(0, "cannot put it aside", 8.0, 8.9, 0.97)], [metrics(0)],
                  dia_turns, dia_starts),
      [None])
check("near-zero confidence inside speech — not evidence",
      reasons_for([seg_p(0, "so", 1.0, 1.3, 0.001)], [metrics(0)], dia_turns, dia_starts),
      [None])
check("near-zero confidence AND outside speech — evidence",
      reasons_for([seg_p(0, "Thanks.", 8.0, 8.4, 0.02)], [metrics(0)],
                  dia_turns, dia_starts),
      [["near-zero confidence outside diarized speech"]])
check("mixed scripts inside a word are flagged",
      reasons_for([seg(0, "HampиEM化.")], [metrics(0)]),
      [["mixed scripts in a word"]])
check("a latin loanword is not evidence",
      reasons_for([seg(0, "bought an iPhone and installed Windows")], [metrics(0)]),
      [None])

six = [seg(i, "To be continued.") for i in range(6)]
check("six identical lines in a row is a decoder loop",
      reasons_for(six, [metrics(i) for i in range(6)]),
      [["stuck on one phrase"]] * 6)
four = [seg(i, "Yes.") for i in range(4)]
check("four identical lines is live agreement, not evidence",
      reasons_for(four, [metrics(i) for i in range(4)]), [None] * 4)

one = [seg(0, "Subtitles by DimaTorzok")]
flag_suspect_segments(one, [metrics(0)])
flag_suspect_segments(one, [metrics(0)])
check("re-flagging does not duplicate reasons", one[0]["suspect"], ["caption artefact"])


# =====================================================================
#  utterances
# =====================================================================
def sg(id_, start, end, speaker="A"):
    return {"id": id_, "start": start, "end": end, "speaker": speaker, "text": "x"}


short = [sg(i, i * 0.8, i * 0.8 + 0.7) for i in range(10)]
units = build_emotion_units(short, build_turns(short))
check("short segments are merged into utterances",
      [(len(u["segments"]), round(u["end"] - u["start"], 1)) for u in units],
      [(7, 5.5), (3, 2.3)])
check("every segment knows its utterance", [s["unit"] for s in short], [0] * 7 + [1] * 3)

mixed = [sg(0, 0.0, 0.5, "A"), sg(1, 0.6, 1.1, "A"), sg(2, 1.2, 1.7, "B"),
         sg(3, 1.8, 2.3, "A")]
check("an utterance never crosses a speaker change",
      [(u["speaker"], u["segments"])
       for u in build_emotion_units(mixed, build_turns(mixed))],
      [("A", [0, 1]), ("B", [2]), ("A", [3])])

long_turn = [sg(i, i * 3.0, i * 3.0 + 2.9) for i in range(8)]
units = build_emotion_units(long_turn, build_turns(long_turn))
check("no utterance exceeds the ceiling",
      all(u["end"] - u["start"] <= 10.0 for u in units), True)
check("a long turn is broken into several utterances", len(units) > 1, True)


# =====================================================================
#  speaker refinement
# =====================================================================
VOICE_A = np.array([1.0, 0.0, 0.0])
VOICE_B = np.array([0.0, 1.0, 0.0])


def make_embed(table, default=None):
    def embed(start, end):
        v = table.get(round(start, 2), default)
        return None if v is None else np.asarray(v, dtype=float)
    return embed


long_turns = [[0.0, 8.0, "A"], [10.0, 18.0, "B"], [20.0, 28.0, "A"], [30.0, 38.0, "B"]]
turns_in = long_turns + [[40.0, 40.9, "A"]]
table = {0.0: VOICE_A, 20.0: VOICE_A, 10.0: VOICE_B, 30.0: VOICE_B, 40.0: VOICE_B}
out, report = refine_speaker_turns(turns_in, make_embed(table))
check("a short turn is reassigned against the references",
      [t[2] for t in out], ["A", "B", "A", "B", "B"])
check("the report counts reassignments", (report["changed"], report["abstained"]), (1, 0))

middle = (VOICE_A + VOICE_B) / np.linalg.norm(VOICE_A + VOICE_B)
table_mid = dict(table)
table_mid[40.0] = middle
out, report = refine_speaker_turns(turns_in, make_embed(table_mid))
check("with no margin the diarizer's label is kept",
      [t[2] for t in out], ["A", "B", "A", "B", "A"])
check("such a case counts as no margin", report["abstained"], 1)
check("a turn with no margin is marked uncertain",
      [t[3] for t in out], [False, False, False, False, True])

near = np.array([1.0, 0.05, 0.0])
near = near / np.linalg.norm(near)
table_same = {0.0: VOICE_A, 20.0: VOICE_A, 10.0: near, 30.0: near, 40.0: VOICE_B}
out, report = refine_speaker_turns(turns_in, make_embed(table_same))
check("indistinguishable references mean no refinement",
      [t[2] for t in out], [t[2] for t in turns_in])
check("the reason for refusing is named", bool(report["skipped"]), True)

solo = [[0.0, 8.0, "A"], [10.0, 18.0, "A"], [40.0, 40.9, "A"]]
out, report = refine_speaker_turns(solo, make_embed({0.0: VOICE_A, 10.0: VOICE_A,
                                                     40.0: VOICE_B}))
check("with one speaker refinement is skipped",
      (report["skipped"] is not None, [t[2] for t in out]), (True, ["A", "A", "A"]))

asr_one = [{"id": 0, "start": 40.0, "end": 40.9, "text": "Super.",
            "words": [{"text": " Super.", "start": 40.0, "end": 40.9, "prob": 0.9}]}]
out, _ = refine_speaker_turns(turns_in, make_embed(table_mid))
check("a segment inside an uncertain turn is marked",
      build_segments(asr_one, out)[0].get("speaker_uncertain"), True)
check("a segment inside a confident turn is not",
      build_segments(asr_one, [[40.0, 41.0, "A", False]])[0].get("speaker_uncertain"), None)


# =====================================================================
#  speaker baselines
# =====================================================================
def acoustic(id_, speaker, start, end, loud, pitch=200.0, spread=30.0):
    return {"id": id_, "speaker": speaker, "start": start, "end": end,
            "text": "x", "loudness_db": loud, "pitch_mean": pitch, "pitch_std": spread}


centre, spread = robust_baseline([10.0, 10.0, 10.0, 10.0, 100.0])
check("the median does not follow an outlier", centre, 10.0)
# more than half the values coincide here, mad degenerates, and the standard deviation
# fallback kicks in. that is deliberate: returning None would leave the speaker without
# deviations at all and the outlier would go unnoticed.
check("with a degenerate mad there is still a spread", spread is not None, True)
check("and the outlier still reads as a large deviation",
      abs((100.0 - centre) / spread) > 2.0, True)

normal = [-30.0, -29.0, -31.0, -30.5, -29.5, -30.0, -28.5]
_, spread_clean = robust_baseline(normal)
_, spread_dirty = robust_baseline(normal + [-5.0])
check("a single outlier barely moves the robust spread",
      abs(spread_dirty - spread_clean) < 0.5, True)

segs = [acoustic(i, "A", i * 2.0, i * 2.0 + 1.0, -30.0) for i in range(6)]
segs.append(acoustic(99, "A", 100.0, 100.1, -5.0))
check("a short segment does not enter the baseline",
      build_speaker_stats(segs)["A"]["segments"], 6)

segs_silent = [acoustic(i, "A", i * 2.0, i * 2.0 + 1.0, -30.0) for i in range(6)]
segs_silent.append(acoustic(99, "A", 100.0, 101.0, SILENCE_FLOOR_DB - 40))
stats = build_speaker_stats(segs_silent)
apply_speaker_z(segs_silent, stats)
check("digital silence does not enter the baseline", stats["A"]["segments"], 6)
check("and gets no deviation either", segs_silent[-1].get("loudness_z"), None)

few = [acoustic(i, "B", i * 2.0, i * 2.0 + 1.0, -30.0) for i in range(3)]
stats = build_speaker_stats(few)
apply_speaker_z(few, stats)
check("too few segments means no baseline",
      (stats["B"]["enough"], few[0].get("loudness_z")), (False, None))

values = [-30.0, -28.0, -26.0, -32.0, -34.0, -30.0]
segs = [acoustic(i, "A", i * 2.0, i * 2.0 + 1.0, v) for i, v in enumerate(values)]
segs.append(acoustic(50, "A", 90.0, 91.0, -18.0))
stats = build_speaker_stats(segs)
apply_speaker_z(segs, stats)
expected = round((-18.0 - stats["A"]["loudness_db"]["median"])
                 / stats["A"]["loudness_db"]["spread"], 2)
check("deviation is measured from that speaker's own median",
      segs[-1]["loudness_z"], expected)

mixed = ([acoustic(i, "A", i * 2.0, i * 2.0 + 1.0, -30.0 + (i % 3)) for i in range(6)]
         + [acoustic(10 + i, "B", 100 + i * 2.0, 101 + i * 2.0, -45.0 + (i % 3))
            for i in range(6)])
probe_a = acoustic(20, "A", 200.0, 201.0, -38.0)
probe_b = acoustic(21, "B", 210.0, 211.0, -38.0)
mixed += [probe_a, probe_b]
stats = build_speaker_stats(mixed)
apply_speaker_z(mixed, stats)
check("the same loudness gives different deviations for different people",
      probe_a["loudness_z"] < 0 < probe_b["loudness_z"], True)

stale = [acoustic(i, "C", i * 2.0, i * 2.0 + 1.0, -30.0) for i in range(3)]
stale[0]["loudness_z"] = 99.0
apply_speaker_z(stale, build_speaker_stats(stale))
check("a stale deviation is erased", stale[0].get("loudness_z"), None)

check("a small deviation stays out of the line", format_z(0.9), "")
check("a notable one goes in", format_z(2.4), "+2.4σ")
check("a negative one keeps its sign", format_z(-3.2), "-3.2σ")
check("a missing one prints nothing", format_z(None), "")


# =====================================================================
#  deviations in the unit each measure has
# =====================================================================

check("a small level change stays out of the line", format_delta_db(4.0), "")
check("six decibels is where it starts", format_delta_db(6.0), "↑6dB")
check("quieter points down", format_delta_db(-8.4), "↓8dB")
check("a missing level prints nothing", format_delta_db(None), "")

check("four semitones is not worth a mark", format_delta_semitones(4.0), "")
check("five semitones is", format_delta_semitones(5.0), "↑5st")
check("lower points down", format_delta_semitones(-12.3), "↓12st")

# the point of semitones: the same interval reads the same for any voice, which hertz
# cannot do. a fifth above 100 Hz and a fifth above 250 Hz are 50 and 125 Hz apart.
low = acoustic(0, "bass", 0.0, 2.0, -30.0)
low["pitch_mean"] = 150.0
high = acoustic(1, "soprano", 0.0, 2.0, -30.0)
high["pitch_mean"] = 375.0
bass = [acoustic(i, "bass", i * 3.0, i * 3.0 + 2.0, -30.0) for i in range(6)]
for seg in bass:
    seg["pitch_mean"] = 100.0
bass.append(low)
soprano = [acoustic(i + 10, "soprano", i * 3.0, i * 3.0 + 2.0, -30.0) for i in range(6)]
for seg in soprano:
    seg["pitch_mean"] = 250.0
soprano.append(high)
both = bass + soprano
apply_speaker_z(both, build_speaker_stats(both))
check("a fifth above a bass and above a soprano give the same number of semitones",
      (round(low["pitch_delta_st"], 1), round(high["pitch_delta_st"], 1)), (7.0, 7.0))
check("in hertz the same interval would not match",
      low["pitch_mean"] - 100.0 == high["pitch_mean"] - 250.0, False)

# every measure the transcript can mark must be described in the guide that ships in the json
described = set(DEVIATION_GUIDE["measures"])
check("the guide covers every marked measure",
      described, {"loudness", "pitch", "pitch_spread", "emotion_scales"})
check("the guide quotes the thresholds actually used",
      (DEVIATION_GUIDE["measures"]["loudness"]["printed_from"],
       DEVIATION_GUIDE["measures"]["pitch"]["printed_from"]),
      (LOUDNESS_NOTABLE_DB, PITCH_NOTABLE_SEMITONES))


# =====================================================================
#  emotion scales
# =====================================================================
check("the scales are named as expected", EMOTION_DIMENSIONS,
      ("arousal", "dominance", "valence"))
check("the reading key names all three scales",
      sorted(EMOTION_DIMENSION_GUIDE["dimensions"]), ["arousal", "dominance", "valence"])
check("the key describes all four quadrants", len(EMOTION_DIMENSION_GUIDE["quadrants"]), 4)


def dim_unit(id_, speaker, arousal=0.5, dominance=0.5, valence=0.5):
    return {"id": id_, "speaker": speaker, "start": id_ * 6.0, "end": id_ * 6.0 + 5.0,
            "segments": [id_], "arousal": arousal, "dominance": dominance,
            "valence": valence}


calm = [dim_unit(i, "A", 0.50 + (i % 3) * 0.01, 0.50, 0.50 + (i % 3) * 0.01)
        for i in range(8)]
calm.append(dim_unit(50, "A", arousal=0.92, valence=0.15))
stats = build_dimension_stats(calm, {})
apply_dimension_z(calm, stats)
check("the scale baseline is built", stats["A"]["dimension_units"], 9)
check("an arousal spike reads as a deviation", calm[-1]["arousal_z"] > 2.0, True)
check("a valence drop reads as a deviation", calm[-1]["valence_z"] < -2.0, True)

mixed = ([dim_unit(i, "A", 0.30 + (i % 3) * 0.02) for i in range(8)]
         + [dim_unit(20 + i, "B", 0.70 + (i % 3) * 0.02) for i in range(8)])
probe_a = dim_unit(40, "A", arousal=0.50)
probe_b = dim_unit(41, "B", arousal=0.50)
mixed += [probe_a, probe_b]
stats = build_dimension_stats(mixed, {})
apply_dimension_z(mixed, stats)
check("the same arousal gives opposite signs for different people",
      probe_a["arousal_z"] > 0 > probe_b["arousal_z"], True)

partial = [dim_unit(i, "C") for i in range(6)]
partial.append({"id": 90, "speaker": "C", "start": 100.0, "end": 105.0, "segments": [90]})
stats = build_dimension_stats(partial, {})
apply_dimension_z(partial, stats)
check("an utterance without measurements is not counted",
      stats["C"]["dimension_units"], 6)
check("and gets no deviations", partial[-1].get("arousal_z"), None)


def voice_unit(id_, speaker, jitter=2.0, shimmer=12.0, hnr=12.0):
    return {"id": id_, "speaker": speaker, "start": id_ * 6.0, "end": id_ * 6.0 + 5.0,
            "segments": [id_], "jitter": jitter, "shimmer": shimmer, "hnr": hnr}


voice = [voice_unit(i, "A", 2.0 + (i % 3) * 0.1, 12.0 + (i % 3) * 0.2,
                    12.0 - (i % 3) * 0.3) for i in range(8)]
voice.append(voice_unit(50, "A", jitter=4.0, shimmer=18.0, hnr=6.0))
stats = build_voice_stats(voice, {})
apply_voice_z(voice, stats)
check("the voice baseline is built over utterances", stats["A"]["voice_units"], 9)
check("an atypical utterance gets notable deviations",
      all(abs(voice[-1][f + "_z"]) > 1.5 for f in ("jitter", "shimmer", "hnr")), True)


# =====================================================================
#  dialogue statistics
# =====================================================================
check("the union does not count overlap twice",
      union_seconds([(0.0, 10.0), (5.0, 12.0), (20.0, 22.0)]), 14.0)
check("adjacent intervals join", union_seconds([(0.0, 5.0), (5.0, 9.0)]), 9.0)
check("intersection of two sets",
      overlap_seconds([(0.0, 10.0)], [(8.0, 15.0), (20.0, 21.0)]), 2.0)
check("disjoint sets give zero", overlap_seconds([(0.0, 5.0)], [(6.0, 9.0)]), 0.0)
check("syllables are counted by vowel groups", count_syllables("hello how are you"), 5)

merged = merge_speaker_intervals([[0.0, 5.0, "A"], [6.0, 10.0, "A"]], max_gap=3.0)
check("a pause inside one's own speech is glued", merged["A"], [[0.0, 10.0]])
merged = merge_speaker_intervals([[0.0, 5.0, "A"], [6.0, 10.0, "A"], [5.2, 5.8, "B"]],
                                 max_gap=3.0)
check("someone else speaking in the pause breaks the turn",
      (merged["A"], merged["B"]), ([[0.0, 5.0], [6.0, 10.0]], [[5.2, 5.8]]))
merged = merge_speaker_intervals([[0.0, 5.0, "A"], [20.0, 25.0, "A"]], max_gap=3.0)
check("a long pause breaks the turn", merged["A"], [[0.0, 5.0], [20.0, 25.0]])


def dialogue_state(diarization, segments=None, duration=100.0):
    segments = segments or []
    turns = build_turns(segments) if segments else []
    return {"source": {"duration_sec": duration}, "diarization": diarization,
            "segments": segments, "turns": turns}


def spoken(id_, speaker, start, end, text="yes yes"):
    return {"id": id_, "speaker": speaker, "start": start, "end": end, "text": text,
            "word_count": len(text.split())}


data = analyze_dialogue(dialogue_state([[0.0, 20.0, "A"], [25.0, 35.0, "B"]], duration=50.0))
check("speech time from the union", data["speech_sec"], 30.0)
check("silence from the recording length", data["silence_share"], round(20.0 / 50.0, 3))
check("speaking shares",
      (data["speakers"]["A"]["talk_share"], data["speakers"]["B"]["talk_share"]),
      (round(20 / 30, 3), round(10 / 30, 3)))
check("one floor change", data["transitions"], 1)
check("response latency", data["response_latency_median"], 5.0)
check("no simultaneous speech", data["simultaneous_sec"], 0.0)

data = analyze_dialogue(dialogue_state([[0.0, 60.0, "A"], [30.0, 30.6, "B"]], duration=70.0))
check("a nested turn is not a floor change", data["transitions"], 0)
check("a short nested turn counts as a backchannel",
      data["speakers"]["B"]["backchannels_made"], 1)
check("simultaneous speech equals the insert length", data["simultaneous_sec"], 0.6)

data = analyze_dialogue(dialogue_state([[0.0, 10.0, "A"], [8.0, 20.0, "B"]], duration=25.0))
check("an interruption is a floor change", data["transitions"], 1)
check("the change happened over someone else's speech",
      data["overlapping_transitions"], 1)
check("a long cut-in is an interruption, not a backchannel",
      (data["speakers"]["B"]["interruptions_made"],
       data["speakers"]["B"]["backchannels_made"]), (1, 0))
check("latency is negative on an interruption", data["response_latency_median"], -2.0)
check("simultaneous speech time", data["simultaneous_sec"], 2.0)

segments = [spoken(0, "A", 0.0, 2.0, "mother washed the frame"),
            spoken(1, "A", 8.0, 10.0, "once more")]
data = analyze_dialogue(dialogue_state([[0.0, 10.0, "A"]], segments, duration=12.0))
check("rate is measured against time spent producing words",
      data["speakers"]["A"]["voiced_sec"], 4.0)
check("syllables per second",
      data["speakers"]["A"]["syllables_per_sec"],
      round(count_syllables("mother washed the frame once more") / 4.0, 2))

segments = [spoken(0, "A", 0.0, 4.0),
            dict(spoken(1, "A", 5.0, 6.0), speaker_uncertain=True)]
data = analyze_dialogue(dialogue_state([[0.0, 6.0, "A"]], segments, duration=10.0))
check("the share of uncertain time is reported", data["uncertain_share"], round(1 / 5, 3))

data = analyze_dialogue({"source": {"duration_sec": 10.0}, "diarization": [],
                         "segments": [spoken(0, None, 0.0, 5.0)], "turns": []})
check("without diarization nothing is invented", data.get("speakers"), {})


# =====================================================================
#  input from a url
# =====================================================================

check("a url is recognised", [is_url(v) for v in
      ("https://youtu.be/x", "http://a.b/c", "ytsearch3:two hosts",
       "C:/video.mp4", "/home/a/v.mp4", "video.mp4")],
      [True, True, True, False, False, False])


# =====================================================================
#  where the txt goes
# =====================================================================

out = Path("out")
check("a transcript with speakers is named apart",
      transcript_path({"speakers": ["SPEAKER_00", "SPEAKER_01"]}, out, "meeting").name,
      "meeting_speakers.txt")
check("without speakers the name stays plain",
      transcript_path({"speakers": []}, out, "meeting").name, "meeting.txt")
check("a state that never got that far is plain too",
      transcript_path({}, out, "meeting").name, "meeting.txt")



# =====================================================================
#  the recognition backend contract
# =====================================================================
# an over-long segment becomes an over-long utterance, the emotion model reads only its first
# MAX_EMOTION_DURATION seconds, and the label is then attached to all of it. whisper usually
# stays under the ceiling, but nothing in it guarantees that.

words = [{"text": f" w{i}", "start": float(i), "end": float(i + 1), "prob": 0.9}
         for i in range(28)]
long_seg = {"id": 0, "start": 0.0, "end": 28.0, "text": "x", "no_speech_prob": 0.1,
            "avg_logprob": -0.2, "compression_ratio": 1.5, "words": words}

pieces = split_long_segments([dict(long_seg)])
check("an over-long segment is cut to the measurement ceiling",
      max(p["end"] - p["start"] for p in pieces) <= 10.0, True)
check("no word is lost in the cutting",
      sum(len(p["words"]) for p in pieces), 28)
check("the cut lands on a word boundary",
      all(p["start"] == p["words"][0]["start"] and p["end"] == p["words"][-1]["end"]
          for p in pieces), True)
check("ids are renumbered so the pipeline can index them",
      [p["id"] for p in pieces], list(range(len(pieces))))

short = split_long_segments([dict(long_seg, end=5.0, words=words[:5])])
check("a segment already short enough is untouched", len(short), 1)

no_words = split_long_segments([dict(long_seg, words=[])])
check("a segment with no word timings cannot be cut safely, so it is left alone",
      len(no_words), 1)

check("an unknown backend name is an error, not a silent fallback",
      _raises(lambda: make_backend("gpt-whisper", "turbo", "cpu"), ValueError), True)

# =====================================================================
#  what produced a result
# =====================================================================
# read from package metadata, so this runs where nothing but numpy is installed — which is
# exactly where a missing package must come out as absent rather than as a crash.

versions = software_versions()
check("the record names python and this package",
      ("python" in versions, "transcriber" in versions), (True, True))
check("every heavy dependency has an entry, installed or not",
      all(name in versions for name in ("openai-whisper", "pyannote.audio", "torch")), True)
check("an absent package is recorded as None, never as an error",
      all(v is None or isinstance(v, str) for v in versions.values()), True)

# =====================================================================
#  no network, cached models
# =====================================================================
# without this, a run with no network waits out the hub's retries before every model load:
# five minutes of a six-minute run in a container. nothing here opens a real connection.

env = {}
note = prefer_local_models(env, reachable=lambda: False)
check("an unreachable hub switches the libraries to the local cache",
      (note is not None, env.get("HF_HUB_OFFLINE")), (True, "1"))
env = {}
check("a reachable hub changes nothing",
      (prefer_local_models(env, reachable=lambda: True), env), (None, {}))
probed = []
env = {"HF_HUB_OFFLINE": "0"}
check("an explicit setting is the user's decision: not probed, not changed",
      (prefer_local_models(env, reachable=lambda: probed.append(1) or False), env, probed),
      (None, {"HF_HUB_OFFLINE": "0"}, []))


def _refuse(address, timeout):
    raise ConnectionRefusedError()


class _Accepted:
    def close(self):
        pass


check("a refused connection counts as unreachable", hub_reachable(connect=_refuse), False)
check("an accepted connection counts as reachable",
      hub_reachable(connect=lambda address, timeout: _Accepted()), True)

# =====================================================================
#  the token argument that changed its name
# =====================================================================
# pyannote.audio 4 takes `token`, the 3.1.1 fork inside diarizen `use_auth_token`. the wrong
# one is a TypeError that the loaders report as a model that could not be loaded.


class _Pyannote4:
    @classmethod
    def from_pretrained(cls, checkpoint, token=None, cache_dir=None):
        return {"token": token}


class _Pyannote3:
    @classmethod
    def from_pretrained(cls, checkpoint, hparams_file=None, use_auth_token=None):
        return {"use_auth_token": use_auth_token}


check("pyannote 4 gets the token as `token`",
      from_pretrained(_Pyannote4, "m", "hf_x"), {"token": "hf_x"})
check("pyannote 3.x gets it as `use_auth_token`",
      from_pretrained(_Pyannote3, "m", "hf_x"), {"use_auth_token": "hf_x"})

# =====================================================================
#  is this machine ready
# =====================================================================
# --check has to say what is missing and what to do about it, and it has to be honest about
# whether a run can start at all: a warning is not a refusal.

sample = [line(OK, "python", "3.10.19"),
          line(WARN, "torch", "no card in use", "reinstall from the cu128 index"),
          line(FAIL, "ffmpeg", "cannot be run", "pip install imageio-ffmpeg")]
report = render(sample)
check("every check is one line, with its mark",
      [row.split()[0] for row in report.splitlines() if row and not row.startswith(" ")][:3],
      ["✓", "⚠", "✗"])
check("a hint is printed for what is not ok, and only for that",
      (report.count("→"), "cu128" in report, "imageio-ffmpeg" in report), (2, True, True))
check("a missing essential is reported as blocking", ready(sample), False)
check("warnings alone still allow a run", ready(sample[:2]), True)
check("nothing missing reads as ready", render(sample[:1]).splitlines()[-1], "ready to run.")

# =====================================================================
#  where diarizen is
# =====================================================================
# three layouts have to work without asking: diarizen in this environment (the default install
# and the image), a separate environment named by flag or variable, one next to the project.
# the benchmark once kept its own copy of this and never learned the first case — every
# subprocess
# failed and it reported DER 100% while the pipeline was labelling the same audio correctly.

never = lambda path: False          # noqa: E731 — a stand-in filesystem, not a definition
always = lambda path: True          # noqa: E731

check("an interpreter given by hand wins over everything else",
      str(interpreter("/given/python", environ={"DIARIZEN_PYTHON": "/from/var"},
                      exists=always, importable=lambda name: True)),
      str(Path("/given/python")))
check("then the variable",
      str(interpreter(None, environ={"DIARIZEN_PYTHON": "/from/var"},
                      exists=always, importable=lambda name: True)),
      str(Path("/from/var")))
check("then an environment next to the project",
      interpreter(None, environ={}, root="/project", exists=always,
                  importable=lambda name: True).parts[-3:],
      (".venv-diarizen", "Scripts", "python.exe"))
check("this environment, when diarizen is installed here",
      interpreter(None, environ={}, exists=never, importable=lambda name: True),
      Path(sys.executable))
check("nothing anywhere is None, not a guess",
      interpreter(None, environ={}, exists=never, importable=lambda name: False), None)

# =====================================================================
#  the live mode: what it promises to emit, and when
# =====================================================================
# four kinds of event, because four different moments — a word stops changing about a second
# after it is spoken, the line it belongs to only ends when the speaker pauses, the rough tail
# changes constantly, and a speaker label can be corrected after it was shown. an interface
# draws them differently, so the contract is worth a test that needs no model, no microphone.


class _ScriptedWhisper:
    """Stands in for whisper: two words near the end of whatever window it is given.

    it cannot know where the window sits in the recording — neither does whisper — so the words
    are placed relative to the chunk and the engine is the one that shifts them into the
    recording's clock. that is exactly the arithmetic worth testing.
    """

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, language=None, verbose=False, word_timestamps=True,
                   condition_on_previous_text=False, **whisper_options):
        # **whisper_options because the engine passes what whisper takes — temperature, a
        # prompt — and a stand-in must not care which of them arrive
        seconds = len(audio) / 16000
        self.calls += 1
        return {"segments": [{"words": [
            {"word": f" settled{self.calls}", "start": seconds - 2.0, "end": seconds - 1.5},
            {"word": f" rough{self.calls}", "start": seconds - 0.5, "end": seconds - 0.2},
        ]}]}


scripted = _ScriptedWhisper()
# polish=False here: re-recognising a finished line is measured separately below, and with a
# stub that answers differently every call it would replace the very text under test
live = Live(scripted, language="ru", speakers=None, window=8.0, step=1.0, settle=1.0,
            recluster=0, polish=False)
seen = []
for _ in range(4):
    seen.extend(live.push(np.zeros(16000, dtype="float32")))

check("a pass runs once per step of audio", scripted.calls, 4)
check("words below the horizon are reported as final, in order",
      [word["text"] for event in seen if event["kind"] == "final" for word in event["words"]],
      ["settled2", "settled3", "settled4"])
# the first pass has one second of audio, so its word would start before the recording did.
# a word cannot be dated outside the stream, and the engine drops it rather than moving it.
check("a word that would start before the recording is dropped, not shifted",
      "settled1" not in [word["text"] for event in seen if event["kind"] == "final"
                         for word in event["words"]], True)
check("the tail is reported as rough, and changes every pass",
      [event["text"] for event in seen if event["kind"] == "partial"],
      ["rough1", "rough2", "rough3", "rough4"])
check("a final word is dated in the recording's clock, not the window's",
      [round(word["end"], 1) for event in seen if event["kind"] == "final"
       for word in event["words"]], [0.5, 1.5, 2.5])
check("no line is finished while the speaker keeps going",
      [event for event in seen if event["kind"] == "settled"], [])

closing = live.finish()
check("the stream ending finishes the line that was open",
      [(event["kind"], event["text"]) for event in closing],
      [("settled", "settled2 settled3 settled4"), ("settled", "rough4")])
check("finishing twice does not duplicate the last words", live.finish(), [])
check("a line with no speaker says so rather than inventing one",
      closing[0]["speaker"], None)


# re-recognising a finished line: the words of a live pass come from the edge of a sliding
# window, where the model cannot hear how the phrase ends. a closed line is a whole phrase.
# what has to hold is that its text is the second answer, with the first kept for comparison.
polished = _ScriptedWhisper()
second = Live(polished, language="ru", speakers=None, window=8.0, step=1.0, settle=1.0,
              recluster=0, polish=True)
for _ in range(4):
    second.push(np.zeros(16000, dtype="float32"))
line = second.finish()[0]
check("a finished line is recognised again, as a whole",
      polished.calls > scripted.calls, True)
check("the line carries the second answer",
      line["text"], "settled5")
check("and keeps the first one, so the two can be compared",
      line["rough"], "settled2 settled3 settled4")


# what the terminal does with those events: the rough tail is redrawn in place, a finished line
# is printed, and a correction is a separate note — a terminal cannot rewrite a printed line.
screen = Screen(out=io.StringIO())
screen.show({"kind": "final", "words": [{"text": "ну", "start": 0.0, "end": 0.4}]})
screen.show({"kind": "partial", "text": "вот", "words": []})
screen.show({"kind": "settled", "start": 61.0, "end": 62.0, "text": "ну вот", "speaker": None})
screen.show({"kind": "relabel", "pieces": [{"start": 61.0, "speaker": "SPEAKER_01"}]})
printed = screen.out.getvalue()
check("the rough tail is drawn after the words that are already final",
      "ну вот" in printed.split("\r")[2], True)
check("a finished line carries its speaker and its time", "[? 01:01] ну вот" in printed, True)
check("an unknown speaker is printed as a question mark, not as a name",
      "[SPEAKER" in printed.split("speakers corrected")[0], False)
check("a correction names the line it applies to",
      "speakers corrected: 01:01 → SPEAKER_01" in printed, True)
check("minutes and seconds, not raw seconds", (clock(0), clock(61), clock(3599)),
      ("00:00", "01:01", "59:59"))

# =====================================================================
#  what a live session leaves on disk
# =====================================================================
# the live mode kept nothing at first, which was surprising rather than safe: people talk into
# it and then look for the text. it writes a txt named by the date. two things have to hold — a
# speaker corrected later must reach the file, and an earlier session in the same file must
# survive being rewritten.

from datetime import datetime  # noqa: E402

when = datetime(2026, 9, 27, 14, 32)
before = ["--- live, 09:00, ru ---", "[SPEAKER_00 00:01] раньше"]
paper = LiveTranscript(Path("nowhere") / "x.txt", language="ru", when=when,
                       existing=chr(10).join(before))
paper.add(61.4, "SPEAKER_00", "ну что")
paper.add(75.0, None, "как ваши дела")
rendered = paper.render().splitlines()

check("the date names the file, the folder is asked of the system",
      live_path("C:/somewhere", when).name, "2026-09-27.txt")
# the file reads: the earlier session, a blank line, this session's header, then its lines
check("a session marks where it begins", rendered[3], "--- live, 14:32, ru ---")
check("what was in the file before is kept", rendered[0], "--- live, 09:00, ru ---")
check("lines carry the speaker and the time",
      rendered[4], "[SPEAKER_00 01:01] ну что")
check("a line with no speaker yet says so rather than inventing one",
      rendered[5], "[? 01:15] как ваши дела")

paper.relabel([{"start": 75.0, "speaker": "SPEAKER_01"}])
check("a speaker corrected later reaches the file, matched by the line's own time",
      paper.render().splitlines()[5], "[SPEAKER_01 01:15] как ваши дела")
paper.relabel([{"start": 999.0, "speaker": "SPEAKER_09"}])
check("a correction for a line that is not there changes nothing",
      [line["speaker"] for line in paper.lines], ["SPEAKER_00", "SPEAKER_01"])

print()
print("ALL PASSED" if ok else "FAILURES PRESENT")
raise SystemExit(0 if ok else 1)
