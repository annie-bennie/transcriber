# -*- coding: utf-8 -*-
"""The json file: the only artefact, and the checkpoint at the same time.

it holds two layers. the raw one — whisper's output with words and confidence metrics, plus
diarization turns — never changes after its stage. the derived one — segments cut by speaker
change, turns, utterances, statistics, flags — is rebuilt from the raw one on demand,
without models and without audio.

that split is what makes it cheap to change cutting rules or filter thresholds: recognising
an hour costs two and a half minutes, rebuilding the derived layer costs four seconds.
"""
import json
import os
from datetime import datetime
from pathlib import Path

from .analysis import (analyze_dialogue, apply_dimension_z, apply_speaker_z, apply_voice_z,
                       build_dimension_stats, build_speaker_stats, build_voice_stats)
from .config import EMOTION_UNIT_TARGET
from .emotion import DEVIATION_GUIDE, EMOTION_DIMENSION_GUIDE
from .logging import log
from .speakers import build_emotion_units, build_segments, build_turns

SCHEMA_VERSION = 8


def state_path(output_dir: Path, base_name: str) -> Path:
    return output_dir / f"{base_name}.json"


def new_state(media_path: Path, params: dict) -> dict:
    stat = media_path.stat()
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "name": media_path.name,
            "path": str(media_path),
            "size_bytes": stat.st_size,
            "duration_sec": None,
        },
        "params": params,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": None,
        # done / failed / partial / None (not started)
        "stages": {"transcribe": None, "diarize": None, "analyze": None},
        "speakers": [],
        "text": "",
        # raw whisper output with words and confidence metrics. never touched after its own
        # stage: segments are rebuilt from it for a new diarization without re-recognising.
        "asr_segments": [],
        # working units: whisper segments cut where the speaker changes.
        "segments": [],
        # turns: consecutive segments of one person, by segment id.
        "turns": [],
        # utterances: merged to a usable length for emotion measurement.
        "emotion_units": [],
        # raw diarization turns [start, end, speaker, uncertain].
        "diarization": [],
        # per-speaker baselines: median and spread of every measured value.
        "speaker_stats": {},
        # dialogue statistics: speaking shares, pauses, latency, interruptions, speech rate.
        "dialogue": {},
        # how to read the emotion scales. written into the file on purpose so that whoever
        # picks it up later — a person or a language model — does not have to guess.
        "emotion_guide": EMOTION_DIMENSION_GUIDE,
        "deviation_guide": DEVIATION_GUIDE,
    }


def write_text_atomic(path: Path, text: str):
    """Write via a temporary file and a rename. every artefact goes through this.

    two reasons. a checkpoint is written often and sometimes right before a likely crash —
    if the process dies mid-write, what stays on disk must be the previous whole version,
    not a stump. and on windows a file with the hidden attribute cannot be opened for
    writing at all, while a rename over it succeeds; the attribute gets set by antivirus or
    sync software often enough to matter.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def load_state(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        log(f"  ⚠ {path.name} is unreadable ({type(e).__name__}: {e}) — starting over")
        return None
    state.setdefault("stages", {})
    state.setdefault("asr_segments", [])
    state.setdefault("diarization", [])
    state.setdefault("segments", [])
    state.setdefault("turns", [])
    state.setdefault("emotion_units", [])
    state.setdefault("speakers", [])
    state.setdefault("speaker_stats", {})
    state.setdefault("dialogue", {})
    return state


def save_state(path: Path, state: dict):
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_text_atomic(path, json.dumps(state, ensure_ascii=False, indent=1))


def state_progress(state: dict) -> str:
    """One line about what is already done — printed when resuming."""
    stages = state.get("stages", {})
    segments = state.get("segments", [])
    units = state.get("emotion_units") or []
    parts = [f"whisper: {stages.get('transcribe') or 'no'} ({len(segments)} segments)"]
    parts.append(f"diarization: {stages.get('diarize') or 'no'}"
                 + (f" ({len(state.get('turns') or [])} turns)" if state.get("turns") else ""))
    analyzed = sum(1 for u in units if u.get("analyzed"))
    parts.append(f"measurement: {stages.get('analyze') or 'no'} "
                 f"({analyzed}/{len(units)} utterances)")
    return " | ".join(parts)


def state_is_complete(state, want_diarize: bool, want_emotion: bool) -> bool:
    """Is the file finished for every stage this run asks for?

    the schema version is checked here rather than only when resuming: otherwise a file
    finished under an older format would silently count as done and be skipped, leaving
    yesterday's result on disk.
    """
    if not state or state.get("schema_version") != SCHEMA_VERSION:
        return False
    if state.get("stages", {}).get("transcribe") != "done":
        return False
    stages = state["stages"]
    # failed means diarization was already tried and did not work; no point retrying it on
    # every run.
    if want_diarize and stages.get("diarize") not in ("done", "failed"):
        return False
    if want_emotion and stages.get("analyze") != "done":
        return False
    return True


def state_matches_request(state, args) -> bool:
    """Was the saved result produced with the same settings this run asks for?

    the readiness check and the compatibility check are one question asked in two places,
    and they drifted apart three times: a saved result counted as done, the file was
    skipped, and what stayed on disk was the answer to a different question. so "ready"
    here means "finished AND computed with what is being asked for", while the decision of
    which stages to redo is still made later, in one place.
    """
    old = state.get("params") or {}
    if old.get("model") != args.model or old.get("language") != args.language:
        return False
    if old.get("word_timestamps") != (not args.no_word_timestamps):
        return False
    if (old.get("num_speakers"), old.get("min_speakers"), old.get("max_speakers")) != \
            (args.num_speakers, args.min_speakers, args.max_speakers):
        return False
    if args.no_emotion:
        return True
    if bool(old.get("voice_quality")) != args.voice_quality:
        return False
    if (old.get("emotion_dimensions") or None) != (
            None if args.no_emotion_dimensions else args.emotion_dim_model):
        return False
    return True


# ---------------------------------------------------------------- carrying results over
def carry_acoustics(old_segments, new_segments) -> int:
    """Carry loudness and pitch onto rebuilt segments with the same boundaries.

    rebuilding makes new objects, and without carrying, measurements would be lost on every
    threshold change. the key is the source whisper segment plus the boundaries: if they
    match, the audio chunk is the same one and what was measured on it still holds.
    """
    index = {(s.get("asr_id"), round(s["start"], 2), round(s["end"], 2)): s
             for s in old_segments if s.get("loudness_db") is not None}
    carried = 0
    for seg in new_segments:
        source = index.get((seg.get("asr_id"), round(seg["start"], 2), round(seg["end"], 2)))
        if source is None:
            continue
        for key in ("loudness_db", "pitch_mean", "pitch_std"):
            if key in source:
                seg[key] = source[key]
        carried += 1
    return carried


def carry_unit_measurements(old_units, new_units) -> int:
    """Carry emotion scales and voice quality onto rebuilt utterances."""
    index = {(u.get("speaker"), round(u["start"], 2), round(u["end"], 2)): u
             for u in old_units if u.get("analyzed")}
    carried = 0
    for unit in new_units:
        source = index.get((unit.get("speaker"), round(unit["start"], 2),
                            round(unit["end"], 2)))
        if source is None:
            continue
        for key in ("arousal", "dominance", "valence", "jitter", "shimmer", "hnr",
                    "analyzed"):
            if key in source:
                unit[key] = source[key]
        carried += 1
    return carried


def rebuild_derived(state) -> bool:
    """Rebuild everything derived from the json: segments, turns, utterances, flags, stats.

    needs neither models nor audio — the raw whisper output and the diarization turns are
    already in the file. that is why they are kept there: cutting rules and filter
    thresholds change often, and recognising an hour costs two and a half minutes.
    """
    if not state.get("asr_segments"):
        return False
    # the derived layer has been rebuilt in full, so the state now matches the current schema.
    state["schema_version"] = SCHEMA_VERSION
    state["emotion_guide"] = EMOTION_DIMENSION_GUIDE
    state["deviation_guide"] = DEVIATION_GUIDE
    segments = build_segments(state["asr_segments"], state.get("diarization") or None)
    carry_acoustics(state.get("segments") or [], segments)
    turns = build_turns(segments)
    target = (state.get("params") or {}).get("emotion_unit_target") or EMOTION_UNIT_TARGET
    units = build_emotion_units(segments, turns, target)
    carried = carry_unit_measurements(state.get("emotion_units") or [], units)
    state["segments"] = segments
    state["turns"] = turns
    state["emotion_units"] = units
    state["speakers"] = sorted({s["speaker"] for s in segments if s.get("speaker")})
    refresh_statistics(state)
    stages = state.setdefault("stages", {})
    if stages.get("analyze") == "done" and carried < len(units):
        stages["analyze"] = "partial" if carried else None
    return True


def refresh_statistics(state) -> int:
    """Recompute baselines, deviations and dialogue statistics from what is measured."""
    segments = state.get("segments") or []
    units = state.get("emotion_units") or []
    state["speaker_stats"] = build_speaker_stats(segments)
    build_voice_stats(units, state["speaker_stats"])
    build_dimension_stats(units, state["speaker_stats"])
    marked = apply_speaker_z(segments, state["speaker_stats"])
    apply_voice_z(units, state["speaker_stats"])
    apply_dimension_z(units, state["speaker_stats"])
    state["dialogue"] = analyze_dialogue(state)
    return marked


def plain_text(state: dict, drop_suspect: bool = True) -> str:
    """Continuous text from segments — for the folder-wide file and for a language model.

    built from segments rather than state["text"]: that field holds whisper's output as it
    came, hallucinations included. it is the source record, not the readable text.
    """
    return " ".join(seg["text"] for seg in state["segments"]
                    if seg["text"] and not (drop_suspect and seg.get("suspect")))
