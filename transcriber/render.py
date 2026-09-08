# -*- coding: utf-8 -*-
"""Drawing the txt out of the json.

the txt is a view, never a source. everything here can be regenerated with --rebuild-txt,
which is why nothing is stored only in this format.
"""
from pathlib import Path

from .analysis import format_delta_db, format_delta_semitones, format_z
from .emotion import EMOTION_DIMENSION_GUIDE, EMOTION_DIMENSION_SHORT, EMOTION_DIMENSIONS
from .logging import log
from .speakers import describe_suspects, units_by_segment
from .state import plain_text, write_text_atomic


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_measurements(seg, unit=None) -> str:
    """Compact tag for a transcript line.

    emotion scales belong to the utterance a segment sits in, loudness and pitch to the
    segment itself. only notable deviations are printed: "arousal 0.55" on its own says
    nothing, and a mark on every line stops being a mark.
    """
    parts = []
    if seg.get("suspect"):
        parts.append(", ".join(seg["suspect"]))
    if unit is not None:
        for name in EMOTION_DIMENSIONS:
            deviation = format_z(unit.get(name + "_z"))
            if deviation:
                parts.append(f"{EMOTION_DIMENSION_SHORT[name]} {deviation}")
    if seg.get("loudness_db") is not None:
        deviation = format_delta_db(seg.get("loudness_delta_db"))
        parts.append(f"{seg['loudness_db']:.0f}dB" + (f" {deviation}" if deviation else ""))
    if seg.get("pitch_mean") is not None:
        deviation = format_delta_semitones(seg.get("pitch_delta_st"))
        parts.append(f"pitch {seg['pitch_mean']:.0f}Hz"
                     + (f" {deviation}" if deviation else ""))
    # pitch spread has no unit of its own — it is a spread of a spread — so it stays in
    # sigma, and only appears when notable rather than on every line.
    deviation = format_z(seg.get("pitch_var_z"))
    if deviation:
        parts.append(f"pitch spread {deviation}")
    return " | ".join(parts)


def render_emotion_key() -> list:
    """The reading key for the scales, printed at the top of the transcript.

    the same block sits in the json. it is here so that a person opening the txt, or a model
    given only the txt, can interpret the numbers without going looking for documentation.
    """
    guide = EMOTION_DIMENSION_GUIDE
    lines = ["Emotion scales (per utterance, 0 to 1, model trained on natural speech):"]
    for name in EMOTION_DIMENSIONS:
        info = guide["dimensions"][name]
        lines.append(f"  {name} — {info['measures']}: low = {info['low']}, "
                     f"high = {info['high']}")
    lines.append("  the two main axes read together as:")
    for quadrant in guide["quadrants"]:
        lines.append(f"    arousal {quadrant['arousal']:>4}, valence {quadrant['valence']:>4}"
                     f"  ->  {quadrant['reads_as']}")
    lines.extend(render_deviation_key())
    return lines


# each measure is compared to that speaker's own median in this recording, and the distance
# is printed in the unit the measure has. thresholds come from the units, not from a target
# share of marked lines — see config.py.
DEVIATION_KEY = (
    ("loudness", "↑7dB / ↓7dB", "decibels from their usual level; 6 dB is half or double "
                                "the amplitude"),
    ("pitch", "↑6st / ↓6st", "semitones from their usual pitch; 5 st is a musical fourth"),
    ("pitch spread", "+2.4σ", "how much wider or narrower the intonation of this line "
                              "is; skewed, so -2σ is rarer than +2σ"),
    ("emotion scales", "+2.4σ", "distance in units of that speaker's own spread"),
)


def render_deviation_key() -> list:
    """How to read the deviation marks. printed once, above the transcript."""
    lines = ["  deviations are always from that speaker's own median in this recording,",
             "  never from other speakers or other files:"]
    for name, example, meaning in DEVIATION_KEY:
        lines.append(f"    {name:14} {example:12} {meaning}")
    lines.append("  smaller deviations are not printed. absolute values (dB, Hz, the 0..1 "
                 "scales)")
    lines.append("  are not comparable between recordings; the deviations are.")
    return lines


# baseline layout: what is printed, in which units, at what precision. every sigma in the
# transcript is measured against exactly these numbers, so they are all shown — a line
# reading "dom +2.5σ" is unreadable when dominance has no printed baseline at all.
BASELINE_ROWS = (
    (("loudness_db", "loudness", "dB", 1),
     ("pitch_mean", "pitch", "Hz", 0),
     ("pitch_std", "pitch spread", "Hz", 0)),
    (("arousal", "arousal", "", 2),
     ("dominance", "dominance", "", 2),
     ("valence", "valence", "", 2)),
)


def render_speaker_baseline(speaker, entry):
    """One speaker in the header: median ± spread for everything measured on them."""
    if not entry.get("enough"):
        return [f"  - {speaker}"]
    lines = [f"  - {speaker} — {entry['segments']} segments"]
    for row in BASELINE_ROWS:
        pieces = []
        for key, label, unit, digits in row:
            values = entry.get(key) or {}
            centre, spread = values.get("median"), values.get("spread")
            if centre is None:
                continue
            text = f"{label} {centre:.{digits}f}"
            if spread:
                text += f" ±{spread:.{digits}f}"
            pieces.append(text + (f" {unit}" if unit else ""))
        if pieces:
            lines.append("      " + " | ".join(pieces))
    return lines


def render_transcript(state: dict, drop_suspect: bool = False) -> str:
    lines = [f"File: {state['source']['name']}"]
    duration = state["source"].get("duration_sec")
    if duration:
        lines.append(f"Duration: {format_time(duration)}")
    if state.get("speakers"):
        lines.append(f"Speakers found: {len(state['speakers'])}")
        stats = state.get("speaker_stats") or {}
        for spk in state["speakers"]:
            lines.extend(render_speaker_baseline(spk, stats.get(spk) or {}))

    units = state.get("emotion_units") or []
    if any(unit.get("arousal") is not None for unit in units):
        lines.append("")
        lines.extend(render_emotion_key())

    unsure = sum(1 for seg in state["segments"] if seg.get("speaker_uncertain"))
    if unsure:
        lines.append(f"Speaker in doubt: {unsure} lines, marked «?» — diarization found no "
                     f"confident answer and the label was left as it was")
    flagged = sum(1 for seg in state["segments"] if seg.get("suspect"))
    if flagged and drop_suspect:
        lines.append(f"Looks invented, removed from the text: {flagged} segments "
                     f"(still in the json under `suspect`)")
    elif flagged:
        lines.append(f"Looks invented: {flagged} of {len(state['segments'])} segments, "
                     f"marked ⚠ with the reason in brackets. remove with --drop-suspect")

    stages = state.get("stages", {})
    if stages.get("transcribe") != "done" or any(v == "partial" for v in stages.values()):
        lines.append(f"⚠ Not finished — {state_progress_line(state)}")

    lines.append("=" * 60)
    # a blank line marks a turn boundary: the dialogue reads as a dialogue while the line
    # format stays the same and still greps.
    turn_starts = {t["segments"][0] for t in (state.get("turns") or []) if t.get("segments")}
    unit_of = units_by_segment(units)
    printed = False
    for seg in state["segments"]:
        if drop_suspect and seg.get("suspect"):
            continue
        if printed and seg["id"] in turn_starts:
            lines.append("")
        speaker = f"{seg['speaker']} " if seg.get("speaker") else ""
        prefix = f"[{speaker}{format_time(seg['start'])}-{format_time(seg['end'])}]"
        tag = format_measurements(seg, unit_of.get(seg["id"]))
        if tag:
            prefix += f" ({tag})"
        marker = ("⚠" if seg.get("suspect") else "") + ("?" if seg.get("speaker_uncertain") else "")
        lines.append(f"{marker + ' ' if marker else ''}{prefix}: {seg['text'].strip()}")
        printed = True

    lines.extend(render_dialogue_report(state))
    return "\n".join(lines) + "\n"


def state_progress_line(state) -> str:
    from .state import state_progress
    return state_progress(state)


def render_dialogue_report(state: dict) -> list:
    """Summary at the end of the transcript. empty when there is nothing to count."""
    data = state.get("dialogue") or {}
    speakers = data.get("speakers") or {}
    if not speakers:
        return []

    lines = ["", "=" * 60, "Dialogue"]
    duration = data.get("duration_sec") or 0
    speech = data.get("speech_sec") or 0
    silence = data.get("silence_share")
    lines.append(f"  duration {format_time(duration)}, speech {format_time(speech)}"
                 + (f", silence {silence * 100:.0f}%" if silence is not None else ""))

    transitions = data.get("transitions")
    if transitions:
        piece = f"  floor changes {transitions}"
        if data.get("overlapping_transitions"):
            piece += f", of them over someone else's speech {data['overlapping_transitions']}"
        lines.append(piece)
    if data.get("simultaneous_sec"):
        share = data.get("simultaneous_share")
        lines.append(f"  spoke at the same time {format_time(data['simultaneous_sec'])}"
                     + (f" ({share * 100:.0f}% of speech)" if share is not None else ""))
    latency = data.get("response_latency_median")
    if latency is not None:
        piece = f"  response latency: median {latency:+.2f} s"
        if data.get("fast_responses"):
            piece += f", latches without a pause {data['fast_responses']}"
        lines.append(piece)

    stats_all = state.get("speaker_stats") or {}
    for speaker, entry in sorted(speakers.items()):
        share = entry.get("talk_share")
        head = f"  {speaker}: spoke {format_time(entry.get('talk_sec') or 0)}"
        if share is not None:
            head += f" ({share * 100:.0f}% of speech)"
        head += f", turns {entry.get('turns')}"
        if entry.get("median_turn_sec") is not None:
            head += f", median {entry['median_turn_sec']:.1f} s"
        lines.append(head)

        details = []
        if entry.get("syllables_per_sec") is not None:
            details.append(f"rate {entry['syllables_per_sec']:.1f} syl/s")
        if entry.get("median_pause_within_sec") is not None:
            details.append(f"pause within a turn {entry['median_pause_within_sec']:.2f} s")
        if entry.get("interruptions_made"):
            details.append(f"interrupted {entry['interruptions_made']}")
        if entry.get("backchannels_made"):
            details.append(f"backchannelled {entry['backchannels_made']}")
        if details:
            lines.append(f"      {', '.join(details)}")

        stats = stats_all.get(speaker) or {}
        if stats.get("arousal") and stats.get("valence"):
            lines.append(f"      scales: arousal {stats['arousal']['median']:.2f}, "
                         f"valence {stats['valence']['median']:.2f}, "
                         f"dominance {stats['dominance']['median']:.2f} "
                         f"(over {stats.get('dimension_units', 0)} utterances)")
        if stats.get("shimmer") and stats.get("hnr"):
            lines.append(f"      voice: shimmer {stats['shimmer']['median']:.1f}%, "
                         f"HNR {stats['hnr']['median']:.1f} dB "
                         f"(over {stats.get('voice_units', 0)} utterances)")

    uncertain = data.get("uncertain_share")
    if uncertain:
        lines.append(f"  ⚠ speaker in doubt for {uncertain * 100:.0f}% of speech time — "
                     f"everything above is imprecise by the same amount")
    return lines


# a transcript with speaker labels and one without are different documents, and telling
# them apart from the filename saves opening both. the json keeps the plain name either
# way — it is the artefact, and its name should not depend on which stages ran.
SPEAKERS_SUFFIX = "_speakers"


def transcript_path(state: dict, output_dir: Path, base_name: str) -> Path:
    """Where the txt goes: `<name>_speakers.txt` when speakers were found."""
    suffix = SPEAKERS_SUFFIX if (state or {}).get("speakers") else ""
    return output_dir / f"{base_name}{suffix}.txt"


def write_transcript_txt(state: dict, path: Path, drop_suspect: bool = False):
    write_text_atomic(path, render_transcript(state, drop_suspect))
    shown = sum(1 for s in state["segments"] if not (drop_suspect and s.get("suspect")))
    log(f"  ✓ saved: {path.name} ({shown} segments)")
    _warn_stale_sibling(path)


def _warn_stale_sibling(path: Path):
    """Say so when the other name is still lying about from an earlier run.

    nothing is deleted: the file is derived and cheap to redraw, but it is also the user's,
    and quietly removing a transcript is not a thing to do on a rename.
    """
    name = path.stem
    other = (path.with_name(name[:-len(SPEAKERS_SUFFIX)] + ".txt")
             if name.endswith(SPEAKERS_SUFFIX)
             else path.with_name(name + SPEAKERS_SUFFIX + ".txt"))
    if other.exists():
        log(f"    note: {other.name} is left over from an earlier run and is now stale")


def write_combined(output_dir: Path, media_files, states, no_combined=False):
    """One continuous text for a whole folder — also entirely from the json.

    states of files this run did not touch have already been read from disk, so the combined
    file does not lose what earlier runs produced.
    """
    if len(media_files) <= 1 or no_combined:
        return
    parts = []
    for f in media_files:
        state = states.get(f)
        text = plain_text(state) if state else ""
        if text:
            parts.append(f"=== {f.name} ===\n{text}")
    if not parts:
        return
    combined_path = output_dir / "full_transcript.txt"
    write_text_atomic(combined_path, "\n\n".join(parts))
    log(f"✓ combined file ({len(parts)} of {len(media_files)}): {combined_path}")
