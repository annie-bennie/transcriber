# -*- coding: utf-8 -*-
"""Command line: arguments, file selection, the run loop."""
import argparse
import gc
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from .config import (DEFAULT_LANGUAGE, DEFAULT_WHISPER_MODEL, EMOTION_UNIT_TARGET,
                     MEDIA_EXTS)
from .emotion import DEFAULT_EMOTION_DIM_MODEL
from .logging import close_log_file, log, log_env, open_log_file
from .pipeline import Transcriber
from .render import write_combined, write_transcript_txt
from .speakers import describe_suspects
from .state import (load_state, rebuild_derived, save_state, state_is_complete,
                    state_matches_request, state_path, state_progress)

DESCRIPTION = """\
Transcribe meetings, lectures and interviews: who spoke, what they said, and how it
sounded — speaker separation, emotion scales, loudness, pitch and dialogue statistics.

Everything runs locally; audio never leaves the machine.

Examples:
    transcribe.py "lecture.mp4"
    transcribe.py "meeting.mp4" --language ru --num-speakers 4
    transcribe.py ~/records --no-diarize --no-emotion

Each file produces one <name>.json and a <name>.txt drawn from it. the json holds a raw
layer (whisper output with words, diarization turns) and a derived layer (segments,
turns, utterances, measurements). the derived layer is rebuilt with --rebuild-txt in
seconds, without models or audio; recognising an hour of audio takes two and a half
minutes, which is why the two are kept apart.

The same json is the checkpoint. it is written after whisper, after diarization and as
measurement proceeds, so after a crash or a Ctrl+C the same command continues where it
stopped rather than starting over.
"""


def collect_media_files(input_path: Path):
    if input_path.is_file():
        return [input_path]
    return sorted(
        Path(folder) / filename
        for folder, _subs, filenames in os.walk(input_path)
        for filename in filenames
        if Path(filename).suffix.lower() in MEDIA_EXTS
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="media file, or a folder to walk")

    group = parser.add_argument_group("output")
    group.add_argument("--output-dir", help="where to put results (default: <input>/transcripts)")
    group.add_argument("--keep-audio", action="store_true",
                       help="save the extracted audio next to the results")
    group.add_argument("--no-combined", action="store_true",
                       help="do not build a folder-wide full_transcript.txt")

    group = parser.add_argument_group("speech recognition")
    group.add_argument("--model", default=DEFAULT_WHISPER_MODEL,
                       help=f"whisper model (default: {DEFAULT_WHISPER_MODEL})")
    group.add_argument("--language", default=DEFAULT_LANGUAGE,
                       help=f"language of the recording (default: {DEFAULT_LANGUAGE}). "
                            f"this is the only setting about the audio; everything the "
                            f"tool writes about itself stays in english")
    group.add_argument("--device", default=None, help="cuda / cpu (default: autodetect)")
    group.add_argument("--no-word-timestamps", action="store_true",
                       help="skip word timestamps: about 40%% faster, but speakers are "
                            "assigned per whole segment and boundaries suffer")

    group = parser.add_argument_group("speaker separation")
    group.add_argument("--no-diarize", action="store_true", help="do not separate speakers")
    group.add_argument("--diarize-model", default="auto",
                       help="auto (diarizen if installed, else pyannote 3.1), diarizen, "
                            "or a pyannote model name on huggingface")
    group.add_argument("--diarizen-model", default=None, metavar="MODEL",
                       help="diarizen model (default: diarizen-wavlm-large-s80-md-v2)")
    group.add_argument("--diarizen-python", default=None, metavar="PATH",
                       help="interpreter of the diarizen environment. DIARIZEN_PYTHON or a "
                            ".venv-diarizen folder next to the project work too")
    group.add_argument("--diarize-batch", type=int, default=None,
                       help="pyannote batch size; lower it if vram runs short")
    group.add_argument("--num-speakers", type=int, default=None,
                       help="exact number of participants, when known")
    group.add_argument("--min-speakers", type=int, default=None)
    group.add_argument("--max-speakers", type=int, default=None)
    group.add_argument("--no-speaker-refine", action="store_true",
                       help="do not refine speakers against voice references")
    group.add_argument("--rediarize", action="store_true",
                       help="label speakers again without touching recognition")
    group.add_argument("--hf-token", default=None,
                       help="hugging face token (else HF_TOKEN or the cli login cache)")

    group = parser.add_argument_group("measurement")
    group.add_argument("--no-emotion", action="store_true",
                       help="skip emotion scales, loudness and pitch entirely")
    group.add_argument("--no-emotion-dimensions", action="store_true",
                       help="do not compute arousal, dominance and valence")
    group.add_argument("--emotion-dim-model", default=DEFAULT_EMOTION_DIM_MODEL,
                       metavar="MODEL", help="model for the emotion scales")
    group.add_argument("--voice-quality", action="store_true",
                       help="compute jitter, shimmer and HNR. needs praat-parselmouth; "
                            "off by default because these are sensitive to recording "
                            "quality — see docs/measurements.md")
    group.add_argument("--emotion-unit-target", type=float, default=EMOTION_UNIT_TARGET,
                       metavar="SEC",
                       help=f"measurement window length (default: {EMOTION_UNIT_TARGET:g} s). "
                            f"shorter is more precise in time but noisier")
    group.add_argument("--emotion-batch", type=int, default=4,
                       help="batch size for the measurement models")

    group = parser.add_argument_group("output and rebuilding")
    group.add_argument("--rebuild-txt", action="store_true",
                       help="rebuild everything derived from existing json files — "
                            "segments, turns, flags and txt. needs no models and no audio")
    group.add_argument("--drop-suspect", action="store_true",
                       help="leave segments flagged as invented out of the txt (they stay "
                            "in the json either way)")
    group.add_argument("--overwrite", action="store_true",
                       help="recompute everything, ignoring saved progress")
    group.add_argument("--checkpoint-every", type=int, default=200, metavar="N",
                       help="how often to flush measurement progress, in utterances")
    group.add_argument("--debug", action="store_true",
                       help="verbose log: print every recognised line with its timecode")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"path not found: {input_path}")
        return 1

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        input_path.parent / "transcripts" if input_path.is_file()
        else input_path / "transcripts"
    )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        open_log_file(output_dir / f"transcribe_log_{datetime.now():%Y%m%d_%H%M%S}.txt")
    except Exception as e:
        print(f"(could not prepare the output folder: {e})", flush=True)

    log_env()

    media_files = collect_media_files(input_path)
    if not media_files:
        log(f"no media files found in {input_path}.")
        return 1

    log(f"files found: {len(media_files)}")
    states = {f: load_state(state_path(output_dir, f.stem)) for f in media_files}

    if args.rebuild_txt:
        rebuilt = 0
        for media_path in media_files:
            state = states[media_path]
            if state is None:
                log(f"  · {media_path.name}: no json yet, skipping")
                continue
            log(f"  · {media_path.name}: {state_progress(state)}")
            if rebuild_derived(state):
                save_state(state_path(output_dir, media_path.stem), state)
                log(f"    rebuilt: {len(state['segments'])} segments, "
                    f"{len(state['turns'])} turns; "
                    f"looks invented: {describe_suspects(state['segments'])}")
            else:
                log("    no raw whisper output in the json — only the txt was redrawn")
            write_transcript_txt(state, output_dir / f"{media_path.stem}.txt",
                                 args.drop_suspect)
            rebuilt += 1
        write_combined(output_dir, media_files, states, args.no_combined)
        log(f"txt rebuilt: {rebuilt} of {len(media_files)}")
        return 0 if rebuilt else 1

    ready = [] if (args.overwrite or args.rediarize) else [
        f for f in media_files
        if state_is_complete(states[f], not args.no_diarize, not args.no_emotion)
        and state_matches_request(states[f], args)
    ]
    pending = [f for f in media_files if f not in ready]

    if ready:
        log(f"already done, skipping ({len(ready)}):")
        for f in ready:
            log(f"  · {f.name}")
    if not pending:
        log("every transcript is in place. recompute from scratch with --overwrite")
        # the txt may be missing or stale — it is derived, and redrawing is cheap.
        for f in ready:
            write_transcript_txt(states[f], output_dir / f"{f.stem}.txt", args.drop_suspect)
        write_combined(output_dir, media_files, states, args.no_combined)
        return 0
    log(f"to process ({len(pending)}):")
    for f in pending:
        state = states[f]
        note = f" — {state_progress(state)}" if state is not None else ""
        log(f"  - {f.name} ({f.stat().st_size / 2**20:.0f} MB){note}")

    transcriber = Transcriber(
        model_name=args.model,
        device=args.device,
        diarize=not args.no_diarize,
        hf_token=args.hf_token,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        emotion=not args.no_emotion,
        emotion_batch=args.emotion_batch,
        diarize_batch=args.diarize_batch,
        diarize_model=args.diarize_model,
        diarizen_model=args.diarizen_model,
        diarizen_python=args.diarizen_python,
        speaker_refine=not args.no_speaker_refine,
        word_timestamps=not args.no_word_timestamps,
        emotion_unit_target=args.emotion_unit_target,
        voice_quality=args.voice_quality,
        emotion_dimensions=not args.no_emotion_dimensions,
        emotion_dim_model=args.emotion_dim_model,
        debug=args.debug,
    )

    failed = []
    for i, media_path in enumerate(pending, 1):
        log("")
        log(f"[{i}/{len(pending)}] {media_path.name}")
        try:
            states[media_path] = transcriber.process_file(
                media_path, output_dir, language=args.language, keep_audio=args.keep_audio,
                resume=not args.overwrite, checkpoint_every=args.checkpoint_every,
                drop_suspect=args.drop_suspect, rediarize=args.rediarize,
            )
        except Exception as e:
            failed.append(media_path.name)
            log(f"  ✗ failed on {media_path.name}: {type(e).__name__}: {e}")
            log(traceback.format_exc())
        finally:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    write_combined(output_dir, media_files, states, args.no_combined)

    if failed:
        log(f"⚠ not processed: {len(failed)} — {', '.join(failed)}")
    log(f"done. results in: {output_dir}")
    return 1 if failed else 0


def run():
    """Entry point for the console script and for transcribe.py."""
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    try:
        return main()
    finally:
        close_log_file()
