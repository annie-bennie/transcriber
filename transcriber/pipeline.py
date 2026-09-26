# -*- coding: utf-8 -*-
"""Orchestration: which stage runs, in what order, and what gets written when.

models are not loaded in the constructor. a stage may turn out to be already computed, and
raising a model for it costs a dozen seconds and several gigabytes for nothing — on a
resume or a --rediarize that was pure waste. each model comes up on first use.
"""
import gc
import json
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

from .analysis import analyze_units
from .asr import DEFAULT_BACKEND, make_backend
from .audio import decode_audio, voice_quality_available
from .compat import allow_pyannote_checkpoints, from_pretrained, patch_torchaudio
from .config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_WHISPER_MODEL,
    DIARIZE_MODEL_31,
    DIARIZE_MODEL_COMMUNITY,
    DIARIZE_MODEL_DIARIZEN,
    EMOTION_UNIT_MIN,
    EMOTION_UNIT_TARGET,
    SAMPLE_RATE,
    SPEAKER_REFINE_MIN_MARGIN,
)
from .emotion import DEFAULT_EMOTION_DIM_MODEL, DimensionalEmotion
from .logging import Progress, log, mem_snapshot, software_versions
from .render import transcript_path, write_transcript_txt
from .speakers import (
    build_emotion_units,
    build_segments,
    build_turns,
    describe_suspects,
    diarization_turns,
    refine_speaker_turns,
)
from .state import (
    SCHEMA_VERSION,
    load_state,
    new_state,
    rebuild_derived,
    refresh_statistics,
    save_state,
    state_path,
    state_progress,
)


def _round(value, digits):
    """round() that survives None: some whisper metrics arrive unset."""
    return round(value, digits) if isinstance(value, (int, float)) else None


class _PyannoteBar:
    """Adapter between pyannote's hook and our bar.

    pyannote calls this with a step name and how far through that step it is. the steps run
    one after another — segmentation, embeddings, clustering — so the bar is relabelled and
    restarted at each boundary rather than trying to show one total across all of them,
    which pyannote does not know either.
    """

    def __init__(self, bar):
        self.bar = bar
        self.step = None

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if step_name != self.step:
            self.step = step_name
            self.bar.retitle(f"diarization: {step_name}")
        if completed is not None and total:
            self.bar.set(completed, total=total)
        else:
            self.bar.tick()


class Transcriber:
    def __init__(self, model_name=DEFAULT_WHISPER_MODEL, device=None, diarize=True,
                 hf_token=None, num_speakers=None, min_speakers=None, max_speakers=None,
                 emotion=True, emotion_batch=4, diarize_batch=None, diarize_model="auto",
                 diarizen_model=None, diarizen_python=None, speaker_refine=True,
                 embedding_model=DEFAULT_EMBEDDING_MODEL, word_timestamps=True,
                 emotion_unit_target=EMOTION_UNIT_TARGET, voice_quality=False,
                 emotion_dimensions=True, emotion_dim_model=DEFAULT_EMOTION_DIM_MODEL,
                 asr_backend=DEFAULT_BACKEND, debug=False):
        import torch

        self.debug = debug
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.diarize = diarize
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.emotion = emotion
        self.emotion_batch = emotion_batch
        self.diarize_batch = diarize_batch
        self.diarize_model = diarize_model
        self.diarizen_model = diarizen_model
        self.diarizen_python = diarizen_python
        self.diarize_model_used = None
        self.speaker_refine = speaker_refine
        self.embedding_model = embedding_model
        self.embedder = None
        self.plda = None
        self.plda_available = None
        self.hf_token = None
        self.hf_token_arg = hf_token
        self.word_timestamps = word_timestamps
        self.emotion_unit_target = emotion_unit_target
        self.voice_quality = voice_quality
        self.emotion_dimensions = emotion_dimensions
        self.emotion_dim_model = emotion_dim_model

        self.asr_backend = asr_backend
        self.asr = make_backend(asr_backend, model_name, self.device)
        self.whisper_model = None
        self.diarization_pipeline = None
        self.diarization_tried = False
        self.dimensions = None
        self.dimensions_tried = False

        log(f"device: {self.device}")

    # ------------------------------------------------------------ lazy model loading
    def _get_whisper(self):
        """The recogniser, loaded on first use. kept under the old name: every stage that
        parks models between passes refers to it."""
        if self.whisper_model is None:
            self.asr.load()
            self.whisper_model = self.asr
        return self.whisper_model

    def _get_diarization(self):
        if not self.diarization_tried:
            self.diarization_tried = True
            self._load_diarization(self.hf_token_arg)
            if self.diarization_pipeline is not None:
                self._park(self.diarization_pipeline)
        return self.diarization_pipeline

    def _get_dimensions(self):
        if not self.emotion_dimensions or self.dimensions_tried:
            return self.dimensions
        self.dimensions_tried = True
        log(f"loading the emotion scales model ({self.emotion_dim_model})...")
        try:
            self.dimensions = DimensionalEmotion(self.emotion_dim_model, self.device)
            log(f"scales loaded: arousal, dominance, valence. {mem_snapshot()}")
        except Exception as e:
            log(f"  ⚠ could not load the scales model: {type(e).__name__}: {e}")
            log("    continuing without them")
            self.emotion_dimensions = False
        return self.dimensions

    def _free_vram(self):
        import torch
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    def _park(self, *models):
        """Move models off the gpu while they are not needed.

        whisper, pyannote and the scales model run strictly in turn, but all of them used to
        sit on the card at once. on long recordings the last stage ran out of room and the
        allocator spilled into system memory through wddm: rss went past 40 GB and the stage
        slowed down sevenfold. moving a model there and back costs a second or two per file.
        """
        import torch
        if self.device != "cuda":
            return
        for model in models:
            try:
                model.to(torch.device("cpu"))
            except Exception:
                pass
        self._free_vram()

    # ------------------------------------------------------------ diarization
    def _diarizen_python(self):
        """Interpreter of the environment that has diarizen — this one, or another.

        the order and the reasons live in transcriber/diarizen.py, so that the benchmark tools
        resolve it exactly the same way.
        """
        from .diarizen import interpreter

        return interpreter(self.diarizen_python)

    def _diarize_diarizen(self, audio):
        """Run diarizen as a subprocess. returns a list of turns or None."""
        python = self._diarizen_python()
        if python is None:
            return None
        helper = Path(__file__).resolve().parent.parent / "tools" / "diarize_diarizen.py"
        if not helper.exists():
            log(f"  ⚠ {helper.name} not found — skipping diarizen")
            return None

        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            turns_path = Path(tmp) / "turns.json"
            sf.write(str(wav), audio, SAMPLE_RATE)
            command = [str(python), str(helper), "--input", str(wav),
                       "--output", str(turns_path)]
            if self.diarizen_model:
                command += ["--model", self.diarizen_model]
            # subprocess.run would block with a frozen bar on screen for minutes, so the
            # process is polled instead. its output goes to files rather than to pipes:
            # polling a pipe without draining it deadlocks as soon as the child writes
            # more than the os buffer holds, and diarizen is talkative on stderr.
            bar = Progress(f"diarization (diarizen, {python.parent.parent.name})", unit="")
            out_path = Path(tmp) / "stdout.log"
            err_path = Path(tmp) / "stderr.log"
            with open(out_path, "wb") as out_file, open(err_path, "wb") as err_file:
                process = subprocess.Popen(command, stdout=out_file, stderr=err_file)
                try:
                    while process.poll() is None:
                        bar.tick()
                        time.sleep(0.1)
                finally:
                    bar.close("")
            result = subprocess.CompletedProcess(
                command, process.returncode,
                out_path.read_bytes(), err_path.read_bytes())
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", "replace")
                log(f"  ⚠ diarizen failed (code {result.returncode}):")
                if stderr.strip():
                    log("    " + stderr.strip().splitlines()[-1])
                return None
            payload = json.loads(turns_path.read_text(encoding="utf-8"))
        turns = payload["turns"]
        self.diarize_model_used = f"diarizen:{payload['model']}"
        log(f"  diarization done: {len(turns)} turns. {mem_snapshot()}")
        return turns

    def _load_diarization(self, hf_token):
        import torch

        log("loading the pyannote diarization pipeline...")
        patch_torchaudio()
        allow_pyannote_checkpoints()

        import pyannote.audio.pipelines.speaker_diarization as _sd_mod
        from pyannote.audio import Pipeline

        # pyannote 4.x pulls plda from the gated community-1 even for pipeline 3.1, which
        # does not need it, and without access that kills the whole load. the error is
        # swallowed, but not silently: there used to be a bare except here, and the pipeline
        # could run for a long time without plda and never say so. it turned out it did.
        # get_plda only exists in 4.x — on 3.x there is nothing to patch.
        _orig_get_plda = getattr(_sd_mod, "get_plda", None)
        self.plda_available = None

        def _safe_get_plda(plda, token=None, cache_dir=None):
            try:
                result = _orig_get_plda(plda, token=token, cache_dir=cache_dir)
                self.plda_available = result is not None
                return result
            except Exception as e:
                self.plda_available = False
                log(f"  ⚠ plda unavailable ({type(e).__name__}) — short turns will be "
                    f"harder to tell apart")
                return None

        if _orig_get_plda is not None:
            _sd_mod.get_plda = _safe_get_plda

        token = hf_token or os.environ.get("HF_TOKEN") or True
        self.hf_token = token
        # 3.1 by default rather than community-1. measured: on an hour of audio they split
        # it identically, but on a two-minute excerpt community-1 collapsed two speakers
        # into one. plda is taken from community-1 separately and works with both — the
        # embedding model is the same, values match to three decimals.
        wanted = self.diarize_model
        candidates = ([DIARIZE_MODEL_31, DIARIZE_MODEL_COMMUNITY] if wanted == "auto"
                      else [wanted])
        last_error = None
        for name in candidates:
            try:
                self.diarization_pipeline = from_pretrained(Pipeline, name, token)
                self.diarize_model_used = name
                break
            except Exception as e:
                last_error = e
                log(f"  ⚠ {name} did not load ({type(e).__name__})"
                    + (f" — trying {candidates[-1]}" if name != candidates[-1] else ""))

        if self.diarization_pipeline is None:
            log(f"⚠ could not load a diarization pipeline: "
                f"{type(last_error).__name__}: {last_error}")
            log("  check the licence terms for pyannote/speaker-diarization-3.1 and")
            log("  pyannote/segmentation-3.0 on huggingface.co, and that a token is set.")
            # pyannote is not always required: with diarizen installed there is another way
            # to label speakers. this used to switch off the whole stage.
            if (self.diarize_model in ("auto", DIARIZE_MODEL_DIARIZEN)
                and self._diarizen_python()):
                log("  speakers will be labelled by diarizen.")
                return
            log("  continuing without diarization.")
            self.diarize = False
            return

        self.diarization_pipeline.to(torch.device(self.device))
        if self.diarize_batch:
            self.diarization_pipeline.segmentation_batch_size = self.diarize_batch
            self.diarization_pipeline.embedding_batch_size = self.diarize_batch
        plda = {True: "yes", False: "NO", None: "not requested"}[self.plda_available]
        log(f"diarization: {self.diarize_model_used} (plda: {plda}; segmentation batch "
            f"{getattr(self.diarization_pipeline, 'segmentation_batch_size', '?')}, "
            f"embedding batch "
            f"{getattr(self.diarization_pipeline, 'embedding_batch_size', '?')}). "
            f"{mem_snapshot()}")

    def _load_embedder(self):
        """Voice embedding model — loaded only when refinement needs it.

        if the diarization pipeline already carries one (community-1), take that: it is the
        same wespeaker and a second copy on the card is pointless. plda comes from there too.
        """
        if self.embedder is not None or not self.speaker_refine:
            return self.embedder
        # refinement can run after diarizen, which never touches the pyannote loader, so the
        # shims have to be applied here too.
        patch_torchaudio()
        allow_pyannote_checkpoints()
        self.plda = getattr(self.diarization_pipeline, "_plda", None)
        if self.plda is None:
            # plda lives in community-1 but applies to 3.1 embeddings as well: the voice
            # model is shared. no reason to load all of community-1 for one file.
            try:
                from pyannote.audio.core.plda import PLDA
                self.plda = PLDA.from_pretrained(
                    DIARIZE_MODEL_COMMUNITY, subfolder="plda", token=self.hf_token)
            except ImportError:
                # pyannote 3.x has no plda at all, and that is the version a single environment
                # with diarizen carries. not a missing licence, so not worth a warning sign.
                log("  plda needs pyannote.audio 4; this environment has an older one — "
                    "speaker refinement runs without it")
            except Exception as e:
                log(f"  ⚠ plda unavailable ({type(e).__name__}) — speaker refinement will "
                    f"be less confident. terms: https://hf.co/{DIARIZE_MODEL_COMMUNITY}")
        own = getattr(self.diarization_pipeline, "_embedding", None)
        if own is not None:
            self.embedder = own
            log(f"  voice embeddings taken from the diarization pipeline "
                f"(plda: {'yes' if self.plda is not None else 'no'})")
            return self.embedder
        try:
            import torch
            from pyannote.audio.pipelines.speaker_verification import (
                PretrainedSpeakerEmbedding,
            )
            self.embedder = PretrainedSpeakerEmbedding(
                self.embedding_model, device=torch.device(self.device))
            log(f"  voice embedding model loaded ({self.embedding_model}; "
                f"plda: {'yes' if self.plda is not None else 'no'})")
        except Exception as e:
            log(f"  ⚠ could not load the embedding model: {type(e).__name__}: {e}")
            log("    speaker refinement will be skipped")
            self.speaker_refine = False
        return self.embedder

    def _refine_speakers(self, turns, audio):
        """Refine speaker labels against voice references."""
        import numpy as np
        import torch

        embedder = self._load_embedder()
        if embedder is None:
            return turns

        total = audio.shape[-1]

        def embed(start, end):
            chunk = audio[max(0, int(start * SAMPLE_RATE)):min(total, int(end * SAMPLE_RATE))]
            if chunk.size < int(0.3 * SAMPLE_RATE):
                return None
            try:
                with torch.no_grad():
                    vector = embedder(
                        torch.from_numpy(chunk.copy()).unsqueeze(0).unsqueeze(0))
            except Exception:
                return None
            vector = np.asarray(vector).reshape(1, -1)
            if np.isnan(vector).any():
                return None
            if self.plda is not None:
                # plda brings embeddings into a space where short chunks become comparable:
                # on the reference exchange it was the only thing that actually raised
                # accuracy.
                try:
                    vector = np.asarray(self.plda(vector)).reshape(1, -1)
                except Exception:
                    pass
            vector = vector.reshape(-1)
            return vector / (np.linalg.norm(vector) + 1e-9)

        log(f"  refining speakers against voice references ({len(turns)} turns, "
            f"margin threshold {SPEAKER_REFINE_MIN_MARGIN})...")
        refined, report = refine_speaker_turns(turns, embed)
        if report["skipped"]:
            log(f"    skipped: {report['skipped']}")
        else:
            refs = ", ".join(f"{spk}: {n} turns / {sec:.0f} s"
                             for spk, (n, sec) in sorted(report["references"].items()))
            log(f"    references — {refs}; distance between them {report['separation']}")
            log(f"    rounds {report['rounds']}; checked {report['checked']}, "
                f"reassigned {report['changed']}, no margin {report['abstained']}; "
                f"marked «speaker in doubt» {report.get('uncertain', 0)}")
        self._park(getattr(embedder, "model_", None) or embedder)
        return refined

    def _diarize_pyannote(self, audio):
        """Pyannote with a fallback to cpu when vram runs out.

        on long recordings pyannote is the hungriest stage: segmentation and embeddings run
        a sliding window over the whole file. a vram shortage used to kill the file
        outright; now the worst case is that it is slower.
        """
        import torch

        kwargs = {}
        if self.num_speakers:
            kwargs["num_speakers"] = self.num_speakers
        else:
            if self.min_speakers:
                kwargs["min_speakers"] = self.min_speakers
            if self.max_speakers:
                kwargs["max_speakers"] = self.max_speakers

        payload = {"waveform": torch.from_numpy(audio).unsqueeze(0),
                   "sample_rate": SAMPLE_RATE}
        attempts = [self.device] + (["cpu"] if self.device != "cpu" else [])
        try:
            for attempt_device in attempts:
                try:
                    log(f"  diarization (pyannote, {attempt_device})... {mem_snapshot()}")
                    self.diarization_pipeline.to(torch.device(attempt_device))
                    bar = Progress("diarization", unit="")
                    try:
                        output = self.diarization_pipeline(
                            payload, hook=_PyannoteBar(bar), **kwargs)
                    finally:
                        bar.close("")
                    # pyannote 4.x returns DiarizeOutput instead of a bare Annotation
                    diarization = getattr(output, "speaker_diarization", output)
                    log(f"  diarization done. {mem_snapshot()}")
                    return diarization
                except torch.cuda.OutOfMemoryError as e:
                    log(f"  ⚠ not enough vram for diarization: {e}")
                    self._free_vram()
                    if attempt_device != attempts[-1]:
                        log("  retrying on cpu — slower, but the memory is there.")
                except Exception as e:
                    log(f"  ⚠ diarization failed: {type(e).__name__}: {e}")
                    log(traceback.format_exc())
                    break
            return None
        finally:
            self._park(self.diarization_pipeline)

    # ------------------------------------------------------------ parameters and state
    def _params(self, language):
        """Settings the content of the json depends on."""
        return {
            "model": self.model_name,
            "asr": self.asr_backend,
            "language": language,
            "diarize": self.diarize,
            # what was asked for — this decides whether recomputation is needed. the
            # resolved name is filled in later, once the engine actually loads: with lazy
            # loading it is unknown at this point, and comparing by it made diarization
            # rerun on every single launch.
            "diarize_request": self.diarize_model,
            "diarize_model": self.diarize_model_used,
            "speaker_refine": self.speaker_refine,
            "num_speakers": self.num_speakers,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "emotion": self.emotion,
            "emotion_unit_target": self.emotion_unit_target if self.emotion else None,
            "voice_quality": self.voice_quality if self.emotion else None,
            "emotion_dimensions": (self.emotion_dim_model if
                                   (self.emotion and self.emotion_dimensions) else None),
            "word_timestamps": self.word_timestamps,
        }

    def _reusable_state(self, state, media_path, params):
        """Can the saved state be continued, or must it be recomputed?

        only what genuinely invalidates a result is checked: a different file format, a
        replaced source, a different whisper model or language. changing a measurement model
        resets only that measurement — the whisper segments do not depend on it.
        """
        if state.get("schema_version") != SCHEMA_VERSION:
            # only the derived layer changed between versions; the raw one — whisper output
            # with words, and diarization turns — did not. so instead of recomputing from
            # scratch it is enough to rebuild what is derived.
            if state.get("asr_segments"):
                log(f"  json format updated to v{SCHEMA_VERSION} — rebuilding the derived "
                    f"layer, recognition stays")
                state["schema_version"] = SCHEMA_VERSION
                state["stages"]["analyze"] = None
                for unit in state.get("emotion_units") or []:
                    unit.pop("analyzed", None)
                rebuild_derived(state)
            else:
                log(f"  ⚠ {media_path.stem}.json is a different format version — "
                    f"recomputing")
                return None
        if state.get("source", {}).get("size_bytes") != media_path.stat().st_size:
            log("  ⚠ the source file changed since the last run — recomputing")
            return None
        old = state.get("params") or {}
        # the engine counts as much as the model does. an older json has no "asr" field at
        # all and is treated as whisper.
        was = (old.get("model"), old.get("language"), old.get("asr") or "whisper")
        now = (params["model"], params["language"], params["asr"])
        if was != now:
            log(f"  ⚠ different engine, model or language (was {was[2]}/{was[0]}/{was[1]}) "
                f"— the transcript is recomputed")
            return None
        if old.get("word_timestamps") != params["word_timestamps"]:
            log("  ⚠ word timestamp mode changed — the transcript is recomputed")
            return None

        diarize_keys = ("num_speakers", "min_speakers", "max_speakers", "diarize_request",
                        "speaker_refine")
        if tuple(old.get(k) for k in diarize_keys) != tuple(params[k] for k in diarize_keys):
            log("  ⚠ different diarization settings — speakers are labelled again")
            # this is exactly what the raw whisper output is stored for: re-cutting a
            # recording under a different diarization costs seconds, recognition stays.
            state["stages"]["diarize"] = None
            state["stages"]["analyze"] = None
        for key, message in (("emotion_dimensions", "emotion scales"),
                             ("voice_quality", "voice quality"),
                             ("emotion_unit_target", "measurement window")):
            if params["emotion"] and old.get(key) != params[key]:
                log(f"  ⚠ {message} setting changed — measurement is redone")
                # marks must be cleared from utterances: that is where measurements live.
                for unit in state.get("emotion_units") or []:
                    unit.pop("analyzed", None)
                state["stages"]["analyze"] = None
        return state

    # ------------------------------------------------------------ the run itself
    def process_file(self, media_path: Path, output_dir: Path, language=DEFAULT_LANGUAGE,
                     keep_audio=False, resume=True, checkpoint_every=200,
                     drop_suspect=False, rediarize=False):
        """Compute what is missing, appending to the json after every stage."""
        import torch

        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = media_path.stem
        json_path = state_path(output_dir, base_name)
        params = self._params(language)

        state = load_state(json_path) if resume else None
        if state is not None:
            state = self._reusable_state(state, media_path, params)
        if state is None:
            state = new_state(media_path, params)
        else:
            log(f"  continuing from the last run — {state_progress(state)}")
            state["params"] = params
        if rediarize:
            # segments are rebuilt from the raw whisper output, so relabelling speakers
            # costs one diarization rather than a full recomputation of the file.
            state["stages"]["diarize"] = None
            state["stages"]["analyze"] = None

        stages = state["stages"]
        # what to do is decided by intent, not by whether a model is loaded: otherwise the
        # check itself would load it and laziness would mean nothing.
        todo_transcribe = stages.get("transcribe") != "done"
        todo_diarize = self.diarize and stages.get("diarize") != "done"
        todo_analyze = self.emotion and (
            todo_transcribe or any(not unit.get("analyzed")
                                   for unit in state["emotion_units"]))

        if not (todo_transcribe or todo_diarize or todo_analyze or keep_audio):
            log("  everything was computed earlier — only redrawing the txt from the json.")
            write_transcript_txt(state, transcript_path(state, output_dir, base_name),
                             drop_suspect)
            return state

        log(f"  decoding audio... {mem_snapshot()}")
        audio = decode_audio(media_path)
        state["source"]["duration_sec"] = round(audio.shape[-1] / SAMPLE_RATE, 2)
        log(f"    audio in memory: {audio.shape[-1] / SAMPLE_RATE / 60:.0f} min "
            f"({audio.nbytes / 2**20:.0f} MB). {mem_snapshot()}")

        if keep_audio:
            import soundfile as sf
            sf.write(str(output_dir / f"{base_name}.wav"), audio, SAMPLE_RATE)

        if todo_transcribe:
            log(f"  transcribing ({self.asr.name})...")
            self._get_whisper()
            if self.device == "cuda":
                self.asr.to(self.device)
            # the backend returns segments already shaped for the json, confidence numbers
            # and all: the hallucination filter reads no_speech_prob, avg_logprob and
            # compression_ratio, and a backend that dropped one would break the filter
            # rather than the transcript — quietly.
            result = self.asr.transcribe(
                audio, language=language, word_timestamps=self.word_timestamps,
                verbose=bool(self.debug))
            state["asr_segments"] = result.segments
            state["text"] = result.text
            state["segments"] = build_segments(state["asr_segments"])
            state["turns"] = build_turns(state["segments"])
            state["emotion_units"] = build_emotion_units(
                state["segments"], state["turns"], self.emotion_unit_target)
            # the transcript was rewritten — old measurements belong to other segments.
            stages.update({"transcribe": "done", "diarize": None, "analyze": None})
            state.setdefault("software", {})["transcribe"] = software_versions()
            state["speakers"] = []
            save_state(json_path, state)
            words = result.words
            log(f"  {self.asr.name} done: {len(state['asr_segments'])} segments, "
                f"{words} words "
                f"with timestamps, checkpoint in {json_path.name}. {mem_snapshot()}")
            if self.word_timestamps and not words:
                log("  ⚠ the model returned no word timestamps — speakers will be assigned "
                    "per whole segment, as before")
            log(f"  looks invented: {describe_suspects(state['segments'])}")
        else:
            log(f"  whisper skipped: {json_path.name} already has "
                f"{len(state['segments'])} segments")

        # free the card for pyannote: whisper is not needed again until the next file.
        if self.whisper_model is not None:
            self._park(self.whisper_model)

        if todo_diarize:
            turns = None
            if self.diarize_model in ("auto", DIARIZE_MODEL_DIARIZEN):
                turns = self._diarize_diarizen(audio)
                if turns is None:
                    if self.diarize_model == DIARIZE_MODEL_DIARIZEN:
                        log("  ⚠ diarizen unavailable — falling back to pyannote")
                    elif self._diarizen_python() is None:
                        log("  diarizen is not installed, labelling with pyannote. with it "
                            "speakers come out noticeably better — see docs/diarization.md")
            if turns is None and self._get_diarization() is not None:
                diarization = self._diarize_pyannote(audio)
                turns = diarization_turns(diarization) if diarization is not None else None
            if turns is not None:
                if self.speaker_refine:
                    turns = self._refine_speakers(turns, audio)
                state["diarization"] = turns
                state["params"]["diarize_model"] = self.diarize_model_used
                log(f"  labelling speakers per word "
                    f"({len(state['asr_segments'])} whisper segments)...")
                was_analyzed = sum(1 for u in state["emotion_units"] if u.get("analyzed"))
                before = len(state["segments"])
                # rebuilt from the raw whisper output rather than patched: otherwise a
                # second diarization would land on already-cut segments.
                state["segments"] = build_segments(state["asr_segments"], turns)
                state["turns"] = build_turns(state["segments"])
                state["emotion_units"] = build_emotion_units(
                    state["segments"], state["turns"], self.emotion_unit_target)
                state["speakers"] = sorted({s["speaker"] for s in state["segments"]
                                            if s.get("speaker")})
                stages["diarize"] = "done"
                state.setdefault("software", {})["diarize"] = software_versions()
                log(f"  speakers: {len(state['speakers'])}, turns: {len(state['turns'])}, "
                    f"segments {before} → {len(state['segments'])} "
                    f"(split at speaker changes: {len(state['segments']) - before})")
                log(f"  looks invented: {describe_suspects(state['segments'])}")
                if was_analyzed:
                    stages["analyze"] = None
                    log(f"  ⚠ measurement ({was_analyzed} utterances) is redone — "
                        f"boundaries changed")
            else:
                stages["diarize"] = "failed"
            save_state(json_path, state)

        if self.emotion:
            units = state["emotion_units"]
            if any(not unit.get("analyzed") for unit in units):
                if self.voice_quality and not voice_quality_available():
                    log("  ⚠ praat-parselmouth is not installed — voice quality "
                        "(jitter/shimmer/HNR) will not be computed")
                    self.voice_quality = False
                dimensions = self._get_dimensions()
                if dimensions is not None and self.device == "cuda":
                    dimensions.to(torch.device(self.device))
                too_short = sum(1 for u in units
                                if u["end"] - u["start"] < EMOTION_UNIT_MIN)
                log(f"  measuring: {len(units)} utterances from "
                    f"{len(state['segments'])} segments, shorter than "
                    f"{EMOTION_UNIT_MIN:g} s (no emotion): {too_short}. {mem_snapshot()}")
                stages["analyze"] = "partial"
                save_state(json_path, state)
                analyze_units(
                    units, state["segments"], audio, SAMPLE_RATE,
                    dimensions=dimensions, block_size=checkpoint_every,
                    on_block=lambda: save_state(json_path, state),
                    voice=self.voice_quality,
                )
                if dimensions is not None:
                    self._park(dimensions.model)
                log(f"  measurement done. {mem_snapshot()}")
            state.setdefault("software", {})["analyze"] = software_versions()
            stages["analyze"] = ("done" if all(u.get("analyzed") for u in units)
                                 else "partial")
            # baselines are built from loudness and pitch, so they come after them. on an
            # unfinished stage they end up computed from a partial sample — that is fine,
            # the value is derived and recomputed on every following pass.
            marked = refresh_statistics(state)
            enough = sum(1 for e in state["speaker_stats"].values() if e.get("enough"))
            log(f"  speaker baselines: {enough} of {len(state.get('speakers') or [])} "
                f"speakers, deviations computed {marked}")
            save_state(json_path, state)

        del audio
        write_transcript_txt(state, transcript_path(state, output_dir, base_name),
                             drop_suspect)
        return state
