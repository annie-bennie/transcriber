# -*- coding: utf-8 -*-
"""What `--check` answers: is this machine ready, and if not, what exactly is missing.

without it a person learns about a missing piece the way the author did — an hour into a run,
from a traceback ten frames deep, or worse: not at all, because the pipeline fell back to the
cpu and only took six times longer. every line below is something that has actually gone
wrong: torch installed without cuda, ffmpeg absent, models not downloaded yet, no disk left
for the 2.5 GB of weights, diarizen sitting in an environment nobody pointed at.

nothing here loads a model. the heaviest thing it does is import torch, which it has to: the
question "does this machine see the card" cannot be answered from metadata.
"""
import importlib.metadata as metadata
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

OK = "ok"
WARN = "warn"
FAIL = "fail"

MARK = {OK: "✓", WARN: "⚠", FAIL: "✗"}

# models the default setup downloads on first use, by the name they have in the hugging face
# cache. whisper keeps its own cache elsewhere and is checked separately.
CACHED_MODELS = {
    "BUT-FIT/diarizen-wavlm-large-s80-md-v2": "speaker separation (diarizen)",
    "pyannote/wespeaker-voxceleb-resnet34-LM": "voice embeddings for refinement",
    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim": "emotion scales",
}

OPTIONAL = {
    "parselmouth": ("voice quality: jitter, shimmer, HNR", "pip install praat-parselmouth"),
    "yt_dlp": ("a youtube link or playlist as the input", "pip install yt-dlp"),
    "jiwer": ("WER and DER benchmarks in tools/", "pip install datasets jiwer"),
    "sounddevice": ("--live: listening to a microphone", "pip install sounddevice"),
}


def line(level, name, detail, hint=""):
    return {"level": level, "name": name, "detail": detail, "hint": hint}


def version_of(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def installed(module):
    """is the module importable — without importing it. cheap, and safe for torch."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def cached_repos():
    """repo ids already in the hugging face cache, or None if that cannot be read."""
    try:
        from huggingface_hub import scan_cache_dir
        return {repo.repo_id for repo in scan_cache_dir().repos}
    except Exception:
        return None


def whisper_model_cached(model_name):
    """is the whisper checkpoint on disk. returns (cached, path or None)."""
    try:
        import whisper
        url = whisper._MODELS.get(model_name)
    except Exception:
        return None, None
    if not url:
        return None, None
    root = os.getenv("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    path = Path(root) / "whisper" / os.path.basename(url)
    return path.exists(), path


def free_gb(path):
    try:
        return shutil.disk_usage(path).free / 2 ** 30
    except OSError:
        return None


def collect(model_name="turbo", diarizen_python=None):
    """Every check, in the order a run would need them."""
    checks = []

    # ---- the interpreter this is running under ------------------------------------------
    version = ".".join(str(n) for n in sys.version_info[:3])
    if sys.version_info < (3, 10):
        checks.append(line(FAIL, "python", f"{version} — too old",
                           "python 3.10, 3.11 or 3.12"))
    elif sys.version_info >= (3, 13):
        checks.append(line(WARN, "python", f"{version} — diarizen needs numpy 1.26.4, "
                                           f"which has no build for this version",
                           "python 3.10, 3.11 or 3.12 for speaker separation"))
    else:
        checks.append(line(OK, "python", f"{version} — {sys.executable}"))

    # ---- torch and the card ------------------------------------------------------------
    try:
        import torch
    except Exception as e:
        checks.append(line(FAIL, "torch", f"not importable ({type(e).__name__})",
                           "see docs/install.md — torch comes first, at 2.8.0"))
    else:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            card = (f"{torch.__version__} — {torch.cuda.get_device_name(0)}, "
                    f"vram {free / 2 ** 30:.1f}/{total / 2 ** 30:.1f} GB free")
            checks.append(line(OK, "torch", card))
        else:
            checks.append(line(WARN, "torch", f"{torch.__version__} — no card in use, "
                                              f"cuda build {torch.version.cuda}",
                               "on a machine with an nvidia card this means the wrong torch: "
                               "reinstall it from the cu128 index"))

    # ---- ffmpeg: one on the PATH, or the static build that comes with the install -------
    from .audio import ffmpeg_executable
    exe = ffmpeg_executable()
    try:
        out = subprocess.run([exe, "-version"], capture_output=True)
        first = out.stdout.decode("utf-8", "replace").splitlines()[0]
        checks.append(line(OK, "ffmpeg", f"{first.split(' Copyright')[0]} — {exe}"))
    except Exception:
        checks.append(line(FAIL, "ffmpeg", "cannot be run",
                           "pip install imageio-ffmpeg, or put ffmpeg on the PATH"))

    # ---- recognition -------------------------------------------------------------------
    if not installed("whisper"):
        checks.append(line(FAIL, "whisper", "not installed",
                           "pip install -r requirements.txt"))
    else:
        cached, path = whisper_model_cached(model_name)
        if cached:
            checks.append(line(OK, "whisper", f"{version_of('openai-whisper')}, model "
                                              f"{model_name} in the cache"))
        elif cached is False:
            checks.append(line(WARN, "whisper", f"{version_of('openai-whisper')}, model "
                                                f"{model_name} not downloaded yet",
                               f"about 1.5 GB on the first run, into {path.parent}"))
        else:
            checks.append(line(WARN, "whisper", f"{version_of('openai-whisper')}, "
                                                f"unknown model name {model_name!r}"))

    # ---- speaker separation ------------------------------------------------------------
    pyannote = version_of("pyannote.audio")
    if installed("diarizen"):
        checks.append(line(OK, "diarizen",
                           f"in this environment, with pyannote.audio {pyannote}"))
    elif diarizen_python:
        checks.append(line(OK, "diarizen", f"a separate environment: {diarizen_python}"))
    elif pyannote and pyannote.startswith("4"):
        checks.append(line(WARN, "diarizen", "not installed — speakers will be labelled by "
                                             f"pyannote {pyannote}",
                           "diarizen is more accurate on real recordings: "
                           "see docs/install.md"))
    else:
        checks.append(line(FAIL, "speaker separation", "neither diarizen nor pyannote.audio",
                           "pip install -r requirements.txt, or run with --no-diarize"))

    # ---- the emotion scales ------------------------------------------------------------
    if installed("transformers"):
        checks.append(line(OK, "transformers",
                           f"{version_of('transformers')} — the emotion scales"))
    else:
        checks.append(line(FAIL, "transformers", "not installed",
                           "pip install -r requirements.txt, or run with --no-emotion"))

    # ---- weights already on disk -------------------------------------------------------
    repos = cached_repos()
    if repos is None:
        checks.append(line(WARN, "model cache", "cannot be read",
                           "huggingface_hub is part of the install"))
    else:
        missing = [f"{name}" for repo, name in CACHED_MODELS.items() if repo not in repos]
        if not missing:
            checks.append(line(OK, "model cache",
                               "every model of the default run is downloaded"))
        else:
            checks.append(line(WARN, "model cache",
                               "not downloaded yet: " + ", ".join(missing),
                               "the first run fetches about 1 GB more; it needs the network, "
                               "but no hugging face token"))

    # ---- room for all of that ----------------------------------------------------------
    temp_free = free_gb(tempfile.gettempdir())
    if temp_free is None:
        checks.append(line(WARN, "disk", "cannot be read"))
    elif temp_free < 5:
        checks.append(line(WARN, "disk", f"{temp_free:.1f} GB free where temporary files go",
                           "models and a decoded hour of audio need a few gigabytes"))
    else:
        checks.append(line(OK, "disk", f"{temp_free:.1f} GB free for temporary files"))

    # ---- extras, each of which only some runs need -------------------------------------
    for module, (what, how) in OPTIONAL.items():
        if installed(module):
            checks.append(line(OK, f"optional: {module}", what))
        else:
            checks.append(line(WARN, f"optional: {module}", f"absent — {what}", how))

    return checks


def render(checks) -> str:
    """The report, as one string. pure, so the wording is testable without a machine."""
    width = max(len(c["name"]) for c in checks) if checks else 0
    out = []
    for c in checks:
        out.append(f"{MARK[c['level']]} {c['name']:<{width}}  {c['detail']}")
        if c["hint"] and c["level"] != OK:
            out.append(f"  {'':<{width}}  → {c['hint']}")
    failed = [c for c in checks if c["level"] == FAIL]
    warned = [c for c in checks if c["level"] == WARN]
    out.append("")
    if failed:
        out.append(f"{len(failed)} thing(s) must be fixed before a run; "
                   f"{len(warned)} worth knowing about.")
    elif warned:
        out.append(f"ready to run. {len(warned)} thing(s) worth knowing about.")
    else:
        out.append("ready to run.")
    return "\n".join(out)


def ready(checks) -> bool:
    """Nothing essential is missing."""
    return not any(c["level"] == FAIL for c in checks)
