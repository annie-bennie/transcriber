# -*- coding: utf-8 -*-
"""Timestamped logging with an immediate flush.

nothing here is clever. the flush matters though: on a long recording the process can die
hard — access violation, driver failure — and a buffered log loses exactly the lines that
would say where.
"""
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime

_T0 = time.monotonic()
_LOG_FILE = None


def elapsed() -> float:
    return time.monotonic() - _T0


def log(msg=""):
    line = f"[{elapsed():8.1f}s] {msg}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        try:
            _LOG_FILE.write(line + "\n")
            _LOG_FILE.flush()
        except Exception:
            pass


def open_log_file(path):
    """Start writing a copy of the log to a file. failure here is not fatal."""
    global _LOG_FILE
    try:
        _LOG_FILE = open(path, "w", encoding="utf-8")
    except Exception as e:
        print(f"(could not open log file: {e})", flush=True)


def close_log_file():
    global _LOG_FILE
    if _LOG_FILE is not None:
        try:
            _LOG_FILE.close()
        except Exception:
            pass
        _LOG_FILE = None


def mem_snapshot() -> str:
    """process ram, free system memory and vram."""
    parts = []
    try:
        import psutil
        vm = psutil.virtual_memory()
        parts.append(f"process RAM {psutil.Process().memory_info().rss / 2**30:.2f} GB")
        parts.append(f"free RAM {vm.available / 2**30:.1f}/{vm.total / 2**30:.1f} GB")
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            parts.append(f"free VRAM {free / 2**30:.2f}/{total / 2**30:.2f} GB")
    except Exception:
        pass
    return " | ".join(parts) if parts else "no data"


def log_env():
    """versions and hardware, so the log shows what changed between runs."""
    log("=" * 70)
    log(f"started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"command: {' '.join(sys.argv)}")
    log(f"python: {sys.version.split()[0]} — {sys.executable}")
    try:
        import torch
        log(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | "
            f"available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            log(f"GPU: {torch.cuda.get_device_name(0)} — VRAM "
                f"{free / 2**30:.2f}/{total / 2**30:.2f} GB free")
    except Exception as e:
        log(f"torch: could not query ({e})")
    for mod in ("whisper", "pyannote.audio", "transformers", "torchaudio", "soundfile",
                "numpy"):
        try:
            m = __import__(mod, fromlist=["__version__"])
            log(f"{mod}: {getattr(m, '__version__', '?')}")
        except Exception as e:
            log(f"{mod}: not importable ({e})")
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True)
        log(f"ffmpeg: {out.stdout.decode('utf-8', 'replace').splitlines()[0]}")
    except Exception as e:
        log(f"ffmpeg: not found ({e})")
    try:
        usage = shutil.disk_usage(tempfile.gettempdir())
        log(f"disk (temp): {usage.free / 2**30:.1f} of {usage.total / 2**30:.1f} GB free")
    except Exception:
        pass
    log(f"memory at start: {mem_snapshot()}")
    log("=" * 70)
