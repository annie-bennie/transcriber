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
    # a bar sits on the last line of the terminal and a plain print lands on top of it,
    # leaving half a bar wedged into the middle of the log. tqdm.write scrolls the line
    # above the bar and redraws it underneath.
    if _ACTIVE_BARS:
        try:
            from tqdm import tqdm
            tqdm.write(line, file=sys.stdout)
        except Exception:
            print(line, flush=True)
    else:
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


# ---------------------------------------------------------------------------- progress
# tqdm, the same bar whisper already draws, so every stage looks alike. it also solves the
# two things a hand-rolled bar gets wrong: it disables itself when the output is not a
# terminal, and `tqdm.write` lets log lines scroll past without tearing the bar apart.
#
# the log file never sees a frame of it. a bar rewritten in place becomes a thousand
# near-identical lines in a file, and the log is the thing you read after a crash — so the
# file gets one line when the stage ends.
PIPED_STEP = 0.1          # without a terminal, report every 10%

_ACTIVE_BARS = []


def _fmt_time(seconds):
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


class Progress:
    """A stage in progress. `total` may be None when the size is not known in advance.

        with Progress("measuring", total=len(units), unit="utterances") as bar:
            for unit in units:
                ...
                bar.step()

    closing prints the summary line and takes the bar off the screen.
    """

    def __init__(self, label, total=None, unit="", quiet=False):
        self.label = label
        self.total = total or None
        self.unit = unit
        self.done = 0
        self.started = time.monotonic()
        self._last_reported = 0.0
        self._last_tick = 0.0
        self._closed = False
        self._bar = None
        if not quiet:
            self._bar = self._open()
        if self._bar is None:
            log(f"  {label}..." + (f" ({total} {unit})".rstrip() if total else ""))

    # a stage with no total to count against — a subprocess that reports nothing — has no
    # rate and no percentage, and tqdm's default line then reads "0it [00:07, ?it/s]",
    # which is three pieces of punctuation around one useful number. show the number.
    OPEN_ENDED_FORMAT = "{desc}: {elapsed}"

    def _open(self):
        try:
            from tqdm import tqdm
        except Exception:
            return None
        try:
            # disable=None is tqdm's own "only when attached to a terminal"
            bar = tqdm(total=self.total, desc=f"  {self.label}",
                       unit=self.unit or "it", leave=False, dynamic_ncols=True,
                       file=sys.stdout, disable=None,
                       bar_format=None if self.total else self.OPEN_ENDED_FORMAT)
        except Exception:
            return None
        if bar.disable:
            return None
        _ACTIVE_BARS.append(bar)
        return bar

    # ------------------------------------------------------------------ updating
    def step(self, n=1, note=""):
        self.set(self.done + n, note=note)

    def set(self, done, total=None, note=""):
        if total and total != self.total:
            # a stage that starts open-ended and later learns its size — pyannote reports
            # a total only once a step begins — swaps back to the counting layout.
            self.total = total
            if self._bar is not None:
                self._bar.total = total
                self._bar.bar_format = None
        delta = done - self.done
        self.done = done
        if self._bar is not None:
            if note:
                self._bar.set_postfix_str(note, refresh=False)
            self._bar.update(delta)
            return
        if self.total:
            share = self.done / self.total
            if share - self._last_reported >= PIPED_STEP:
                self._last_reported = share
                log(f"    {self.label}: {self.done}/{self.total} {self.unit} "
                    f"({share * 100:.0f}%)".rstrip())

    def retitle(self, label):
        """A stage that turns out to have named sub-steps relabels the same bar."""
        self.label = label
        self.done = 0
        self.total = None
        if self._bar is not None:
            self._bar.set_description_str(f"  {label}", refresh=False)
            self._bar.bar_format = self.OPEN_ENDED_FORMAT
            self._bar.total = None
            self._bar.reset()

    def tick(self):
        """Keep an indeterminate bar moving while nothing has progress to report."""
        if self._bar is None:
            return
        now = time.monotonic()
        if now - self._last_tick < 0.2:
            return
        self._last_tick = now
        self._bar.refresh()

    # ------------------------------------------------------------------- finishing
    def close(self, summary=None):
        if self._closed:
            return
        self._closed = True
        if self._bar is not None:
            try:
                _ACTIVE_BARS.remove(self._bar)
            except ValueError:
                pass
            self._bar.close()
            self._bar = None
        spent = time.monotonic() - self.started
        tail = summary if summary is not None else (
            f"{self.done} {self.unit}".strip() if self.done else "")
        log(f"  {self.label} done in {_fmt_time(spent)}" + (f" — {tail}" if tail else ""))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        elif self._bar is not None:
            try:
                _ACTIVE_BARS.remove(self._bar)
            except ValueError:
                pass
            self._bar.close()
            self._bar = None
        return False


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
