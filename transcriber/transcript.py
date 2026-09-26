# -*- coding: utf-8 -*-
"""Where a live session is written, and what it looks like on disk.

the live mode used to keep nothing, which was a defensible default and a surprising one: people
speak into it for ten minutes and then look for the text. so it writes a txt, by default
into the documents folder, named by the date.

two things this has to get right.

the folder is asked of the system, not guessed. `~/Documents` is wrong on any machine where the
folder was moved — with onedrive that is most windows machines, and the file would quietly land
somewhere the person does not look.

the file is rewritten, not appended to. speaker labels are corrected as the recording goes on —
about one line in six in a group — so a line already written can change. anything already
in the file from an earlier session is kept as it is, and this session's lines are rewritten
under it every time something changes. the write is atomic, so an interrupted session leaves
the previous whole version rather than half a line.
"""
import os
from datetime import datetime
from pathlib import Path

from .state import write_text_atomic


def documents_dir():
    """The documents folder, as the system understands it, or the home directory."""
    if os.name == "nt":
        try:
            import winreg

            key = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as handle:
                value, _ = winreg.QueryValueEx(handle, "Personal")
            path = Path(os.path.expandvars(value))
            if path.is_dir():
                return path
        except Exception:
            pass
    else:
        # the same idea on linux: the folder can be renamed or moved, and the system knows
        for line in (Path.home() / ".config" / "user-dirs.dirs").read_text(
                encoding="utf-8", errors="replace").splitlines() if (
                Path.home() / ".config" / "user-dirs.dirs").exists() else []:
            if line.startswith("XDG_DOCUMENTS_DIR="):
                value = line.split("=", 1)[1].strip().strip('"')
                path = Path(os.path.expandvars(value.replace("$HOME", str(Path.home()))))
                if path.is_dir():
                    return path
    fallback = Path.home() / "Documents"
    return fallback if fallback.is_dir() else Path.home()


def live_path(output_dir=None, when=None):
    """<documents>/2026-09-27.txt — the date is the name, as asked."""
    when = when or datetime.now()
    folder = Path(output_dir) if output_dir else documents_dir()
    return folder / f"{when:%Y-%m-%d}.txt"


class LiveTranscript:
    """This session's lines, and the file they are kept in.

    `header` marks where a session begins, because a file named by the date holds every session
    of that day. lines are held in memory so that a corrected speaker changes the file too.
    """

    def __init__(self, path, language=None, when=None, existing=None):
        self.path = Path(path)
        self.when = when or datetime.now()
        self.language = language
        self.lines = []
        if existing is None:
            existing = ""
            if self.path.exists():
                existing = self.path.read_text(encoding="utf-8", errors="replace").rstrip("\n")
        self.existing = existing

    def header(self):
        language = f", {self.language}" if self.language else ""
        return f"--- live, {self.when:%H:%M}{language} ---"

    def add(self, start, speaker, text):
        self.lines.append({"start": start, "speaker": speaker, "text": text})

    def relabel(self, pieces):
        """Apply corrected speakers, matched by the line's own start time."""
        corrections = {round(piece["start"], 2): piece["speaker"] for piece in pieces}
        for line in self.lines:
            speaker = corrections.get(round(line["start"], 2))
            if speaker:
                line["speaker"] = speaker

    def render(self):
        clock = []
        for line in self.lines:
            minutes, seconds = divmod(int(line["start"]), 60)
            who = line["speaker"] or "?"
            clock.append(f"[{who} {minutes:02d}:{seconds:02d}] {line['text']}")
        parts = []
        if self.existing:
            parts.append(self.existing)
            parts.append("")
        parts.append(self.header())
        parts.extend(clock)
        return "\n".join(parts) + "\n"

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.path, self.render())
        return self.path
