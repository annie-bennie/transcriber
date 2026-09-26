# -*- coding: utf-8 -*-
"""Which interpreter runs diarizen, and where its helper script lives.

diarizen can be in three places, and nothing should have to ask which: in this very environment
(the default install and the docker image), in a separate environment named by
`--diarizen-python` or `DIARIZEN_PYTHON` (the optional pyannote 4 layout), or in a folder
next to the project.

this used to be decided inside the pipeline, with a second copy in `tools/benchmark.py` that
never learned about the first case. the benchmark then reported DER 100% on every recording —
every subprocess failed with "the system cannot find the file specified" — while the
pipeline was labelling the same recordings correctly. one place, one answer.
"""
import importlib.util
import os
import sys
from pathlib import Path

# an environment kept next to the project, under either of the names the docs mention
FOLDERS = (".venv-diarizen", "diarizen-venv")


def project_root():
    return Path(__file__).resolve().parent.parent


def _importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def interpreter(explicit=None, environ=None, root=None, exists=None, importable=None):
    """The interpreter to run the diarizen helper with, or None if there is none.

    order: what was asked for, the variable, an environment beside the project, this one. this
    one comes last on purpose — when someone names an interpreter they mean it, even if
    diarizen is also installed here.

    the last four arguments exist so the order can be tested without a filesystem.
    """
    environ = os.environ if environ is None else environ
    exists = (lambda path: Path(path).exists()) if exists is None else exists
    importable = _importable if importable is None else importable
    root = project_root() if root is None else Path(root)

    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if environ.get("DIARIZEN_PYTHON"):
        candidates.append(Path(environ["DIARIZEN_PYTHON"]))
    for name in FOLDERS:
        candidates.append(root / name / "Scripts" / "python.exe")
        candidates.append(root / name / "bin" / "python")

    for candidate in candidates:
        if exists(candidate):
            return candidate
    if importable("diarizen"):
        return Path(sys.executable)
    return None


def helper(root=None):
    """The script the interpreter above is given: it prints turns as json and exits."""
    root = project_root() if root is None else Path(root)
    return root / "tools" / "diarize_diarizen.py"
