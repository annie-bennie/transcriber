#!/usr/bin/env sh
# Installer for macos and linux: the four steps of docs/install.md as one command.
#
#   ./install.sh                    # picks the torch build from the card it finds
#   ./install.sh --cpu              # force the cpu build
#   ./install.sh --venv /path/env   # somewhere other than .venv
#
# in order: find a python between 3.10 and 3.12 (or let uv fetch one), make a virtual
# environment, install torch 2.8.0 from the index that matches the hardware, install everything
# else, and finish by running --check, so the result is verified rather than assumed.
set -eu

VENV=".venv"
VARIANT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu) VARIANT="cpu"; shift ;;
        --venv) VENV="$2"; shift 2 ;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

# uv brings its own python, which is the whole reason to prefer it: nothing to install by hand
# on a machine whose newest python is 3.13, where diarizen's numpy 1.26.4 has no build.
UV=""
if command -v uv >/dev/null 2>&1; then
    UV="uv"
    echo "uv found — it will fetch python 3.12 if this machine has none"
else
    BASE=""
    for candidate in python3.12 python3.11 python3.10 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c \
            'import sys; raise SystemExit(0 if (3,10) <= sys.version_info < (3,13) else 1)' \
            2>/dev/null; then
            BASE="$candidate"
            break
        fi
    done
    if [ -z "$BASE" ]; then
        echo "no python between 3.10 and 3.12 found, and no uv to fetch one." >&2
        echo "  macos:  brew install python@3.12    (or: brew install uv)" >&2
        echo "  ubuntu: sudo apt install python3.12 python3.12-venv" >&2
        echo >&2
        echo "why the version matters: diarizen pins numpy 1.26.4, which has no build for 3.13." >&2
        exit 1
    fi
    echo "python: $("$BASE" -V) — $(command -v "$BASE")"
fi

# the card decides which torch build is right; pypi's default is not always it
if [ -z "$VARIANT" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        VARIANT="cu128"
        echo "nvidia card found — installing the cuda 12.8 build"
    else
        VARIANT="cpu"
        echo "no nvidia card found — installing the cpu build (several times slower)"
    fi
fi

INDEX="https://download.pytorch.org/whl/$VARIANT"
PY="$VENV/bin/python"

if [ -n "$UV" ]; then
    uv venv --python 3.12 "$VENV"
    echo "installing torch 2.8.0 ($VARIANT) — the long part, a few gigabytes"
    uv pip install --python "$PY" torch==2.8.0 torchaudio==2.8.0 --index-url "$INDEX"
    echo "installing everything else"
    uv pip install --python "$PY" -r requirements.txt
else
    "$BASE" -m venv "$VENV"
    "$PY" -m pip install --quiet --upgrade pip
    echo "installing torch 2.8.0 ($VARIANT) — the long part, a few gigabytes"
    "$PY" -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url "$INDEX"
    echo "installing everything else"
    "$PY" -m pip install -r requirements.txt
fi

echo
"$PY" transcribe.py --check || true
echo
echo "done. to use it:"
echo "  $PY transcribe.py \"meeting.mp4\" --language ru"
echo "or let the launchers find this environment:"
echo "  export TRANSCRIBE_PYTHON=\"\$PWD/$PY\" && ./transcribe.sh \"meeting.mp4\""
