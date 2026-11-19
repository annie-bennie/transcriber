#!/usr/bin/env sh
# launcher for macos and linux.
#
#   ./transcribe.sh "lecture.mp4"
#   ./transcribe.sh ~/records --language ru --num-speakers 2
#
# the interpreter is looked up in this order: TRANSCRIBE_PYTHON, a .venv or venv inside the
# project, an already activated environment, the system python3.
set -eu

DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ -n "${TRANSCRIBE_PYTHON:-}" ]; then
    PY="$TRANSCRIBE_PYTHON"
elif [ -x "$DIR/.venv/bin/python" ]; then
    PY="$DIR/.venv/bin/python"
elif [ -x "$DIR/venv/bin/python" ]; then
    PY="$DIR/venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PY="$VIRTUAL_ENV/bin/python"
else
    PY=python3
fi

# a plain "no module named whisper" ten frames deep is a poor way to learn that the wrong
# interpreter was picked, so the dependencies are checked here while it is still obvious.
if ! "$PY" -c "import whisper, numpy" >/dev/null 2>&1; then
    echo "$PY has no dependencies of this project installed." >&2
    echo "  either activate the environment they live in," >&2
    echo "  or point TRANSCRIBE_PYTHON at its interpreter:" >&2
    echo "    export TRANSCRIBE_PYTHON=/path/to/env/bin/python" >&2
    echo "  a variable set in a shell profile only reaches shells started after it." >&2
    exit 1
fi

# -u turns off buffering so progress shows as it happens, not after a stage finishes.
exec "$PY" -u "$DIR/transcribe.py" "$@"
