#!/bin/sh
# the input folder is mounted read-only, so the default output location — a transcripts/
# folder next to the input — cannot be written. results go to /out unless the command names
# another place.
set -eu

case " $* " in
    *" --output-dir "* | *" --output-dir="*)
        exec python -u /app/transcribe.py "$@" ;;
    *" --help "* | *" -h "*)
        exec python -u /app/transcribe.py "$@" ;;
esac

exec python -u /app/transcribe.py "$@" --output-dir /out
