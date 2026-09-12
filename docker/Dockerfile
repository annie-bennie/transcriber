# syntax=docker/dockerfile:1
#
# transcriber: whisper, diarizen and the emotion scales in one environment — the one
# requirements.txt describes for an install without docker.
#
#   docker build -f docker/Dockerfile -t transcriber:cpu .
#   docker build -f docker/Dockerfile -t transcriber:gpu --build-arg TORCH=cu128 .
#
# the gpu image also runs on a machine without a card, as slowly as the cpu one does.

FROM python:3.10-slim-bookworm
ARG TORCH=cpu

# uv is mounted for this step rather than copied in: the image carries the environment, not the
# tool that built it. no download cache either — kept in the layer it would double the size of
# the build for nothing. torch first, from the index that matches the build; cuda wheels run to
# hundreds of megabytes, hence the longer timeout. whisper asks for triton on any linux
# machine: on a cpu that is 900 MB that can never run, so the cpu build leaves it out, while
# the gpu build keeps it because torch itself depends on it there.
RUN --mount=from=ghcr.io/astral-sh/uv:0.12.12,source=/uv,target=/bin/uv \
    --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    export UV_NO_CACHE=1 UV_COMPILE_BYTECODE=1 UV_HTTP_TIMEOUT=300 \
 && uv venv /opt/venv \
 && uv pip install --python /opt/venv/bin/python torch==2.8.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/${TORCH} \
 && echo triton > /tmp/cpu-excludes.txt \
 && uv pip install --python /opt/venv/bin/python -r /tmp/requirements.txt \
        $([ "${TORCH}" = cpu ] && echo --excludes /tmp/cpu-excludes.txt) \
 && useradd --create-home --uid 1000 transcriber \
 && install -d -o transcriber /out /models

COPY transcribe.py /app/
COPY transcriber/ /app/transcriber/
COPY tools/diarize_diarizen.py /app/tools/
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/transcribe

USER transcriber
# every model cache here follows XDG_CACHE_HOME — whisper, hugging face, torch hub — so one
# volume at /models keeps 2.5 GB of weights from downloading again on every run
ENV PATH=/opt/venv/bin:$PATH XDG_CACHE_HOME=/models PYTHONUNBUFFERED=1 \
    TRANSCRIBER_IMAGE_VARIANT=${TORCH}
WORKDIR /in
ENTRYPOINT ["transcribe"]
CMD ["--help"]
