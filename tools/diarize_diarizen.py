# -*- coding: utf-8 -*-
"""Diarization through diarizen. runs under a SEPARATE interpreter.

diarizen brings its own fork of pyannote-audio 3.1.1, which cannot coexist with the
pyannote.audio 4.x of the main environment. so it lives in its own venv and hands the result
back as plain json: a list of [start, end, speaker]. the main pipeline calls this file as a
subprocess and needs to know nothing else about diarizen.

Setting up that environment (python 3.10 is required; diarizen does not build on 3.13):

    uv venv --python 3.10 /path/to/diarizen-venv
    git clone https://github.com/BUTSpeechFIT/DiariZen.git
    cd DiariZen
    uv pip install --python /path/to/diarizen-venv/bin/python \\
        torch torchaudio --index-url https://download.pytorch.org/whl/cu130
    uv pip install --python /path/to/diarizen-venv/bin/python \\
        numpy==1.26.4 einops h5py librosa soundfile pyyaml tqdm toml transformers \\
        omegaconf tensorboardX asteroid-filterbanks lightning speechbrain torchmetrics \\
        rich semver psutil scikit-learn accelerate==1.6.0 torch_audiomentations \\
        pytorch-metric-learning "pyannote.core==5.0.0" "pyannote.database==5.1.3" \\
        "pyannote.metrics==3.2.1" "pyannote.pipeline==3.0.1"
    uv pip install --python /path/to/diarizen-venv/bin/python --no-deps -e ./pyannote-audio -e .

Then point DIARIZEN_PYTHON at that interpreter.

Usage (normally invoked from transcribe.py):
    python tools/diarize_diarizen.py --input audio.wav --output turns.json
"""
import argparse
import json
import sys
from pathlib import Path

import torchaudio as _ta

# --- patches for a modern torch ------------------------------------------------
# diarizen is pinned to torch 2.1, but that version does not support recent gpus,
# so the environment is built on a current torch and the mismatches are patched over here.

if not hasattr(_ta, "list_audio_backends"):
    # removed in torchaudio 2.9; speechbrain still calls it on import.
    _ta.list_audio_backends = lambda: ["soundfile"]

if not hasattr(_ta, "AudioMetaData"):
    # removed in torchaudio 2.2; pyannote 3.1.1 imports it for annotations in its
    # training code, so a stub is enough for inference.
    from dataclasses import dataclass

    @dataclass
    class AudioMetaData:
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = ""

    _ta.AudioMetaData = AudioMetaData

if not hasattr(_ta, "_load_original"):
    # torchaudio 2.9 reads files only through torchcodec, which does not always build
    # on windows. all we need is one mono wav, so soundfile does.
    import soundfile as _sf
    import torch as _torch

    def _load_via_soundfile(path, *args, **kwargs):
        data, rate = _sf.read(str(path), dtype="float32", always_2d=True)
        return _torch.from_numpy(data.T.copy()), rate

    _ta._load_original = _ta.load
    _ta.load = _load_via_soundfile

import torch
import torch.torch_version

# torch 2.6+ loads checkpoints in safe mode. allow the specific harmless types that sit
# in the weights instead of switching the check off entirely.
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

DEFAULT_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"


def main():
    parser = argparse.ArgumentParser(description="diarization through diarizen")
    parser.add_argument("--input", required=True, help="wav, mono, 16 kHz")
    parser.add_argument("--output", required=True, help="where to put the json with the turns")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    from diarizen.pipelines.inference import DiariZenPipeline
    from pyannote.audio.core.task import Problem, Resolution, Specifications

    torch.serialization.add_safe_globals([Specifications, Problem, Resolution])

    pipeline = DiariZenPipeline.from_pretrained(args.model)
    try:
        pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        pass

    result = pipeline(str(Path(args.input)))
    turns = [[round(turn.start, 3), round(turn.end, 3), f"SPEAKER_{str(spk):0>2}"]
             for turn, _, spk in result.itertracks(yield_label=True)]

    Path(args.output).write_text(
        json.dumps({"model": args.model, "turns": turns}, ensure_ascii=False),
        encoding="utf-8")
    print(f"turns: {len(turns)}, speakers: {len({t[2] for t in turns})}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
