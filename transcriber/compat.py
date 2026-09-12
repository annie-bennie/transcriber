# -*- coding: utf-8 -*-
"""Shims for versions that moved apart.

these must run before anything imports pyannote. they used to live inside the diarization
loader, which meant that when diarizen handled diarization the loader never ran and the
voice embedding model failed on a missing torchaudio attribute.
"""


def patch_torchaudio():
    """Restore two names pyannote still imports and torchaudio no longer provides.

    idempotent — safe to call from every place that is about to touch pyannote.
    """
    import torchaudio

    if not hasattr(torchaudio, "list_audio_backends"):
        # dropped in torchaudio 2.9; speechbrain, a pyannote dependency, still calls it.
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    if not hasattr(torchaudio, "AudioMetaData"):
        # dropped in torchaudio 2.2. pyannote 3.x imports it for type annotations in its
        # training code, and a stub is enough for inference. needed to run inside an
        # environment that has pyannote 3.x — the one diarizen installs, for instance.
        from dataclasses import dataclass

        @dataclass
        class AudioMetaData:
            sample_rate: int = 0
            num_frames: int = 0
            num_channels: int = 0
            bits_per_sample: int = 0
            encoding: str = ""

        torchaudio.AudioMetaData = AudioMetaData


def from_pretrained(cls, name, token, **kwargs):
    """`cls.from_pretrained` with the hugging face token under the name this version takes.

    pyannote.audio 4 calls it `token`, 3.x — the fork diarizen installs — `use_auth_token`.
    the wrong one is a TypeError, which the loaders around it report as a model that could not
    be loaded, so it has to be right rather than guessed.
    """
    import inspect

    params = inspect.signature(cls.from_pretrained).parameters
    key = "token" if "token" in params else "use_auth_token"
    return cls.from_pretrained(name, **{key: token}, **kwargs)


def allow_pyannote_checkpoints():
    """Let torch's safe loader open pyannote 3.x checkpoints.

    since torch 2.6 `torch.load` refuses anything but plain tensors unless told otherwise, and
    a pyannote 3.x checkpoint also holds a torch version object and the task description
    classes. pyannote 4 deals with this itself; 3.x — the fork inside diarizen — raises an
    UnpicklingError, which the embedding loader reports as "speaker refinement will be
    skipped". allowing these specific types is narrower than switching the check off.
    idempotent.
    """
    import torch
    import torch.torch_version

    allowed = [torch.torch_version.TorchVersion]
    try:
        from pyannote.audio.core.task import Problem, Resolution, Specifications
        allowed += [Specifications, Problem, Resolution]
    except Exception:
        pass
    torch.serialization.add_safe_globals(allowed)
