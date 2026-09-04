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
