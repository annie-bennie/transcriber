# -*- coding: utf-8 -*-
"""Decoding and everything measured straight from the waveform.

no machine learning here: loudness, pitch and voice quality are arithmetic over samples.
"""
import math
import shutil
import subprocess
from pathlib import Path

from .config import (
    MAX_PITCH_DURATION,
    MIN_ACOUSTIC_DURATION,
    PITCH_MAX_HZ,
    PITCH_MIN_HZ,
    SAMPLE_RATE,
)
from .logging import log

VOICE_PITCH_FLOOR, VOICE_PITCH_CEILING = 60.0, 500.0
VOICE_MIN_DURATION = 1.0
VOICE_MAX_DURATION = 10.0


def ffmpeg_executable() -> str:
    """ffmpeg from the PATH if there is one, else the static build imageio-ffmpeg installs.

    the second is what lets requirements.txt stand for the whole install: nobody has to put a
    separate ffmpeg on the PATH first, in docker or out of it. if neither exists the plain name
    is returned, so the call fails with the familiar "not found".
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def decode_audio(src: Path, sample_rate: int = SAMPLE_RATE):
    """Decode any media file into a mono float32 array — once per file.

    audio used to be decoded three times: ffmpeg to a temporary wav, then whisper ran that
    wav through ffmpeg again, then soundfile read the same file for pyannote. now it is one
    ffmpeg pass into memory and every stage reuses the array.
    """
    import numpy as np

    cmd = [
        ffmpeg_executable(), "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "0",
        "-i", str(src), "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(sample_rate), "-",
    ]
    # no text=True on purpose. otherwise subprocess decodes ffmpeg's output with the system
    # codepage and the reader thread dies on non-ascii metadata, hiding the real error.
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ffmpeg could not decode {src.name} (code {result.returncode}):"
            f"\n{stderr[-2000:]}")
    if not result.stdout:
        raise RuntimeError(f"ffmpeg returned an empty audio stream for {src.name} "
                           "— no audio track?")
    audio = np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768.0
    del result
    return audio


class PitchTrack:
    """Fundamental frequency over the whole recording, computed in one pass.

    detect_pitch_frequency used to be called separately for every segment — 2624 calls on
    an hour of audio, 12 ms each. measured: one pass over ten minutes costs 0.3 s against
    2.3 s piecewise, eight times cheaper, and the values agree (median difference 2 Hz).
    almost all of the time was call overhead rather than computation.

    accuracy improves as a side effect: at a segment boundary the detector now sees the
    context that piecewise slicing hid from it.
    """

    def __init__(self, audio, sample_rate):
        import torch
        import torchaudio

        self.values = None
        self.hop = None
        self.voiced = None
        try:
            track = torchaudio.functional.detect_pitch_frequency(
                torch.from_numpy(audio.copy()).unsqueeze(0), sample_rate)
            values = track.squeeze(0).numpy()
            if values.size:
                self.values = values
                self.hop = (audio.shape[-1] / sample_rate) / values.size
                self.voiced = (values > PITCH_MIN_HZ) & (values < PITCH_MAX_HZ)
        except Exception as e:
            log(f"  ⚠ could not build the pitch track ({type(e).__name__}) — "
                f"falling back to per-segment, which is slower")

    def ready(self) -> bool:
        return self.values is not None

    def stats(self, start, end):
        """(mean pitch, spread) over a span, or (None, None) if no voicing was found."""
        import numpy as np

        if self.values is None:
            return None, None
        low = max(0, int(start / self.hop))
        high = min(self.values.size, max(low + 1, int(end / self.hop)))
        window = self.values[low:high]
        voiced = window[self.voiced[low:high]]
        if voiced.size < 2:
            return None, None
        return float(np.mean(voiced)), float(np.std(voiced, ddof=1))


def analyze_acoustics(chunk, sample_rate, pitch_track=None, start=None, end=None):
    """Loudness in dBFS and pitch statistics for one chunk.

    if a ready pitch track is passed, pitch comes from it; otherwise it is computed on the
    chunk itself.
    """
    import numpy as np
    import torch
    import torchaudio

    if chunk.size < MIN_ACOUSTIC_DURATION * sample_rate:
        return None, None, None

    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
    loudness_db = 20.0 * math.log10(max(rms, 1e-9))

    if pitch_track is not None and pitch_track.ready() and start is not None:
        pitch_mean, pitch_std = pitch_track.stats(start, end)
        return loudness_db, pitch_mean, pitch_std

    pitch_mean = pitch_std = None
    try:
        head = chunk[: int(MAX_PITCH_DURATION * sample_rate)]
        f0 = torchaudio.functional.detect_pitch_frequency(
            torch.from_numpy(head.copy()).unsqueeze(0), sample_rate)
        voiced = f0[(f0 > PITCH_MIN_HZ) & (f0 < PITCH_MAX_HZ)]
        if voiced.numel() >= 2:
            pitch_mean = voiced.mean().item()
            pitch_std = voiced.std().item()
    except Exception:
        pass

    return loudness_db, pitch_mean, pitch_std


def voice_quality(chunk, sample_rate):
    """(jitter %, shimmer %, HNR dB) via praat, or None if it could not be computed.

    computed per utterance rather than per segment, and that matters. measured on an hour
    of audio: on segments (0.1 to 10 s long) the value depended on chunk length more than
    on the voice — correlation −0.40, −0.39, +0.33. on utterances, where lengths are
    uniform, it drops to −0.16…+0.13.

    how meaningful these are at all was checked by splitting: halves of one person's speech
    against different people. the ratio of "difference between people" to "noise within a
    person" came out as shimmer 18, HNR 7.9, jitter 2.1. shimmer and HNR describe the voice
    reliably; jitter is close to noise and should not be relied on.

    absolute values must not be compared to clinical norms: running speech through a video
    call codec gave shimmer 12% against a norm of 4% and HNR 12 dB against 20. norms are
    measured on a sustained vowel in a quiet room.
    """
    try:
        import parselmouth
    except ImportError:
        return None
    if chunk.size < VOICE_MIN_DURATION * sample_rate:
        return None
    import numpy as np

    head = chunk[: int(VOICE_MAX_DURATION * sample_rate)]
    try:
        sound = parselmouth.Sound(head.astype(np.float64), sampling_frequency=sample_rate)
        periods = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)",
                                         VOICE_PITCH_FLOOR, VOICE_PITCH_CEILING)
        jitter = parselmouth.praat.call(periods, "Get jitter (local)",
                                        0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([sound, periods], "Get shimmer (local)",
                                         0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)",
                                             0.01, VOICE_PITCH_FLOOR, 0.1, 1.0)
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    except Exception:
        return None
    values = (jitter * 100, shimmer * 100, hnr)
    # praat returns NaN where it found no periodicity.
    return values if all(v == v for v in values) else None


def voice_quality_available() -> bool:
    try:
        import parselmouth  # noqa: F401
    except ImportError:
        return False
    return True
