# -*- coding: utf-8 -*-
"""Tunable constants, all in one place.

Every threshold here was picked from a measurement, not from taste. the comment above
each one says which measurement, so that changing it is an informed decision rather
than a guess.
"""

SAMPLE_RATE = 16000

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"}
MEDIA_EXTS = VIDEO_EXTS | AUDIO_EXTS

# ---------------------------------------------------------------- speech recognition
DEFAULT_WHISPER_MODEL = "turbo"
# english by default: whisper handles any language, and the recording language is a
# property of the file, not of the tool. everything the tool says about itself — logs,
# reports, field names — is english regardless.
DEFAULT_LANGUAGE = "en"

# ---------------------------------------------------------------- diarization
DIARIZE_MODEL_31 = "pyannote/speaker-diarization-3.1"
DIARIZE_MODEL_COMMUNITY = "pyannote/speaker-diarization-community-1"
DIARIZE_MODEL_DIARIZEN = "diarizen"
DEFAULT_DIARIZEN_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
DEFAULT_EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

# a word is confidently attributed when a diarization turn covers at least this share of
# its duration. at turn boundaries pyannote and whisper disagree by fractions of a second,
# so a word fragment clipped by a neighbouring turn is not evidence.
WORD_MIN_OVERLAP = 0.5
# a word the diarizer never reached still gets attached to the nearest turn within this gap.
WORD_SPEAKER_GAP = 0.75
# a pause longer than this ends a turn, even for the same speaker.
TURN_MAX_GAP = 3.0

# speaker refinement against voice references.
#
# clustering compares short turns against each other, and their embeddings are close to
# noise. measured on a one-hour recording: on three-second chunks the same/different
# speaker distance is 0.31 vs 0.53, on one-second chunks it collapses to 0.82 vs 0.94.
# a greeting exchange made of one-second turns therefore ends up assigned to one person,
# and no clustering setting fixes it.
#
# references change the question: a short chunk is compared not to an equally noisy
# neighbour but to an averaged voice built from long turns, where the embedding is
# reliable. on the same exchange that gave six correct decisions out of eight instead of
# four, and both errors landed on margins of 0.000 and 0.010 — the method shows where it
# does not know.
SPEAKER_REFERENCE_MIN = 6.0
SPEAKER_REFINE_MIN_SEPARATION = 0.30
SPEAKER_REFINE_ROUNDS = 3
# margins of correct decisions fell between 0.07 and 0.40, the two unrecognisable turns
# got 0.026 and 0.004 — the boundary has room to spare. raising it to 0.08 under plda
# rejected a correct decision at 0.071, so one threshold for both scoring modes.
SPEAKER_REFINE_MIN_MARGIN = 0.05

# ---------------------------------------------------------------- units of analysis
# below this a chunk is not handed to the emotion model at all. measured on a one-hour
# recording: chunks of 0.3–0.5 s produced 8% junk labels, 0.5–1 s produced 2.7%, from two
# seconds on it was zero. shorter than a second is usually a backchannel, not an emotion.
EMOTION_UNIT_MIN = 1.0
# merging target. shorter and the model guesses, longer and one chunk holds too many
# different things.
EMOTION_UNIT_TARGET = 5.0
# hard ceiling: attention cost grows quadratically with length, and thirty-second batches
# once reserved 31 GB on a 16 GB card.
MAX_EMOTION_DURATION = 10.0
# round chunk length up to a multiple of this before batching. whisper boundaries are
# fractional, so without rounding almost every batch arrives with a new input shape; cuda
# caches them per shape and never reuses them.
EMOTION_BUCKET_SECONDS = 0.5

# ---------------------------------------------------------------- acoustics
PITCH_MIN_HZ, PITCH_MAX_HZ = 60, 500
MAX_PITCH_DURATION = 15.0
# below this a chunk is a few hundred samples and both loudness and pitch describe noise.
MIN_ACOUSTIC_DURATION = 0.1

# ---------------------------------------------------------------- speaker baselines
# absolute values say almost nothing: −27 dBFS is a recording level and a distance to the
# microphone, 232 Hz is simply whose voice it is. what means something is how far a person
# moved from their own habit, so each speaker gets their own median.
SPEAKER_BASELINE_MIN_SEGMENTS = 5
SPEAKER_BASELINE_MIN_DURATION = 0.5
# below this level a chunk is digital silence, not quiet speech. such segments appear
# where a whisper word timestamp slipped off the audio: they read −100 dB and produced
# deviations of −27σ, which is a meaningless number.
SILENCE_FLOOR_DB = -70.0
# median and mad instead of mean and standard deviation: a cough or a single shout moves
# the mean noticeably and the median almost not at all. the multiplier brings mad to the
# scale of a standard deviation so that "σ" reads the usual way.
MAD_TO_SIGMA = 1.4826
# a deviation is printed in the unit the measure actually has, and each threshold comes
# from what that unit means — not from a target share of marked lines. tuning six numbers
# to hit "5% each" on one recording would bake that recording's room, microphone and voices
# into every future file; a decibel means the same thing everywhere.
#
# loudness: 6 dB is amplitude halved or doubled — unmistakably quieter or louder.
LOUDNESS_NOTABLE_DB = 6.0
# pitch: 5 semitones is a musical fourth. semitones rather than hertz because 40 Hz is a
# different thing entirely for a 100 Hz voice and a 250 Hz one, and a semitone is not.
PITCH_NOTABLE_SEMITONES = 5.0
# what is left has no natural unit — a spread of pitch, and three unitless 0..1 scales — so
# it stays in sigma. at 1.0 sigma marked 54% of lines, and a mark on half the lines stops
# meaning "look here": 1.5σ → 32%, 2.0σ → 20%, 2.5σ → 12%.
Z_NOTABLE = 2.0
# measured on an hour of dialogue, share of lines each threshold marks: loudness 12.1%,
# pitch 6.2%, pitch spread 4.3%, arousal 3.9%, dominance 6.4%, valence 7.3%. together
# 18.8% of lines carry at least one acoustic mark — the same density as the old flat 2σ,
# but now a mark on loudness and a mark on valence mean comparable things.

# ---------------------------------------------------------------- dialogue statistics
DIALOGUE_FLOOR_GAP = TURN_MAX_GAP
# a short cut-in is a backchannel, not an attempt to take the floor. the threshold is
# rough, so the two cases are counted separately in the report.
BACKCHANNEL_MAX_SEC = 1.0
# an answer faster than this is a latch: the person knew what to say before the other
# finished.
FAST_RESPONSE_SEC = 0.15
RU_VOWELS = "аеёиоуыэюя"
EN_VOWELS = "aeiouy"

# ---------------------------------------------------------------- hallucination filter
# the same thresholds whisper uses internally to reject a window
# (no_speech_threshold=0.6, logprob_threshold=-1.0, compression_ratio_threshold=2.4).
# whisper drops its window only when the first two fire together, and so do we: each one
# alone produces plenty of false alarms on quiet but real speech.
SUSPECT_NO_SPEECH_PROB = 0.6
SUSPECT_AVG_LOGPROB = -1.0
SUSPECT_COMPRESSION_RATIO = 2.4
# raised from four after measuring: runs of exactly four appeared twice on a one-hour
# recording, and both were a live "yes, yes, yes, yes" — fast agreement, not a decoder
# loop. a real loop repeats a phrase far longer.
SUSPECT_REPEAT_RUN = 6
# mean word confidence. on its own it is not evidence: the lowest values across the whole
# recording belonged to ordinary fillers. it only counts together with falling outside
# diarized speech.
SUSPECT_WORD_PROB = 0.05

# whisper hallucinates youtube captions in silence — it heard enough of them during
# training that with no speech present it produces the most frequent ones.
HALLUCINATION_PHRASES = (
    "субтитры сделал", "субтитры создавал", "субтитры и перевод", "редактор субтитров",
    "продолжение следует", "спасибо за просмотр", "спасибо за внимание",
    "подписывайтесь на канал", "ставьте лайк", "до новых встреч",
    "amara.org", "dimatorzok",
    "subtitles by", "thanks for watching", "thank you for watching",
    "subscribe to", "see you next time", "closed captioning",
)
