# -*- coding: utf-8 -*-
"""Emotion as three numbers instead of one label.

arousal, dominance and valence, each 0 to 1, from the dimensional model of affect. the
model is trained on MSP-Podcast — natural conversational speech, not acted recordings.

why dimensions and not a label. measured on annotated sets (RESD in russian, RAVDESS in
english, four-class core):

  * the scales behave according to theory and identically in both languages: anger has the
    highest arousal and the lowest valence, neutral speech has the lowest arousal, joy has
    the highest valence;
  * a decision rule transfers across languages: tuned on russian it gives 45% on english,
    tuned on english 43% on russian, against 49–51% on its own language. that small drop is
    the whole argument;
  * a logistic regression transfers *worse* than nearest centroids (38% against 45%) — it
    overfits to the training language. the value of the dimensional approach is the
    robustness of a simple rule, not peak accuracy.

but the real point is that the value is continuous, so it can be normalised within a
speaker the same way loudness and pitch are. "arousal 2σ above this person's own norm" is
a statement about a person; a label like "positive 0.9" is a statement about what a chunk
of audio resembles.
"""
from .config import (LOUDNESS_NOTABLE_DB, PITCH_NOTABLE_SEMITONES, Z_NOTABLE)
from .logging import log

DEFAULT_EMOTION_DIM_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
EMOTION_DIMENSIONS = ("arousal", "dominance", "valence")
EMOTION_DIMENSION_SHORT = {"arousal": "aro", "dominance": "dom", "valence": "val"}

# a reading key, written into the json and the txt so that a downstream reader — a person
# or a language model — can interpret the numbers without guessing what they mean.
EMOTION_DIMENSION_GUIDE = {
    "model": DEFAULT_EMOTION_DIM_MODEL,
    "trained_on": "MSP-Podcast, natural conversational speech",
    "scale": "0 to 1, measured per utterance",
    "dimensions": {
        "arousal": {
            "measures": "energy and activation of speech",
            "low": "sluggish, sleepy, monotone",
            "high": "fast, tense, keyed up",
        },
        "dominance": {
            "measures": "sense of control over the situation",
            "low": "hesitant, uncertain, yielding",
            "high": "assertive, confident, pressing",
        },
        "valence": {
            "measures": "pleasantness of the experience",
            "low": "downcast, irritated, dejected",
            "high": "lively, glad, pleased",
        },
    },
    # the two main axes combine into the familiar emotion names. this is what makes the
    # numbers readable without a categorical model.
    "quadrants": [
        {"arousal": "high", "valence": "low",
         "reads_as": "anger, anxiety, panic"},
        {"arousal": "high", "valence": "high",
         "reads_as": "joy, excitement, enthusiasm"},
        {"arousal": "low", "valence": "low",
         "reads_as": "sadness, fatigue, boredom"},
        {"arousal": "low", "valence": "high",
         "reads_as": "calm, contentment"},
    ],
    "how_to_read": (
        "absolute values are not comparable between recordings or between people. compare "
        "the per-speaker deviations instead: fields ending in _z are how far an utterance "
        "sits from that speaker's own median in this recording, in robust standard "
        "deviations. |z| >= 2 is notable."
    ),
    "known_limits": (
        "four-class accuracy is 49-51%, below a language-specific categorical model "
        "(59-61%), so a single utterance should not be read as a verdict. the scales are "
        "for tracking change within one speaker, which is what a label cannot do."
    ),
    "not_independent": (
        "arousal and dominance are nearly the same measurement: r = 0.96 on an hour of "
        "dialogue and 0.96 on ninety minutes of a class, on raw values. both track vocal "
        "effort, which is what the acoustics carry. read a matching pair as one "
        "observation rather than two agreeing ones. valence is the axis that carries "
        "separate information, though how separate depends on the recording (r = 0.63 "
        "with arousal on the dialogue, 0.12 on the class)."
    ),
}

# every deviation is from that speaker's own median in this recording — the baselines live
# in speaker_stats, one set per speaker. the unit differs by measure, and so does the
# threshold above which the transcript prints it: a decibel and a semitone mean the same
# thing in any recording, which a percentile of one file does not.
#
# normalising per speaker removes differences of level and of spread, so a bass and a
# soprano are handled alike. what it cannot remove is the shape of the distribution — a
# lecture has thin tails, an argument fat ones — which is exactly why the thresholds are
# taken from the units rather than tuned to mark a fixed share of lines.
DEVIATION_GUIDE = {
    "compared_to": "the same speaker's median in this recording, never another speaker "
                   "or another file",
    "baselines_in": "speaker_stats — median and spread of every measure, per speaker",
    "measures": {
        "loudness": {
            "field": "loudness_delta_db",
            "unit": "decibels",
            "printed_from": LOUDNESS_NOTABLE_DB,
            "reference": "6 dB is half or double the amplitude",
        },
        "pitch": {
            "field": "pitch_delta_st",
            "unit": "semitones",
            "printed_from": PITCH_NOTABLE_SEMITONES,
            "reference": "5 semitones is a musical fourth, 12 an octave. semitones rather "
                         "than hertz because 40 Hz is a different thing for a 100 Hz voice "
                         "and a 250 Hz one",
        },
        "pitch_spread": {
            "field": "pitch_var_z",
            "unit": "robust standard deviations",
            "printed_from": Z_NOTABLE,
            "reference": "how much wider or narrower the intonation of this line is. a "
                         "spread of a spread has no unit of its own",
            "one_sided": "intonation can widen far but can only narrow to flat, so this "
                         "one is skewed: measured at +0.5 to +0.7 across two recordings, "
                         "where the emotion scales sit near zero. a -2σ here is a rarer "
                         "event than a +2σ, which is not true of the scales",
        },
        "emotion_scales": {
            "field": "arousal_z, dominance_z, valence_z",
            "unit": "robust standard deviations",
            "printed_from": Z_NOTABLE,
            "reference": "the scales are unitless 0..1 values, so distance is only "
                         "expressible in spreads",
        },
    },
    "note": "absolute values (dB, Hz, the 0..1 scales) are not comparable between "
            "recordings; the deviations are.",
}



class DimensionalEmotion:
    """Wrapper around the audeering regression model.

    the classification head is not a separate file in the model repository, so its
    description has to be repeated here — this is exactly the architecture given on the
    model card.
    """

    def __init__(self, model_name=DEFAULT_EMOTION_DIM_MODEL, device="cpu"):
        import torch
        from torch import nn
        from transformers import (Wav2Vec2Model, Wav2Vec2PreTrainedModel,
                                  Wav2Vec2Processor)

        class RegressionHead(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, features):
                x = self.dropout(features)
                x = self.dense(x)
                x = torch.tanh(x)
                x = self.dropout(x)
                return self.out_proj(x)

        class EmotionRegressor(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = RegressionHead(config)
                self.init_weights()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0]
                return self.classifier(torch.mean(hidden, dim=1))

        self.device = torch.device(device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = EmotionRegressor.from_pretrained(model_name).to(self.device).eval()
        self.name = model_name

    def measure(self, chunks, sample_rate, batch_size=4):
        """List of chunks -> list of dicts with the scales (or None where it failed)."""
        import numpy as np
        import torch

        out = []
        for start in range(0, len(chunks), max(1, batch_size)):
            batch = chunks[start:start + max(1, batch_size)]
            try:
                inputs = self.processor(batch, sampling_rate=sample_rate,
                                        return_tensors="pt", padding=True)
                with torch.no_grad():
                    values = self.model(inputs["input_values"].to(self.device))
                values = values.cpu().numpy()
            except Exception as e:
                log(f"    ⚠ emotion scales failed on {len(batch)} chunks: "
                    f"{type(e).__name__}")
                out.extend([None] * len(batch))
                continue
            for row in values:
                if np.isnan(row).any():
                    out.append(None)
                else:
                    out.append({name: round(float(v), 4)
                                for name, v in zip(EMOTION_DIMENSIONS, row)})
        return out

    def to(self, device):
        self.model.to(device)
        return self
