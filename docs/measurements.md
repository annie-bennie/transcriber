# what the numbers mean

[← back to readme](../README.md)

## first, about deviations

almost every value below comes with a mark like `↑7dB`, `↓5st` or `+2.3σ`. that is not the
value itself but how far a person moved from their own habit in this recording.

for each speaker the pipeline takes their median and typical spread across the whole file.
the baselines are printed at the top of every transcript, one line per speaker, so every
mark in the text can be traced to what it was measured against.

this exists because absolute values say almost nothing. −27 dBFS is a recording level and a
distance to the microphone. 232 Hz is simply whose voice it is. "arousal 0.55" is anyone's
guess. none of them compare between people or between recordings — but a person against
themselves does compare, and that is what the deviation shows.

### each measure in its own unit

| measure | unit | printed from | why that number |
|---|---|---|---|
| loudness | decibels | 6 dB | half or double the amplitude |
| pitch | semitones | 5 st | a musical fourth; 12 st is an octave |
| pitch spread | σ | 2σ | a spread of a spread has no unit of its own |
| arousal, dominance, valence | σ | 2σ | the scales are unitless 0…1 values |

σ is one symbol over four measures, which is fair only if it means the same rarity in each.
measured on two recordings — an hour of dialogue and ninety minutes of a class — the four
sit between 1.2% and 5.4% of lines at |z| ≥ 2, with skew from −0.46 to +0.67. that is close
enough to read them together. loudness and pitch were the ones that did not belong: before
the split, loudness marked 11–15% of lines with a skew of −5.

one asymmetry survives. pitch spread is skewed positive on both recordings (+0.5, +0.7),
because intonation can widen a long way but can only narrow to flat. a −2σ there is a rarer
event than a +2σ, which is not true of the scales.

σ is a last resort, not a default: it is used only where the quantity has no unit anyone can
picture. "seven decibels quieter than usual" is a statement about the world; "−2.1σ" is a
statement about a distribution.

pitch is in semitones rather than hertz because hertz do not transfer between voices. 40 Hz
is a large move for a 100 Hz voice and a small one for a 250 Hz voice; a semitone is the
same interval for both. a fifth above a bass and a fifth above a soprano are both 7 st,
where in hertz they are 50 and 125 apart.

### why the thresholds are not tuned

each threshold comes from what its unit means, never from a target share of marked lines.
that distinction matters. normalising per speaker removes differences of level and of
spread, so a bass and a soprano are handled alike — but it cannot remove the *shape* of a
distribution. a lecture has thin tails, an argument fat ones, a call with automatic gain
control has almost no loudness spread left at all. a threshold set to mark "5% of lines"
on one recording would mark 1% on the first and 15% on the second, carrying the properties
of the file it was fitted on into every other file. a decibel does not do that.

measured on an hour of dialogue, these thresholds mark: loudness 12.1% of lines, pitch 6.2%,
pitch spread 4.3%, arousal 3.9%, dominance 6.4%, valence 7.3% — 18.8% of lines carry at
least one acoustic mark. that is a check on the result, not the way it was chosen. full
values are always in the json, along with a `deviation_guide` block that repeats all of
this for anything reading the file without this page.

## emotion scales

speech is described by three numbers from 0 to 1 rather than one label. this is the standard
dimensional model of affect, where a state is given by coordinates instead of a name.

| scale | what it measures | low | high |
|---|---|---|---|
| **arousal** | energy, activation | sluggish, sleepy, monotone | fast, tense, keyed up |
| **valence** | pleasantness | downcast, irritated, dejected | lively, glad, pleased |
| **dominance** | sense of control | hesitant, uncertain, yielding | assertive, confident, pressing |

the first two are the main ones — almost any familiar emotion name is a combination of them:

| | low valence | high valence |
|---|---|---|
| **high arousal** | anger, anxiety, panic | joy, excitement, enthusiasm |
| **low arousal** | sadness, fatigue, boredom | calm, contentment |

that table is why the scales are more useful than a label. anger and anxiety share one cell
and a label confuses them; but moving from sadness to anger is arousal rising at constant
valence, and on the scales it reads as movement rather than a swapped tag. the same way it
shows when a person did not change emotion but simply got tired: valence stays, arousal
falls.

the table also goes into the json (`emotion_guide`) and into the top of the txt, so that a
reader — a person or a language model — can interpret the numbers without going looking for
this page.

### what this answers that a label does not

- *how did the state change through the conversation?* a label jumps between classes, the
  scales give a continuous curve, and a build-up is visible as a build-up.
- *can two recordings be compared?* with labels no — their distribution depends on the
  window setting. with scales yes, comparing each person's deviation from their own norm.

### the third scale is mostly the first one again

measured on two recordings, arousal and dominance correlate at **r = 0.96** on raw values —
0.966 on an hour of dialogue, 0.961 on ninety minutes of a class. the difference between
them spans 0.16 where arousal itself spans 0.56. as far as the audio is concerned they are
one measurement: both track vocal effort, which is what a microphone can hear.

so a line reading `aro -2.3σ | dom -2.4σ` is one observation printed twice, not two
agreeing ones. both stay in the output — the pair is informative when it *fails* to hold,
and dominance still belongs in the json for anything downstream — but they should not be
counted as separate evidence, and "who led the conversation" is better answered from the
speaking shares and interruptions in the dialogue block than from this scale.

valence is the axis carrying separate information, and how separate depends on the
recording: r = 0.63 with arousal on the dialogue, 0.12 on the class. that is also why the
quadrant table is built on arousal and valence and not on the third scale.

### how much to trust them

the model is trained on MSP-Podcast — natural conversational speech, not acted recordings.
checked on annotated sets: RESD in russian, RAVDESS in english, four-class core.

| what was checked | result |
|---|---|
| agreement with theory | holds in both languages: anger has the highest arousal and the lowest valence, neutral speech the lowest arousal, joy the highest valence |
| transfer between languages | a rule tuned on russian gives 45% on english, tuned on english 43% on russian, against 49–51% on its own language |
| four-class accuracy | 49–51%, against 59–61% for a language-specific categorical model |

a single utterance is not a verdict. the scales are for tracking change within one speaker,
which is the thing a label cannot do.

one caveat worth keeping: RAVDESS is acted emotion, more expressive than the real thing. on
a live english conversation the accuracy will be lower, and there is nothing to check that
against — free corpora of natural annotated english emotional speech are not something i
found.

## loudness and pitch

**loudness (dB)** is the rms level in dBFS, relative to the loudest thing that can be
recorded. zero is as loud as it gets, −60 is nearly silence. it describes the person and the
microphone at once, which is why on its own it means little.

**pitch (Hz)** is the mean fundamental frequency — how fast the vocal folds vibrate. an adult
man is usually 85–155 Hz, a woman 165–255 Hz. within one person pitch rises with excitement
and questions, and falls with fatigue.

loudness carries its deviation in decibels and pitch in semitones, and that is the part worth reading.

## voice quality — optional, `--voice-quality`

three values from clinical phonetics. all three describe how *evenly* the vocal folds work.
needs `pip install praat-parselmouth`.

**jitter (%)** — period wobble. the folds close tens to hundreds of times a second and each
cycle differs slightly in length from the last; jitter is the average of that difference. it
rises with hoarseness, vocal fatigue, a cold. under 1% on a sustained vowel is healthy.

**shimmer (%)** — amplitude wobble. the same, but about loudness: how much each cycle differs
in strength from its neighbour. high shimmer is heard as breathiness and rasp. under 4% is
the norm on a sustained vowel.

**HNR (dB)** — harmonics-to-noise ratio. how much of the sound is voice proper and how much
is turbulent noise from folds that do not close cleanly. higher is clearer and more ringing,
lower is breathier and hoarser. above 20 dB is the norm on a sustained vowel.

### why it is off by default

all three were designed for a sustained vowel recorded in a quiet room with a good
microphone. on running speech through a video-call codec they measure the recording channel
along with the voice. checked on an hour of audio by splitting — halves of one person's
speech against different people:

| value | signal-to-noise ratio | verdict |
|---|---:|---|
| shimmer | 18.0 | describes the voice reliably |
| HNR | 7.9 | describes the voice reliably |
| **jitter** | **2.1** | close to noise, do not rely on it |

absolute values must not be compared to the norms: this recording gave shimmer 12% against a
norm of 4% and HNR 12 dB against 20 — a property of the material, not of the voices. so they
are normalised within a speaker like everything else.

computed per utterance rather than per line: on short chunks the value depends on chunk
length more than on the voice (correlation with duration −0.40 against −0.16).

### what they are for at all

these three answer a question neither emotion nor loudness does: *how does the voice itself
sound*. did the person go hoarse by the end of a two-hour meeting, did breathiness appear on
a difficult topic, did anything shift between today's recording and the last one. an emotion
label is about what someone expresses; these are about the state of the vocal apparatus.

## dialogue statistics

| value | how it is computed | what it tells you |
|---|---|---|
| share of speaking | a person's speech time over total speech time. can sum above 100% by the amount of simultaneous speech | who led. a skew is expected in an interview; 80/20 in a meeting is worth a look at who is not being heard |
| silence | share of the recording where nobody speaks | thinking pauses, technical hitches. a very low share means a conversation with no breathing room |
| floor change | the previous speaker finished and the next continued past them. a turn nested entirely inside another does not count | how live the exchange is. many changes with short turns is fast dialogue; few with long ones is monologues taking turns |
| response latency | from the end of a turn to the start of the answer. negative means the answer started over the other person's speech | a median around 0.2 s is ordinary conversational pace. noticeably more means people are thinking, or the connection lags |
| latch without a pause | an answer starting within 0.15 s | engagement or argument: the person knew what to say before the other finished |
| interruption / backchannel | cutting into someone else's speech: under a second is a backchannel, longer is an interruption | backchannels are a sign of listening, interruptions a fight for the floor. the one-second threshold is rough, so the two are counted separately |
| speech rate | syllables over the time a person actually produced words, not over floor time | 4–6 syllables per second is ordinary for conversational speech. within one person a rising rate is an arousal marker |
| pause within a turn | median gap between one's own consecutive segments | how fluent the speech is. long pauses inside one's own turn mean searching for words |

if speaker labelling is uncertain anywhere, the summary says on what share of time — and
everything else in it is imprecise by the same amount.
