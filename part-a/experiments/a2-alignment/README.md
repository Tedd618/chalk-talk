# A.2 — Speech-stroke alignment

Goal: produce, per video, a single time-ordered event stream that
interleaves spoken words and pen events. This is the training-shaped
data we need for the model.

## Pipeline

```
videos/<tag>.mp4 (from a1)
    │
    ▼  stt.py (faster-whisper)
output/<tag>.words.jsonl  — {word, t_start, t_end} per line
                              │
a1/output/<tag>.strokes.jsonl │  ← from A.1
                              ▼
                   merge.py
                              │
                              ▼
output/<tag>.events.jsonl  — time-ordered: PEN events + WORD events
```

## Usage

The Whisper model and venv live with A.1 (we share dependencies). From
the project root:

```bash
cd part-a/experiments/a1-stroke-extraction

# 1. Transcribe
.venv/bin/python ../a2-alignment/stt.py oct

# 2. Merge with A.1 strokes
.venv/bin/python ../a2-alignment/merge.py oct

# 3. Visualize (strokes + scrolling captions)
.venv/bin/python ../a2-alignment/viz_events.py oct
```

## Status

**Pilot working.** OCT 3-min Pythagorean theorem video produces a
1442-event stream (425 words + 880 pen events + 137 stroke ends).
Speech-stroke alignment is qualitatively correct: when the tutor says
"A squared plus B squared is equal to C squared", the matching
formula is being written on the canvas in the same time window.

## Pilot numbers

```
words   : 425   (Whisper small @ int8, ~142 wpm)
pen ev. : 880   (one per stroke point)
strokes : 137   (matches a1's stroke count)
total   : 1442 events, time-ordered
```

Whisper-small CPU inference: ~53s for 3 minutes of audio.

## Decision log

- **Whisper backend**: faster-whisper (CTranslate2). Apple Silicon
  friendly, supports word-level timestamps, fast on CPU at int8.
- **Model size**: `small` (244 MB). Math-tutor speech is clear; small
  is enough for word boundaries within ~100-200 ms. Easy to switch to
  `medium` if alignment errors show up systematically.
- **Event schema**: time-ordered stream of `word`, `pen`, `stroke_end`
  events. `pen` events carry `(x, y, t, stroke_id, first/last_in_stroke)`,
  enough to derive pen-state implicitly. This is the shape downstream
  training data takes.
- **VAD on**: `vad_filter=True`, 300 ms min silence. Helps with
  long pauses where no speech is happening.
