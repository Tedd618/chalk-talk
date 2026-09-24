# Output Format Reference

This document describes every file that the pipeline produces and
exactly what each field means. Read this before writing any training
or evaluation code that consumes the pipeline output.

---

## 1. strokes.jsonl — the primary stroke data

**Location:** `output/<tag>.strokes.jsonl`  
(also saved as `output/<tag>.v4.strokes.jsonl` — identical content)

One JSON object per line. Each object is one stroke — a single
continuous pen-down to pen-up motion.

### Example

```json
{
  "stroke_id": 0,
  "page": 0,
  "t_start": 6.3,
  "t_end": 6.6,
  "points": [
    {"x": 321.0, "y": 119.0, "t": 6.3},
    {"x": 327.0, "y": 115.0, "t": 6.3},
    {"x": 348.0, "y": 111.0, "t": 6.4},
    {"x": 351.0, "y": 114.0, "t": 6.4}
  ]
}
```

### Fields

| Field | Type | Description |
|---|---|---|
| `stroke_id` | int | Global stroke index, 0-based, ordered by `t_start` across the whole video |
| `page` | int | Which page (canvas clear session) this stroke belongs to. 0-based. **Critical for training — see below.** |
| `t_start` | float | Time in seconds when this stroke began (first point drawn) |
| `t_end` | float | Time in seconds when this stroke ended (last point drawn) |
| `points` | list | The stroke path — see below |

### Points

Each point in `points` is one location along the stroke path:

| Field | Type | Description |
|---|---|---|
| `x` | float | Horizontal position in pixels. Left = 0, right = 1280. |
| `y` | float | Vertical position in pixels. Top = 0, bottom = 720. |
| `t` | float | Time in seconds when this specific point was drawn |

**Coordinate system:** 1280 × 720 pixels, origin at top-left.  
**Point count:** 2–68 points per stroke, average ~8 (RDP-simplified).  
**Time precision:** ~0.1 seconds (10fps subsampling).

### The `page` field — why it matters

The tutor clears the board between problems. Without the `page` field
you would not know which strokes belong together.

For training, **treat each page as a separate episode.** Strokes on
different pages are not in the same context — they belong to different
problems. Never train on strokes that span a page boundary as if they
were continuous.

Example (oct-fractions, 10 min video):
```
page 0:  47 strokes,  t=6.3s  – 44.9s   (problem 1)
page 1:  47 strokes,  t=51.6s – 79.6s   (problem 2)
page 2: 139 strokes,  t=104.7s – 198.7s (problem 3)
...
page 7:  91 strokes,  t=546.8s – 604.6s (problem 8)
```

---

## 2. events.jsonl — words and strokes merged in time order

**Location:** `output/a2-alignment/<tag>.events.jsonl`

One JSON object per line. Words and pen movements are interleaved in
time order — this is the file you feed to the sequence model.

### Three event types

#### Word event
```json
{"type": "word", "t": 1.46, "t_end": 1.66, "word": "this"}
```
| Field | Description |
|---|---|
| `type` | `"word"` |
| `t` | Time the word started being spoken (seconds) |
| `t_end` | Time the word finished being spoken (seconds) |
| `word` | The spoken word (raw Whisper output, may include punctuation) |

#### Pen event
```json
{
  "type": "pen",
  "t": 6.23,
  "x": 321.0,
  "y": 119.0,
  "stroke_id": 0,
  "first_in_stroke": true,
  "last_in_stroke": false
}
```
| Field | Description |
|---|---|
| `type` | `"pen"` |
| `t` | Time this pen point was drawn (seconds) |
| `x`, `y` | Position in pixels (same coordinate system as strokes.jsonl) |
| `stroke_id` | Which stroke this point belongs to (matches stroke_id in strokes.jsonl) |
| `first_in_stroke` | True if this is the first point of a stroke (pen touched board) |
| `last_in_stroke` | True if this is the last point of a stroke (pen lifted) |

#### Page break event
```json
{"type": "page_break", "t": 48.43, "kind": "clear"}
```
| Field | Description |
|---|---|
| `type` | `"page_break"` |
| `t` | Time the board was cleared (seconds) |
| `kind` | Always `"clear"` for now |

### How to read a training sequence

A 5-second window from events.jsonl looks like:

```
word("In")          t=0.84
word("this")        t=1.46
word("video,")      t=1.66
word("we're")       t=2.10
word("going")       t=2.26
word("to")          t=2.38
pen(x=321, y=119, stroke_id=0, first=True)   t=6.23  ← pen touches board
pen(x=327, y=115, stroke_id=0)               t=6.23
pen(x=348, y=111, stroke_id=0)               t=6.33
pen(x=351, y=114, stroke_id=0, last=True)    t=6.60  ← pen lifts
```

This is the natural training sequence: the model reads events left to
right and learns to predict what comes next.

---

## 3. pages.jsonl — page break timestamps

**Location:** `output/<tag>.pages.jsonl`

One JSON object per page break. Used by `reconstruct.py` for videos
when the `page` field is absent from strokes.

```json
{"t": 48.43, "kind": "clear"}
{"t": 87.10, "kind": "clear"}
```

> **Note:** For v4 strokes (which have the `page` field), this file
> is not needed for reconstruction or training. The `page` field
> in strokes.jsonl is the authoritative source of page boundaries.

---

## 4. words.jsonl — raw speech transcript

**Location:** `output/a2-alignment/<tag>.words.jsonl`

One word per line with timestamps. Intermediate file — `events.jsonl`
is the merged version you should use for training.

```json
{"word": "In", "t": 0.84, "t_end": 1.46}
{"word": "this", "t": 1.46, "t_end": 1.66}
```

---

## Quick reference — which file to use for what

| Task | File |
|---|---|
| Training the sequence model | `events.jsonl` |
| Training stroke shape encoder | `strokes.jsonl` (points only) |
| Knowing which problem each stroke belongs to | `strokes.jsonl` → `page` field |
| Reconstruction / visualization | `strokes.jsonl` + `pages.jsonl` |
| Audio transcript only | `words.jsonl` |

---

## Important notes for training code

1. **Always group by `page`** before building training sequences.
   Strokes on different pages are different problems.

2. **The trigger sentence** is the first few words of each page's
   speech (usually the first 5–10 word events before the first pen
   event on that page). This is what you give the model at inference
   time to start generation.

3. **Coordinates are absolute pixels** (1280×720). Normalize to
   [0, 1] before feeding to a model:
   `x_norm = x / 1280`, `y_norm = y / 720`

4. **Timestamps are seconds from the start of the video**, not
   from the start of the page. If you want page-relative time,
   subtract the `t_start` of the first stroke on that page.

5. **Point timestamps within one stroke are nearly identical**
   (spread over < 0.5 seconds). Don't treat them as independent
   events — they are all part of one pen gesture.
