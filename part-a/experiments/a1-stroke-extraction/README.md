# A.1 — Stroke Extraction Pilot

Pilot videos:

| Tag | Source | URL |
|---|---|---|
| `ka-pythagoras` | Khan Academy | https://www.youtube.com/watch?v=AA6RfgP-AHU |
| `oct` | Organic Chemistry Tutor | https://www.youtube.com/watch?v=pbf4lcJhIfI |

## Pipeline

```
YouTube URL
    │
    ▼  download.py (yt-dlp)
videos/<tag>.mp4 + videos/<tag>.meta.json (title, description, duration)
    │
    ▼  frames.py (ffmpeg)
frames/<tag>/0001.png ... NNNN.png  (30 fps, downscaled)
    │
    ▼  extract.py (frame diff + cursor mask + vectorize)
output/<tag>.strokes.jsonl  (one stroke per line: list of {x,y,t})
    │
    ▼  viz.py
output/<tag>.viz.mp4  (extracted strokes overlaid on original — eyeball test)
```

## Usage

```bash
# 0. Install tools (one-time)
brew install yt-dlp ffmpeg
pip install -r requirements.txt

# 1. Download both pilot videos
python3 download.py

# 2. Extract frames
python3 frames.py ka-pythagoras
python3 frames.py oct

# 3. Extract strokes
python3 extract.py ka-pythagoras
python3 extract.py oct

# 4. Visualize for eyeball check
python3 viz.py ka-pythagoras
python3 viz.py oct
```

## Status

**v2 pipeline shipped.** Pen-tip = centroid of new persistent ink;
pen-down = sticky-on-signal; segmentation = greedy cubic-Bézier fit
within a max-error budget.

**OCT extracts cleanly. KA does not.** Reconstruction (strokes only,
replayed on a blank canvas via `reconstruct.py`) is recognizable on
the OCT pilot but unreadable on the KA pilot. KA's cursive flow
defeats per-frame pen-tip recovery from new-ink centroids.

Per-source policy locked in: **OCT-style for v1**, KA deferred until
extraction is better. See [../docs/PLAN.md](../docs/PLAN.md) data
sources section.

## Decision log

- **v0** (frame-diff + centroid + gap/jump segmentation): cursor
  motion dominated, every active frame contributed regardless of
  whether ink was being added. KA collapsed to 1 stroke for 30s.
- **v1** (added persistence check: pixel only counts if it's still
  inked LOOKAHEAD frames later → cursor filtered out, real ink
  centroids tracked): cursor problem solved; segmentation under-cut
  KA cursive (5 strokes for "Pythagorean Theorem") and over-cut OCT
  slow lines. Threshold tuning couldn't satisfy both styles.
- **v2** (replaced gap/jump with greedy cubic-Bézier-fit segmentation
  + sticky pen-down to bridge brief detection gaps): KA → 23 strokes,
  OCT → 4. Numbers look right.
- **Reconstruction validation**: OCT readable, KA not. Pivoted v1
  data-curation policy to OCT-style only.
- **Cursor-tracking attempt** (briefly tried): cursor pixels and
  freshly-deposited ink overlap at the pen tip, so the
  "diff-minus-persistent" cursor mask was empty exactly when needed.
  Abandoned for now; flagged for future revisit if KA support
  becomes a priority.
- **v3** (replaced Bézier-fit error with **cusp-based segmentation**:
  only cut a pen-down run at sharp direction changes, not where one
  cubic can't cover the curve; LOOKAHEAD = 90 = 3 sec): on 3-min OCT
  produced 137 strokes from 61 pen-down runs. Visibly better but
  still had residual phantom long lines connecting unrelated regions.
- **v4–v8** (explorations): net pixel-growth detection, shape-based
  component filtering, size-band line filter, dot/cursor-blob filter
  on `new_persist`. None of these matched v3 quality; some were
  significantly worse.
- **v3 + path-verification continuation check** (current baseline):
  the residual phantom long lines came from sticky pen-down bridging
  two physically separate strokes that fell in the same sticky
  window. Fix: before appending point B to a run that ended at A,
  sample the straight line A→B against the current persistent ink
  mask. If ≥ 40% of path samples lie on existing ink, the points are
  a real continuation; otherwise commit the current run and start
  fresh. Principled (uses ground truth, not a distance heuristic),
  decisive on phantom lines, leaves real continuous strokes intact.
- **Tiny-letter capture**: lowered `MIN_NEW_INK_PIXELS` from 2 to 1.
  Slow, careful strokes (superscripts, decimal points) deposit very
  few pixels per frame — the threshold of 2 was filtering them.
- **Reconstruction smoothing**: `reconstruct.py` now applies
  Chaikin's algorithm (two iterations) per stroke. Visually approximates
  a quadratic B-spline; output looks like flowing handwriting rather
  than jagged polylines. Cosmetic only — does not change the data.

## Current baseline

```
extract.py settings (current preferred):
  INK_THRESHOLD = 28
  LOOKAHEAD = 90 (3 sec)
  BASELINE_FRAMES = 60
  MIN_NEW_INK_PIXELS = 1
  STICKY_FRAMES = 4
  INKED_PATH_RATIO = 0.4
  CUSP_ANGLE_DEG = 110
  CUSP_MIN_SEPARATION = 6
  MIN_STROKE_PTS = 3
```

3-min OCT pilot:
- 1172 frames with raw ink signal
- 111 pen-down runs (after path-verification splits)
- 137 final strokes (after cusp segmentation)

Reconstruction is recognizably the original writing.
