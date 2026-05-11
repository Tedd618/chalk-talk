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

**v4 skeleton pipeline shipped.** Complete rewrite from centroid-tracking
(v3) to skeleton-based extraction. Instead of tracking where the pen IS
each frame, v4 records when each pixel BECAME ink, then skeletonizes
the ink shapes and traces them as stroke polylines.

**Key improvements over v3:**
- **Per-page processing** — detects page breaks automatically and extracts
  each page independently. No ghost strokes from previous pages.
- **Temporal-cluster skeletonization** — strokes that cross (e.g. "1"
  over a fraction bar) are skeletonized separately by when they were
  drawn, preventing fragmentation.
- **Max-channel color capture** — uses max(R,G,B) instead of grayscale,
  so colored annotations (arrows, boxes, highlights) are preserved.
- **10fps subsampling** — reads every 3rd frame for 2.5× speedup with
  no quality loss.
- **~1.5 min for 10 min of video** (was ~4 min before subsampling).

**End-to-end pipeline (`pipeline.py`)** — single command from YouTube
URL to training-ready events.jsonl.

Per-source policy: **OCT-style for v1**, KA deferred.

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
- **v4 skeleton rewrite** — completely new approach. Instead of
  tracking the pen-tip centroid frame-by-frame (v3), v4 builds a
  temporal pixel map (when did each pixel first become persistent ink),
  skeletonizes the ink shapes to get 1px-wide centerlines, and traces
  those as stroke polylines. Much denser, more shape-accurate strokes.
- **Max-channel brightness** — `max(R, G, B)` instead of grayscale
  conversion. Colored annotations (blue arrows, red boxes) that were
  invisible in grayscale are now captured as solid strokes.
- **Temporal-cluster skeletonization** — the key insight. Skeletonizing
  ALL ink at once creates junction artifacts wherever strokes cross
  (e.g. a "1" crossing a fraction bar splits the bar into fragments).
  Fix: within each connected component of ink, group pixels by when
  they were drawn, and skeletonize each temporal cluster independently.
  Strokes drawn at different times never share junctions.
- **Per-page temporal maps** — in a 10-min video, the tutor clears the
  board multiple times. Without page awareness, old ink from page 1
  pollutes page 5's extraction (ghost strokes). Fix: detect page breaks
  during the temporal-map build (sharp drop in persistent pixel count)
  and snapshot + reset. Each page gets a completely fresh extraction.
- **10fps subsampling** — Phase 1 (building the temporal map) is 95%
  imread time. Reading every 3rd frame (10fps instead of 30fps) gives
  2.5× speedup with no quality difference. Temporal precision goes
  from ±33ms to ±100ms per stroke — plenty for training data.

## Current baseline

```
extract_v4.py settings:
  INK_THRESHOLD = 28
  LOOKAHEAD = 30 (3 sec at 10fps)
  BASELINE_FRAMES = 20
  FRAME_STEP = 3 (30fps → 10fps subsampling)
  TEMPORAL_CLUSTER_GAP = 8
  TEMPORAL_GAP_FRAMES = 10
  RDP_EPSILON = 0.8
  MIN_STROKE_POINTS = 2
  MIN_STROKE_DISTANCE = 5.0
  PAGE_BREAK_DROP = 0.4
```

Verified on:
- `oct` (Pythagorean, 3 min): 2 pages, 153 strokes, 28s
- `oct-fractions` (Adding Fractions, 10 min): 8 pages, 767 strokes, 94s

Reconstruction matches original at all tested timestamps.
Colored annotations (red boxes, blue arrows, green lines) captured.
No ghost strokes across page breaks.
