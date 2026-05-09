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

Pipeline scaffolded; **not yet validated**. The MVP extractor uses
naive frame-diff with a simple cursor mask. Expect to iterate on the
extractor based on what the visualization shows. Formal metrics
(recall / precision / endpoint error) and ground-truth annotation
are deferred until the eyeball test says the basic approach is
viable.

## Decision log

This file gets appended as we iterate. Each entry: date, what was
tried, what we saw, what to try next.

- *(empty — first run pending)*
