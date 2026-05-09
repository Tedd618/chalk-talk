"""Naive stroke extractor: frame-diff + simple cursor mask + vectorize.

This is a v0 — deliberately simple — so we can eyeball whether the
basic approach is viable on KA and OCT before investing in SAM2 +
proper hand masking. Outputs one stroke per JSONL line:

    {"stroke_id": 7, "t_start": 1.234, "t_end": 1.401,
     "points": [{"x":..., "y":..., "t":...}, ...]}

A "stroke" here = a continuous run of frames where new ink appeared
in roughly-connected locations. Stroke breaks are detected by frames
with no new ink for > GAP_MS, or by spatial jumps.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
FRAMES = ROOT / "frames"
OUTPUT = ROOT / "output"

# tuning knobs (v0 defaults — tune by eye)
DIFF_THRESHOLD = 30        # pixel value diff to count as "new ink"
MIN_PIXELS_PER_FRAME = 3   # need at least N changed pixels to count as drawing
GAP_MS = 250               # > this gap with no drawing -> end of stroke
JUMP_PX = 60               # > this spatial jump -> end of stroke
SMOOTH_KERNEL = 3          # blur before diff to reduce codec noise


def load_frames(tag: str) -> list[Path]:
    d = FRAMES / tag
    if not d.exists():
        sys.exit(f"no frames at {d}. run frames.py {tag} first.")
    return sorted(d.glob("*.png"))


def frame_time(idx: int, fps: int = 30) -> float:
    return idx / fps


def diff_mask(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Return a boolean mask of pixels that got *darker* (new ink on light bg)
    or *lighter* (new ink on dark bg). KA is dark bg + light strokes;
    OCT can be either. So we take absolute diff above threshold."""
    p = cv2.GaussianBlur(prev, (SMOOTH_KERNEL, SMOOTH_KERNEL), 0)
    c = cv2.GaussianBlur(cur,  (SMOOTH_KERNEL, SMOOTH_KERNEL), 0)
    d = cv2.absdiff(c, p)
    if d.ndim == 3:
        d = d.max(axis=2)
    return d > DIFF_THRESHOLD


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) < MIN_PIXELS_PER_FRAME:
        return None
    return float(xs.mean()), float(ys.mean())


def extract(tag: str, fps: int = 30) -> None:
    frame_paths = load_frames(tag)
    if len(frame_paths) < 2:
        sys.exit(f"need at least 2 frames, got {len(frame_paths)}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / f"{tag}.strokes.jsonl"
    out = out_path.open("w")

    prev = cv2.imread(str(frame_paths[0]))
    cur_stroke: list[dict] = []
    last_t: float | None = None
    last_xy: tuple[float, float] | None = None
    stroke_id = 0
    n_active = 0
    n_total = len(frame_paths) - 1

    def flush():
        nonlocal cur_stroke, stroke_id
        if len(cur_stroke) >= 2:
            obj = {
                "stroke_id": stroke_id,
                "t_start": cur_stroke[0]["t"],
                "t_end":   cur_stroke[-1]["t"],
                "points":  cur_stroke,
            }
            out.write(json.dumps(obj) + "\n")
            stroke_id += 1
        cur_stroke = []

    for i, path in enumerate(frame_paths[1:], start=1):
        cur = cv2.imread(str(path))
        if cur is None or cur.shape != prev.shape:
            prev = cur
            continue
        m = diff_mask(prev, cur)
        c = centroid(m)
        t = frame_time(i, fps)

        if c is None:
            # no drawing in this frame
            if last_t is not None and (t - last_t) * 1000 > GAP_MS:
                flush()
                last_t = None
                last_xy = None
        else:
            x, y = c
            if last_xy is not None:
                dx, dy = x - last_xy[0], y - last_xy[1]
                if (dx * dx + dy * dy) ** 0.5 > JUMP_PX:
                    flush()
            cur_stroke.append({"x": round(x, 2), "y": round(y, 2),
                               "t": round(t, 4)})
            last_t = t
            last_xy = (x, y)
            n_active += 1
        prev = cur

    flush()
    out.close()
    print(f"[{tag}] {stroke_id} strokes from {n_active}/{n_total} active frames "
          f"-> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    extract(args.tag, fps=args.fps)


if __name__ == "__main__":
    main()
