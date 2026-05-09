"""Replay extracted strokes on a blank canvas in their recorded timing.

This is the honest validation: if the reconstruction looks like the
original writing, the strokes are good training data. If it looks like
gibberish, the overlay viz was hiding extraction gaps.

Output: output/<tag>.recon.mp4 — black background, strokes drawn at
their recorded times. Optional --color-per-stroke to see segmentation.

Usage:
    python3 reconstruct.py ka-pythagoras
    python3 reconstruct.py ka-pythagoras --color-per-stroke
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

PALETTE = [
    (0, 0, 255), (0, 255, 255), (0, 255, 0),
    (255, 200, 0), (255, 0, 255), (255, 128, 0),
]


def color_for(stroke_id: int, mode: str) -> tuple[int, int, int]:
    if mode == "single":
        return (255, 255, 255)
    return PALETTE[stroke_id % len(PALETTE)]


def chaikin_smooth(points: list[tuple[float, float]],
                   iterations: int = 2) -> list[tuple[float, float]]:
    """Chaikin's corner-cutting algorithm: smoothly approximates a
    quadratic B-spline through the input polyline. Each iteration
    inserts two points at 25% and 75% of every segment, dropping the
    sharp corner at each interior vertex.

    Two iterations are usually enough for hand-drawn aesthetics."""
    pts = list(points)
    if len(pts) < 3:
        return pts
    for _ in range(iterations):
        new_pts: list[tuple[float, float]] = [pts[0]]
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            q = (0.75 * p0[0] + 0.25 * p1[0],
                 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0],
                 0.25 * p0[1] + 0.75 * p1[1])
            new_pts.append(q)
            new_pts.append(r)
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


def reconstruct(tag: str, fps: int = 30, color_mode: str = "single",
                tail_seconds: float = 1.0) -> None:
    strokes_path = OUTPUT / f"{tag}.strokes.jsonl"
    if not strokes_path.exists():
        sys.exit(f"missing: {strokes_path}. run extract.py first.")

    # canvas size from any frame
    sample_paths = sorted((FRAMES / tag).glob("*.png"))
    if not sample_paths:
        sys.exit(f"no frames for {tag} — need one for canvas size")
    sample = cv2.imread(str(sample_paths[0]))
    h, w = sample.shape[:2]

    strokes = [json.loads(line) for line in strokes_path.read_text().splitlines()
               if line.strip()]
    if not strokes:
        sys.exit(f"no strokes in {strokes_path}")

    max_t = max(s["t_end"] for s in strokes)
    n_frames = int((max_t + tail_seconds) * fps)

    out_path = OUTPUT / f"{tag}.recon.mp4"
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    # Pre-build per-stroke (t, x, y) arrays for speed
    stroke_arrays: list[tuple[int, list[tuple[float, float, float]]]] = [
        (s["stroke_id"],
         [(p["t"], float(p["x"]), float(p["y"])) for p in s["points"]])
        for s in strokes
    ]

    for i in range(n_frames):
        t = i / fps
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        for sid, pts in stroke_arrays:
            if not pts or pts[0][0] > t:
                continue
            shown_xy = [(x, y) for tt, x, y in pts if tt <= t]
            if len(shown_xy) < 2:
                if len(shown_xy) == 1:
                    cv2.circle(canvas, (int(shown_xy[0][0]), int(shown_xy[0][1])),
                               2, color_for(sid, color_mode), -1)
                continue
            # smooth via Chaikin only if stroke has enough sample points
            if len(shown_xy) >= 3:
                smoothed = chaikin_smooth(shown_xy, iterations=2)
            else:
                smoothed = shown_xy
            poly = np.array([[int(round(x)), int(round(y))]
                             for x, y in smoothed], dtype=np.int32)
            col = color_for(sid, color_mode)
            cv2.polylines(canvas, [poly], isClosed=False,
                          color=col, thickness=2, lineType=cv2.LINE_AA)

        cv2.putText(
            canvas, f"reconstruction  t={t:5.2f}s",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (180, 180, 180), 1, cv2.LINE_AA,
        )
        writer.write(canvas)

    writer.release()
    print(f"[{tag}] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--color-per-stroke", action="store_true",
                    help="One color per stroke (default: all white)")
    args = ap.parse_args()
    reconstruct(args.tag, fps=args.fps,
                color_mode="palette" if args.color_per_stroke else "single")


if __name__ == "__main__":
    main()
