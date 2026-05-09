"""Visualize extracted strokes overlaid on the original frames.

Outputs an MP4 to output/<tag>.viz.mp4 — open it and eyeball whether
the extracted strokes match the teacher's actual writing. This is the
qualitative gate before any formal metrics.

Each stroke is drawn in a distinct color, growing as time advances.
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


def color_for(stroke_id: int) -> tuple[int, int, int]:
    # cycle through a fixed palette in BGR
    palette = [
        (0, 0, 255),    # red
        (0, 255, 255),  # yellow
        (0, 255, 0),    # green
        (255, 255, 0),  # cyan
        (255, 0, 255),  # magenta
        (255, 128, 0),  # blue-orange
    ]
    return palette[stroke_id % len(palette)]


def load_strokes(tag: str) -> list[dict]:
    p = OUTPUT / f"{tag}.strokes.jsonl"
    if not p.exists():
        sys.exit(f"no strokes at {p}. run extract.py {tag} first.")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def build(tag: str, fps: int = 30) -> None:
    frame_paths = sorted((FRAMES / tag).glob("*.png"))
    if not frame_paths:
        sys.exit(f"no frames for {tag}")
    strokes = load_strokes(tag)

    sample0 = cv2.imread(str(frame_paths[0]))
    h, w = sample0.shape[:2]
    out_path = OUTPUT / f"{tag}.viz.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    # for each stroke, list of (t, x, y, color)
    points_per_stroke: list[tuple[int, list]] = [
        (s["stroke_id"], [(p["t"], int(p["x"]), int(p["y"])) for p in s["points"]])
        for s in strokes
    ]

    for i, fp in enumerate(frame_paths):
        t = i / fps
        frame = cv2.imread(str(fp))
        # overlay strokes that have started by time t
        for sid, pts in points_per_stroke:
            shown = [(x, y) for tt, x, y in pts if tt <= t]
            if len(shown) >= 2:
                col = color_for(sid)
                for j in range(1, len(shown)):
                    cv2.line(frame, shown[j - 1], shown[j], col, 2,
                             cv2.LINE_AA)
            elif len(shown) == 1:
                x, y = shown[0]
                cv2.circle(frame, (x, y), 3, color_for(sid), -1)
        cv2.putText(frame, f"t={t:5.2f}s  strokes={sum(1 for sid,pts in points_per_stroke if pts and pts[0][0] <= t)}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    print(f"[{tag}] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    build(args.tag, fps=args.fps)


if __name__ == "__main__":
    main()
