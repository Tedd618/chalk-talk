"""Visualize the merged event stream: animated strokes (as in a1's
reconstruct) plus a scrolling caption that highlights the word being
spoken at each moment.

This is the eyeball check for A.2: do the words and strokes sync?
When the teacher says "five squared", does the corresponding `5²` get
written in the same window?

Output: output/<tag>.events.mp4
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

A2_DIR = Path(__file__).resolve().parent
A1_DIR = A2_DIR.parent / "a1-stroke-extraction"
A1_FRAMES = A1_DIR / "frames"
OUTPUT = A2_DIR / "output"

PALETTE = [
    (0, 0, 255), (0, 255, 255), (0, 255, 0),
    (255, 200, 0), (255, 0, 255), (255, 128, 0),
]


def chaikin_smooth(points: list[tuple[float, float]],
                   iterations: int = 2) -> list[tuple[float, float]]:
    pts = list(points)
    if len(pts) < 3:
        return pts
    for _ in range(iterations):
        new_pts: list[tuple[float, float]] = [pts[0]]
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            new_pts.append((0.75 * p0[0] + 0.25 * p1[0],
                            0.75 * p0[1] + 0.25 * p1[1]))
            new_pts.append((0.25 * p0[0] + 0.75 * p1[0],
                            0.25 * p0[1] + 0.75 * p1[1]))
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


def color_for(stroke_id: int) -> tuple[int, int, int]:
    return PALETTE[stroke_id % len(PALETTE)]


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        sys.exit(f"missing: {p}")
    return [json.loads(line) for line in p.read_text().splitlines()
            if line.strip()]


def build(tag: str, fps: int = 30,
          caption_window: float = 4.0,
          tail_seconds: float = 1.0) -> None:
    events = load_jsonl(OUTPUT / f"{tag}.events.jsonl")
    if not events:
        sys.exit(f"no events for {tag}")

    # canvas size from a1 frames
    sample_paths = sorted((A1_FRAMES / tag).glob("*.png"))
    if not sample_paths:
        sys.exit(f"no a1 frames for {tag} — need one for canvas size")
    sample = cv2.imread(str(sample_paths[0]))
    h, w = sample.shape[:2]

    # Pre-organize: per-stroke list of (t, x, y), plus all word events.
    strokes_by_id: dict[int, list[tuple[float, float, float]]] = {}
    words: list[tuple[float, float, str]] = []
    for e in events:
        if e["type"] == "pen":
            strokes_by_id.setdefault(e["stroke_id"], []).append(
                (e["t"], float(e["x"]), float(e["y"])))
        elif e["type"] == "word":
            words.append((e["t"], e["t_end"], e["word"]))

    max_t = max((max(p[0] for p in pts) for pts in strokes_by_id.values()),
                default=0.0)
    max_t = max(max_t, max((w[1] for w in words), default=0.0))
    n_frames = int((max_t + tail_seconds) * fps)

    out_path = OUTPUT / f"{tag}.events.mp4"
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    for i in range(n_frames):
        t = i / fps
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # strokes (smoothed)
        for sid, pts in strokes_by_id.items():
            shown_xy = [(x, y) for tt, x, y in pts if tt <= t]
            if not shown_xy:
                continue
            if len(shown_xy) >= 3:
                smoothed = chaikin_smooth(shown_xy, iterations=2)
            else:
                smoothed = shown_xy
            poly = np.array([[int(round(x)), int(round(y))]
                             for x, y in smoothed], dtype=np.int32)
            if len(poly) >= 2:
                cv2.polylines(canvas, [poly], isClosed=False,
                              color=color_for(sid), thickness=2,
                              lineType=cv2.LINE_AA)
            else:
                cv2.circle(canvas, (poly[0][0], poly[0][1]), 2,
                           color_for(sid), -1)

        # caption — words spoken in the last `caption_window` seconds
        recent = [(ts, te, ww) for ts, te, ww in words
                  if ts <= t and t - ts < caption_window]
        if recent:
            line = " ".join(ww for _, _, ww in recent[-12:])
            # active word (currently being spoken) highlighted at the end
            active = [ww for ts, te, ww in words if ts <= t <= te]
            line_text = line
            cv2.rectangle(canvas, (0, h - 40), (w, h),
                          (32, 32, 32), -1)
            cv2.putText(canvas, line_text, (8, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (240, 240, 240), 1, cv2.LINE_AA)
            if active:
                cv2.putText(canvas, "▸ " + active[-1], (8, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (90, 230, 255), 2, cv2.LINE_AA)

        cv2.putText(canvas, f"t={t:5.2f}s", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (180, 180, 180), 1, cv2.LINE_AA)
        writer.write(canvas)

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
