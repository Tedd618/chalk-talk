"""Visualize cursor track + extracted strokes overlaid on original frames.

Two layers:
  - Faint cyan dot at every detected cursor position (cursor track,
    regardless of pen-state). Lets you see whether cursor detection
    is following the pen reliably.
  - Bright per-stroke colored polylines for pen-down sequences. These
    are the actual training-signal strokes.

Output: output/<tag>.viz.mp4
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
FRAMES = ROOT / "frames"
OUTPUT = ROOT / "output"


def color_for(stroke_id: int) -> tuple[int, int, int]:
    palette = [
        (0, 0, 255),    # red
        (0, 255, 255),  # yellow
        (0, 255, 0),    # green
        (255, 200, 0),  # cyan-blue
        (255, 0, 255),  # magenta
        (255, 128, 0),  # orange
    ]
    return palette[stroke_id % len(palette)]


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        sys.exit(f"missing: {p}. run extract.py first.")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def build(tag: str, fps: int = 30) -> None:
    frame_paths = sorted((FRAMES / tag).glob("*.png"))
    if not frame_paths:
        sys.exit(f"no frames for {tag}")
    strokes = load_jsonl(OUTPUT / f"{tag}.strokes.jsonl")
    trace = load_jsonl(OUTPUT / f"{tag}.trace.jsonl")

    # build per-time lookup for fast access
    cursor_by_t: dict[float, tuple[int, int] | None] = {}
    pen_by_t: dict[float, bool] = {}
    for r in trace:
        t = r["t"]
        c = r["cursor"]
        cursor_by_t[t] = (int(c[0]), int(c[1])) if c else None
        pen_by_t[t] = bool(r["pen_down"])

    sample0 = cv2.imread(str(frame_paths[0]))
    h, w = sample0.shape[:2]
    out_path = OUTPUT / f"{tag}.viz.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    # for each stroke, list of (t, x, y)
    points_per_stroke = [
        (s["stroke_id"], [(p["t"], int(p["x"]), int(p["y"])) for p in s["points"]])
        for s in strokes
    ]

    # rolling cursor trail (last ~1 s) so the cursor track is readable
    TRAIL_LEN = fps  # 1 second

    cursor_trail: list[tuple[int, int]] = []

    for i, fp in enumerate(frame_paths):
        t = round(i / fps, 4)
        frame = cv2.imread(str(fp))

        # cursor trail (faint cyan)
        c = cursor_by_t.get(t)
        if c is not None:
            cursor_trail.append(c)
            if len(cursor_trail) > TRAIL_LEN:
                cursor_trail.pop(0)
        # draw trail
        for j in range(1, len(cursor_trail)):
            cv2.line(frame, cursor_trail[j - 1], cursor_trail[j],
                     (200, 200, 100), 1, cv2.LINE_AA)
        if c is not None:
            radius = 5 if pen_by_t.get(t, False) else 3
            cv2.circle(frame, c, radius, (255, 255, 255), 1, cv2.LINE_AA)

        # strokes (bright)
        active_strokes = 0
        for sid, pts in points_per_stroke:
            shown = [(x, y) for tt, x, y in pts if tt <= t]
            if not shown:
                continue
            active_strokes += 1
            col = color_for(sid)
            if len(shown) >= 2:
                for j in range(1, len(shown)):
                    cv2.line(frame, shown[j - 1], shown[j], col, 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, shown[0], 3, col, -1)

        cv2.putText(
            frame,
            f"t={t:5.2f}s  strokes={active_strokes}  "
            f"pen={'DOWN' if pen_by_t.get(t, False) else 'up'}",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
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
