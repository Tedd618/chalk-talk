"""Detect 'clear page' events from sudden drops in inked-pixel count.

A tutor working on a tablet writes a problem, solves it, then erases or
moves to a fresh page before the next problem. Without modeling this,
our reconstruction accumulates every stroke ever drawn into one
overlapping mess. We treat each clear-page moment as a `page_break`
event so downstream tools (and eventually the model) can reset the
canvas at that point.

Detection: walk through frames, compute the count of pixels that
differ from baseline (the inked-pixel count). When that count drops
to less than DROP_RATIO of its value LOOKBACK_FRAMES ago — and was
above a substantial floor before — we record a page_break.

Output: output/<tag>.pages.jsonl
        one JSON per line: {"t": <seconds>, "kind": "clear",
                            "ink_before": <int>, "ink_after": <int>}
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

INK_THRESHOLD = 28
BASELINE_FRAMES = 60
LOOKBACK_FRAMES = 30           # 1 sec @ 30 fps
DROP_RATIO = 0.30              # current ink < this fraction of prev ink → break
MIN_PREV_INK = 1500            # only detect drops from a substantial canvas
MIN_GAP_BETWEEN_BREAKS_SEC = 4.0


def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def estimate_baseline(paths: list[Path]) -> np.ndarray:
    stack = [to_gray(cv2.imread(str(p))) for p in paths[:BASELINE_FRAMES]]
    return np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)


def detect(tag: str, fps: int = 30) -> None:
    paths = sorted((FRAMES / tag).glob("*.png"))
    if len(paths) < BASELINE_FRAMES + LOOKBACK_FRAMES + 2:
        sys.exit(f"need at least {BASELINE_FRAMES + LOOKBACK_FRAMES + 2} frames")
    print(f"[{tag}] estimating baseline from first {BASELINE_FRAMES} frames")
    baseline = estimate_baseline(paths)

    # streaming bright-pixel count per frame
    bright_counts: list[int] = []
    for p in paths:
        g = to_gray(cv2.imread(str(p)))
        bright = cv2.absdiff(g, baseline) > INK_THRESHOLD
        bright_counts.append(int(bright.sum()))

    breaks: list[dict] = []
    last_break_frame = -10**9
    for i in range(LOOKBACK_FRAMES, len(bright_counts)):
        prev = bright_counts[i - LOOKBACK_FRAMES]
        cur = bright_counts[i]
        if prev < MIN_PREV_INK:
            continue
        if cur < DROP_RATIO * prev:
            if (i - last_break_frame) / fps < MIN_GAP_BETWEEN_BREAKS_SEC:
                continue
            breaks.append({
                "t": round(i / fps, 4),
                "kind": "clear",
                "ink_before": prev,
                "ink_after": cur,
            })
            last_break_frame = i

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / f"{tag}.pages.jsonl"
    with out_path.open("w") as f:
        for b in breaks:
            f.write(json.dumps(b) + "\n")
    print(f"[{tag}] {len(breaks)} page breaks -> {out_path}")


def main() -> None:
    global DROP_RATIO, MIN_PREV_INK, MIN_GAP_BETWEEN_BREAKS_SEC
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--drop-ratio", type=float, default=DROP_RATIO,
                    help="Current ink < this fraction of previous → break "
                         "(default %(default)s)")
    ap.add_argument("--min-prev-ink", type=int, default=MIN_PREV_INK,
                    help="Floor for previous ink (avoids false breaks early)")
    ap.add_argument("--min-gap-sec", type=float, default=MIN_GAP_BETWEEN_BREAKS_SEC,
                    help="Minimum seconds between consecutive breaks")
    args = ap.parse_args()
    DROP_RATIO = args.drop_ratio
    MIN_PREV_INK = args.min_prev_ink
    MIN_GAP_BETWEEN_BREAKS_SEC = args.min_gap_sec
    detect(args.tag, fps=args.fps)


if __name__ == "__main__":
    main()
