"""Stroke extractor v3: persistence pen-tip + cusp-based segmentation.

Per frame T:
  baseline       = median of first BASELINE_FRAMES frames
  ink_now        = pixel differs from baseline by > INK_THRESHOLD
  ink_future     = same at T + LOOKAHEAD
  persistent(T)  = ink_now & ink_future
  new_persist    = persistent(T) & ~persistent(T-1)
  pen_tip(T)     = centroid of new_persist  (None if too few pixels)
  pen_down(T)    = pen_tip(T) is not None  (sticky for STICKY_FRAMES)

Stroke segmentation: collect maximal pen_down runs, then split each
run only at sharp direction changes (cusps). No Bezier-fit error
cutting; smooth curves stay as a single stroke.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
FRAMES = ROOT / "frames"
OUTPUT = ROOT / "output"

# pen-tip extraction
INK_THRESHOLD = 28
LOOKAHEAD = 90              # 3 sec — real ink survives, hovering cursors don't
BASELINE_FRAMES = 60
MIN_NEW_INK_PIXELS = 1     # capture tiny letters / superscripts
STICKY_FRAMES = 4

# continuation check: line A→B must lie on existing ink to be a real
# continuation; otherwise it's a fake bridge across blank canvas
# (sticky linked two unrelated strokes).
INKED_PATH_RATIO = 0.4      # min fraction of path samples that must be inked
CONTINUATION_DIST_FREE = 5  # paths shorter than this skip the check

# cusp-based segmentation
CUSP_WINDOW = 4
CUSP_ANGLE_DEG = 110
CUSP_MIN_SEPARATION = 6
MIN_STROKE_PTS = 3


def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def estimate_baseline(paths: list[Path]) -> np.ndarray:
    stack = [to_gray(cv2.imread(str(p))) for p in paths[:BASELINE_FRAMES]]
    return np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)


def ink_mask(path: Path, baseline: np.ndarray) -> np.ndarray:
    g = to_gray(cv2.imread(str(path)))
    return cv2.absdiff(g, baseline) > INK_THRESHOLD


def centroid_of(mask: np.ndarray, min_pixels: int) -> tuple[float, float] | None:
    if mask.sum() < min_pixels:
        return None
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def is_real_continuation(persistent_mask: np.ndarray,
                         p1: tuple[float, float],
                         p2: tuple[float, float]) -> bool:
    """True iff the straight line from p1 to p2 lies on existing ink in
    `persistent_mask` (i.e., the user actually drew between these
    points). False means it's a fake bridge across blank canvas."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < CONTINUATION_DIST_FREE:
        return True
    H, W = persistent_mask.shape
    n_samples = min(max(int(dist), 5), 100)
    sampled = 0
    inked = 0
    for i in range(1, n_samples):  # skip exact endpoints
        t = i / n_samples
        x = int(p1[0] + dx * t)
        y = int(p1[1] + dy * t)
        if 0 <= x < W and 0 <= y < H:
            sampled += 1
            if persistent_mask[y, x]:
                inked += 1
    if sampled == 0:
        return True
    return (inked / sampled) >= INKED_PATH_RATIO


def detect_cusps(pts: np.ndarray,
                 window: int = CUSP_WINDOW,
                 angle_thresh_deg: float = CUSP_ANGLE_DEG) -> list[int]:
    n = len(pts)
    if n < 2 * window + 1:
        return []
    cusps: list[int] = []
    thresh = np.radians(angle_thresh_deg)
    for i in range(window, n - window):
        v_in = pts[i] - pts[i - window]
        v_out = pts[i + window] - pts[i]
        n_in = float(np.linalg.norm(v_in))
        n_out = float(np.linalg.norm(v_out))
        if n_in < 1.0 or n_out < 1.0:
            continue
        cos_a = float(np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0))
        if np.arccos(cos_a) > thresh:
            cusps.append(i)
    return cusps


def dedup_close(indices: list[int], min_sep: int = CUSP_MIN_SEPARATION) -> list[int]:
    if not indices:
        return []
    out = [indices[0]]
    for x in indices[1:]:
        if x - out[-1] >= min_sep:
            out.append(x)
    return out


def cusp_segment(points: list[dict]) -> list[list[dict]]:
    n = len(points)
    if n < MIN_STROKE_PTS:
        return []
    pts_arr = np.asarray([(p["x"], p["y"]) for p in points], dtype=float)
    cusps = dedup_close(detect_cusps(pts_arr))
    if not cusps:
        return [points]
    segments: list[list[dict]] = []
    start = 0
    for c in cusps:
        if c - start >= MIN_STROKE_PTS:
            segments.append(points[start:c])
        start = c
    if n - start >= MIN_STROKE_PTS:
        segments.append(points[start:])
    return segments


def extract(tag: str, fps: int = 30) -> None:
    paths = sorted((FRAMES / tag).glob("*.png"))
    if len(paths) < BASELINE_FRAMES + LOOKAHEAD + 2:
        sys.exit(f"need at least {BASELINE_FRAMES + LOOKAHEAD + 2} frames")
    print(f"[{tag}] estimating baseline from first {BASELINE_FRAMES} frames")
    baseline = estimate_baseline(paths)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    strokes_path = OUTPUT / f"{tag}.strokes.jsonl"
    trace_path = OUTPUT / f"{tag}.trace.jsonl"

    masks: deque[np.ndarray] = deque(maxlen=LOOKAHEAD + 1)
    persistent_prev: np.ndarray | None = None

    pen_down_runs: list[list[dict]] = []
    cur_run: list[dict] = []
    trace: list[dict] = []
    in_pen_down = False
    inactive_count = 0
    n_signal = 0
    n_processed = 0

    def commit_run() -> None:
        nonlocal cur_run
        if len(cur_run) >= MIN_STROKE_PTS:
            pen_down_runs.append(cur_run)
        cur_run = []

    for i, path in enumerate(paths):
        masks.append(ink_mask(path, baseline))
        if len(masks) < LOOKAHEAD + 1:
            continue

        decision_idx = i - LOOKAHEAD
        ink_now = masks[0]
        ink_future = masks[-1]
        persistent = ink_now & ink_future
        if persistent_prev is None:
            new_persist = persistent.copy()
        else:
            new_persist = persistent & ~persistent_prev

        c = centroid_of(new_persist, MIN_NEW_INK_PIXELS)
        t = decision_idx / fps
        signal = c is not None

        if signal:
            cx, cy = c
            # If the line from the last recorded point to here goes across
            # blank canvas (no ink along the path), this is a fake bridge —
            # two separate strokes that fell within the sticky window.
            # Commit the current run and start a new one.
            if cur_run:
                last = cur_run[-1]
                if not is_real_continuation(
                    persistent, (last["x"], last["y"]), (cx, cy)
                ):
                    commit_run()
            if not in_pen_down:
                in_pen_down = True
            inactive_count = 0
            cur_run.append({"x": round(cx, 2), "y": round(cy, 2),
                            "t": round(t, 4)})
            n_signal += 1
        elif in_pen_down:
            inactive_count += 1
            if inactive_count > STICKY_FRAMES:
                in_pen_down = False
                inactive_count = 0
                commit_run()

        trace.append({
            "t": round(t, 4),
            "cursor": [round(c[0], 2), round(c[1], 2)] if c else None,
            "pen_down": in_pen_down,
        })

        persistent_prev = persistent
        n_processed += 1

    if in_pen_down:
        commit_run()

    strokes: list[dict] = []
    for run in pen_down_runs:
        for seg in cusp_segment(run):
            strokes.append({
                "stroke_id": len(strokes),
                "t_start": seg[0]["t"],
                "t_end":   seg[-1]["t"],
                "points":  seg,
            })

    with strokes_path.open("w") as f:
        for s in strokes:
            f.write(json.dumps(s) + "\n")
    with trace_path.open("w") as f:
        for tt in trace:
            f.write(json.dumps(tt) + "\n")

    print(f"[{tag}] {len(strokes)} strokes "
          f"(from {len(pen_down_runs)} pen-down runs); "
          f"raw signal on {n_signal}/{n_processed} frames "
          f"-> {strokes_path.name}")


def main() -> None:
    global CUSP_ANGLE_DEG
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cusp-angle", type=float, default=CUSP_ANGLE_DEG,
                    help="Cut where trajectory turns more than this many degrees")
    args = ap.parse_args()
    CUSP_ANGLE_DEG = args.cusp_angle
    extract(args.tag, fps=args.fps)


if __name__ == "__main__":
    main()
