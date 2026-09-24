"""Stroke extractor v4: skeleton + temporal-map approach (per-page).

Instead of tracking the pen-tip centroid per frame (v3), this approach:
  1. Records WHEN each pixel first became persistent ink (temporal map)
     - Subsampled at 10fps (every 3rd frame) for 2.5x speedup
     - Automatically detects page breaks (canvas clears) and builds
       a separate temporal map per page, so no ghost strokes leak
       across pages
  2. Skeletonizes each page's ink independently using temporal-cluster
     decomposition (strokes drawn at different times never share
     junctions, preventing fragmentation of fraction bars etc.)
  3. Traces the skeleton graph into ordered paths
  4. Assigns timestamps from the temporal map to each path point
  5. Segments into strokes by temporal gaps

Key design choices:
  - Max-channel brightness (max of R,G,B) instead of grayscale, so
    colored annotations (arrows, boxes, highlights) are captured
  - Per-page processing: each page between canvas clears gets its own
    fresh temporal map → clean extraction with no cross-page contamination
  - Temporal-cluster skeletonization: within each connected component,
    pixels are grouped by when they were drawn; each group is
    skeletonized independently to avoid junction artifacts where
    strokes cross (e.g. a "1" crossing a fraction bar)

Result: dense, shape-accurate (x, y, t) strokes that follow the actual
ink centerline — not the 3-5 point centroid approximations of v3.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent
_scratch = os.environ.get("CHALK_SCRATCH")
FRAMES = Path(_scratch) / "frames" if _scratch else ROOT / "frames"
OUTPUT = ROOT / "output"

# ── Inherited from v3 (persistence detection) ─────────────────────────────────
INK_THRESHOLD = 28
LOOKAHEAD = 30          # 3 sec at 10fps — real ink survives, cursors don't
BASELINE_FRAMES = 20    # 2 sec at 10fps for baseline estimation
FRAME_STEP = 3          # subsample: read every 3rd frame (30fps → 10fps)

# ── New for v4 ────────────────────────────────────────────────────────────────
MORPH_CLOSE_KERNEL = 0      # 0 = skip morphological closing (preserve thin gaps like =)
SPUR_MIN_LENGTH = 8         # prune skeleton branches shorter than this
TEMPORAL_GAP_FRAMES = 10    # ~0.33s gap between points = new stroke (original frame units)
RDP_EPSILON = 0.8           # Douglas-Peucker simplification (pixels) — keep detail
JUNCTION_TIME_THRESH = 10   # max frame gap to chain segments through junction
MIN_STROKE_POINTS = 2       # a 2-point line segment is a valid stroke (e.g. = bar)
MIN_STROKE_DISTANCE = 5.0   # total path length in pixels; shorter = junction noise


# ── Phase 1: Temporal pixel map ───────────────────────────────────────────────

def to_bright(frame: np.ndarray) -> np.ndarray:
    """Max of R,G,B — captures colored ink (arrows, highlights) as well as white."""
    return np.max(frame, axis=2)


def estimate_baseline(paths: list[Path]) -> np.ndarray:
    # Sample BASELINE_FRAMES frames from the start, using FRAME_STEP spacing
    indices = list(range(0, min(BASELINE_FRAMES * FRAME_STEP, len(paths)), FRAME_STEP))
    stack = [to_bright(cv2.imread(str(paths[i]))) for i in indices]
    return np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)


def ink_mask(path: Path, baseline: np.ndarray) -> np.ndarray:
    b = to_bright(cv2.imread(str(path)))
    return cv2.absdiff(b, baseline) > INK_THRESHOLD


PAGE_BREAK_DROP = 0.4   # persistent ink drops below 40% of peak → page break
PAGE_BREAK_COOLDOWN = 20  # ignore new page breaks within 20 subsampled steps


def build_temporal_maps(paths: list[Path], baseline: np.ndarray,
                        start: int = 0, end: int | None = None,
                        ) -> list[np.ndarray]:
    """Build per-page arrival_time maps.

    Reads every FRAME_STEP-th frame (subsampling) for speed.
    Detects page breaks (sharp drops in persistent pixel count) and
    snapshots the current temporal map, then resets for the next page.
    Returns a list of arrival_time arrays, one per page.
    Arrival times are stored as original frame indices (not subsampled).
    """
    if end is None:
        end = len(paths)
    sample = to_bright(cv2.imread(str(paths[0])))
    H, W = sample.shape
    arrival_time = np.zeros((H, W), dtype=np.int32)

    masks: deque[np.ndarray] = deque(maxlen=LOOKAHEAD + 1)
    persistent_prev: np.ndarray | None = None

    pages: list[np.ndarray] = []
    peak_ink = 0
    last_break_step = -PAGE_BREAK_COOLDOWN * 2  # allow first break

    # Subsample: read every FRAME_STEP-th frame
    frame_indices = list(range(start, min(end, len(paths)), FRAME_STEP))

    for step, fi in enumerate(frame_indices):
        masks.append(ink_mask(paths[fi], baseline))
        if len(masks) < LOOKAHEAD + 1:
            continue

        # Map back to original frame index for timestamps
        decision_fi = frame_indices[step - LOOKAHEAD]
        persistent = masks[0] & masks[-1]
        n_persist = int(persistent.sum())

        if persistent_prev is not None:
            new_persist = persistent & ~persistent_prev
        else:
            new_persist = persistent.copy()

        # Track peak ink for page-break detection
        if n_persist > peak_ink:
            peak_ink = n_persist

        # Detect page break: persistent count drops below threshold of peak
        n_stamped = int((arrival_time > 0).sum())
        if (peak_ink > 1000 and n_persist < peak_ink * PAGE_BREAK_DROP
                and n_stamped > 500
                and (step - last_break_step) > PAGE_BREAK_COOLDOWN):
            # Save current page
            pages.append(arrival_time.copy())
            # Reset for next page
            arrival_time = np.zeros((H, W), dtype=np.int32)
            peak_ink = n_persist
            last_break_step = step

        # Stamp newly-persistent pixels with original frame index
        arrival_time[new_persist] = decision_fi

        persistent_prev = persistent

    # Save final page (if it has any ink)
    if (arrival_time > 0).sum() > 0:
        pages.append(arrival_time)

    return pages


# ── Phase 2: Temporal-component skeletonization ─────────────────────────────
#
# The old approach skeletonized the entire ink mask at once. This caused
# junction artifacts wherever strokes cross (e.g. the "1" crossing a
# fraction bar splits the bar into fragments).
#
# New approach: find connected components of ink, split each CC into
# temporal clusters (groups of pixels drawn close in time), and
# skeletonize each cluster independently. Strokes drawn at different
# times never share junctions even if they touch spatially.

TEMPORAL_CLUSTER_GAP = 8  # frames gap within a CC → separate cluster


def skeletonize_ink(arrival_time: np.ndarray) -> np.ndarray:
    """Skeletonize ink per temporal cluster within each connected component.

    Returns boolean skeleton (union of all per-cluster skeletons)."""
    from scipy import ndimage

    ink = (arrival_time > 0).astype(np.uint8)
    H, W = ink.shape
    skeleton_out = np.zeros((H, W), dtype=bool)

    # Find connected components of ink
    labels, n_cc = ndimage.label(ink, structure=np.ones((3, 3)))

    for cc_id in range(1, n_cc + 1):
        cc_mask = (labels == cc_id)
        cc_times = arrival_time[cc_mask]
        cc_times = cc_times[cc_times > 0]

        if len(cc_times) == 0:
            continue

        # Cluster the timestamps within this CC
        unique_times = np.unique(cc_times)
        if len(unique_times) <= 1:
            # Single time → skeletonize as-is
            skel = skeletonize(cc_mask)
            skeleton_out |= skel
            continue

        # Split into temporal clusters by gaps
        sorted_t = np.sort(unique_times)
        gaps = np.diff(sorted_t)
        split_indices = np.where(gaps > TEMPORAL_CLUSTER_GAP)[0]

        if len(split_indices) == 0:
            # All drawn within one burst → skeletonize as-is
            skel = skeletonize(cc_mask)
            skeleton_out |= skel
            continue

        # Build cluster boundaries
        boundaries = [sorted_t[0]]
        for si in split_indices:
            boundaries.append(sorted_t[si])
            boundaries.append(sorted_t[si + 1])
        boundaries.append(sorted_t[-1])

        # Process each temporal cluster
        for i in range(0, len(boundaries), 2):
            t_lo, t_hi = boundaries[i], boundaries[i + 1]
            cluster_mask = cc_mask & (arrival_time >= t_lo) & (arrival_time <= t_hi)

            if cluster_mask.sum() < 4:
                continue

            # Optionally close tiny gaps within the cluster
            if MORPH_CLOSE_KERNEL > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL))
                cluster_ink = cv2.morphologyEx(
                    cluster_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                cluster_mask = cluster_ink > 0

            skel = skeletonize(cluster_mask)
            skeleton_out |= skel

    return skeleton_out


# ── Phase 3: Skeleton graph tracing ──────────────────────────────────────────

def _neighbors8(y: int, x: int, H: int, W: int):
    """Yield 8-connected neighbor coordinates."""
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                yield ny, nx


def classify_skeleton(skeleton: np.ndarray):
    """Return (endpoints, junctions) as boolean masks."""
    skel = skeleton.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    nbr_count = cv2.filter2D(skel, cv2.CV_16U, kernel)
    nbr_count = nbr_count * skel  # only for skeleton pixels

    endpoints = (nbr_count == 1) & skeleton
    junctions = (nbr_count >= 3) & skeleton
    return endpoints, junctions


def trace_paths(skeleton: np.ndarray, endpoints: np.ndarray,
                junctions: np.ndarray) -> list[list[tuple[int, int]]]:
    """Trace all path segments between stop-points (endpoints or junctions).
    Returns list of paths, each a list of (y, x) coordinates."""
    H, W = skeleton.shape
    stop_mask = endpoints | junctions
    visited = np.zeros_like(skeleton, dtype=bool)
    paths: list[list[tuple[int, int]]] = []

    # Collect all start pixels: endpoints first, then junctions
    start_pixels = list(zip(*np.where(endpoints)))
    junction_pixels = list(zip(*np.where(junctions)))

    def walk_from(sy: int, sx: int, ny: int, nx: int) -> list[tuple[int, int]]:
        """Walk from (sy,sx) through (ny,nx) until hitting a stop point
        or dead end. Mark interior pixels as visited."""
        path = [(sy, sx)]
        cy, cx = ny, nx
        while True:
            path.append((cy, cx))
            if stop_mask[cy, cx]:
                # reached another endpoint or junction — don't mark it visited
                # (other paths may also start/end here)
                break
            visited[cy, cx] = True
            # find next unvisited skeleton neighbor
            found = False
            for nny, nnx in _neighbors8(cy, cx, H, W):
                if skeleton[nny, nnx] and not visited[nny, nnx]:
                    cy, cx = nny, nnx
                    found = True
                    break
            if not found:
                break
        return path

    # Walk from each endpoint
    for sy, sx in start_pixels:
        for ny, nx in _neighbors8(sy, sx, H, W):
            if skeleton[ny, nx] and not visited[ny, nx]:
                path = walk_from(sy, sx, ny, nx)
                if len(path) >= 2:
                    paths.append(path)

    # Walk from junctions for any unvisited directions
    for sy, sx in junction_pixels:
        for ny, nx in _neighbors8(sy, sx, H, W):
            if skeleton[ny, nx] and not visited[ny, nx]:
                path = walk_from(sy, sx, ny, nx)
                if len(path) >= 2:
                    paths.append(path)

    # Handle loops: unvisited skeleton pixels (no endpoints)
    unvisited_skel = skeleton & ~visited & ~endpoints & ~junctions
    ys, xs = np.where(unvisited_skel)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx]:
            continue
        # trace the loop starting from this pixel
        path = [(sy, sx)]
        visited[sy, sx] = True
        cy, cx = sy, sx
        while True:
            found = False
            for ny, nx in _neighbors8(cy, cx, H, W):
                if skeleton[ny, nx] and not visited[ny, nx]:
                    path.append((ny, nx))
                    visited[ny, nx] = True
                    cy, cx = ny, nx
                    found = True
                    break
            if not found:
                break
        if len(path) >= 2:
            paths.append(path)

    return paths


def prune_spurs(paths: list[list[tuple[int, int]]],
                endpoints: np.ndarray,
                junctions: np.ndarray) -> list[list[tuple[int, int]]]:
    """Remove short dead-end branches (skeleton spurs)."""
    pruned = []
    for p in paths:
        if len(p) < SPUR_MIN_LENGTH:
            sy, sx = p[0]
            ey, ex = p[-1]
            start_is_ep = endpoints[sy, sx]
            end_is_ep = endpoints[ey, ex]
            # if one end is an endpoint and the other is a junction,
            # this is a spur — drop it
            start_is_junc = junctions[sy, sx]
            end_is_junc = junctions[ey, ex]
            if (start_is_ep and end_is_junc) or (end_is_ep and start_is_junc):
                continue
        pruned.append(p)
    return pruned


def orient_by_time(path: list[tuple[int, int]],
                   arrival_time: np.ndarray) -> None:
    """Orient path so earlier-written end comes first. In-place."""
    n = len(path)
    q = max(n // 4, 1)
    times_start = [arrival_time[y, x] for y, x in path[:q]]
    times_end = [arrival_time[y, x] for y, x in path[-q:]]
    # filter zeros (unlikely but defensive)
    ts = [t for t in times_start if t > 0]
    te = [t for t in times_end if t > 0]
    if not ts or not te:
        return
    if np.median(ts) > np.median(te):
        path.reverse()


# ── Phase 3b: Chain segments through junctions ───────────────────────────────

def chain_segments(paths: list[list[tuple[int, int]]],
                   junctions: np.ndarray,
                   arrival_time: np.ndarray,
                   ) -> list[list[tuple[int, int]]]:
    """Chain path segments through junctions by temporal continuity.
    Returns a list of (potentially longer) stroke polylines."""

    # Build junction adjacency
    # junc_adj[junc_pixel] = [(path_idx, 'start'|'end', median_time_near_junc)]
    junc_adj: dict[tuple[int, int], list[tuple[int, str, float]]] = defaultdict(list)
    for i, p in enumerate(paths):
        for label, pixel, time_slice in [
            ('start', p[0], p[:5]),
            ('end', p[-1], p[-5:]),
        ]:
            y, x = pixel
            if junctions[y, x]:
                times = [arrival_time[yy, xx] for yy, xx in time_slice
                         if arrival_time[yy, xx] > 0]
                med_t = float(np.median(times)) if times else 0.0
                junc_adj[(y, x)].append((i, label, med_t))

    # Greedy chaining: for each junction, pair end→start by temporal proximity
    # A path's 'end' at the junction chains to another path's 'start' at the junction
    used = set()  # (path_idx, label) pairs already chained
    chain_links: dict[int, int] = {}  # path_idx end → path_idx start (successor)
    reverse_links: dict[int, int] = {}

    for junc, conns in junc_adj.items():
        ends = [(i, t) for i, lab, t in conns if lab == 'end' and (i, 'end') not in used]
        starts = [(i, t) for i, lab, t in conns if lab == 'start' and (i, 'start') not in used]

        # pair by temporal proximity
        ends.sort(key=lambda x: x[1])
        starts.sort(key=lambda x: x[1])

        for ei, et in ends:
            best_si, best_dt = -1, float('inf')
            for si, st in starts:
                if si == ei:
                    continue
                if (si, 'start') in used:
                    continue
                dt = abs(st - et)
                if dt < best_dt and dt < JUNCTION_TIME_THRESH:
                    best_dt = dt
                    best_si = si
            if best_si >= 0:
                chain_links[ei] = best_si
                reverse_links[best_si] = ei
                used.add((ei, 'end'))
                used.add((best_si, 'start'))

    # Build chains by following links
    chain_starts = set(range(len(paths))) - set(reverse_links.keys())
    chains: list[list[tuple[int, int]]] = []

    visited_paths = set()
    for start_idx in sorted(chain_starts):
        if start_idx in visited_paths:
            continue
        chain_coords: list[tuple[int, int]] = []
        idx = start_idx
        while True:
            visited_paths.add(idx)
            chain_coords.extend(paths[idx])
            if idx in chain_links:
                next_idx = chain_links[idx]
                if next_idx in visited_paths:
                    break
                idx = next_idx
            else:
                break
        chains.append(chain_coords)

    # add any orphaned paths
    for i, p in enumerate(paths):
        if i not in visited_paths:
            chains.append(list(p))

    return chains


# ── Phase 4: Segmentation + output ───────────────────────────────────────────

def segment_by_time(coords: list[tuple[int, int]],
                    arrival_time: np.ndarray,
                    gap_thresh: int = TEMPORAL_GAP_FRAMES,
                    ) -> list[list[tuple[int, int]]]:
    """Split a polyline where temporal gap exceeds threshold."""
    if len(coords) < 2:
        return [coords] if coords else []

    times = [int(arrival_time[y, x]) for y, x in coords]
    segments: list[list[tuple[int, int]]] = []
    seg_start = 0

    for i in range(1, len(coords)):
        t_cur = times[i] if times[i] > 0 else times[i - 1]
        t_prev = times[i - 1] if times[i - 1] > 0 else t_cur
        if t_cur > 0 and t_prev > 0 and abs(t_cur - t_prev) > gap_thresh:
            segments.append(coords[seg_start:i])
            seg_start = i

    segments.append(coords[seg_start:])
    return [s for s in segments if len(s) >= MIN_STROKE_POINTS]


def simplify_stroke(coords: list[tuple[int, int]],
                    epsilon: float = RDP_EPSILON,
                    ) -> list[tuple[float, float]]:
    """RDP simplification on (y, x) coords. Returns (x, y) points."""
    if len(coords) < 2:
        return [(float(x), float(y)) for y, x in coords]
    # cv2.approxPolyDP wants (x, y) float32
    pts = np.array([(x, y) for y, x in coords], dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(pts, epsilon, closed=False)
    return [(float(p[0, 0]), float(p[0, 1])) for p in simplified]


def assign_times(simplified_xy: list[tuple[float, float]],
                 original_yx: list[tuple[int, int]],
                 arrival_time: np.ndarray,
                 fps: int) -> list[float]:
    """For each simplified point, find the closest original point and
    use its arrival time."""
    if not simplified_xy:
        return []
    orig_xy = np.array([(x, y) for y, x in original_yx], dtype=np.float32)
    orig_times = np.array([arrival_time[y, x] for y, x in original_yx],
                          dtype=np.float32)
    times_out = []
    for sx, sy in simplified_xy:
        dists = (orig_xy[:, 0] - sx) ** 2 + (orig_xy[:, 1] - sy) ** 2
        closest = int(np.argmin(dists))
        t = orig_times[closest]
        # if the closest point has no timestamp, search nearby
        if t == 0:
            for offset in range(1, min(10, len(orig_times))):
                for idx in (closest - offset, closest + offset):
                    if 0 <= idx < len(orig_times) and orig_times[idx] > 0:
                        t = orig_times[idx]
                        break
                if t > 0:
                    break
        times_out.append(round(float(t) / fps, 4))
    return times_out


# ── Main ─────────────────────────────────────────────────────────────────────

def _process_page(arrival_time: np.ndarray, fps: int, page_idx: int,
                   tag: str) -> list[dict]:
    """Run phases 2-4 on a single page's temporal map. Returns stroke dicts."""
    n_inked = int((arrival_time > 0).sum())
    if n_inked == 0:
        return []

    # ── Phase 2: skeletonize ──
    skeleton = skeletonize_ink(arrival_time)
    n_skel = int(skeleton.sum())
    print(f"[{tag}]   page {page_idx}: {n_inked:,} ink px → {n_skel:,} skel px")

    if n_skel == 0:
        return []

    # ── Phase 3: trace ──
    endpoints, junctions = classify_skeleton(skeleton)
    raw_paths = trace_paths(skeleton, endpoints, junctions)
    paths_pruned = prune_spurs(raw_paths, endpoints, junctions)

    for p in paths_pruned:
        orient_by_time(p, arrival_time)

    chains = chain_segments(paths_pruned, junctions, arrival_time)

    # ── Phase 4: segment + simplify ──
    strokes: list[dict] = []
    for chain in chains:
        segments = segment_by_time(chain, arrival_time)
        for seg in segments:
            simplified = simplify_stroke(seg, RDP_EPSILON)
            if len(simplified) < MIN_STROKE_POINTS:
                continue
            path_len = sum(
                ((simplified[i][0] - simplified[i-1][0])**2 +
                 (simplified[i][1] - simplified[i-1][1])**2) ** 0.5
                for i in range(1, len(simplified))
            )
            if path_len < MIN_STROKE_DISTANCE:
                continue
            times = assign_times(simplified, seg, arrival_time, fps)

            strokes.append({
                "stroke_id": 0,  # renumbered later
                "page": page_idx,
                "t_start": times[0],
                "t_end": times[-1],
                "points": [
                    {"x": round(x, 2), "y": round(y, 2), "t": round(t, 4)}
                    for (x, y), t in zip(simplified, times)
                ],
            })
    return strokes


def extract_v4(tag: str, fps: int = 30,
               start_frame: int = 0, end_frame: int | None = None) -> None:
    frame_dir = FRAMES / tag
    paths = sorted(frame_dir.glob("*.png"))
    if not paths:
        sys.exit(f"no frames in {frame_dir}")

    if end_frame is None:
        end_frame = len(paths)

    n_subsampled = len(range(start_frame, end_frame, FRAME_STEP))
    print(f"[{tag}] v4 extraction: frames {start_frame}–{end_frame} "
          f"({(end_frame - start_frame) / fps:.1f}s, "
          f"{n_subsampled} sampled at {fps / FRAME_STEP:.0f}fps)")

    # ── Phase 1: build per-page temporal maps ──
    print(f"[{tag}] Phase 1: building temporal pixel maps (per-page)...")
    baseline = estimate_baseline(paths)
    page_maps = build_temporal_maps(paths, baseline, start_frame, end_frame)
    print(f"[{tag}]   detected {len(page_maps)} page(s)")

    if not page_maps:
        print(f"[{tag}] no ink found — nothing to extract")
        return

    # ── Phases 2-4: process each page independently ──
    print(f"[{tag}] Phases 2-4: skeleton → trace → segment (per-page)...")
    all_strokes: list[dict] = []
    for pi, atmap in enumerate(page_maps):
        page_strokes = _process_page(atmap, fps, pi, tag)
        all_strokes.extend(page_strokes)

    # Sort all strokes by start time and renumber
    all_strokes.sort(key=lambda s: s["t_start"])
    for i, s in enumerate(all_strokes):
        s["stroke_id"] = i

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / f"{tag}.v4.strokes.jsonl"
    with out_path.open("w") as f:
        for s in all_strokes:
            f.write(json.dumps(s) + "\n")

    total_pts = sum(len(s["points"]) for s in all_strokes)
    pts_per = total_pts / len(all_strokes) if all_strokes else 0
    print(f"\n[{tag}] DONE")
    print(f"  pages:         {len(page_maps)}")
    print(f"  strokes:       {len(all_strokes)}")
    print(f"  total points:  {total_pts:,}")
    print(f"  avg pts/stroke: {pts_per:.1f}")
    print(f"  output:        {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Skeleton-based stroke extraction (v4)")
    ap.add_argument("tag", help="video tag (subdirectory in frames/)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=None)
    args = ap.parse_args()
    extract_v4(args.tag, fps=args.fps,
               start_frame=args.start_frame, end_frame=args.end_frame)


if __name__ == "__main__":
    main()
