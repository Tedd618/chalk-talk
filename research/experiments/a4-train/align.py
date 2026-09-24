"""align.py — Convert events.jsonl into word-anchored training sequences.

Each output token is one spoken word, with any strokes drawn during
that word attached as a list of point sequences. Silent pen movements
(no overlapping speech) become <silent> tokens.

Input:  a2-alignment/output/<tag>.events.jsonl
Output: a4-train/output/<tag>.training.jsonl

Each line of training.jsonl is one token:

  {"type": "lesson_start", "tag": "oct-algebra", "page": 0}
  {"type": "word", "word": "squared", "strokes": [[[x, y, p], ...]]}
  {"type": "silent", "strokes": [[[x, y, p], ...]]}
  {"type": "page_break"}
  {"type": "end"}

Coordinates are normalized to [0, 1]:
    x_norm = x / 1280
    y_norm = y / 720

Pen state p:
    0 = pen still down (mid-stroke point)
    1 = pen up (last point of this stroke)

Usage:
    python3 align.py oct-algebra
    python3 align.py --all          # process all videos in a2 output dir
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

A4_DIR   = Path(__file__).resolve().parent
A2_DIR   = A4_DIR.parent / "a2-alignment"
A1_DIR   = A4_DIR.parent / "a1-stroke-extraction"
A2_OUT   = A2_DIR / "output"
MANIFEST = A1_DIR / "queue_manifest.json"
OUTPUT   = A4_DIR / "output"

W = 1280.0  # canvas width  in pixels
H = 720.0   # canvas height in pixels


def load_topics() -> dict[str, str]:
    """Load tag -> topic mapping from queue_manifest.json."""
    if not MANIFEST.exists():
        return {}
    data = json.loads(MANIFEST.read_text())
    return {v["tag"]: v.get("topic", "") for v in data.get("videos", [])}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_events(tag: str) -> list[dict]:
    path = A2_OUT / f"{tag}.events.jsonl"
    if not path.exists():
        sys.exit(f"missing: {path}")
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def collect_strokes(events: list[dict]) -> dict[int, dict]:
    """Group pen events by stroke_id into complete strokes.

    Returns {stroke_id: {points: [{x,y,t},...], t_start, t_end}}.
    t_end is updated to stroke_end time if available, otherwise last pen time.
    """
    strokes: dict[int, dict] = {}
    for e in events:
        if e["type"] == "pen":
            sid = e["stroke_id"]
            if sid not in strokes:
                strokes[sid] = {"points": [], "t_start": e["t"], "t_end": e["t"]}
            strokes[sid]["points"].append({"x": e["x"], "y": e["y"], "t": e["t"]})
            strokes[sid]["t_end"] = e["t"]
        elif e["type"] == "stroke_end":
            sid = e["stroke_id"]
            if sid in strokes:
                strokes[sid]["t_end"] = e["t"]
    return strokes


def time_overlap(s_start: float, s_end: float,
                 w_start: float, w_end: float) -> float:
    return max(0.0, min(s_end, w_end) - max(s_start, w_start))


def assign_strokes(strokes: dict[int, dict],
                   words: list[dict]) -> dict[int, int | None]:
    """Assign each stroke to the word with maximum time overlap.

    Returns {stroke_id: word_index or None (silent)}.
    """
    assignments: dict[int, int | None] = {}
    for sid, stroke in strokes.items():
        best_wi: int | None = None
        best_ov = 0.0
        for wi, word in enumerate(words):
            ov = time_overlap(stroke["t_start"], stroke["t_end"],
                              word["t"], word["t_end"])
            if ov > best_ov:
                best_ov = ov
                best_wi = wi
        assignments[sid] = best_wi
    return assignments


def encode_stroke(stroke: dict) -> list[list[float]]:
    """Encode one stroke as [[x_norm, y_norm, p], ...].

    x, y are normalized to [0, 1].  p = 0 for mid-stroke, 1 for last point.
    """
    pts = stroke["points"]
    encoded = []
    for i, pt in enumerate(pts):
        p_state = 1 if i == len(pts) - 1 else 0
        encoded.append([pt["x"] / W, pt["y"] / H, p_state])
    return encoded


# ── per-page processing ───────────────────────────────────────────────────────

def process_page(page_events: list[dict], tag: str,
                 page_idx: int, topic: str = "") -> list[dict]:
    """Convert one page's events into a word-anchored token sequence.

    Returns a list of token dicts (lesson_start, word, silent, end).
    """
    words = [e for e in page_events if e["type"] == "word"]
    strokes = collect_strokes(page_events)

    if not strokes and not words:
        return []

    assignments = assign_strokes(strokes, words)

    # Build word_strokes: for each word index, collect its strokes in time order
    word_strokes: dict[int, list[dict]] = {i: [] for i in range(len(words))}
    silent_strokes: list[dict] = []  # strokes with no matching word

    for sid, wi in assignments.items():
        if wi is not None:
            word_strokes[wi].append(strokes[sid])
        else:
            silent_strokes.append(strokes[sid])

    # Sort strokes within each word by t_start
    for wi in word_strokes:
        word_strokes[wi].sort(key=lambda s: s["t_start"])
    silent_strokes.sort(key=lambda s: s["t_start"])

    # Build the interleaved output sequence.
    # We walk words in time order and insert any silent strokes that
    # started before the current word's end.
    tokens: list[dict] = []
    tokens.append({"type": "lesson_start", "tag": tag, "page": page_idx,
                   "topic": topic})

    silent_idx = 0  # pointer into sorted silent_strokes

    def flush_silents_before(t: float) -> None:
        nonlocal silent_idx
        while silent_idx < len(silent_strokes):
            s = silent_strokes[silent_idx]
            if s["t_start"] >= t:
                break
            tokens.append({
                "type": "silent",
                "strokes": [encode_stroke(s)],
            })
            silent_idx += 1

    for wi, word in enumerate(words):
        # Flush any silent strokes that happen before this word starts
        flush_silents_before(word["t"])

        tokens.append({
            "type": "word",
            "word": word["word"],
            "strokes": [encode_stroke(s) for s in word_strokes[wi]],
        })

    # Flush remaining silent strokes after last word
    flush_silents_before(float("inf"))

    tokens.append({"type": "end"})
    return tokens


# ── main ──────────────────────────────────────────────────────────────────────

def align(tag: str, topics: dict[str, str] | None = None) -> None:
    events = load_events(tag)
    topic = (topics or {}).get(tag, "")

    # Split events into pages by page_break
    pages: list[list[dict]] = []
    current: list[dict] = []
    for e in events:
        if e["type"] == "page_break":
            if current:
                pages.append(current)
                current = []
        else:
            current.append(e)
    if current:
        pages.append(current)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / f"{tag}.training.jsonl"

    total_words = 0
    total_strokes = 0
    total_silent = 0

    with out_path.open("w") as f:
        for pi, page_events in enumerate(pages):
            tokens = process_page(page_events, tag, pi, topic)
            for tok in tokens:
                f.write(json.dumps(tok, ensure_ascii=False) + "\n")
            if pi < len(pages) - 1:
                f.write(json.dumps({"type": "page_break"}) + "\n")

            total_words   += sum(1 for t in tokens if t["type"] == "word")
            total_strokes += sum(len(t["strokes"]) for t in tokens
                                 if t["type"] in ("word", "silent"))
            total_silent  += sum(1 for t in tokens if t["type"] == "silent")

    print(f"[{tag}] {len(pages)} pages | "
          f"{total_words} words | "
          f"{total_strokes} strokes assigned | "
          f"{total_silent} silent tokens "
          f"→ {out_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default=None,
                    help="Video tag to process (e.g. oct-algebra)")
    ap.add_argument("--all", action="store_true",
                    help="Process all videos found in a2-alignment/output/")
    args = ap.parse_args()

    topics = load_topics()

    if args.all:
        tags = sorted({p.stem.replace(".events", "")
                       for p in A2_OUT.glob("*.events.jsonl")})
        if not tags:
            sys.exit(f"no events files found in {A2_OUT}")
        print(f"processing {len(tags)} videos...")
        for i, tag in enumerate(tags, 1):
            print(f"[{i}/{len(tags)}] {tag}", flush=True)
            align(tag, topics)
    elif args.tag:
        align(args.tag, topics)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
