"""Merge a2's words.jsonl with a1's strokes.jsonl into a single
time-ordered event stream.

Output: output/<tag>.events.jsonl — one event per line, ordered by `t`.

  WORD event:
    {"type": "word", "t": <start>, "t_end": <end>, "word": "..."}

  PEN event (one per stroke point):
    {"type": "pen", "t": <t>, "x": <x>, "y": <y>,
     "stroke_id": <int>, "pen_state": "down" | "up",
     "first_in_stroke": bool, "last_in_stroke": bool}

  STROKE_END event (synthesized at the end-time of each stroke):
    {"type": "stroke_end", "t": <t_end>, "stroke_id": <int>}

The PEN/STROKE_END events together encode pen-state implicitly:
between `first_in_stroke=True` (pen-down start) and `stroke_end` the
pen is down; otherwise up.

Usage:
    python3 merge.py oct
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

A2_DIR = Path(__file__).resolve().parent
A1_DIR = A2_DIR.parent / "a1-stroke-extraction"
A1_OUT = A1_DIR / "output"
OUTPUT = A2_DIR / "output"


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        sys.exit(f"missing: {p}")
    return [json.loads(line) for line in p.read_text().splitlines()
            if line.strip()]


def merge(tag: str) -> None:
    strokes = load_jsonl(A1_OUT / f"{tag}.strokes.jsonl")
    words = load_jsonl(OUTPUT / f"{tag}.words.jsonl")

    events: list[dict] = []

    for stroke in strokes:
        sid = stroke["stroke_id"]
        pts = stroke["points"]
        for i, p in enumerate(pts):
            events.append({
                "type": "pen",
                "t": p["t"],
                "x": p["x"],
                "y": p["y"],
                "stroke_id": sid,
                "first_in_stroke": (i == 0),
                "last_in_stroke": (i == len(pts) - 1),
            })
        events.append({
            "type": "stroke_end",
            "t": stroke["t_end"],
            "stroke_id": sid,
        })

    for w in words:
        events.append({
            "type": "word",
            "t": w["t_start"],
            "t_end": w["t_end"],
            "word": w["word"],
        })

    events.sort(key=lambda e: (e["t"], 0 if e["type"] == "word" else 1))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / f"{tag}.events.jsonl"
    with out_path.open("w") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    n_words = sum(1 for e in events if e["type"] == "word")
    n_pen = sum(1 for e in events if e["type"] == "pen")
    n_strokes = sum(1 for e in events if e["type"] == "stroke_end")
    print(f"[{tag}] {len(events)} events: "
          f"{n_words} word + {n_pen} pen + {n_strokes} stroke_end "
          f"-> {out_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    args = ap.parse_args()
    merge(args.tag)


if __name__ == "__main__":
    main()
