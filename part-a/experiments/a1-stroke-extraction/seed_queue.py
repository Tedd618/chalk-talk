"""One-time script: seed queue/pending/ from corpus.json.

Skips videos that already have strokes extracted (output/<tag>.strokes.jsonl exists).
Safe to run multiple times — never overwrites existing queue entries.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
CORPUS  = ROOT / "corpus.json"
QUEUE   = ROOT / "queue"
OUTPUT  = ROOT / "output"

for d in ("pending", "running", "done", "failed"):
    (QUEUE / d).mkdir(parents=True, exist_ok=True)

videos = json.loads(CORPUS.read_text())["videos"]

for v in videos:
    tag = v["tag"]
    already_done = (OUTPUT / f"{tag}.strokes.jsonl").exists()

    for folder in ("pending", "running", "done", "failed"):
        if (QUEUE / folder / f"{tag}.json").exists():
            print(f"[{tag}] already in queue/{folder}/  — skip")
            break
    else:
        target_dir = "done" if already_done else "pending"
        job = {
            "tag":              tag,
            "url":              v["url"],
            "topic":            v.get("topic", ""),
            "channel":          v.get("channel", ""),
            "duration_cap_sec": v.get("duration_cap_sec", 0),
            "status":           target_dir,
            "worker":           None,
            "added_at":         None,
            "started_at":       None,
            "finished_at":      "pre-existing" if already_done else None,
            "error":            None,
        }
        dest = QUEUE / target_dir / f"{tag}.json"
        dest.write_text(json.dumps(job, indent=2, ensure_ascii=False))
        print(f"[{tag}] → queue/{target_dir}/")
