"""Populate queue/pending/ from queue_manifest.json.

Run this once on any new machine after git pull.
Safe to run multiple times — skips videos already in the queue
(in any state: pending, running, done, failed).

Usage:
    python3 seed_queue.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
MANIFEST = ROOT / "queue_manifest.json"
QUEUE    = ROOT / "queue"
OUTPUT   = ROOT / "output"

for d in ("pending", "running", "done", "failed"):
    (QUEUE / d).mkdir(parents=True, exist_ok=True)

videos = json.loads(MANIFEST.read_text())["videos"]

added = skipped = already_done = 0
for v in videos:
    tag = v["tag"]

    # skip if already in queue (any state)
    for folder in ("pending", "running", "done", "failed"):
        if (QUEUE / folder / f"{tag}.json").exists():
            skipped += 1
            break
    else:
        # mark as done if output already exists on this filesystem
        already_extracted = (OUTPUT / f"{tag}.strokes.jsonl").exists()
        target = "done" if already_extracted else "pending"

        job = {
            "tag":              tag,
            "url":              v["url"],
            "topic":            v.get("topic", ""),
            "channel":          v.get("channel", ""),
            "duration_cap_sec": v.get("duration_cap_sec", 0),
            "status":           target,
            "worker":           None,
            "added_at":         None,
            "started_at":       None,
            "finished_at":      "pre-existing" if already_extracted else None,
            "error":            None,
        }
        (QUEUE / target / f"{tag}.json").write_text(
            json.dumps(job, indent=2, ensure_ascii=False)
        )
        if already_extracted:
            already_done += 1
        else:
            added += 1

print(f"seeded: {added} pending, {already_done} pre-marked done, {skipped} already in queue")
