"""Add all videos from a YouTube playlist to the worker queue.

Usage:
    python3 enqueue_playlist.py <playlist_url> --channel oct --prefix oct-pa
    python3 enqueue_playlist.py <playlist_url> --channel mrh --prefix mrh
    python3 enqueue_playlist.py <playlist_url> --skip-longer-than 3600  # skip >1hr

Generates tags from video IDs: <prefix>-<video_id>.
Skips videos already in the queue (any state).
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
QUEUE   = ROOT / "queue"
PENDING = QUEUE / "pending"

SKIP_KEYWORDS = ["patreon", "studying tips", "study tips", "my channel"]

def slug(title: str) -> str:
    t = title.lower()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    return t.strip("-")[:40]

def all_queued_ids() -> set[str]:
    ids = set()
    for folder in QUEUE.iterdir():
        if folder.is_dir():
            for f in folder.glob("*.json"):
                ids.add(f.stem)
    return ids


def fetch_playlist(url: str) -> list[dict]:
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist",
         "--print", "%(id)s|%(title)s|%(duration)s",
         url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.exit(f"yt-dlp failed:\n{result.stderr}")

    videos = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        vid_id, title, dur_str = parts
        try:
            duration = int(float(dur_str))
        except ValueError:
            duration = 0
        videos.append({"id": vid_id, "title": title, "duration": duration})
    return videos


def main() -> None:
    for d in ("pending", "running", "done", "failed"):
        (QUEUE / d).mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("url",                         help="Playlist URL")
    ap.add_argument("--channel",   default="oct",  help="Channel label (oct / mrh / …)")
    ap.add_argument("--prefix",    default="oct",  help="Tag prefix, e.g. oct-pa")
    ap.add_argument("--skip-longer-than", type=int, default=0,
                    help="Skip videos longer than N seconds (0 = keep all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be added without writing files")
    args = ap.parse_args()

    print(f"fetching playlist …")
    videos = fetch_playlist(args.url)
    print(f"found {len(videos)} videos")

    already = all_queued_ids()
    added = skipped = 0
    new_jobs = []

    for v in videos:
        tag = f"{args.prefix}-{v['id']}"

        # skip non-tutorials
        if any(kw in v["title"].lower() for kw in SKIP_KEYWORDS):
            print(f"  skip (non-tutorial): {v['title'][:60]}")
            skipped += 1
            continue

        # skip if too long
        if args.skip_longer_than and v["duration"] > args.skip_longer_than:
            print(f"  skip (>{args.skip_longer_than}s): {v['title'][:60]}")
            skipped += 1
            continue

        # skip if already queued
        if tag in already:
            print(f"  skip (already queued): {tag}")
            skipped += 1
            continue

        job = {
            "tag":              tag,
            "url":              f"https://www.youtube.com/watch?v={v['id']}",
            "topic":            v["title"],
            "channel":          args.channel,
            "duration_cap_sec": 0,
            "status":           "pending",
            "worker":           None,
            "added_at":         None,
            "started_at":       None,
            "finished_at":      None,
            "error":            None,
        }

        if args.dry_run:
            print(f"  would add: {tag}  ({v['duration']//60}m)  {v['title'][:50]}")
        else:
            dest = PENDING / f"{tag}.json"
            dest.write_text(json.dumps(job, indent=2, ensure_ascii=False))
            new_jobs.append(job)
            print(f"  added: {tag}  ({v['duration']//60}m)  {v['title'][:50]}")
        added += 1

    print(f"\nadded={added}  skipped={skipped}")

    if not args.dry_run and added > 0:
        # update the committed manifest so other machines can seed from git
        manifest_path = Path(__file__).resolve().parent / "queue_manifest.json"
        existing = []
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text()).get("videos", [])
        existing_tags = {e["tag"] for e in existing}
        new_entries = [
            {"tag": j["tag"], "url": j["url"], "topic": j["topic"],
             "channel": j["channel"], "duration_cap_sec": j["duration_cap_sec"]}
            for j in new_jobs if j["tag"] not in existing_tags
        ]
        all_entries = sorted(existing + new_entries, key=lambda x: x["tag"])
        manifest_path.write_text(json.dumps({"videos": all_entries}, indent=2, ensure_ascii=False))
        print(f"updated queue_manifest.json ({len(all_entries)} total entries)")


if __name__ == "__main__":
    main()
