"""Fetch all math playlists from an OCT-style channel and enqueue every video.

Steps:
  1. List all playlists on the channel
  2. Keep only math playlists (keyword filter)
  3. Fetch all video IDs from each playlist
  4. Deduplicate across playlists (same video can appear in multiple)
  5. Write one job file per unique video into queue/pending/

Usage:
    python3 enqueue_channel.py https://www.youtube.com/@TheOrganicChemistryTutor/playlists \
        --channel oct

Tags are:  <channel>-<video_id>   (globally unique, no playlist prefix needed)
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
QUEUE   = ROOT / "queue"
PENDING = QUEUE / "pending"

# Keywords that identify math playlists (case-insensitive, any match = keep)
MATH_KEYWORDS = [
    "math", "algebra", "geometry", "calculus", "trigonometry", "trig",
    "precalculus", "pre-calculus", "statistics", "stat", "fraction",
    "arithmetic", "number", "equation", "integral", "derivative",
    "probability", "percentage", "sat math", "ged math", "linear",
    "quadratic", "polynomial", "logarithm", "sequence", "series",
    "integration", "differentiation", "prealgebra", "pre-algebra",
]

# Keywords that hard-exclude a playlist even if a math keyword also matches
EXCLUDE_KEYWORDS = [
    "chemistry", "physics", "biology", "organic", "biochem",
    "excel", "finance", "stock", "bond", "study tip", "growth tip",
    "circuit", "electric", "science", "patreon",
]

SKIP_TITLE_KEYWORDS = ["patreon", "study tip", "studying tip", "my channel"]


def run_yt(args: list[str]) -> str:
    r = subprocess.run(["yt-dlp"] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  yt-dlp warning: {r.stderr.strip()[:200]}", file=sys.stderr)
    return r.stdout


def fetch_playlists(channel_url: str) -> list[dict]:
    raw = run_yt(["--flat-playlist", "--print", "%(id)s|%(title)s", channel_url])
    playlists = []
    for line in raw.strip().splitlines():
        parts = line.split("|", 1)
        if len(parts) != 2:
            continue
        pl_id, title = parts
        playlists.append({"id": pl_id, "title": title})
    return playlists


def is_math_playlist(title: str) -> bool:
    t = title.lower()
    if any(kw in t for kw in EXCLUDE_KEYWORDS):
        return False
    return any(kw in t for kw in MATH_KEYWORDS)


def fetch_videos(playlist_id: str) -> list[dict]:
    url = f"https://www.youtube.com/playlist?list={playlist_id}"
    raw = run_yt(["--flat-playlist", "--print", "%(id)s|%(title)s|%(duration)s", url])
    videos = []
    for line in raw.strip().splitlines():
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


def all_queued_tags() -> set[str]:
    tags = set()
    for folder in QUEUE.iterdir():
        if folder.is_dir():
            for f in folder.glob("*.json"):
                tags.add(f.stem)
    return tags


def main() -> None:
    for d in ("pending", "running", "done", "failed"):
        (QUEUE / d).mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("channel_url", help="Channel playlists URL")
    ap.add_argument("--channel", default="oct", help="Channel label for tags")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("fetching playlist list …")
    all_playlists = fetch_playlists(args.channel_url)
    print(f"found {len(all_playlists)} playlists total")

    math_playlists = [p for p in all_playlists if is_math_playlist(p["title"])]
    skipped_pl = [p for p in all_playlists if not is_math_playlist(p["title"])]

    print(f"\nmath playlists ({len(math_playlists)}):")
    for p in math_playlists:
        print(f"  ✓  {p['title']}")
    print(f"\nskipping ({len(skipped_pl)}):")
    for p in skipped_pl:
        print(f"  ✗  {p['title']}")

    # fetch all videos, deduplicate by video ID
    seen_ids: dict[str, dict] = {}   # video_id → job dict
    already_queued = all_queued_tags()

    print(f"\nfetching videos from {len(math_playlists)} playlists …")
    for pl in math_playlists:
        print(f"  [{pl['title']}] ", end="", flush=True)
        videos = fetch_videos(pl["id"])
        new = 0
        for v in videos:
            if any(kw in v["title"].lower() for kw in SKIP_TITLE_KEYWORDS):
                continue
            tag = f"{args.channel}-{v['id']}"
            if tag in already_queued:
                continue
            if v["id"] not in seen_ids:
                seen_ids[v["id"]] = {
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
                new += 1
        print(f"{len(videos)} videos, {new} new")

    total = len(seen_ids)
    total_hours = sum(0 for _ in seen_ids)   # duration not stored; just count
    print(f"\n{total} unique new videos to enqueue")

    if args.dry_run:
        for job in list(seen_ids.values())[:10]:
            print(f"  {job['tag']}  {job['topic'][:60]}")
        if total > 10:
            print(f"  … and {total-10} more")
        return

    for job in seen_ids.values():
        dest = PENDING / f"{job['tag']}.json"
        dest.write_text(json.dumps(job, indent=2, ensure_ascii=False))

    print(f"wrote {total} job files to queue/pending/")

    # update the committed manifest so other machines can seed from git
    manifest_path = ROOT / "queue_manifest.json"
    existing = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text()).get("videos", [])
    existing_tags = {e["tag"] for e in existing}
    new_entries = [
        {"tag": j["tag"], "url": j["url"], "topic": j["topic"],
         "channel": j["channel"], "duration_cap_sec": j["duration_cap_sec"]}
        for j in seen_ids.values() if j["tag"] not in existing_tags
    ]
    all_entries = sorted(existing + new_entries, key=lambda x: x["tag"])
    manifest_path.write_text(json.dumps({"videos": all_entries}, indent=2, ensure_ascii=False))
    print(f"updated queue_manifest.json ({len(all_entries)} total entries)")


if __name__ == "__main__":
    main()
