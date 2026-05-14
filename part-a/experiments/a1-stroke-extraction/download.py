"""Download corpus videos with yt-dlp; capture metadata.

Reads the video list from corpus.json. Skips downloads that already
exist. Writes a sidecar <tag>.meta.json per video with title, duration,
uploader, etc.

Usage:
    python3 download.py            # download all corpus entries
    python3 download.py oct-algebra  # download only this tag
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# CHALK_SCRATCH lets worker.py redirect large files to /tmp or similar
_scratch = os.environ.get("CHALK_SCRATCH")
VIDEOS = Path(_scratch) if _scratch else ROOT / "videos"
CORPUS = ROOT / "corpus.json"


def require(cmd: str) -> str:
    p = shutil.which(cmd)
    if not p:
        sys.exit(f"missing tool: {cmd}. install with: brew install {cmd}")
    return p


def load_corpus() -> list[dict]:
    if not CORPUS.exists():
        sys.exit(f"missing corpus file: {CORPUS}")
    return json.loads(CORPUS.read_text())["videos"]


def download_one(entry: dict) -> None:
    tag = entry["tag"]
    url = entry["url"]
    VIDEOS.mkdir(parents=True, exist_ok=True)
    out_video = VIDEOS / f"{tag}.mp4"
    out_meta = VIDEOS / f"{tag}.meta.json"

    if out_video.exists() and out_meta.exists():
        print(f"[{tag}] already downloaded, skipping")
        return

    print(f"[{tag}] downloading {url}")
    cmd = [
        require("yt-dlp"),
        "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "--merge-output-format", "mp4",
        "-o", str(out_video),
        "--write-info-json",
        "--no-write-playlist-metafiles",
    ]
    cookies_file = ROOT / "yt-cookies.txt"
    if cookies_file.exists():
        cmd += ["--cookies", str(cookies_file)]
    cmd.append(url)
    subprocess.run(cmd, check=True)

    info_json = VIDEOS / f"{tag}.info.json"
    if info_json.exists():
        info = json.loads(info_json.read_text())
        meta = {
            "tag": tag,
            "url": url,
            "topic": entry.get("topic", ""),
            "duration_cap_sec": entry.get("duration_cap_sec", 0),
            "title": info.get("title"),
            "description": info.get("description"),
            "duration_sec": info.get("duration"),
            "uploader": info.get("uploader"),
            "upload_date": info.get("upload_date"),
            "width": info.get("width"),
            "height": info.get("height"),
            "fps": info.get("fps"),
        }
        out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        info_json.unlink()
    print(f"[{tag}] done: {out_video.name}")


def main() -> None:
    require("yt-dlp")
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default=None,
                    help="Single tag to download (default: all)")
    ap.add_argument("--url", default=None,
                    help="URL to download directly (skips corpus.json lookup)")
    args = ap.parse_args()

    if args.url:
        # called by worker.py with explicit URL
        download_one({"tag": args.tag, "url": args.url})
        return

    corpus = load_corpus()
    if args.tag:
        corpus = [e for e in corpus if e["tag"] == args.tag]
        if not corpus:
            sys.exit(f"no entry with tag={args.tag} in {CORPUS}")
    for entry in corpus:
        download_one(entry)


if __name__ == "__main__":
    main()
