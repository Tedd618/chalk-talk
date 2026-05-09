"""Download pilot videos with yt-dlp; capture metadata for topic conditioning."""
from __future__ import annotations
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEOS = ROOT / "videos"

PILOTS = {
    "ka-pythagoras": "https://www.youtube.com/watch?v=AA6RfgP-AHU",
    "oct":           "https://www.youtube.com/watch?v=pbf4lcJhIfI",
}


def require(cmd: str) -> str:
    p = shutil.which(cmd)
    if not p:
        sys.exit(f"missing tool: {cmd}. install with: brew install {cmd}")
    return p


def download_one(tag: str, url: str) -> None:
    VIDEOS.mkdir(parents=True, exist_ok=True)
    out_video = VIDEOS / f"{tag}.mp4"
    out_meta = VIDEOS / f"{tag}.meta.json"

    if out_video.exists() and out_meta.exists():
        print(f"[{tag}] already downloaded, skipping")
        return

    print(f"[{tag}] downloading {url}")
    subprocess.run(
        [
            require("yt-dlp"),
            "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
            "--merge-output-format", "mp4",
            "-o", str(out_video),
            "--write-info-json",
            "--no-write-playlist-metafiles",
            url,
        ],
        check=True,
    )

    info_json = VIDEOS / f"{tag}.info.json"
    if info_json.exists():
        info = json.loads(info_json.read_text())
        meta = {
            "tag": tag,
            "url": url,
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
    for tag, url in PILOTS.items():
        download_one(tag, url)


if __name__ == "__main__":
    main()
