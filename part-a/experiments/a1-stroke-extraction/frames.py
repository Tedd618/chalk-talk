"""Extract frames from a downloaded pilot video at 30 fps, downscaled.

Usage:
    python3 frames.py <tag>
    python3 frames.py ka-pythagoras --start 0 --duration 60
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEOS = ROOT / "videos"
FRAMES = ROOT / "frames"

DEFAULT_FPS = 30
DEFAULT_HEIGHT = 720  # downscale tall axis to 720 px max


def require(cmd: str) -> str:
    p = shutil.which(cmd)
    if not p:
        sys.exit(f"missing tool: {cmd}. install with: brew install {cmd}")
    return p


def extract(tag: str, start: float, duration: float | None,
            fps: int, height: int) -> None:
    src = VIDEOS / f"{tag}.mp4"
    if not src.exists():
        sys.exit(f"video not found: {src}. run download.py first.")

    out_dir = FRAMES / tag
    if out_dir.exists():
        # nuke prior run for this tag — frames are cheap to regenerate
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    cmd = [require("ffmpeg"), "-hide_banner", "-loglevel", "warning"]
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(src)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += [
        "-vf", f"fps={fps},scale=-2:{height}:flags=bicubic",
        "-frame_pts", "1",
        str(out_dir / "%06d.png"),
    ]
    print(f"[{tag}] extracting frames -> {out_dir}")
    subprocess.run(cmd, check=True)
    n = sum(1 for _ in out_dir.glob("*.png"))
    print(f"[{tag}] {n} frames written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--start", type=float, default=0.0,
                    help="seconds into the video to start (default: 0)")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="seconds to extract (default: 30; use 0 for full)")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    args = ap.parse_args()

    extract(
        tag=args.tag,
        start=args.start,
        duration=None if args.duration == 0 else args.duration,
        fps=args.fps,
        height=args.height,
    )


if __name__ == "__main__":
    main()
