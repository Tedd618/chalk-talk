"""End-to-end extraction pipeline: YouTube URL → training-ready events.

Given a YouTube URL (OCT-style tablet tutor), runs:
  1. Download video with yt-dlp
  2. Extract frames at 30 fps (720p)
  3. Extract strokes with v4 skeleton pipeline (+ saturation filter)
  4. Detect page breaks (canvas-clear events)
  5. Transcribe speech with Whisper (word-level timestamps)
  6. Merge strokes + words + page breaks → events.jsonl
  7. Reconstruct strokes → validation video

Usage:
    # Full video
    python pipeline.py https://www.youtube.com/watch?v=VIDEO_ID --tag my-video

    # First 5 minutes only
    python pipeline.py https://www.youtube.com/watch?v=VIDEO_ID --tag my-video --duration 300

    # Skip steps that already completed (e.g. re-extract after code change)
    python pipeline.py https://www.youtube.com/watch?v=VIDEO_ID --tag my-video --from extract

    # Just list what would run
    python pipeline.py https://www.youtube.com/watch?v=VIDEO_ID --tag my-video --dry-run

    # Process an already-downloaded video by tag
    python pipeline.py --tag oct-algebra

    # Add to corpus.json for batch processing later
    python pipeline.py https://www.youtube.com/watch?v=VIDEO_ID --tag my-video --add-to-corpus
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
A2_DIR = ROOT.parent / "a2-alignment"
VIDEOS = ROOT / "videos"
FRAMES = ROOT / "frames"
OUTPUT = ROOT / "output"
CORPUS = ROOT / "corpus.json"

# Prefer the venv python if it exists; otherwise use the system python
VENV_PY = ROOT / ".venv" / "bin" / "python"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable

STEPS = ["download", "frames", "extract", "pages", "stt", "merge", "reconstruct"]


def require_tool(cmd: str) -> str:
    p = shutil.which(cmd)
    if not p:
        sys.exit(f"missing tool: {cmd}. install it first (e.g. brew install {cmd})")
    return p


def step_done_marker(tag: str, step: str) -> bool:
    """Check if a step's output already exists."""
    checks = {
        "download": lambda: (VIDEOS / f"{tag}.mp4").exists(),
        "frames": lambda: (FRAMES / tag).exists() and any((FRAMES / tag).glob("*.png")),
        "extract": lambda: (OUTPUT / f"{tag}.strokes.jsonl").exists(),
        "pages": lambda: (OUTPUT / f"{tag}.pages.jsonl").exists(),
        "stt": lambda: (A2_DIR / "output" / f"{tag}.words.jsonl").exists(),
        "merge": lambda: (A2_DIR / "output" / f"{tag}.events.jsonl").exists(),
        "reconstruct": lambda: (OUTPUT / f"{tag}.recon.mp4").exists(),
    }
    return checks.get(step, lambda: False)()


def run_step(label: str, args: list[str], cwd: Path) -> bool:
    print(f"  ▸ {label} … ", end="", flush=True)
    t0 = time.time()
    res = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"FAIL ({dt:.1f}s)")
        if res.stderr.strip():
            # Show last few lines of stderr
            lines = res.stderr.strip().split("\n")
            for line in lines[-10:]:
                print(f"    {line}")
        return False
    print(f"ok ({dt:.1f}s)")
    return True


def download(tag: str, url: str | None) -> bool:
    if not url:
        if not (VIDEOS / f"{tag}.mp4").exists():
            print(f"  ✗ no URL and no existing video for '{tag}'")
            return False
        print(f"  ▸ download … skip (video exists)")
        return True
    return run_step("download", [PY, "download.py", tag], ROOT)


def extract_frames(tag: str, duration: int | None) -> bool:
    args = [PY, "frames.py", tag, "--duration", str(duration or 0)]
    return run_step("frames", args, ROOT)


def extract_strokes(tag: str) -> bool:
    ok = run_step("extract (v4 skeleton)", [PY, "extract_v4.py", tag], ROOT)
    if ok:
        # v4 outputs to <tag>.v4.strokes.jsonl; copy to canonical name
        # so downstream tools (merge, reconstruct) find it
        v4_path = OUTPUT / f"{tag}.v4.strokes.jsonl"
        canonical = OUTPUT / f"{tag}.strokes.jsonl"
        if v4_path.exists():
            shutil.copy2(v4_path, canonical)
    return ok


def detect_pages(tag: str) -> bool:
    return run_step("pages", [PY, "pages.py", tag], ROOT)


def run_stt(tag: str, duration: int | None) -> bool:
    args = [PY, str(A2_DIR / "stt.py"), tag]
    if duration:
        args += ["--duration", str(duration)]
    return run_step("stt", args, ROOT)


def merge_events(tag: str) -> bool:
    return run_step("merge", [PY, str(A2_DIR / "merge.py"), tag], ROOT)


def reconstruct(tag: str) -> bool:
    return run_step("reconstruct", [PY, "reconstruct.py", tag], ROOT)


def add_to_corpus(tag: str, url: str, duration: int | None, topic: str) -> None:
    """Add or update an entry in corpus.json."""
    if CORPUS.exists():
        data = json.loads(CORPUS.read_text())
    else:
        data = {"videos": []}

    # Remove existing entry with same tag
    data["videos"] = [v for v in data["videos"] if v["tag"] != tag]
    data["videos"].append({
        "tag": tag,
        "url": url,
        "duration_cap_sec": duration or 0,
        "topic": topic,
    })
    CORPUS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"  ▸ added '{tag}' to corpus.json")


def ensure_corpus_entry(tag: str, url: str | None) -> None:
    """Make sure download.py can find the entry in corpus.json."""
    if not url:
        return
    if CORPUS.exists():
        data = json.loads(CORPUS.read_text())
        for v in data["videos"]:
            if v["tag"] == tag:
                return  # already there
    else:
        data = {"videos": []}

    data["videos"].append({
        "tag": tag,
        "url": url,
        "duration_cap_sec": 0,
        "topic": "",
    })
    CORPUS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def run_pipeline(tag: str, url: str | None, duration: int | None,
                 from_step: str | None, topic: str,
                 add_corpus: bool, dry_run: bool) -> None:
    print(f"\n{'='*60}")
    print(f"  Pipeline: {tag}")
    if url:
        print(f"  URL: {url}")
    if duration:
        print(f"  Duration cap: {duration}s ({duration/60:.1f} min)")
    print(f"{'='*60}\n")

    start_idx = STEPS.index(from_step) if from_step else 0

    if dry_run:
        for i, step in enumerate(STEPS):
            status = "SKIP (before --from)" if i < start_idx else "RUN"
            if i >= start_idx and step_done_marker(tag, step):
                status = "RUN (output exists, will overwrite)"
            print(f"  {'→' if i >= start_idx else '·'} {step}: {status}")
        return

    # Ensure corpus entry exists for download.py
    if url:
        ensure_corpus_entry(tag, url)

    step_fns = {
        "download": lambda: download(tag, url),
        "frames": lambda: extract_frames(tag, duration),
        "extract": lambda: extract_strokes(tag),
        "pages": lambda: detect_pages(tag),
        "stt": lambda: run_stt(tag, duration),
        "merge": lambda: merge_events(tag),
        "reconstruct": lambda: reconstruct(tag),
    }

    t_total = time.time()
    for i, step in enumerate(STEPS):
        if i < start_idx:
            print(f"  · {step}: skip (before --from)")
            continue
        if not step_fns[step]():
            print(f"\n  ✗ Pipeline stopped at '{step}'")
            return
    dt = time.time() - t_total

    # Summary
    strokes_path = OUTPUT / f"{tag}.strokes.jsonl"
    words_path = A2_DIR / "output" / f"{tag}.words.jsonl"
    events_path = A2_DIR / "output" / f"{tag}.events.jsonl"
    pages_path = OUTPUT / f"{tag}.pages.jsonl"

    n_strokes = sum(1 for _ in strokes_path.open()) if strokes_path.exists() else 0
    n_words = sum(1 for _ in words_path.open()) if words_path.exists() else 0
    n_events = sum(1 for _ in events_path.open()) if events_path.exists() else 0
    n_pages = sum(1 for _ in pages_path.open()) if pages_path.exists() else 0

    print(f"\n{'='*60}")
    print(f"  ✓ Pipeline complete for '{tag}' ({dt:.1f}s)")
    print(f"    Strokes:     {n_strokes}")
    print(f"    Words:       {n_words}")
    print(f"    Page breaks: {n_pages}")
    print(f"    Events:      {n_events}")
    print(f"    Output:      {events_path}")
    print(f"    Validation:  {OUTPUT / f'{tag}.recon.mp4'}")
    print(f"{'='*60}")

    if add_corpus:
        add_to_corpus(tag, url or "", duration, topic)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end: YouTube URL → training-ready events.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("url", nargs="?", default=None,
                    help="YouTube URL (optional if video already downloaded)")
    ap.add_argument("--tag", required=True,
                    help="Short identifier (e.g. oct-algebra)")
    ap.add_argument("--duration", type=int, default=None,
                    help="Cap video to this many seconds (default: full)")
    ap.add_argument("--topic", default="",
                    help="Topic description for corpus.json")
    ap.add_argument("--from", dest="from_step", default=None,
                    choices=STEPS,
                    help="Resume from this step (skip earlier steps)")
    ap.add_argument("--add-to-corpus", action="store_true",
                    help="Add this video to corpus.json after processing")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would run without executing")
    args = ap.parse_args()

    if not args.url and not (VIDEOS / f"{args.tag}.mp4").exists():
        ap.error("provide a URL or ensure the video is already downloaded")

    run_pipeline(
        tag=args.tag,
        url=args.url,
        duration=args.duration,
        from_step=args.from_step,
        topic=args.topic,
        add_corpus=args.add_to_corpus,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
