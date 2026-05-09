"""Run Whisper STT on a video and emit word-level timestamps.

Output: <a2_root>/output/<tag>.words.jsonl
        one JSON per line: {"word": str, "t_start": float, "t_end": float}

Reads videos from a1's videos/ directory. We use faster-whisper with
word_timestamps=True; the timestamps are aligned to the audio track,
which is the same time base as the frame index from a1's extractor.

Usage:
    python3 stt.py oct
    python3 stt.py oct --model small
    python3 stt.py oct --duration 180  # only the first 180 seconds
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

A2_DIR = Path(__file__).resolve().parent
A1_DIR = A2_DIR.parent / "a1-stroke-extraction"
VIDEOS = A1_DIR / "videos"
OUTPUT = A2_DIR / "output"


def transcribe(tag: str, model_size: str = "small",
               duration: float | None = None) -> None:
    src = VIDEOS / f"{tag}.mp4"
    if not src.exists():
        sys.exit(f"video not found: {src}. run a1 download.py first.")

    # imported here so a quick --help on this script doesn't pay the cost
    from faster_whisper import WhisperModel

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / f"{tag}.words.jsonl"

    print(f"[{tag}] loading whisper model: {model_size}")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"[{tag}] transcribing {src.name}"
          f"{' (first ' + str(duration) + 's)' if duration else ''}...")
    segments, info = model.transcribe(
        str(src),
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        clip_timestamps=[0, duration] if duration else None,
    )
    print(f"[{tag}] detected language: {info.language} "
          f"(prob {info.language_probability:.2f}); duration {info.duration:.1f}s")

    n_words = 0
    with out_path.open("w") as f:
        for seg in segments:
            if not seg.words:
                continue
            for w in seg.words:
                f.write(json.dumps({
                    "word": w.word.strip(),
                    "t_start": round(float(w.start), 4),
                    "t_end": round(float(w.end), 4),
                }, ensure_ascii=False) + "\n")
                n_words += 1
    print(f"[{tag}] wrote {n_words} words -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--model", default="small",
                    help="Whisper model size (tiny/base/small/medium/large-v3) "
                         "(default %(default)s)")
    ap.add_argument("--duration", type=float, default=None,
                    help="Transcribe only the first N seconds")
    args = ap.parse_args()
    transcribe(args.tag, model_size=args.model, duration=args.duration)


if __name__ == "__main__":
    main()
