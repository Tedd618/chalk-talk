"""Run the full a1+a2 pipeline on every video in corpus.json.

For each video tag:
    1. download (skip if mp4 exists)
    2. frames (capped to corpus duration_cap_sec; 0 = full)
    3. extract strokes
    4. transcribe with whisper-small (cap matches frames)
    5. merge into events.jsonl
    6. reconstruct stroke-only mp4

Stops on the first failed step per tag and continues to the next tag.
Prints a summary table at the end.

Usage:
    .venv/bin/python batch.py          # all tags in corpus.json
    .venv/bin/python batch.py oct-fractions  # one tag
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

A1_DIR = Path(__file__).resolve().parent
A2_DIR = A1_DIR.parent / "a2-alignment"
CORPUS = A1_DIR / "corpus.json"
PY = str(A1_DIR / ".venv" / "bin" / "python")


def load_corpus() -> list[dict]:
    return json.loads(CORPUS.read_text())["videos"]


def run(label: str, args: list[str], cwd: Path) -> tuple[bool, float]:
    print(f"  ▸ {label} … ", end="", flush=True)
    t0 = time.time()
    res = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"FAIL ({dt:.1f}s)")
        print(f"    stderr: {res.stderr.strip()[:300]}")
        print(f"    stdout: {res.stdout.strip()[:300]}")
        return False, dt
    print(f"ok ({dt:.1f}s)")
    return True, dt


def process_one(entry: dict) -> dict:
    tag = entry["tag"]
    cap = int(entry.get("duration_cap_sec") or 0)
    print(f"\n=== [{tag}] {entry.get('topic','')}")
    if cap:
        print(f"    duration cap: {cap}s")
    timing: dict[str, float] = {}

    # 1. download
    ok, dt = run("download", [PY, "download.py", tag], A1_DIR)
    timing["download"] = dt
    if not ok:
        return {"tag": tag, "status": "fail", "step": "download", "timing": timing}

    # 2. frames (with duration cap; 0 = full per frames.py default behavior
    #    but we encode 0 → use a large duration to mean "full" by passing
    #    --duration 0 which frames.py interprets as full).
    frames_args = [PY, "frames.py", tag, "--duration", str(cap)]
    ok, dt = run("frames", frames_args, A1_DIR)
    timing["frames"] = dt
    if not ok:
        return {"tag": tag, "status": "fail", "step": "frames", "timing": timing}

    # 3. extract strokes (v4 skeleton pipeline)
    ok, dt = run("extract", [PY, "extract_v4.py", tag], A1_DIR)
    timing["extract"] = dt
    if not ok:
        return {"tag": tag, "status": "fail", "step": "extract", "timing": timing}
    # Copy v4 output to canonical name for downstream tools
    import shutil
    v4_out = A1_DIR / "output" / f"{tag}.v4.strokes.jsonl"
    canonical = A1_DIR / "output" / f"{tag}.strokes.jsonl"
    if v4_out.exists():
        shutil.copy2(v4_out, canonical)

    # 3b. detect page breaks (canvas-clear events)
    ok, dt = run("pages", [PY, "pages.py", tag], A1_DIR)
    timing["pages"] = dt
    if not ok:
        return {"tag": tag, "status": "fail", "step": "pages", "timing": timing}

    # 4. STT
    stt_args = [PY, str(A2_DIR / "stt.py"), tag]
    if cap:
        stt_args += ["--duration", str(cap)]
    ok, dt = run("stt", stt_args, A1_DIR)
    timing["stt"] = dt
    if not ok:
        return {"tag": tag, "status": "fail", "step": "stt", "timing": timing}

    # 5. merge
    ok, dt = run("merge", [PY, str(A2_DIR / "merge.py"), tag], A1_DIR)
    timing["merge"] = dt
    if not ok:
        return {"tag": tag, "status": "fail", "step": "merge", "timing": timing}

    # 6. reconstruct
    ok, dt = run("reconstruct", [PY, "reconstruct.py", tag], A1_DIR)
    timing["reconstruct"] = dt
    if not ok:
        return {"tag": tag, "status": "fail", "step": "reconstruct", "timing": timing}

    # collect counts
    strokes_path = A1_DIR / "output" / f"{tag}.strokes.jsonl"
    words_path = A2_DIR / "output" / f"{tag}.words.jsonl"
    n_strokes = sum(1 for _ in strokes_path.open()) if strokes_path.exists() else 0
    n_words = sum(1 for _ in words_path.open()) if words_path.exists() else 0
    return {
        "tag": tag,
        "status": "ok",
        "n_strokes": n_strokes,
        "n_words": n_words,
        "timing": timing,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default=None,
                    help="Process only this tag (default: all in corpus.json)")
    args = ap.parse_args()

    corpus = load_corpus()
    if args.tag:
        corpus = [e for e in corpus if e["tag"] == args.tag]
        if not corpus:
            sys.exit(f"no entry with tag={args.tag}")

    results = []
    for entry in corpus:
        results.append(process_one(entry))

    # summary
    print("\n=== SUMMARY ===")
    print(f"{'tag':<18} {'status':<6} {'strokes':>8} {'words':>7} "
          f"{'frames':>7} {'extract':>8} {'stt':>7} total")
    for r in results:
        t = r.get("timing", {})
        total = sum(t.values())
        if r["status"] == "ok":
            print(f"{r['tag']:<18} {'ok':<6} {r['n_strokes']:>8} {r['n_words']:>7} "
                  f"{t.get('frames',0):>7.1f} {t.get('extract',0):>8.1f} "
                  f"{t.get('stt',0):>7.1f} {total:>5.1f}s")
        else:
            print(f"{r['tag']:<18} {'FAIL':<6} step={r['step']}")


if __name__ == "__main__":
    main()
