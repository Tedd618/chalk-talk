"""Distributed video processing queue using atomic filesystem operations.

Each job is a small JSON file living in one of four folders:
    queue/pending/<tag>.json   — waiting to be picked up
    queue/running/<tag>.json   — claimed by a machine, currently processing
    queue/done/<tag>.json      — finished successfully
    queue/failed/<tag>.json    — failed; see the file for error details

Claiming a job = os.rename(pending → running).
This is atomic on any POSIX filesystem (local or NFS), so two machines
can never claim the same job even if they race.

Usage:
    # Add videos to the queue
    python3 worker.py add https://youtube.com/watch?v=XXX --tag mrh-algebra --topic "Algebra basics" --channel mrh

    # Show queue status
    python3 worker.py status

    # Process jobs on this machine (loops until queue is empty)
    python3 worker.py run

    # Rescue jobs stuck in running/ for more than N hours (default 3)
    python3 worker.py rescue --hours 3
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
QUEUE   = ROOT / "queue"
PENDING = QUEUE / "pending"
RUNNING = QUEUE / "running"
DONE    = QUEUE / "done"
FAILED  = QUEUE / "failed"
A2_DIR  = ROOT.parent / "a2-alignment"
PY      = str(ROOT / ".venv" / "bin" / "python")

# Large temporary data (videos, frames) goes here — outside the home quota.
# Uses /tmp if available and has enough space, otherwise falls back to ROOT.
_tmp_candidate = Path("/tmp") / f"chalk-{os.getenv('USER', 'user')}"
SCRATCH = _tmp_candidate if _tmp_candidate.parent.exists() else ROOT

STALE_HOURS = 3   # jobs running longer than this are considered crashed


# ── helpers ──────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def hostname() -> str:
    return platform.node()

def load_job(path: Path) -> dict:
    return json.loads(path.read_text())

def save_job(path: Path, job: dict) -> None:
    path.write_text(json.dumps(job, indent=2, ensure_ascii=False))


# ── add ──────────────────────────────────────────────────────────────────────

def cmd_add(args) -> None:
    tag = args.tag
    for folder in (PENDING, RUNNING, DONE, FAILED):
        if (folder / f"{tag}.json").exists():
            print(f"[{tag}] already in queue ({folder.name}/). skipping.")
            return

    job = {
        "tag":              tag,
        "url":              args.url,
        "topic":            args.topic or "",
        "channel":          args.channel or "",
        "duration_cap_sec": args.cap or 0,
        "status":           "pending",
        "worker":           None,
        "added_at":         now_iso(),
        "started_at":       None,
        "finished_at":      None,
        "error":            None,
    }
    save_job(PENDING / f"{tag}.json", job)
    print(f"[{tag}] added to queue/pending/")


# ── status ────────────────────────────────────────────────────────────────────

def cmd_status(_args) -> None:
    rows = []
    for folder in (PENDING, RUNNING, DONE, FAILED):
        for p in sorted(folder.glob("*.json")):
            job = load_job(p)
            worker = job.get("worker") or "-"
            started = (job.get("started_at") or "")[:16].replace("T", " ")
            rows.append((folder.name, job["tag"], job.get("topic", "")[:40], worker, started))

    if not rows:
        print("queue is empty")
        return

    print(f"\n{'status':<9} {'tag':<28} {'topic':<42} {'worker':<20} {'started'}")
    print("-" * 115)
    for status, tag, topic, worker, started in rows:
        print(f"{status:<9} {tag:<28} {topic:<42} {worker:<20} {started}")

    counts = {f: len(list((QUEUE / f).glob("*.json")))
              for f in ("pending", "running", "done", "failed")}
    print(f"\n  pending={counts['pending']}  running={counts['running']}"
          f"  done={counts['done']}  failed={counts['failed']}\n")


# ── rescue ────────────────────────────────────────────────────────────────────

def cmd_rescue(args) -> None:
    hours = args.hours or STALE_HOURS
    cutoff = time.time() - hours * 3600
    rescued = 0
    for p in RUNNING.glob("*.json"):
        job = load_job(p)
        started = job.get("started_at")
        if not started:
            continue
        started_ts = datetime.fromisoformat(started).timestamp()
        if started_ts < cutoff:
            job["status"]   = "pending"
            job["worker"]   = None
            job["started_at"] = None
            dest = PENDING / p.name
            save_job(p, job)
            os.rename(p, dest)
            print(f"[{job['tag']}] rescued from running/ (stuck >{hours}h)")
            rescued += 1
    if rescued == 0:
        print(f"nothing stale (threshold: {hours}h)")


# ── run ───────────────────────────────────────────────────────────────────────

def claim_next() -> tuple[Path, dict] | tuple[None, None]:
    """Atomically claim the oldest pending job. Returns (running_path, job) or (None, None)."""
    candidates = sorted(PENDING.glob("*.json"))
    for p in candidates:
        dest = RUNNING / p.name
        try:
            os.rename(p, dest)   # atomic — only one machine wins the race
        except FileNotFoundError:
            continue             # another machine claimed it first
        job = load_job(dest)
        job["status"]     = "running"
        job["worker"]     = hostname()
        job["started_at"] = now_iso()
        save_job(dest, job)
        return dest, job
    return None, None


def run_step(label: str, cmd: list[str], cwd: Path,
             extra_env: dict | None = None) -> tuple[bool, str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
    if res.returncode != 0:
        err = (res.stderr + res.stdout).strip()[:500]
        return False, f"step '{label}' failed:\n{err}"
    return True, ""


def process_job(job_path: Path, job: dict) -> None:
    tag = job["tag"]
    cap = int(job.get("duration_cap_sec") or 0)

    # scratch dir for this job's videos and frames
    scratch = SCRATCH / tag
    scratch.mkdir(parents=True, exist_ok=True)
    env = {"CHALK_SCRATCH": str(scratch)}

    print(f"\n=== [{hostname()}] {tag}: {job.get('topic', '')}")

    steps = [
        ("download", [PY, "download.py", tag]),
        ("frames",   [PY, "frames.py",   tag, "--duration", str(cap)]),
        ("extract",  [PY, "extract_v4.py", tag]),
        ("pages",    [PY, "pages.py",    tag]),
        ("stt",      [PY, str(A2_DIR / "stt.py"), tag] + (["--duration", str(cap)] if cap else [])),
        ("merge",    [PY, str(A2_DIR / "merge.py"), tag]),
    ]

    for label, cmd in steps:
        print(f"  ▸ {label} … ", end="", flush=True)
        t0 = time.time()
        ok, err = run_step(label, cmd, ROOT, extra_env=env)
        dt = time.time() - t0
        if ok:
            if label == "extract":
                # copy v4 output to canonical name
                v4 = ROOT / "output" / f"{tag}.v4.strokes.jsonl"
                canon = ROOT / "output" / f"{tag}.strokes.jsonl"
                if v4.exists():
                    shutil.copy2(v4, canon)
            if label == "frames":
                # video no longer needed — delete from scratch
                for ext in (".mp4", ".webm", ".mkv"):
                    vf = scratch / f"{tag}{ext}"
                    if vf.exists():
                        vf.unlink()
                        print(f"(deleted video) ", end="")
            if label == "extract":
                # frames no longer needed — delete from scratch
                frames_dir = scratch / "frames"
                if frames_dir.exists():
                    shutil.rmtree(frames_dir)
                    print(f"(deleted frames) ", end="")
            print(f"ok ({dt:.1f}s)")
        else:
            shutil.rmtree(scratch, ignore_errors=True)  # clean up on failure too
            print(f"FAIL ({dt:.1f}s)")
            job["status"]      = "failed"
            job["finished_at"] = now_iso()
            job["error"]       = err
            save_job(job_path, job)
            dest = FAILED / job_path.name
            os.rename(job_path, dest)
            print(f"  [{tag}] moved to queue/failed/")
            return

    shutil.rmtree(scratch, ignore_errors=True)  # final cleanup

    job["status"]      = "done"
    job["finished_at"] = now_iso()
    save_job(job_path, job)
    dest = DONE / job_path.name
    os.rename(job_path, dest)
    print(f"  [{tag}] done → queue/done/")


def cmd_run(_args) -> None:
    print(f"worker starting on {hostname()}")
    processed = 0
    while True:
        job_path, job = claim_next()
        if job is None:
            print(f"no pending jobs. processed {processed} this session. exiting.")
            break
        process_job(job_path, job)
        processed += 1


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    for d in (PENDING, RUNNING, DONE, FAILED):
        d.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser(description="Distributed OCT corpus processing queue")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # add
    p_add = sub.add_parser("add", help="Add a video to the queue")
    p_add.add_argument("url",      help="YouTube URL")
    p_add.add_argument("--tag",    required=True, help="Short identifier, e.g. mrh-algebra")
    p_add.add_argument("--topic",  default="",    help="Human-readable topic description")
    p_add.add_argument("--channel",default="",    help="Channel name, e.g. mrh / oct")
    p_add.add_argument("--cap",    type=int, default=0, help="Duration cap in seconds (0=full)")

    # status
    sub.add_parser("status", help="Show queue state across all machines")

    # run
    sub.add_parser("run", help="Claim and process jobs until queue is empty")

    # rescue
    p_rescue = sub.add_parser("rescue", help="Return stale running jobs to pending")
    p_rescue.add_argument("--hours", type=float, default=STALE_HOURS,
                          help=f"Hours before a running job is considered stale (default {STALE_HOURS})")

    args = ap.parse_args()
    {"add": cmd_add, "status": cmd_status, "run": cmd_run, "rescue": cmd_rescue}[args.cmd](args)


if __name__ == "__main__":
    main()
