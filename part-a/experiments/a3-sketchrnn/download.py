"""Download QuickDraw stroke datasets in Sketch-RNN format.

These .npz files are pre-processed by Google for the Sketch-RNN paper:
each split ('train', 'valid', 'test') is an object array of variable-
length stroke sequences, where each step is (Δx, Δy, pen_state).
pen_state is 0 (pen down to next point) or 1 (pen up; next point starts
a new stroke).

Usage:
    python3 download.py cat            # one class
    python3 download.py cat dog face   # several
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/sketchrnn"


def download(klass: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"{klass}.npz"
    if out.exists():
        print(f"[{klass}] already exists ({out.stat().st_size//1024} KB), skipping")
        return
    url = f"{BASE_URL}/{klass}.npz"
    print(f"[{klass}] downloading {url}")
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        sys.exit(f"failed: HTTP {r.status_code} on {url}")
    total = int(r.headers.get("content-length", 0))
    with out.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                   desc=klass) as bar:
        for chunk in r.iter_content(64 * 1024):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"[{klass}] -> {out} ({out.stat().st_size//1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("classes", nargs="+", help="QuickDraw classes (e.g. cat dog face)")
    args = ap.parse_args()
    for k in args.classes:
        download(k)


if __name__ == "__main__":
    main()
