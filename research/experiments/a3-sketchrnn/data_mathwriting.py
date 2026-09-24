"""MathWriting (Google 2024) handwritten-math stroke dataset.

InkML format: each .inkml file is one math expression with multiple
<trace> elements. Each trace is one continuous pen-down stroke
captured at sub-pixel precision with timestamps.

Data download:
    https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz
The CC BY-NC-SA license is research-friendly. See ./data/mathwriting-2024
or ./data/mathwriting-2024-excerpt for extracted contents.

Conversion to our 5-element representation:
    [Δx_norm, Δy_norm, p_down, p_up, p_end]
- Concatenate all traces into one sequence.
- Within a trace: (Δx, Δy) of consecutive points, with p_down=1.
- Between traces: emit one (Δx, Δy, p_up) point bridging the gap.
- After the last trace: synthesize a (0, 0, 0, 0, 1) end token.
- Δs are normalized by a single global std computed on the train set.
"""
from __future__ import annotations
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATA = Path(__file__).resolve().parent / "data"
INKML_NS = "{http://www.w3.org/2003/InkML}"


def parse_inkml(path: Path) -> list[list[tuple[float, float]]]:
    """Return list of strokes; each stroke is a list of (x, y) points."""
    tree = ET.parse(path)
    root = tree.getroot()
    strokes: list[list[tuple[float, float]]] = []
    for trace in root.findall(f"{INKML_NS}trace"):
        text = (trace.text or "").strip()
        if not text:
            continue
        pts: list[tuple[float, float]] = []
        for tok in text.split(","):
            parts = tok.strip().split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            pts.append((x, y))
        if len(pts) >= 2:
            strokes.append(pts)
    return strokes


def strokes_to_deltas(strokes: list[list[tuple[float, float]]]) -> np.ndarray:
    """Convert a list of (x, y) strokes to (N, 3) (Δx, Δy, pen_state).
    pen_state: 0 = pen down (line drawn to next point); 1 = pen up
    (next point starts a new stroke). Following the same convention
    as QuickDraw."""
    if not strokes:
        return np.zeros((0, 3), dtype=np.float32)
    rows: list[list[float]] = []
    prev_x, prev_y = strokes[0][0]
    for si, stroke in enumerate(strokes):
        # If this is not the first stroke, the FIRST point of this stroke
        # is reached by a pen-up move from the previous stroke's end.
        if si > 0:
            x0, y0 = stroke[0]
            rows.append([x0 - prev_x, y0 - prev_y, 1.0])  # pen-up
            prev_x, prev_y = x0, y0
        for (x, y) in stroke[1:]:
            rows.append([x - prev_x, y - prev_y, 0.0])  # pen-down
            prev_x, prev_y = x, y
    return np.asarray(rows, dtype=np.float32)


def find_split_dir(root: Path, split: str) -> Path:
    """Find the named split directory under data/, supporting both the
    full release and the excerpt (which is nested one level deeper)."""
    direct = root / split
    if direct.exists():
        return direct
    # try excerpt or named release
    for candidate in root.glob(f"mathwriting-*"):
        sub = candidate / split
        if sub.exists():
            return sub
    raise FileNotFoundError(f"split '{split}' not found under {root}")


def load_split_strokes(split: str = "train",
                       limit: int | None = None) -> list[np.ndarray]:
    """Return a list of (N, 3) (Δx, Δy, pen_state) arrays, one per
    InkML file in the split."""
    split_dir = find_split_dir(DATA, split)
    paths = sorted(split_dir.glob("*.inkml"))
    if limit:
        paths = paths[:limit]
    out: list[np.ndarray] = []
    for p in paths:
        try:
            strokes = parse_inkml(p)
        except ET.ParseError:
            continue
        deltas = strokes_to_deltas(strokes)
        if len(deltas) >= 2:
            out.append(deltas)
    return out


def compute_global_scale(samples: list[np.ndarray]) -> float:
    deltas = np.concatenate([s[:, :2] for s in samples], axis=0)
    return float(deltas.std())


def to_5d(seq: np.ndarray, max_len: int, scale: float) -> tuple[np.ndarray, int]:
    """Convert (N, 3) → (max_len + 1, 5) padded — same format as
    QuickDraw data.py.to_5d so the model code works unchanged."""
    n = min(len(seq), max_len)
    out = np.zeros((max_len + 1, 5), dtype=np.float32)
    out[:n, 0] = seq[:n, 0] / scale
    out[:n, 1] = seq[:n, 1] / scale
    pen_states = seq[:n, 2].astype(np.int64)
    out[:n, 2] = (pen_states == 0).astype(np.float32)  # p_down
    out[:n, 3] = (pen_states == 1).astype(np.float32)  # p_up
    out[n:, 4] = 1.0                                    # p_end
    return out, n


class MathWritingDataset(Dataset):
    def __init__(self, split: str = "train", max_len: int = 600,
                 scale: float | None = None, limit: int | None = None):
        samples = load_split_strokes(split, limit=limit)
        # filter overly long sequences (training stability)
        samples = [s for s in samples if len(s) <= max_len]
        self.samples = samples
        self.max_len = max_len
        if scale is None:
            scale = compute_global_scale(samples) if samples else 1.0
        self.scale = scale

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seq, n = to_5d(self.samples[idx], self.max_len, self.scale)
        return torch.from_numpy(seq), n


def collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = torch.stack([b[0] for b in batch], dim=0)
    lens = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return seqs, lens


if __name__ == "__main__":
    # smoke test on the excerpt
    ds = MathWritingDataset("train", max_len=600)
    print(f"{len(ds)} samples, scale={ds.scale:.3f}")
    if len(ds) > 0:
        s, n = ds[0]
        print(f"shape: {s.shape}, length: {n}")
        # show stroke length distribution
        lens = [len(s) for s in ds.samples[:200]]
        if lens:
            import statistics
            print(f"stroke seq lengths: min/median/max = "
                  f"{min(lens)} / {int(statistics.median(lens))} / {max(lens)}")
