"""QuickDraw stroke dataset for Sketch-RNN-style training.

Loads a class .npz file produced by Google's pre-processing for
Sketch-RNN. Each sample is a sequence of (Δx, Δy, pen_state) where
pen_state is 0 (pen down to next) or 1 (pen up; next point starts a
new stroke).

We expand pen_state to a 5-element representation following the
Sketch-RNN paper:
    [Δx, Δy, p_down, p_up, p_end]
with p_end one-hot on the LAST step (synthesized end-of-sketch token).

Δx, Δy are normalized by a single global scale (std of all offsets in
the training split) so the model sees ~unit-variance inputs.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

DATA = Path(__file__).resolve().parent / "data"


def load_split(klass: str, split: str) -> list[np.ndarray]:
    npz_path = DATA / f"{klass}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}; run download.py {klass}")
    arr = np.load(npz_path, allow_pickle=True, encoding="latin1")
    return [s.astype(np.float32) for s in arr[split]]


def compute_global_scale(samples: list[np.ndarray]) -> float:
    """Single std for all (Δx, Δy) values across the dataset."""
    deltas = np.concatenate([s[:, :2] for s in samples], axis=0)
    return float(deltas.std())


def to_5d(seq: np.ndarray, max_len: int, scale: float) -> tuple[np.ndarray, int]:
    """Convert (N, 3) (Δx, Δy, pen_state) → (max_len+1, 5) padded sequence.

    Padding past the actual length uses (0, 0, 0, 0, 1) — a 'pen-end'
    state, so the loss can be masked simply by the recorded length.
    """
    n = min(len(seq), max_len)
    out = np.zeros((max_len + 1, 5), dtype=np.float32)
    # one-hot p_down=1 by default for valid steps; we overwrite below
    out[:n, 0] = seq[:n, 0] / scale
    out[:n, 1] = seq[:n, 1] / scale
    pen_states = seq[:n, 2].astype(np.int64)
    out[:n, 2] = (pen_states == 0).astype(np.float32)  # p_down
    out[:n, 3] = (pen_states == 1).astype(np.float32)  # p_up
    # synthesize end-of-sketch on the position right after the last point
    out[n:, 4] = 1.0
    return out, n


class StrokeDataset(Dataset):
    def __init__(self, klass: str, split: str = "train",
                 max_len: int = 200, scale: float | None = None):
        samples = load_split(klass, split)
        # filter out sequences longer than max_len for training stability
        samples = [s for s in samples if len(s) <= max_len]
        self.samples = samples
        self.max_len = max_len
        if scale is None:
            scale = compute_global_scale(samples)
        self.scale = scale

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seq, n = to_5d(self.samples[idx], self.max_len, self.scale)
        return torch.from_numpy(seq), n


def collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = torch.stack([b[0] for b in batch], dim=0)   # (B, T+1, 5)
    lens = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return seqs, lens


if __name__ == "__main__":
    # smoke test
    ds = StrokeDataset("cat", "train", max_len=200)
    print(f"{len(ds)} samples, scale={ds.scale:.3f}")
    s, n = ds[0]
    print(f"shape: {s.shape}, length: {n}")
    print(f"first 3 rows:\n{s[:3]}")
