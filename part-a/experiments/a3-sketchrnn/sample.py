"""Sample sketches from a trained checkpoint and render them as a grid.

Usage:
    .venv/bin/python sample.py cat                 # uses cat.final.pt
    .venv/bin/python sample.py cat --step 5000     # specific checkpoint
    .venv/bin/python sample.py cat --temp 0.6 --n 16
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import StrokeTransformer, sample as model_sample

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints"
SAMPLES = ROOT / "samples"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def render_sketch(ax, pts: list[tuple[float, float, int]], scale: float) -> None:
    """Draw a single sketch onto a matplotlib axis."""
    if not pts:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center")
        ax.set_axis_off()
        return
    x = y = 0.0
    cur_x = [x]
    cur_y = [y]
    strokes_x = []
    strokes_y = []
    for dx, dy, pen in pts:
        x += dx * scale
        y += dy * scale
        if pen == 0:
            cur_x.append(x); cur_y.append(y)
        elif pen == 1:
            cur_x.append(x); cur_y.append(y)
            strokes_x.append(cur_x); strokes_y.append(cur_y)
            cur_x = [x]; cur_y = [y]
        else:  # end
            cur_x.append(x); cur_y.append(y)
            break
    if cur_x:
        strokes_x.append(cur_x); strokes_y.append(cur_y)

    for sx, sy in zip(strokes_x, strokes_y):
        # flip y: image-coordinate convention (y down) for QuickDraw
        ax.plot(sx, [-v for v in sy], color="black", linewidth=1)
    ax.set_aspect("equal")
    ax.set_axis_off()


def load_checkpoint(klass: str, step: int | None) -> tuple[dict, str]:
    if step is None:
        path = CKPT / f"{klass}.final.pt"
    else:
        path = CKPT / f"{klass}.step_{step}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    return torch.load(path, weights_only=False, map_location="cpu"), str(path.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("klass")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step (default: final)")
    ap.add_argument("--n", type=int, default=16, help="number of samples")
    ap.add_argument("--temp", type=float, default=0.4)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or pick_device()
    print(f"device: {device}")
    ckpt, name = load_checkpoint(args.klass, args.step)
    cfg = ckpt["config"]
    scale = float(ckpt["scale"])
    print(f"loaded {name} (scale={scale:.2f}, step={ckpt.get('step','?')})")

    model = StrokeTransformer(
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["layers"],
        num_components=cfg["num_components"],
        max_len=cfg["max_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    SAMPLES.mkdir(parents=True, exist_ok=True)

    cols = int(np.ceil(np.sqrt(args.n)))
    rows = int(np.ceil(args.n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.atleast_1d(axes).flatten()

    print(f"sampling {args.n} sketches at temperature {args.temp}...")
    for i in range(args.n):
        pts = model_sample(model, max_steps=args.max_steps,
                           temperature=args.temp, device=device)
        render_sketch(axes[i], pts, scale)
        axes[i].set_title(f"{i}", fontsize=8)
    for j in range(args.n, len(axes)):
        axes[j].set_axis_off()

    fig.suptitle(f"{args.klass} samples (T={args.temp})  "
                 f"ckpt={name}", fontsize=10)
    fig.tight_layout()
    out = SAMPLES / f"{args.klass}_T{args.temp}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
