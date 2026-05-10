"""Render raw MathWriting InkML samples to a grid image.

This shows the *training data* — what the model is learning to imitate.
Each panel is one math expression, drawn from the actual stylus
trajectories in the .inkml files.

Usage:
    .venv/bin/python viz_data.py                    # 12 random samples from train
    .venv/bin/python viz_data.py --n 24 --seed 7
    .venv/bin/python viz_data.py --split valid
    .venv/bin/python viz_data.py --color-per-stroke
"""
from __future__ import annotations
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
NS = "{http://www.w3.org/2003/InkML}"

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def parse_inkml_full(path: Path) -> tuple[list[list[tuple[float, float]]], str]:
    tree = ET.parse(path)
    root = tree.getroot()
    label = ""
    for ann in root.findall(f"{NS}annotation"):
        if ann.attrib.get("type") == "normalizedLabel":
            label = (ann.text or "").strip()
    if not label:
        for ann in root.findall(f"{NS}annotation"):
            if ann.attrib.get("type") == "label":
                label = (ann.text or "").strip()
    strokes: list[list[tuple[float, float]]] = []
    for trace in root.findall(f"{NS}trace"):
        pts: list[tuple[float, float]] = []
        for tok in (trace.text or "").split(","):
            parts = tok.strip().split()
            if len(parts) >= 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        if len(pts) >= 2:
            strokes.append(pts)
    return strokes, label


def render(ax, strokes, label: str, color_mode: str) -> None:
    for i, stroke in enumerate(strokes):
        xs = [p[0] for p in stroke]
        ys = [-p[1] for p in stroke]   # flip y for image-coord
        col = PALETTE[i % len(PALETTE)] if color_mode == "palette" else "black"
        ax.plot(xs, ys, color=col, linewidth=1.4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    title = label[:48] + ("..." if len(label) > 48 else "")
    # plain text — matplotlib mathtext can't parse complex LaTeX
    # (\begin{matrix}, etc.). The raw source is more useful anyway.
    ax.set_title(title, fontsize=7, family="monospace")


def find_split_dir(split: str) -> Path:
    for candidate in [
        ROOT / "data" / "mathwriting-2024" / split,
        ROOT / "data" / "mathwriting-2024-excerpt" / split,
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"split '{split}' not found under data/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--color-per-stroke", action="store_true",
                    help="One color per stroke (default: all black)")
    args = ap.parse_args()

    split_dir = find_split_dir(args.split)
    files = sorted(split_dir.glob("*.inkml"))
    if not files:
        raise SystemExit(f"no inkml files in {split_dir}")
    rng = random.Random(args.seed)
    chosen = rng.sample(files, min(args.n, len(files)))

    cols = int(np.ceil(np.sqrt(args.n)))
    rows = int(np.ceil(args.n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.4))
    axes = np.atleast_1d(axes).flatten()

    color_mode = "palette" if args.color_per_stroke else "single"
    for i, p in enumerate(chosen):
        try:
            strokes, label = parse_inkml_full(p)
            render(axes[i], strokes, label, color_mode)
        except ET.ParseError:
            axes[i].set_axis_off()
            axes[i].set_title("(parse error)", fontsize=8)
    for j in range(args.n, len(axes)):
        axes[j].set_axis_off()

    n_strokes_each = [len(parse_inkml_full(p)[0]) for p in chosen]
    fig.suptitle(
        f"MathWriting — {args.split} samples ({args.n})  "
        f"strokes per sample (median): {int(np.median(n_strokes_each))}",
        fontsize=10,
    )
    fig.tight_layout()
    out = ROOT / "samples" / f"mathwriting_{args.split}_data.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
