"""Primed generation: feed a real stroke prefix, let the model continue.

Given a real MathWriting expression, show the first N strokes as a blue
prefix and let the model autoregressively complete the rest in red.
Produces a grid of K completions so you can see variance.

Usage:
    # random expression, first 2 strokes as prefix, 6 completions
    .venv/bin/python prime.py mathwriting

    # show what the model completes after the fraction bar
    .venv/bin/python prime.py mathwriting --prime-strokes 3 --n 6

    # specific expression index
    .venv/bin/python prime.py mathwriting --idx 42 --prime-strokes 4

    # search for fractions (expressions containing \\frac in ground truth)
    .venv/bin/python prime.py mathwriting --search frac --prime-strokes 3

    # show full reference expression alongside completions
    .venv/bin/python prime.py mathwriting --show-full
"""
from __future__ import annotations
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F

from model import StrokeTransformer
from data_mathwriting import parse_inkml, strokes_to_deltas, find_split_dir, DATA

INKML_NS = "{http://www.w3.org/2003/InkML}"
ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints"
SAMPLES = ROOT / "samples"


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# InkML helpers
# ---------------------------------------------------------------------------

def get_truth_label(path: Path) -> str:
    """Extract ground-truth LaTeX from an InkML file (if present)."""
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        for ann in root.findall(f"{INKML_NS}annotation"):
            if ann.get("type") in ("label", "truth", "normalizedLabel"):
                return (ann.text or "").strip()
    except Exception:
        pass
    return ""


def find_inkml_files(split: str = "train") -> list[Path]:
    split_dir = find_split_dir(DATA, split)
    return sorted(split_dir.glob("*.inkml"))


def pick_file(paths: list[Path], idx: int | None, search: str | None,
              seed: int | None) -> tuple[Path, str]:
    """Select one InkML file. Returns (path, truth_label)."""
    if search:
        # filter to expressions whose truth label contains search string
        matched = []
        for p in paths:
            label = get_truth_label(p)
            if search.lower() in label.lower():
                matched.append((p, label))
        if not matched:
            raise ValueError(f"No expressions found containing '{search}'")
        print(f"  Found {len(matched)} expressions matching '{search}'")
        if idx is not None:
            p, label = matched[idx % len(matched)]
        else:
            rng = random.Random(seed)
            p, label = rng.choice(matched)
        return p, label

    if idx is not None:
        p = paths[idx % len(paths)]
    else:
        rng = random.Random(seed)
        p = rng.choice(paths)
    return p, get_truth_label(p)


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

def strokes_to_5d_raw(strokes: list[list[tuple[float, float]]],
                      scale: float) -> np.ndarray:
    """Convert stroke list → (N, 5) normalized, no padding."""
    deltas = strokes_to_deltas(strokes)  # (N, 3)
    n = len(deltas)
    out = np.zeros((n, 5), dtype=np.float32)
    out[:, 0] = deltas[:, 0] / scale
    out[:, 1] = deltas[:, 1] / scale
    pen = deltas[:, 2].astype(int)
    out[:, 2] = (pen == 0).astype(np.float32)   # p_down
    out[:, 3] = (pen == 1).astype(np.float32)   # p_up
    # p_end is 0 everywhere; we add it only at the actual end
    return out


def split_at_stroke(seq5: np.ndarray, n_strokes: int) -> int:
    """Return the index of the first pen event AFTER the n-th complete stroke.

    A stroke ends at a pen-up event (p_up=1). If there are fewer than
    n_strokes pen-ups, returns the full length.
    """
    count = 0
    for i, row in enumerate(seq5):
        if row[3] > 0.5:  # p_up
            count += 1
            if count >= n_strokes:
                return i + 1  # include the pen-up event itself
    return len(seq5)


# ---------------------------------------------------------------------------
# Autoregressive continuation
# ---------------------------------------------------------------------------

@torch.no_grad()
def continue_from_prefix(
    model: StrokeTransformer,
    prefix: np.ndarray,         # (P, 5) — real events, already normalized
    max_steps: int,
    temperature: float,
    device: str,
) -> list[tuple[float, float, int]]:
    """Run the model on prefix, then autoregressively continue.

    Returns the *generated* portion as (Δx, Δy, pen_state) triples
    (does NOT include the prefix itself).
    """
    model.eval()
    P = len(prefix)
    max_len = model.max_len

    # Build the initial context tensor from prefix
    # If prefix > max_len, take the tail
    start = max(0, P - max_len)
    ctx = torch.tensor(prefix[start:], dtype=torch.float32,
                       device=device).unsqueeze(0)  # (1, P', 5)

    out_pts: list[tuple[float, float, int]] = []

    for _ in range(max_steps):
        gauss, mix_logits, pen_logits = model(ctx)
        last_g = gauss[0, -1]        # (M, 5)
        last_m = mix_logits[0, -1]   # (M,)
        last_p = pen_logits[0, -1]   # (3,)

        # Sample mixture component
        log_mix = F.log_softmax(last_m / max(temperature, 1e-6), dim=-1)
        comp = torch.distributions.Categorical(logits=log_mix).sample().item()
        mu_x = last_g[comp, 0].item()
        mu_y = last_g[comp, 1].item()
        sigma_x = last_g[comp, 2].exp().item() * (temperature ** 0.5)
        sigma_y = last_g[comp, 3].exp().item() * (temperature ** 0.5)
        rho = torch.tanh(last_g[comp, 4]).item()

        z1, z2 = torch.randn(2).tolist()
        dx = mu_x + sigma_x * z1
        dy = mu_y + sigma_y * (rho * z1 + (1 - rho ** 2) ** 0.5 * z2)

        pen_t = last_p / max(temperature, 1e-6)
        pen = torch.distributions.Categorical(logits=pen_t).sample().item()

        out_pts.append((float(dx), float(dy), int(pen)))
        if pen == 2:
            break

        next_row = torch.zeros((1, 1, 5), device=device)
        next_row[0, 0, 0] = dx
        next_row[0, 0, 1] = dy
        next_row[0, 0, 2 + pen] = 1.0
        ctx = torch.cat([ctx, next_row], dim=1)
        if ctx.shape[1] >= max_len:
            ctx = ctx[:, -max_len:]

    return out_pts


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_events(ax, events: list[tuple[float, float, int]],
                  scale: float, color: str,
                  x0: float = 0.0, y0: float = 0.0,
                  lw: float = 1.5) -> tuple[float, float]:
    """Render (Δx, Δy, pen) events onto ax.

    Returns the final (x, y) position so continuations can start there.
    """
    x, y = x0, y0
    cur_x = [x]
    cur_y = [y]
    for dx, dy, pen in events:
        x += dx * scale
        y += dy * scale
        if pen == 0:
            cur_x.append(x)
            cur_y.append(y)
        elif pen == 1:
            cur_x.append(x)
            cur_y.append(y)
            if len(cur_x) >= 2:
                ax.plot(cur_x, [-v for v in cur_y], color=color, linewidth=lw)
            cur_x = [x]
            cur_y = [y]
        else:  # end
            break
    if len(cur_x) >= 2:
        ax.plot(cur_x, [-v for v in cur_y], color=color, linewidth=lw)
    return x, y


def render_panel(ax, prefix_events, prefix_scale,
                 gen_events, gen_scale, title: str = ""):
    """Render one subplot: prefix in blue, continuation in red."""
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Render prefix
    x_end, y_end = render_events(ax, prefix_events, prefix_scale,
                                 color="#2563eb", lw=1.8)
    # Render continuation (starts where prefix ended)
    if gen_events:
        render_events(ax, gen_events, gen_scale,
                      color="#dc2626", lw=1.5, x0=x_end, y0=y_end)

    if title:
        ax.set_title(title, fontsize=7, pad=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prime the stroke model with a real prefix, see continuations.")
    ap.add_argument("klass", help="checkpoint name (e.g. mathwriting)")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step (default: final)")
    ap.add_argument("--split", default="train",
                    help="dataset split to draw primes from")
    ap.add_argument("--idx", type=int, default=None,
                    help="expression index within matching set")
    ap.add_argument("--search", default=None,
                    help="filter expressions whose label contains this string")
    ap.add_argument("--prime-strokes", type=int, default=2,
                    help="number of complete strokes to use as prefix")
    ap.add_argument("--prime-events", type=int, default=None,
                    help="use first N pen events as prefix (overrides --prime-strokes)")
    ap.add_argument("--n", type=int, default=6,
                    help="number of independent completions to generate")
    ap.add_argument("--temp", type=float, default=0.4)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--show-full", action="store_true",
                    help="add a column showing the full ground-truth expression")
    ap.add_argument("--file", default=None,
                    help="path to a specific InkML file (overrides --search/--idx)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or pick_device()
    print(f"device: {device}")

    # ---- load checkpoint ----
    if args.step is None:
        ckpt_path = CKPT / f"{args.klass}.final.pt"
    else:
        ckpt_path = CKPT / f"{args.klass}.step_{args.step}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    scale = float(ckpt["scale"])
    print(f"loaded {ckpt_path.name} (scale={scale:.4f})")

    model = StrokeTransformer(
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["layers"],
        num_components=cfg["num_components"],
        max_len=cfg["max_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- pick expression ----
    if args.file:
        inkml_path = Path(args.file)
        truth = get_truth_label(inkml_path)
    else:
        all_paths = find_inkml_files(args.split)
        if not all_paths:
            all_paths = find_inkml_files("test")
        if not all_paths:
            raise FileNotFoundError(f"No InkML files found for split '{args.split}'")
        inkml_path, truth = pick_file(all_paths, args.idx, args.search, args.seed)

    print(f"expression: {inkml_path.name}")
    if truth:
        print(f"  ground truth: {truth}")

    strokes = parse_inkml(inkml_path)
    if not strokes:
        raise ValueError("Expression has no strokes.")

    # ---- build full sequence and prefix ----
    seq5 = strokes_to_5d_raw(strokes, scale)  # (N, 5) normalized

    if args.prime_events is not None:
        split_idx = min(args.prime_events, len(seq5))
    else:
        split_idx = split_at_stroke(seq5, args.prime_strokes)

    if split_idx <= 0:
        split_idx = min(5, len(seq5))
        print(f"  warning: fewer strokes than requested; using {split_idx} events")

    prefix5 = seq5[:split_idx]
    print(f"  total events: {len(seq5)}  |  prefix events: {split_idx}  "
          f"({args.prime_strokes} stroke(s))")

    # Convert prefix to displayable (Δx, Δy, pen_state) list
    prefix_display: list[tuple[float, float, int]] = []
    for row in prefix5:
        pen = 0 if row[2] > 0.5 else (1 if row[3] > 0.5 else 2)
        prefix_display.append((float(row[0]), float(row[1]), pen))

    # Full ground truth for --show-full
    full_display: list[tuple[float, float, int]] = []
    for row in seq5:
        pen = 0 if row[2] > 0.5 else (1 if row[3] > 0.5 else 2)
        full_display.append((float(row[0]), float(row[1]), pen))

    # ---- generate completions ----
    print(f"generating {args.n} completions at T={args.temp}...")
    completions: list[list[tuple[float, float, int]]] = []
    for i in range(args.n):
        gen = continue_from_prefix(
            model, prefix5,
            max_steps=args.max_steps,
            temperature=args.temp,
            device=device,
        )
        completions.append(gen)
        print(f"  [{i+1}/{args.n}] generated {len(gen)} events")

    # ---- render ----
    SAMPLES.mkdir(parents=True, exist_ok=True)

    n_cols = args.n + (1 if args.show_full else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.2, 2.5))
    if n_cols == 1:
        axes = [axes]
    else:
        axes = list(axes)

    col = 0
    if args.show_full:
        ax = axes[col]
        render_panel(ax, full_display, 1.0, [], 1.0, title="ground truth")
        # override: show whole expression in gray
        ax.clear()
        ax.set_aspect("equal")
        ax.set_axis_off()
        render_events(ax, full_display, 1.0, color="#374151", lw=1.5)
        # mark split point (render prefix part in blue on top)
        render_events(ax, prefix_display, 1.0, color="#2563eb", lw=2.0)
        ax.set_title("full expression\n(blue = prefix)", fontsize=7, pad=2)
        col += 1

    for i, gen in enumerate(completions):
        render_panel(axes[col + i], prefix_display, 1.0,
                     gen, 1.0,
                     title=f"completion {i+1}")

    # Legend
    blue_patch = mpatches.Patch(color="#2563eb", label=f"prefix ({split_idx} events)")
    red_patch = mpatches.Patch(color="#dc2626", label="model continuation")
    fig.legend(handles=[blue_patch, red_patch],
               loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))

    title_str = f"Primed generation  |  {args.klass}  |  T={args.temp}"
    if truth:
        # truncate long labels
        disp = truth if len(truth) <= 60 else truth[:57] + "..."
        title_str += f"\n{disp}"
    fig.suptitle(title_str, fontsize=9, y=1.02)
    fig.tight_layout()

    tag = args.search or inkml_path.stem[:12]
    out_name = f"prime_{args.klass}_{tag}_ps{args.prime_strokes}_T{args.temp}.png"
    out = SAMPLES / out_name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
