"""Train the stroke transformer on a QuickDraw class.

Usage:
    .venv/bin/python train.py cat                         # default settings
    .venv/bin/python train.py cat --steps 10000           # short run
    .venv/bin/python train.py cat --device mps            # Apple Silicon GPU
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import StrokeDataset, collate
from model import StrokeTransformer, loss_fn

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train(klass: str, *, steps: int, batch_size: int, lr: float,
          d_model: int, nhead: int, layers: int, num_components: int,
          max_len: int, log_every: int, save_every: int,
          device: str) -> None:
    print(f"loading dataset: {klass}")
    train_ds = StrokeDataset(klass, "train", max_len=max_len)
    val_ds = StrokeDataset(klass, "valid", max_len=max_len, scale=train_ds.scale)
    print(f"train: {len(train_ds)}  valid: {len(val_ds)}  scale: {train_ds.scale:.3f}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0,
                              drop_last=True)

    print(f"device: {device}")
    model = StrokeTransformer(
        d_model=d_model, nhead=nhead,
        num_layers=layers, num_components=num_components,
        max_len=max_len + 2,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                      eta_min=lr * 0.05)

    CKPT.mkdir(parents=True, exist_ok=True)
    log_path = CKPT / f"{klass}.train_log.jsonl"
    log_f = log_path.open("a")

    model.train()
    step = 0
    t0 = time.time()
    pbar = tqdm(total=steps, desc=f"train {klass}")
    train_iter = iter(train_loader)
    last = {"loss": float("nan")}
    while step < steps:
        try:
            seqs, lens = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            seqs, lens = next(train_iter)
        seqs = seqs.to(device)
        lens = lens.to(device)

        opt.zero_grad()
        loss, parts = loss_fn(model, seqs, lens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

        step += 1
        pbar.update(1)

        if step % log_every == 0 or step == 1:
            last = {"step": step, **parts, "lr": sched.get_last_lr()[0]}
            log_f.write(json.dumps(last) + "\n"); log_f.flush()
            pbar.set_postfix(loss=f"{parts['loss']:.3f}",
                             gmm=f"{parts['gmm']:.3f}",
                             pen=f"{parts['pen']:.3f}")

        if save_every and step % save_every == 0:
            ckpt_path = CKPT / f"{klass}.step_{step}.pt"
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "scale": train_ds.scale,
                "config": {
                    "d_model": d_model, "nhead": nhead, "layers": layers,
                    "num_components": num_components,
                    "max_len": max_len + 2,
                },
            }, ckpt_path)
            tqdm.write(f"  saved {ckpt_path.name}")

    # final checkpoint
    final_path = CKPT / f"{klass}.final.pt"
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "scale": train_ds.scale,
        "config": {
            "d_model": d_model, "nhead": nhead, "layers": layers,
            "num_components": num_components,
            "max_len": max_len + 2,
        },
    }, final_path)
    print(f"\nfinal checkpoint: {final_path.name}")
    print(f"elapsed: {(time.time() - t0)/60:.1f} min")
    log_f.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("klass", help="QuickDraw class name (e.g. cat)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--num-components", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--device", default=None,
                    help="cpu / mps / cuda (default auto)")
    args = ap.parse_args()

    device = args.device or pick_device()
    train(
        args.klass,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        d_model=args.d_model, nhead=args.nhead, layers=args.layers,
        num_components=args.num_components, max_len=args.max_len,
        log_every=args.log_every, save_every=args.save_every,
        device=device,
    )


if __name__ == "__main__":
    main()
