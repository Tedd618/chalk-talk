"""Single-phase training for FlatOCTModel (v2 — scaled delta-xy).

No curriculum. One unified transformer trained with composite loss
on interleaved word+stroke sequences.

Usage
-----
    python3 train.py --epochs 60
    python3 train.py --epochs 60 --seq-len 512 --batch-size 4

Checkpoints saved to checkpoints/flat.best.pt.
Training logs go to checkpoints/flat.log.jsonl.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import (
    FlatOCTDataset, build_vocab, save_vocab, load_vocab,
    flat_collate_fn,
)
from model import FlatOCTModel

ROOT       = Path(__file__).resolve().parent
TRAIN_OUT  = ROOT / "output"
CKPT       = ROOT / "checkpoints"
VOCAB_FILE = CKPT / "vocab.json"


# ─── device ──────────────────────────────────────────────────────────────────

def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ─── training step ───────────────────────────────────────────────────────────

def step_batch(
    model:     FlatOCTModel,
    batch:     dict,
    device:    str,
    xy_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one training batch and return (loss_tensor, loss_dict)."""
    token_types = batch["token_types"].to(device)
    word_ids    = batch["word_ids"].to(device)
    xy          = batch["xy"].to(device)
    pen         = batch["pen"].to(device)
    pad_mask    = batch["pad_mask"].to(device)

    outputs = model(token_types, word_ids, xy, pen, pad_mask)
    losses  = model.compute_loss(
        outputs, token_types, word_ids, xy, pen, pad_mask,
        xy_weight=xy_weight,
    )

    return losses["total"], {k: v.item() for k, v in losses.items()}


# ─── main training loop ─────────────────────────────────────────────────────

def train(
    model:      FlatOCTModel,
    train_dl:   DataLoader,
    val_dl:     DataLoader,
    epochs:     int,
    lr:         float,
    device:     str,
    log_path:   Path,
    ckpt_path:  Path,
    xy_weight:  float = 1.0,
    warmup:     int   = 50,
    patience:   int   = 15,
    grad_clip:  float = 1.0,
) -> None:
    """Train the model."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Linear warmup + cosine decay
    total_steps = epochs * len(train_dl)
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + __import__("math").cos(3.14159 * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Mixed precision
    use_amp = (device == "cuda")
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    log_f      = log_path.open("a")
    best_val   = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_totals: dict[str, float] = {}
        n_batches = 0

        pbar = tqdm(train_dl, desc=f"ep{epoch:03d}/{epochs}", leave=False)
        for batch in pbar:
            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    total, parts = step_batch(model, batch, device, xy_weight)
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                total, parts = step_batch(model, batch, device, xy_weight)
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            sched.step()

            for k, v in parts.items():
                train_totals[k] = train_totals.get(k, 0.0) + v
            n_batches += 1
            pbar.set_postfix(
                {k: f"{v:.3f}" for k, v in parts.items() if k != "total"},
                refresh=False,
            )

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        val_totals: dict[str, float] = {}
        n_val = 0
        with torch.no_grad():
            for batch in val_dl:
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        _, parts = step_batch(model, batch, device, xy_weight)
                else:
                    _, parts = step_batch(model, batch, device, xy_weight)
                for k, v in parts.items():
                    val_totals[k] = val_totals.get(k, 0.0) + v
                n_val += 1

        train_avg = {k: v / max(n_batches, 1) for k, v in train_totals.items()}
        val_avg   = {k: v / max(n_val, 1)     for k, v in val_totals.items()}
        val_loss  = val_avg.get("total", float("inf"))
        elapsed   = time.time() - t0

        log_entry = {
            "epoch":   epoch,
            "train":   train_avg,
            "val":     val_avg,
            "lr":      sched.get_last_lr()[0],
            "elapsed": round(elapsed, 1),
        }
        log_f.write(json.dumps(log_entry) + "\n")
        log_f.flush()

        print(
            f"  ep{epoch:03d}  "
            f"train={train_avg.get('total', 0):.4f}  "
            f"val={val_loss:.4f}  "
            f"(type={val_avg.get('type', 0):.3f} "
            f"word={val_avg.get('word', 0):.3f} "
            f"xy={val_avg.get('xy', 0):.4f} "
            f"pen={val_avg.get('pen', 0):.3f})  "
            f"({elapsed:.0f}s)"
        )

        # ── early stopping + checkpointing ───────────────────────────────────
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "config": {
                    "vocab_size": model.vocab_size,
                    "d_model":    model.d_model,
                    "n_layers":   len(model.transformer.layers),
                    "n_heads":    model.transformer.layers[0].self_attn.num_heads,
                },
            }, ckpt_path)
            print(f"    -> saved (val={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    early stop (no improvement for {patience} epochs)")
                break

    log_f.close()
    print(f"\ntraining done. best val loss: {best_val:.4f}")


# ─── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Train FlatOCTModel v2")
    ap.add_argument("--epochs",     type=int,   default=60)
    ap.add_argument("--batch-size", type=int,   default=4)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--seq-len",    type=int,   default=512)
    ap.add_argument("--d-model",    type=int,   default=384)
    ap.add_argument("--n-layers",   type=int,   default=6)
    ap.add_argument("--n-heads",    type=int,   default=8)
    ap.add_argument("--d-ff",       type=int,   default=1536)
    ap.add_argument("--dropout",    type=float, default=0.05)
    ap.add_argument("--xy-weight",  type=float, default=1.0)
    ap.add_argument("--warmup",     type=int,   default=50)
    ap.add_argument("--patience",   type=int,   default=15)
    ap.add_argument("--val-frac",   type=float, default=0.1)
    ap.add_argument("--min-freq",   type=int,   default=2)
    ap.add_argument("--device",     default=None)
    ap.add_argument("--resume",     default=None,
                    help="Path to checkpoint to resume from")
    args = ap.parse_args()

    device = args.device or pick_device()
    CKPT.mkdir(parents=True, exist_ok=True)

    # ── vocabulary ────────────────────────────────────────────────────────────
    if VOCAB_FILE.exists():
        vocab = load_vocab(VOCAB_FILE)
        if len(vocab) < 50:
            print(f"stale vocab ({len(vocab)} tokens), rebuilding...")
            VOCAB_FILE.unlink()
            vocab = build_vocab(TRAIN_OUT, min_freq=args.min_freq)
            save_vocab(vocab, VOCAB_FILE)
        else:
            print(f"loaded vocab: {len(vocab)} tokens")
    else:
        print("building vocabulary...")
        vocab = build_vocab(TRAIN_OUT, min_freq=args.min_freq)
        save_vocab(vocab, VOCAB_FILE)
    print(f"  vocab size: {len(vocab)}")

    pad_id = vocab["<pad>"]

    # ── dataset ───────────────────────────────────────────────────────────────
    print("loading dataset...")
    full_ds = FlatOCTDataset(
        TRAIN_OUT, vocab,
        seq_len=args.seq_len,
        augment=False,
    )

    n_val   = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    # Enable augmentation on training split only
    train_ds.dataset.augment = True  # type: ignore

    coll = partial(flat_collate_fn, pad_id=pad_id)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=coll, num_workers=0, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=coll, num_workers=0,
    )
    print(f"  train: {n_train}  val: {n_val}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = FlatOCTModel(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=max(args.seq_len * 2, 2048),
        dropout=args.dropout,
        pad_idx=pad_id,
    ).to(device)
    print(f"model: {model.n_params/1e6:.2f}M params  device={device}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"resumed from {args.resume}")

    # ── train ─────────────────────────────────────────────────────────────────
    train(
        model     = model,
        train_dl  = train_dl,
        val_dl    = val_dl,
        epochs    = args.epochs,
        lr        = args.lr,
        device    = device,
        log_path  = CKPT / "flat.log.jsonl",
        ckpt_path = CKPT / "flat.best.pt",
        xy_weight = args.xy_weight,
        warmup    = args.warmup,
        patience  = args.patience,
    )


if __name__ == "__main__":
    main()
