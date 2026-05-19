"""Three-phase curriculum training for OCTModel.

Phase 1 — Stroke decoder only
    Trains the stroke decoder to reproduce pen movements given a word
    embedding.  The outer transformer is frozen; only word_embed and
    stroke_decoder are updated.  This grounds the geometry before
    language structure is introduced.

Phase 2 — Outer LM only
    Trains the causal transformer to predict the next word given prior
    words.  The stroke decoder is frozen.

Phase 3 — Joint, lower LR
    Trains everything end-to-end.  Word loss + stroke loss combined.

Usage
-----
    python3 train.py --phase 1 --epochs 20
    python3 train.py --phase 2 --epochs 20
    python3 train.py --phase 3 --epochs 30
    python3 train.py          # runs all three phases sequentially

Checkpoints are saved to checkpoints/<name>.pt after each phase.
Training logs (one JSON per epoch) go to checkpoints/<name>.log.jsonl.
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

from data  import OCTDataset, build_vocab, save_vocab, load_vocab, collate_fn, P_STOP
from model import OCTModel

ROOT  = Path(__file__).resolve().parent
TRAIN_OUT = ROOT / "output"          # training.jsonl files
CKPT      = ROOT / "checkpoints"
VOCAB_FILE = CKPT / "vocab.json"

STROKE_LOSS_WEIGHT = 0.1   # λ in L = L_word + λ·L_stroke


# ─── device ───────────────────────────────────────────────────────────────────

def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ─── loss functions ───────────────────────────────────────────────────────────

def word_loss(
    logits:   torch.Tensor,   # (B, T, V)
    word_ids: torch.Tensor,   # (B, T)
    pad_id:   int,
) -> torch.Tensor:
    """Causal LM loss: predict word at t+1 given words 0..t.

    Target is word_ids shifted left by one; padding is ignored.
    """
    # logits[:, :-1] predicts word_ids[:, 1:]
    logits_flat = logits[:, :-1].reshape(-1, logits.size(-1))
    target_flat = word_ids[:, 1:].reshape(-1)
    return nn.functional.cross_entropy(
        logits_flat, target_flat, ignore_index=pad_id
    )


def stroke_loss(
    xy_pred:  torch.Tensor,   # (N, S, 2)
    p_logits: torch.Tensor,   # (N, S, 3)
    xy_tgt:   torch.Tensor,   # (N, S, 2)
    p_tgt:    torch.Tensor,   # (N, S) long
    pad_mask: torch.Tensor,   # (N, S) bool  True = pad
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stroke reconstruction loss.

    xy loss: MSE on (x,y) at non-pad, non-stop positions.
    p  loss: cross-entropy over pen states at all non-pad positions.
    """
    valid = ~pad_mask                         # (N, S) True = real token
    is_not_stop = (p_tgt != P_STOP)          # don't penalize xy at stop

    # xy loss — only where pen is not at stop and not padding
    xy_mask = (valid & is_not_stop).float()  # (N, S)
    xy_err  = ((xy_pred - xy_tgt) ** 2).sum(-1)  # (N, S)
    n_xy    = xy_mask.sum().clamp_min(1.0)
    l_xy    = (xy_err * xy_mask).sum() / n_xy

    # p loss — all non-pad positions
    p_flat   = p_logits[valid]               # (M, 3)
    tgt_flat = p_tgt[valid]                  # (M,)
    l_p = nn.functional.cross_entropy(p_flat, tgt_flat) if p_flat.size(0) > 0 \
          else p_logits.new_tensor(0.0)

    return l_xy, l_p


# ─── phase helpers ────────────────────────────────────────────────────────────

def set_phase(model: OCTModel, phase: int) -> None:
    """Freeze / unfreeze parameters for each training phase."""
    # Start: freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    if phase == 1:
        # Stroke decoder + word embeddings (so embeddings develop during phase 1)
        for p in model.stroke_decoder.parameters():
            p.requires_grad_(True)
        model.word_embed.weight.requires_grad_(True)

    elif phase == 2:
        # Outer transformer (word_embed, pos_enc, transformer, word_head, style)
        for name, p in model.named_parameters():
            if "stroke_decoder" not in name:
                p.requires_grad_(True)

    elif phase == 3:
        # Everything
        for p in model.parameters():
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = model.n_params
    print(f"  phase {phase}: {trainable/1e6:.2f}M / {total/1e6:.2f}M params trainable")


# ─── training step ────────────────────────────────────────────────────────────

def step_batch(
    model:   OCTModel,
    batch:   dict,
    device:  str,
    phase:   int,
    pad_id:  int,
) -> dict[str, float]:
    """Run one training batch and return loss components."""
    word_ids = batch["word_ids"].to(device)
    word_pad = batch["word_pad_mask"].to(device)
    strokes  = batch["strokes"]

    hidden, w_logits = model.encode_words(word_ids, word_pad)

    losses: dict[str, float] = {}
    total = word_ids.new_tensor(0.0, dtype=torch.float32)

    # ── word loss ─────────────────────────────────────────────────────────────
    if phase in (2, 3):
        l_word = word_loss(w_logits, word_ids, pad_id)
        total  = total + l_word
        losses["word"] = l_word.item()

    # ── stroke loss ───────────────────────────────────────────────────────────
    if phase in (1, 3) and strokes is not None:
        bidx = strokes["batch_idx"].to(device)
        wpos = strokes["word_pos"].to(device)
        sxy  = strokes["xy"].to(device)
        sp   = strokes["p"].to(device)
        spad = strokes["pad_mask"].to(device)

        ctx = model.extract_ctx(hidden, bidx, wpos)   # (N, d_model)
        xy_pred, p_logits = model.decode_strokes(ctx, sxy, sp, spad)

        l_xy, l_p = stroke_loss(xy_pred, p_logits, sxy, sp, spad)
        l_stroke   = l_xy + l_p
        total      = total + STROKE_LOSS_WEIGHT * l_stroke
        losses["stroke_xy"] = l_xy.item()
        losses["stroke_p"]  = l_p.item()

    losses["total"] = total.item()
    return total, losses


# ─── main training loop ───────────────────────────────────────────────────────

def run_phase(
    model:      OCTModel,
    train_dl:   DataLoader,
    val_dl:     DataLoader,
    phase:      int,
    epochs:     int,
    lr:         float,
    device:     str,
    pad_id:     int,
    log_path:   Path,
    ckpt_path:  Path,
    patience:   int = 5,
) -> None:
    set_phase(model, phase)

    opt   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.02,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.05
    )

    log_f     = log_path.open("a")
    best_val  = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_totals: dict[str, float] = {}
        n_batches = 0

        pbar = tqdm(train_dl, desc=f"phase{phase} ep{epoch}/{epochs}", leave=False)
        for batch in pbar:
            opt.zero_grad()
            total, parts = step_batch(model, batch, device, phase, pad_id)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            for k, v in parts.items():
                train_totals[k] = train_totals.get(k, 0.0) + v
            n_batches += 1
            pbar.set_postfix({k: f"{v:.3f}" for k, v in parts.items()
                              if k != "total"})

        sched.step()

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        val_totals: dict[str, float] = {}
        n_val = 0
        with torch.no_grad():
            for batch in val_dl:
                _, parts = step_batch(model, batch, device, phase, pad_id)
                for k, v in parts.items():
                    val_totals[k] = val_totals.get(k, 0.0) + v
                n_val += 1

        train_avg = {k: v / max(n_batches, 1) for k, v in train_totals.items()}
        val_avg   = {k: v / max(n_val,    1) for k, v in val_totals.items()}
        val_loss  = val_avg.get("total", float("inf"))
        elapsed   = time.time() - t0

        log_entry = {
            "phase":  phase,
            "epoch":  epoch,
            "train":  train_avg,
            "val":    val_avg,
            "lr":     sched.get_last_lr()[0],
            "elapsed": round(elapsed, 1),
        }
        log_f.write(json.dumps(log_entry) + "\n")
        log_f.flush()

        print(
            f"  ep{epoch:03d}  "
            f"train {train_avg.get('total', 0):.4f}  "
            f"val {val_loss:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # ── early stopping + checkpointing ────────────────────────────────────
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save({
                "phase":       phase,
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "config": {
                    "vocab_size":      model.vocab_size,
                    "d_model":         model.d_model,
                    "stroke_d_model":  model.stroke_decoder.d_model,
                },
            }, ckpt_path)
            print(f"    ✓ saved checkpoint  (val={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    early stop (no improvement for {patience} epochs)")
                break

    log_f.close()
    print(f"phase {phase} done.  best val loss: {best_val:.4f}")


# ─── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Train OCTModel")
    ap.add_argument("--phase",      type=int,   default=None,
                    help="1=stroke, 2=word, 3=joint (default: run all)")
    ap.add_argument("--epochs-p1",  type=int,   default=20)
    ap.add_argument("--epochs-p2",  type=int,   default=20)
    ap.add_argument("--epochs-p3",  type=int,   default=30)
    ap.add_argument("--batch-size", type=int,   default=32)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--lr-p3",      type=float, default=2e-4)
    ap.add_argument("--val-frac",   type=float, default=0.05,
                    help="Fraction of episodes held out for validation")
    ap.add_argument("--d-model",    type=int,   default=256)
    ap.add_argument("--n-layers",   type=int,   default=4)
    ap.add_argument("--d-stroke",   type=int,   default=128)
    ap.add_argument("--n-stroke-layers", type=int, default=2)
    ap.add_argument("--dropout",    type=float, default=0.3)
    ap.add_argument("--min-freq",   type=int,   default=2)
    ap.add_argument("--patience",   type=int,   default=5)
    ap.add_argument("--device",     default=None)
    ap.add_argument("--resume",     default=None,
                    help="Path to checkpoint to resume from")
    args = ap.parse_args()

    device = args.device or pick_device()
    CKPT.mkdir(parents=True, exist_ok=True)

    # ── vocabulary ────────────────────────────────────────────────────────────
    if VOCAB_FILE.exists():
        print(f"loading vocab from {VOCAB_FILE}")
        vocab = load_vocab(VOCAB_FILE)
    else:
        print("building vocabulary …")
        vocab   = build_vocab(TRAIN_OUT, min_freq=args.min_freq)
        save_vocab(vocab, VOCAB_FILE)
        print(f"  vocab size: {len(vocab)}  → {VOCAB_FILE}")

    pad_id = vocab["<pad>"]

    # ── dataset ───────────────────────────────────────────────────────────────
    print("loading dataset …")
    full_ds = OCTDataset(TRAIN_OUT, vocab, augment=False)

    n_val   = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    # Enable augmentation only on training split
    train_ds.dataset.augment = True   # type: ignore

    coll = partial(collate_fn, pad_id=pad_id)
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
    model = OCTModel(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_stroke=args.d_stroke,
        n_stroke_layers=args.n_stroke_layers,
        dropout=args.dropout,
        pad_idx=pad_id,
    ).to(device)
    print(f"model: {model.n_params/1e6:.2f}M params  device={device}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"resumed from {args.resume}")

    # ── run phases ────────────────────────────────────────────────────────────
    phases = [args.phase] if args.phase else [1, 2, 3]

    for phase in phases:
        lr = args.lr_p3 if phase == 3 else args.lr
        epochs = {1: args.epochs_p1, 2: args.epochs_p2, 3: args.epochs_p3}[phase]

        print(f"\n{'─'*50}")
        print(f"Phase {phase}  |  epochs={epochs}  lr={lr}")

        run_phase(
            model     = model,
            train_dl  = train_dl,
            val_dl    = val_dl,
            phase     = phase,
            epochs    = epochs,
            lr        = lr,
            device    = device,
            pad_id    = pad_id,
            log_path  = CKPT / f"phase{phase}.log.jsonl",
            ckpt_path = CKPT / f"phase{phase}.best.pt",
            patience  = args.patience,
        )

    print("\ntraining complete.")


if __name__ == "__main__":
    main()
