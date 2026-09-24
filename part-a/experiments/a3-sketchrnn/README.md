# A.3 — Stroke Transformer Pretraining

> **Status (Sept 2026): standalone.** This validated the MDN + pen-state
> modeling stack (QuickDraw cat smoke test, then 229k MathWriting
> expressions). The a4 model was ultimately trained from scratch on the
> teacher corpus and does not load this checkpoint. `prime.py` (feed a real
> stroke prefix, let the model complete it) is the most useful tool here.

Pretrain a Sketch-RNN-derived transformer on public stroke datasets.
Validates the modeling stack and produces a checkpoint with clean
math-stroke priors.

## Pipeline

```
QuickDraw .npz / MathWriting .inkml
    │
    ├─ data.py            (QuickDraw)
    └─ data_mathwriting.py (MathWriting InkML parser)
                          ↓
                  5-element format
                  [Δx_n, Δy_n, p_down, p_up, p_end]
                          ↓
                       model.py
              transformer + MDN + pen-state heads
                          ↓
                      train.py
                AdamW + cosine LR schedule
                          ↓
                  checkpoints/<name>.pt
                          ↓
                      sample.py
              temperature-controlled generation
                          ↓
                  samples/<name>_T<n>.png
```

## Files

- `download.py` — fetch QuickDraw class .npz files
- `data.py` — QuickDraw stroke dataset
- `data_mathwriting.py` — MathWriting InkML parser → 5-element format
- `viz_data.py` — render raw training samples for sanity inspection
- `model.py` — transformer + MDN + pen-state heads (~3.3M params)
- `train.py` — training loop with `--dataset {quickdraw,mathwriting}`
- `sample.py` — generation + grid render

## Status

**A.3 done.** Two checkpoints trained:

| Checkpoint | Dataset | Final loss | Notes |
|---|---|---|---|
| `cat.final.pt` | QuickDraw cat (70k) | -0.02 | Smoke test. Generated samples look like cats. Validated the modeling stack. |
| `mathwriting.final.pt` | MathWriting train (229k) | -2.85 | A.3 deliverable. Stroke priors for math handwriting. |

**Sample interpretation note**: unconditional samples from
`mathwriting.final.pt` look like *math-flavored gibberish* — strokes
with the right cadence and density of human math handwriting, but
not forming recognizable expressions. This is the expected outcome
of unconditional generation on a diverse dataset where every sample
is a *different* unique expression. The model learned *how* to write
math (stroke dynamics, pen-up timing, math aesthetics) but not *what*
to write — that signal comes from conditioning, which A.4 adds via
canvas + topic + speech inputs.

The training metrics confirm the model is healthy:
- 98% pen-state classification accuracy
- Tight per-step `(Δx, Δy)` prediction (negative GMM NLL)
- No infinite loops; strokes terminate cleanly
- ~3.3M params, ~13 MB checkpoint

## Decision log

- **CROHME**: dropped because official hosting is 404 in 2026 and
  every accessible mirror was either gated, image-only, or auth-
  required.
- **MathWriting (Google 2024)**: pivoted to it. Larger, cleaner, CC
  BY-NC-SA licensed, publicly hosted on GCS. 229k human-written math
  expressions in InkML format.
- **IAM Online**: dropped. OCT teacher writes math, not English
  sentences. MathWriting variables cover what's needed.
- **SketchAgent synthetic data**: skipped permanently. Quality and
  bias concerns; defeats the "learn from real handwriting" premise.
- **LaTeX-label conditioning**: not used. A.3 is unconditional. A
  label-conditioned mid-stage between A.3 and A.4 is a possible
  future experiment if A.4 needs more stepping stones.

## Reproducing

```bash
# 1. Set up
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Cat smoke test
.venv/bin/python download.py cat
.venv/bin/python train.py cat --steps 10000 --save-every 2000
.venv/bin/python sample.py cat --n 16 --temp 0.4

# 3. MathWriting pretraining
mkdir -p data && cd data
curl -sLO "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz"
tar xzf mathwriting-2024.tgz
cd ..
.venv/bin/python train.py mathwriting --dataset mathwriting \
    --max-len 400 --batch-size 32 --steps 10000 --save-every 2000
.venv/bin/python sample.py mathwriting --n 16 --temp 0.4 --max-steps 600
```

The `mathwriting.final.pt` checkpoint becomes the input to A.4.
