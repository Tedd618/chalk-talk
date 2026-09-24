# a4-train — flat stroke+speech transformer and the context ablation

Everything here trains on Google Colab (T4). The local `.py` files are a
reference copy of the model/data code; the notebooks are self-contained.

## Files

| file | role |
|---|---|
| `align.py` | `a2-alignment/output/*.events.jsonl` → `output/*.training.jsonl` (word-anchored: every token is a spoken word with the strokes drawn during it; silent drawing becomes `<silent>`) |
| `data.py`, `model.py`, `train.py` | local reference implementation of the v3 model: flat interleaved sequence, MDN stroke head (20 bivariate Gaussians), `(dx, dy, x, y)` stroke input |
| `train_colab.ipynb` | single-model v3 training run + generation figures |
| `context_ablation_colab.ipynb` | **the experiment**: 7 conditions × 3 seeds, per-window bootstrap, results table, generation figures. Generated from `tools/build_notebook.py` |
| `results/ctx_ablation_v2/` | the numbers behind the report: `summary.json`, `result_<cond>_s<seed>.json` (per-epoch history; the three untagged `result_<cond>.json` files are seed 42 from the first v2 attempt, reused), `per_window.json`, `vocab.json`, generation PNGs. Checkpoints (`*.pt`) are not committed |
| `tools/` | `build_notebook.py` builds the ablation notebook from raw-string cells and emits a byte-identical library; `smoke_test.py` runs the whole pipeline on CPU with a tiny model before anything goes to Colab (it caught two real bugs) |

## Reproduce

1. Get `output.zip` from the
   [`data-v1` release](https://github.com/Tedd618/chalk-talk/releases/tag/data-v1)
   and upload it to `My Drive/chalk-talk/a4-train/output.zip`.
2. Open `context_ablation_colab.ipynb` in Colab with a T4 runtime, Run all.
   ≈ 40 min per seed; results are saved per (condition, seed) so it resumes
   after a disconnect.
3. The final cells print the results table and the length-controlled
   deltas and write `summary.json`.

To regenerate the data instead: run `a1-stroke-extraction/pipeline.py` per
video, then `a2-alignment/merge.py`, then `python align.py --all` here.

## Conditions

```
full             [w][w][pen dx,dy][pen dx,dy][pen dx,dy][w]   everything visible
mask_strokes     [w][w][pen 0,0 ][pen 0,0 ][pen 0,0 ][w]      pen exists, values zeroed
nostroke_hidden  [w][w][  ---   ][  ---   ][  ---   ][w]      same positions, masked out of attention
drop_strokes     [w][w][w]                                     pen tokens removed
```
and the mirror ladder `mask_words` / `noword_hidden` / `drop_words` for the
stroke loss. Design, confound history and results:
`../../docs/METHOD1_CONTEXT_EXPERIMENT.md`.

## Results (held-out, mean over seeds 42/43/44)

| step | Δ (nats) | verdict |
|---|---|---|
| pen values become real → next word | +0.125 ± 0.077 | robust, helps |
| empty pen slots visible → next word | −0.266 ± 0.119 | robust, hurts |
| words spread to original spacing | −0.151 ± 0.066 | robust, hurts |
| word identities real → next pen move | −0.037 ± 0.029 | robust, hurts (small) |

Positive = adding that information reduced the loss.
