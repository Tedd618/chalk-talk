# Method 1 — Cross-Modal Context Experiment

**Status:** v1 run completed 2026-09-22 (5 conditions). Found a real design confound in
its "events" comparison and a statistical-rigor gap — see §11. v2 (`context_ablation_colab.ipynb`,
7 conditions, length-controlled, bootstrap CIs) fixes both and is ready to run; v1's results
are kept in Drive (`ctx_ablation/`) and reported alongside v2's for methodological transparency.
**Scope:** only the pure stroke+script model from PLAN.md (Method 1). Methods 2/3 are out of scope.

## 1. The question, stated so it can be answered

PLAN.md asks: *can handwriting strokes and spoken words act as context for each other?*

That is two measurable claims about a causal transformer over the interleaved sequence:

- **W←S:** knowing the strokes drawn so far makes the *next spoken word* more predictable.
- **S←W:** knowing the words spoken so far makes the *next pen movement* more predictable.

We measure each as a held-out negative log-likelihood difference between a model that can see the other modality and an otherwise identical model that cannot. Generation quality is secondary evidence; the NLL deltas are the result. A model can generate ugly output and still give a clean, publishable answer to the question.

## 2. What went wrong before (and why it matters for the design)

| Problem in prior runs | Effect | Fix |
|---|---|---|
| xy head was a deterministic `Linear(d,2)` trained with MSE/Huber | Pen deltas are multimodal (next move can go any direction). The optimal point estimate is the mean ≈ 0, so val xy loss plateaued at epoch 3 and generated strokes collapsed to dots. Classic regression-to-the-mean; Graves 2013 and Sketch-RNN both solved it with a mixture density head. | **MDN head**: 20 bivariate Gaussians, sampled at generation. |
| Training windows sampled uniformly from ~9k-token pages | P(window includes page start) ≈ 0.01%. Model never saw `[BOS, intro words…]`, yet every generation started there. | 20% of windows start at position 0, 10% end at EOS. |
| `<silent>` forbidden during generation | 49.4% of stroke points are introduced by `<silent>`. Half of all paths into drawing were banned at inference. | `<silent>` allowed. |
| Pen class weight 4.0 | Over-predicted pen-up at sampling → strokes shredded into dashes. | Weight 2.0. |
| Raw Whisper tokens as vocab (`video.` ≠ `video,` ≠ `Video`) | 17k types, sparse. | Lowercase + strip outer punctuation → ~half the vocab, +0.8pt coverage. |
| Tensor used as dict key in eval (`id2word[tensor]`) | Every trigger word became `<unk>`; ground-truth text rendered as `?`. Evaluation was testing garbage input. | `.item()` before lookup. |
| Only absolute→delta input; no absolute position given to the model | The model must integrate deltas over 1k tokens to know where it is on the board. | Stroke input feature is `(dx, dy, x, y)`; prediction target stays delta. |
| No control condition at all | "Does context help?" was never measurable. | This experiment. |

## 3. Data (audited 2026-09-22)

- 1,191 videos → 1,191 pages (one page per file; board clears did not survive the upstream pipeline, so absolute coordinates are only locally meaningful).
- ~12.5M flat tokens: **78.4% stroke points, 21.6% word tokens.**
- Words: 1.78M tokens. Raw vocab 17.1k; normalized ≈ 8–9k; min_freq=2 coverage ≈ 99.6%.
- Median page: 8,860 tokens (2,150 words + 6,600 points). 95% of pages exceed a 1,024-token window.
- Strokes per drawing token: median 1, p99 4. Points per stroke: median 6 (10 fps sampling).
- Word runs without drawing: median 2, p90 11 consecutive words.
- Stroke geometry (raw [0,1] units): within-stroke step median 0.0056; between-stroke jump median 0.042, p90 0.30. Scaled ×20: 0.11 vs 6.0 — strongly bimodal, which is exactly what a mixture handles and a point estimate cannot.
- Strokes attach to spoken words 50.6% / to `<silent>` 49.4%.
- Per-token stroke caps (8 strokes / 150 points) discard 0.3% of points — removed; no longer needed by the flat model.

Is 1.78M words / 10.7M points enough? For a from-scratch word LM this is Penn-Treebank scale on a narrow domain; for stroke modeling it is comparable to IAM-OnDB. It is enough to answer the context question. It is not enough for a product, which is not the goal.

## 4. Architecture

Single causal transformer, flat interleaved sequence
`[W:"today"] [W:"we"] [S:dx,dy,x,y,MID] [S:…,UP] [W:<silent>] [S:…] …`

- d_model 256, 4 layers, 8 heads, d_ff 1024, dropout 0.1 (≈6M params — "minimum scale", trains 5 conditions in ~1.5 h on a T4)
- Input per position: `type_embed` + (`word_embed` if WORD) + (`Linear(4→d)(dx,dy,x,y)` + `pen_embed` if STROKE) + sinusoidal PE
- Heads at each position predict position t+1: type (2-way), word (vocab, weight-tied), **MDN** (K=20: π, μ, σ, ρ), pen (2-way)
- Loss: `type_CE + word_CE(label_smooth 0.1) + MDN_NLL + pen_CE(w=[1,2])`
- Delta coordinates scaled ×20 so the MDN's initial σ=1 matches the data std (~1.25)

## 5. Conditions

All five conditions are transforms of the **same sampled window** (same page, same start offset, same seed), so contexts are span-matched. Same model, same hyperparameters, same epoch budget, same initialization seed.

| Condition | Word stream | Stroke stream | Read from it |
|---|---|---|---|
| `full` | real | real | word CE, stroke NLL |
| `mask_strokes` | real | tokens present, features zeroed (model knows *that* a pen event happened, not *where*) | word CE |
| `drop_strokes` | real | tokens removed (pure word LM over the same words) | word CE |
| `mask_words` | non-special words → `<mask>` (model knows *that* a word was spoken, not *which*) | real | stroke NLL |
| `drop_words` | word tokens removed (pure stroke model over the same points; BOS/EOS kept) | real | stroke NLL |

This gives two ladders:

- **Words:** `drop_strokes` (no stroke info) → `mask_strokes` (+ stroke *events*) → `full` (+ stroke *content*)
- **Strokes:** `drop_words` (no word info) → `mask_words` (+ word *events*) → `full` (+ word *content*)

## 6. Metrics (held out, unsmoothed, per target token)

- `word_ce` — cross-entropy in nats over WORD targets (perplexity = e^ce)
- `stroke_nll` — MDN negative log-likelihood in nats over STROKE targets (in scaled-delta space; can be negative)
- `pen_ce`, `type_ce` — sanity/secondary

Validation windows are deterministic (seeded per page), 3 windows per validation page, identical across conditions. 10% of pages held out by page.

**Result statements:**
- `ΔW_content = word_ce(mask_strokes) − word_ce(full)` — value of stroke content for words
- `ΔW_events  = word_ce(drop_strokes) − word_ce(mask_strokes)` — value of stroke timing for words
- `ΔS_content = stroke_nll(mask_words) − stroke_nll(full)` — value of word content for strokes
- `ΔS_events  = stroke_nll(drop_words) − stroke_nll(mask_words)` — value of word timing for strokes

Positive Δ = the context helps. Report with the number of target tokens; with ~30k word targets and ~100k stroke targets the standard error on a CE mean is ~0.01–0.02 nats, so differences ≥0.05 nats are real.

## 7. Qualitative evidence (from `full` only)

1. **Cold start:** `[BOS] + first 5 words of a held-out page` → 800 tokens. Render strokes, print words.
2. **Warm start:** first 400 ground-truth tokens of a held-out page as context → generate 600. Render context (blue), true continuation (grey), generated continuation (red). This is the fair test of "does it keep writing sensibly given real board state."

Sampling: word temp 0.8 / top-k 40; MDN σ-temperature 0.65 (Graves' "bias" trick); `<silent>` allowed; pen temp 1.0.

## 8. Expected outcomes and how to read them

- **Both ΔW_content and ΔS_content clearly > 0:** the central claim holds — each modality is informative context for the other. This is the interesting result for the transformer-context story.
- **ΔS > 0 but ΔW ≈ 0:** words condition drawing (you write what you say), but drawings don't help predict speech at 1k-token context. Still a result; suggests speech is the "driver" modality.
- **Both ≈ 0:** cross-modal attention isn't being used; either the alignment (align.py, max-time-overlap) is too noisy or the context window is too short to bridge modalities. Next step would be checking attention maps and alignment quality, not more epochs.
- Events vs content split tells whether the benefit is *timing* (pause structure) or *semantics* (what was said/drawn).

## 9. Runtime

T4, fp16: ≈ 60 s/epoch for the stroke-bearing conditions at seq 1024 × batch 8, 20 epochs → ≈ 20 min each; `drop_strokes` is ~5× shorter. Whole experiment ≈ 80–90 min. Results are checkpointed per condition to Drive so a disconnect resumes where it stopped.

## 10. References

- Graves, A. (2013). *Generating Sequences With Recurrent Neural Networks.* — MDN handwriting synthesis, σ-bias sampling.
- Ha, D. & Eck, D. (2017). *A Neural Representation of Sketch Drawings* (Sketch-RNN). — Δx,Δy + pen-state stroke tokens, GMM output.
- Bishop, C. (1994). *Mixture Density Networks.*

## 11. v1 results, the confound found in them, and the v2 fix

**v1 result (5 conditions, ctx_ablation/):** trained cleanly (T4, 20 epochs, best checkpoint
always in the last 2–3 epochs, no divergence). Headline numbers:

| | value |
|---|---|
| ΔW_content (mask_strokes − full) | +0.074 nats |
| ΔW_events (drop_strokes − mask_strokes) | −0.345 nats |
| ΔS_content (mask_words − full) | −0.058 nats |
| ΔS_events (drop_words − mask_words) | +0.072 nats |

Generated text was word-salad in both cold- and warm-start conditions; generated strokes
were illegible scatter in every generated panel (2 topics × cold/warm), consistent with a
4.6M-param model trained for only ~2,700 steps — not evidence of a bug, Graves' original
synthesis network needed a dedicated soft-attention text↔pen alignment mechanism to get
legible cursive, which this flat shared-attention design doesn't have.

**Confound found on inspection:** `full`/`mask_strokes`/`mask_words` share one window
length (1,022 flat-sequence positions). `drop_strokes` physically removes stroke positions,
so its windows are only **23.5%** as long (measured: 86,078 vs. 366,894 target positions);
`drop_words` windows are **76.5%** as long. So `ΔW_events`/`ΔS_events` don't cleanly measure
"value of knowing an event happened" — they conflate that with "words packed contiguously
in a short window vs. the same words diluted ~4.5× further apart by stroke placeholders,"
a pure attention-distance effect unrelated to information content. `ΔW_content`/`ΔS_content`
don't have this problem (token-for-token identical windows, only feature values differ).

**Statistical-rigor gap found on inspection:** the "|Δ| ≥ 0.05 nats is real" threshold
(§6) assumed token-level independence. Tokens within one lecture are correlated (a
confused model gets correlated extra loss across a whole page), so the effective sample
size is much closer to the 363 validation windows than to the 86k–280k tokens.

**v2 fix — three-way decomposition + bootstrap.** Two new conditions, `nostroke_hidden`
and `noword_hidden`, hold sequence length and word/stroke attention-distance IDENTICAL to
`full`/`mask_*` (the other modality's positions stay in the sequence at their original
index) but are masked out of self-attention entirely — indistinguishable from ordinary
batch padding, so the model can't even tell an event happened there. This splits each
2-way delta into three orthogonal pieces along the chain
`drop_* → {nostroke,noword}_hidden → mask_* → full`:

- **content** = `mask_*` → `full` (same length; feature values real vs. zeroed) — same
  definition as v1's content delta
- **marker** = `{nostroke,noword}_hidden` → `mask_*` (same length; event visible vs. hidden)
- **length** = `drop_*` → `{nostroke,noword}_hidden` (same zero-info; length differs)

Evaluation also switched from a pooled token-level sum to per-validation-window means,
bootstrapped (3,000 resamples) over the 363 windows for a 95% CI per delta, instead of
the token-count heuristic.

**Bug found while implementing the fix (and how it was caught):** under causal attention,
query position 0 can only attend to key position 0. A `nostroke_hidden`/`noword_hidden`
window that doesn't start at the page start (no prepended BOS) begins with whatever token
was there originally — a stroke ~78% of the time — and if that first token is hidden, its
own query row has zero valid keys, giving an all-`-inf` softmax row → NaN, which propagated
through the whole batch. The local CPU smoke test caught this immediately (`nostroke_hidden`
loss came back NaN) before it ever reached Colab. Fix: never hide position 0
(`hide[0] = False` unconditionally in `_apply_condition`) — this guarantees every later
query has at least one fallback causal key (position 0), which transitively fixes the
whole sequence, not just position 0. A dedicated regression test forces 20 mid-page
windows per condition and confirms finite loss *and* finite gradients through backward.

## 12. First v2 attempt (2026-09-23): two more findings

The first v2 run completed `full`, `mask_strokes`, `nostroke_hidden` and crashed at epoch 14
of `drop_strokes`. Two things came out of it.

**Empty-window crash.** A 1,022-token span that is one long silent drawing run contains no
word or `<silent>` token, so `drop_strokes` filters it to an empty sequence and
`hide[0] = False` indexed into an empty tensor. The 24-page smoke sample never drew such a
span; the full corpus did (≈1 in 15k draws). Fix: training windows that come out empty are
resampled, with a deterministic fallback to the page start (BOS is prepended there and kept
by every condition, so it cannot be empty); validation windows are never resampled, so index
*i* stays the same span in every condition, and the metrics skip a window with no targets.
Regression test fabricates a 600-point all-stroke page and checks all three behaviours plus
that an empty row batched with a real one leaves the real row's metrics finite.

**Training-run variance is as large as the effects.** The same `mask_strokes` condition, same
seed, same code semantics, gave word CE 4.2203 (v1) vs 4.3566 (v2) — a 0.136-nat gap. The
two trajectories track to epoch ~9 then diverge (fp16 attention kernels are nondeterministic;
a 4.6M model at lr 3e-4 amplifies small differences). `full` reproduced to 0.005. That spread
exceeds three of the four v1 deltas, and the window bootstrap does not see it — it only
captures evaluation-sampling noise within one trained model. So v2 now trains every condition
once per seed (`SEEDS = [42, 43, 44]`), computes each delta within a seed (paired by window),
and reports mean ± std across seeds. A delta is called **ROBUST** only if its window CI
excludes zero in every seed *and* all seeds agree in sign. The three seed-42 conditions from
the crashed run are reused. Cost: ≈ 40 min per seed on a T4.

## 13. v2 results (2026-09-23, 7 conditions × 3 seeds, Drive `ctx_ablation_v2/`)

Held-out, best epoch, mean ± std over seeds 42/43/44:

| condition | word CE | ppl | stroke NLL |
|---|---|---|---|
| `full` | 4.140 ± 0.003 | 62.8 | −1.144 ± 0.063 |
| `mask_strokes` | 4.268 ± 0.082 | 71.4 | −0.741 ± 0.023 |
| `nostroke_hidden` | 4.002 ± 0.070 | 54.7 | — |
| `drop_strokes` | 3.852 ± 0.010 | 47.1 | — |
| `mask_words` | 4.384 ± 0.002 | 80.2 | −1.181 ± 0.038 |
| `noword_hidden` | — | — | −1.249 ± 0.061 |
| `drop_words` | — | — | −1.175 ± 0.009 |

Length-controlled deltas (loss reduction from adding that information; + = helps; per-seed
window-bootstrap CIs all exclude zero unless noted):

| | mean ± std (seeds) | per seed | verdict |
|---|---|---|---|
| ΔW_length (drop→hidden) | −0.151 ± 0.066 | −0.144 / −0.220 / −0.089 | ROBUST (hurts) |
| ΔW_marker (hidden→mask) | −0.266 ± 0.119 | −0.347 / −0.129 / −0.320 | ROBUST (hurts) |
| ΔW_content (mask→full) | **+0.125 ± 0.077** | +0.209 / +0.057 / +0.110 | **ROBUST (helps)** |
| ΔS_length (drop→hidden) | +0.073 ± 0.051 | +0.103 / +0.014 / +0.101 | ROBUST, but see caveat |
| ΔS_marker (hidden→mask) | −0.067 ± 0.084 | −0.080 / +0.023 / −0.144 | seeds disagree |
| ΔS_content (mask→full) | −0.037 ± 0.029 | −0.042 / −0.005 / −0.063 | ROBUST (hurts, small) |

**Reading.**
- *Stroke content is genuinely informative for the next word*: +0.125 nats, every seed,
  every CI. This is the positive half of PLAN.md's question, and it is the cleanest
  comparison in the design (token-for-token identical windows).
- *But carrying stroke tokens in the same stream costs more than the content is worth.*
  Making the stroke slots visible as empty markers costs −0.266; spreading the words over
  4.5× more positions costs a further −0.151. Net, `full` is 0.29 nats worse at words than a
  word-only model (ppl 62.8 vs 47.1). v1's confounded "events" delta (−0.416) is now split:
  ~36% was the length confound, ~64% is a real attention-dilution effect of interleaving.
- *Words do not help strokes*: real word identities make stroke NLL slightly worse than
  `<mask>` placeholders (−0.037, robust); the marker effect flips sign across seeds; net
  `full` vs `drop_words` is +0.03 (i.e. slightly worse) on average and mixed per seed.
- Generation from `full` is unchanged qualitatively: word salad, illegible stroke scatter
  placed in roughly the right board region.

**Caveats.**
- The stroke ladder's "zero-info" baseline is not fully zero: `mask_words` masks only
  non-special words, so `<silent>` (49% of drawing starts) stays visible in `mask_words`
  and `noword_hidden` but is removed in `drop_words`. ΔS_length therefore includes the value
  of the `<silent>` boundary marker, not just length. The word ladder has no such leak
  (`nostroke_hidden` hides every stroke position). A clean fix is a `drop_words` variant that
  keeps `<silent>`; not run.
- Seed spread on the word-side magnitudes is 2–3× (ΔW_marker −0.13 to −0.35) even though
  signs and CIs agree; report ranges, not just means. Three seeds is the minimum.
- 4.6M params, 20 epochs. The result is about this architecture at this scale.

**One-sentence result:** in a flat interleaved transformer, handwriting strokes carry real
information about the next spoken word (+0.13 nats, robust), but the cost of interleaving
them — attention dilution from the stroke tokens themselves (−0.27) plus positional
spreading (−0.15) — outweighs it, and spoken words carry no usable information about the
next pen movement at this scale.
