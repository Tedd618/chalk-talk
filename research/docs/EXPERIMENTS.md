# Research — Experiments, as actually run

This is the record of what was executed, in order, with what came out.
The original May 2026 plan — which also had a frontier-VLM baseline (A.0),
a symbol-labeling pipeline (A.3.5), four A.4 variants, a stroke-as-token
parallel track and a deployment step (A.5) — is preserved unchanged in
[archive/EXPERIMENTS_plan_2026-05.md](archive/EXPERIMENTS_plan_2026-05.md).
None of those extra items were run; see the last section for why.

---

## A.1 — Stroke extraction from lecture video

**Code:** `experiments/a1-stroke-extraction/`

**Pilot (May 8–11).** One Khan Academy and one Organic Chemistry Tutor
video on the same topic. Extractors v0–v3 (frame difference → persistence
check → Bézier-fit segmentation → cusp detection with path verification)
were judged by *reconstruction*: replay the extracted strokes on a blank
canvas and see if the writing is still readable. OCT reconstructions were;
Khan Academy's cursive was not. Corpus policy became OCT-style only.

**v4, the extractor that shipped (`extract_v4.py`).** Skeleton-based
instead of pen-tip tracking: split the video at page breaks, group new ink
by the time it was drawn, skeletonize each temporal group separately (so
crossing strokes don't fragment each other), capture colored annotations
with max(R,G,B). Subsampled to 10 fps for a 2.5× speedup with no visible
loss: ~1.5 min of extraction per 10 min of video.

**Scale run (summer).** `queue_manifest.json` lists 1,372 OCT videos;
`worker.py` / `enqueue_*.py` / `seed_queue.py` run download → frames →
extract → STT → merge on a shared filesystem. 8–12 lab machines in
parallel; 30 triggered YouTube rate limiting. 1,191 videos completed.

**What did not hold up.** Page-break detection (`pages.py`, drops in inked
pixel count) worked on the pilot but in the final corpus every video came
out as a single page — board clears did not survive into `events.jsonl`.
Absolute board position is therefore only locally meaningful. The plan's
formal recall/precision targets against hand-annotated ground truth were
never measured; reconstruction eyeballing was the acceptance test.

---

## A.2 — Speech–stroke alignment

**Code:** `experiments/a2-alignment/`

`stt.py` runs faster-whisper (`small`, int8, CPU) for word-level timestamps;
`merge.py` interleaves words, strokes and page breaks into a time-ordered
`events.jsonl`; `viz_events.py` animates the reconstruction with a
scrolling caption for eyeballing. Pilot: 425 words, 880 pen events, 137
stroke ends on the 3-minute video, and the caption "a squared plus b
squared…" lines up with `a² + b² = c²` being written.

**Not measured:** the plan's ≥80% word–stroke pairing accuracy. Alignment
is maximum time overlap (`a4-train/align.py`), which is approximate; it is
listed as a limitation in the report.

---

## A.3 — Stroke-only transformer pretraining (standalone)

**Code:** `experiments/a3-sketchrnn/`

A transformer with a mixture-density head over (Δx, Δy) and a pen-state
head, ~3.3M parameters. QuickDraw cat as a smoke test (10k steps on a Mac,
samples are unmistakably cats). Then MathWriting, 229k handwritten
expressions: loss 3.58 → −2.85, pen-state accuracy ~98%. Unconditional
samples are "math-flavoured gibberish" — the expected behaviour of a
healthy unconditional model on diverse data; it learned how math is
written, not what to write. `prime.py` feeds a real stroke prefix and
shows completions.

**Decision later:** the A.4 model trains from scratch on the teacher
corpus; this checkpoint is not loaded. A.3 stands as validation of the
MDN modeling stack.

---

## A.4 — The flat stroke+speech model and the context ablation

**Code:** `experiments/a4-train/` · **Design and every bug:**
[METHOD1_CONTEXT_EXPERIMENT.md](METHOD1_CONTEXT_EXPERIMENT.md) ·
**Report:** [report/report.pdf](report/report.pdf)

**Data.** `align.py` converts `events.jsonl` to a word-anchored
`training.jsonl`: every token is a spoken word carrying the strokes drawn
while it was said; drawing with nobody speaking becomes `<silent>`.
1,191 videos → 12.5M flat tokens (78% pen points, 22% words), 5.5k
vocabulary after normalizing Whisper's case and punctuation. Published as
`output.zip` on the `data-v1` release.

**v1 — word-anchored `OCTModel`** (outer word transformer + stroke
sub-decoder, three-phase curriculum). Words came out coherent, strokes
collapsed to a scribble at one spot. Structurally, strokes could never
feed back into the word context, which was the question being asked.
Replaced.

**v2 — one flat interleaved sequence**, scaled delta-xy, Huber loss, four
heads (type / word / xy / pen). Type head learned well (0.11 CE), word
perplexity ~100, but the xy loss plateaued at epoch 3 and generation was
87% words with a dot cluster of strokes. Four causes found:

1. a deterministic xy head predicts the mean of every direction the pen
   could go — ≈ 0 — so strokes collapse (the classic reason handwriting
   models use a mixture head);
2. uniform random 1k-token windows on ~9k-token pages included a page
   start with probability ≈ 0.01%, yet every generation started there;
3. `<silent>` was forbidden at generation, although 49% of drawing starts
   from it;
4. a tensor used as a dict key silently turned every trigger word into
   `<unk>`, so all earlier generation tests had been run on garbage input.

**v3 — mixture-density head** (20 bivariate Gaussians, verified against
scipy), 20% of windows start at the page start, `<silent>` allowed, pen
class weight 4→2, absolute (x, y) added to the stroke input alongside the
delta.

**The context ablation.** Seven conditions, all transforms of the same
validation windows, same model, three seeds each, bootstrap over windows:

| word ladder | stroke ladder |
|---|---|
| `full` — everything visible | `full` |
| `mask_strokes` — pen slots kept, values zeroed | `mask_words` — word slots kept, identity → `<mask>` |
| `nostroke_hidden` — same positions, hidden from attention | `noword_hidden` |
| `drop_strokes` — pen tokens removed | `drop_words` |

The first version had only `mask` and `drop`; that confounded event
information with sequence length (dropped windows are 23.5% as long), so
the `hidden` rung was added. Two runs of one condition with one seed
differed by 0.14 nats, so single-seed results were discarded.

**Result** (loss reduction from adding that information; + = helps):

| step | Δ nats (3 seeds) | verdict |
|---|---|---|
| pen values become real → next word | **+0.125 ± 0.077** | robust, helps |
| empty pen slots become visible → next word | −0.266 ± 0.119 | robust, hurts |
| words spread to their original spacing | −0.151 ± 0.066 | robust, hurts |
| word identities become real → next pen move | −0.037 ± 0.029 | robust, hurts (small) |
| masked word slots become visible → pen | −0.067 ± 0.084 | seeds disagree |
| strokes spread to their original spacing | +0.073 ± 0.051 | robust, but includes `<silent>` visibility |

The board helps predict the next word; carrying the pen in the same
sequence costs more than that (word-only model: ppl 47 vs 63); speech
does not help the pen. Generation from `full` is word salad plus pen
scatter in roughly the right region — no characters.

**Artifacts.** `context_ablation_colab.ipynb` (generated by
`tools/build_notebook.py`, CPU-smoke-tested by `tools/smoke_test.py`,
which caught an all-masked-causal-row NaN and an empty-window crash before
Colab); `results/ctx_ablation_v2/` (per-run histories, per-window
metrics, summary); `docs/report/`.

---

## Not run, and why

- **A.0 frontier-VLM baseline, A.3.5 symbol labeling, A.4a-1/a-2 (MathWriting
  backbone), A.4b (Qwen-Math), A.4c (symbol-anchored), the stroke-as-token
  track, A.5 deployment, and the May plan's Methods 2/3 (Qwen fine-tunes).** The
  ablation answered the core question for the flat design, and the answer
  argues against continuing down this branch at this scale (see
  [CONCLUSIONS.md](CONCLUSIONS.md)). The next step ([../../PLAN.md](../../PLAN.md))
  changes approach — an agent that plans what to say and draw — rather than
  adding variants of the same model.
- Two cheap checks were considered and left open: a larger stroke-only
  model generating unconditionally (does this data support legible
  handwriting at all?), and a larger `full` model (does word content start
  helping the pen?). About three T4-hours together.
