# Part A — Progress Log

A running diary of what I worked on, what decisions were made, and where
things ended up. Newest entries on top.

---

## 2026-05-09 — A.1 stroke-extraction baseline lands

### Motivation

Part A is the research arm of the project: a trained model that generates
math and text strokes on a virtual blackboard, alongside spoken
explanations, learned from real teacher videos. The goal of today was to
(a) absorb two recent papers that bear directly on this problem, (b)
crystallize the architecture and training plan enough to act on, and (c)
actually start the data pipeline so we have something to train on.

### Reading

Two papers, cover to cover:

- **Sketch-RNN** (Ha & Eck, 2017). A sequence-to-sequence VAE that
  generates vector sketches as pen-action sequences `(Δx, Δy, pen-state)`
  with a Mixture Density Network output head over the offsets. The
  representation is well-tested and small enough to train. This paper
  gives us the **output representation** for our trained model.
- **SketchAgent** (Vinker et al., MIT/Stanford, 2024). A frontier
  multimodal LLM running in a closed loop, sketching on a numbered grid
  via a string-based stroke language with Bézier post-processing. No
  training. This paper gives us the **closed-loop mechanic** and a
  zero-training baseline that any trained model should beat.

Neither paper covers our exact problem (joint stroke + speech generation,
math content, learned from teacher videos), but together they pin down
the architectural and methodological priors. We adopt Sketch-RNN's MDN
output head, drop its LSTM backbone in favor of a transformer, and use
SketchAgent's closed-loop pattern as the inference-time framework.

### Decisions

A handful of decisions crystallized during the day:

1. **Token granularity.** Three options exist for the unit the model
   emits per autoregressive step: stroke-anchored, word-anchored, or
   symbol-anchored. We will pursue all three eventually. The order is
   **stroke-anchored first** (the harder, purer research path), then
   **word-anchored on Qwen-Math** (the practical path with strong
   priors), then **symbol-anchored** (requires HMR labeling, slowest to
   build). The reasoning is recorded in detail in
   [docs/PLAN.md](docs/PLAN.md).
2. **Architecture.** A single causal transformer with multiple output
   heads (action-type categorical, word softmax, MDN over offsets,
   pen-state categorical). No LSTM sub-decoder. The hierarchy is
   implicit in the data and structural separator tokens.
3. **Pretraining strategy.** Training from scratch on KA + OCT alone
   would underfit — there is no single "stroke transformer" base model
   to fine-tune. Instead, a staged plan: warm-start vision and text
   encoders from pretrained models, pretrain the decoder on public
   stroke datasets (QuickDraw, CROHME, IAM Online), then fine-tune on
   extracted teacher-video data.
4. **Repository.** Created `chalk-talk` (private) on GitHub.

### Work — A.1 stroke-extraction pilot

Most of the day went into the stroke-extraction pipeline. The plan was:
pick one Khan Academy video and one Organic Chemistry Tutor video on the
same topic (Pythagorean theorem), build a stroke extractor, and validate
by **reconstruction** — replay the extracted strokes alone on a blank
canvas, with no original ink underneath, and see if the writing is
recognizable. If yes, the data is usable for training; if no, the
pipeline lost information.

This took many iterations. Each one was instructive about a different
failure mode:

- **v0** — Frame-diff plus centroid plus gap/jump segmentation. The
  cursor sprite dominated; KA collapsed to a single 30-second "stroke"
  because the cursor was always present.
- **v1** — Persistence check: a pixel only counts as ink if it is still
  bright `LOOKAHEAD` frames later. Cursor problem solved (real ink
  survives; transient cursor pixels do not). But segmentation went
  wrong in opposite directions for the two videos: KA cursive flow was
  under-cut into too few long strokes; OCT slow line-drawing was
  over-cut into too many fragments. No threshold combination worked
  for both.
- **v2** — Replaced gap/jump segmentation with greedy cubic-Bézier-fit
  segmentation. Better, but the fit-error criterion cuts whenever one
  cubic can't cover a curve, fragmenting smooth strokes that simply
  needed multiple cubics chained together.
- **v3** — Replaced Bézier-fit-error with **cusp detection**: only cut
  a pen-down run where the trajectory turns sharply. This is the right
  unit for a stroke boundary. Combined with sticky pen-down (bridge
  brief detection gaps within a real stroke), v3 produced 137 strokes
  on 3 minutes of OCT — believable counts.
- **Reconstruction validation.** Built `reconstruct.py` to replay
  strokes on a blank canvas, the honest test. OCT reconstructions were
  recognizably the original writing. KA reconstructions were not.
  **We curated the v1 corpus down to OCT-style content only**, deferring
  KA-style until extraction is good enough for cursive.
- **v4–v8** — Several creative attempts to clean up residual phantom
  strokes: net pixel-growth detection, shape-based component filtering,
  size-band line filtering, dot/cursor-blob filter on `new_persist`.
  None matched v3 quality; some were considerably worse.
- **The diagnostic moment.** Inspecting v3's reconstruction at a higher
  zoom revealed that residual phantom long lines were not coming from
  the cursor. They "popped" — the algorithm was bridging two distant
  centroids inside one pen-down run, drawing a straight line across
  blank canvas. The cause was sticky pen-down keeping the run open
  across two separate physical strokes that happened to fall within
  the sticky window.
- **The principled fix — path verification.** Before appending point
  B to a run that ended at A, sample the straight line A→B against
  the current persistent ink mask. If at least 40% of the sampled
  pixels lie on existing ink, the line is a real continuous stroke;
  if it goes through blank canvas, it is a fake bridge. Commit the
  current run and start a new one with B in that case. This uses
  ground-truth ink rather than a distance heuristic. It cleaned up
  phantom long lines decisively while leaving real strokes intact.
- **Tiny-letter capture.** Dropped `MIN_NEW_INK_PIXELS` from 2 to 1 so
  small careful strokes (superscripts, decimal-point-like marks)
  produce a centroid each frame instead of being filtered as noise.
- **Reconstruction smoothing.** `reconstruct.py` now applies Chaikin's
  algorithm (two iterations of corner-cutting, approximating a
  quadratic B-spline) per stroke before drawing. The output looks like
  flowing handwriting rather than jagged polylines. Cosmetic only —
  does not change the data.

### Where this leaves us

A.1 has a working baseline. On 3 minutes of OCT video:

- 1172 frames with raw ink signal (~22% of total)
- 111 pen-down runs after path-verification splits
- 137 final strokes after cusp segmentation

Reconstruction is recognizably the original writing — the equation
`a² + b² = c²`, the worked steps `5² + 12² = x²` → `25 + 144 = x²` →
`√169 = √x²` → `13 = x`, both triangles with their `a / b / c / 12 / 5`
labels. The pipeline is ready to scale to a real OCT-style corpus.

### Tomorrow

- Decide the corpus size and source list for the v1 dataset (suggested
  starting target: 3 hours of OCT-style content from OCT itself plus
  similar tablet-tutor channels).
- Run extraction on each video and validate by reconstruction; accept
  videos that pass.
- Move to **A.2 — speech-stroke alignment**: run Whisper STT on each
  video, align word-level timestamps with the extracted stroke
  timestamps to produce the interleaved sequences we need for training.
