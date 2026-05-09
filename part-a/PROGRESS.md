# Part A — Progress Log

A running diary of what I worked on, what decisions were made, and where
things ended up. Newest entries on top.

---

## 2026-05-10 (later) — Track 2: stroke transformer trains, makes cat-like cats

### Motivation

After A.2 landed, scaling the OCT corpus would have been the obvious
next move. But noticing the chaotic algebra reconstruction surfaced a
deeper concern: video-extracted stroke quality has a hard ceiling.
The fix is the staged pretraining strategy already in PLAN.md —
pretrain on clean public stroke datasets (where strokes were captured
directly, sub-pixel precision), then fine-tune on extracted teacher-
video data for the joint speech-stroke behavior. Track 2 is the
"validate the modeling stack first" step that gates everything else.

### Decisions

- **Start with QuickDraw cat.** Smallest, most-tested public stroke
  dataset; the original Sketch-RNN paper used it. Easy to evaluate
  visually — if generated samples look like cats, the modeling stack
  works. CROHME and IAM Online come next.
- **Architecture: pure transformer** as committed in PLAN.md. d=256,
  4 layers, 8 heads, M=20 GMM components. ~3M params. Trains on Mac
  MPS in under an hour.
- **No conditioning yet.** Unconditional generation only at this
  stage. Adding canvas + topic + speech encoders is A.4's job.
- **Checkpoints stay out of git.** Added `.pt` to .gitignore. They're
  regenerable from the training script + downloaded data. Training
  logs (small JSONL) and sample images stay in git for reproducibility.

### Work — `part-a/experiments/a3-sketchrnn/`

Built four small scripts and trained end-to-end on the cat class:

- `download.py` — fetch QuickDraw class .npz files from Google's
  Sketch-RNN preprocessed dataset.
- `data.py` — convert (Δx, Δy, pen_state) sequences to the 5-element
  representation `[Δx_norm, Δy_norm, p_down, p_up, p_end]` with a
  global scale (std of Δs in the training split).
- `model.py` — causal transformer with multiple per-position output
  heads. MDN over (Δx, Δy) with M=20 components plus categorical over
  pen-state. Loss = GMM negative log likelihood + cross-entropy on
  pen-state, masked by sequence length.
- `train.py` — AdamW + cosine LR schedule + gradient clipping.
  Auto-picks MPS / CUDA / CPU. Saves checkpoints every 2k steps.
- `sample.py` — temperature-controlled MDN + pen-state sampling,
  matplotlib grid render.

### Training results

10k steps on QuickDraw cat, ~50 minutes on Mac MPS at 3.4 steps/sec.
Loss went from 4.07 → -0.02. The GMM term went *negative* — meaning
the model is concentrating probability mass tightly on the right
next-pen-movements, much better than a baseline unit Gaussian. Pen-
state cross-entropy dropped from 1.39 to 0.29 (≈ 75% accuracy on the
3-way pen-state classification).

Generated samples at T=0.4 and T=0.7 look unmistakably like the
wonky cat doodles from QuickDraw — heads, whiskers, occasionally
ears and tails. Not professional, but recognizable. Same quality as
the Sketch-RNN paper reports on the same data with their LSTM
backbone.

### Where this leaves us

The modeling stack is validated. We have:
- A working transformer-based stroke generator with MDN + pen-state
  output heads.
- A training loop that converges in ~50 minutes on a Mac.
- A sampling pipeline that produces recognizable shapes.

This unlocks the next steps with confidence:
- **CROHME** — same architecture, math expressions instead of cats.
  Direct relevance to the project.
- **IAM Online** — English handwriting. Letter shapes for text
  generation.
- Eventually the full pretraining: combined-dataset training on
  CROHME + IAM Online + QuickDraw, then fine-tune on the OCT corpus
  with conditioning added.

### Hardware note

Cat at 3M params trains comfortably on Mac MPS. Same hardware should
handle CROHME and IAM Online (both small datasets). Switching to a
school lab GPU becomes necessary when:
- Single training run > 6 hours on Mac, **or**
- Out-of-memory errors (typically at model size > 50M params, sequence
  length > 2k tokens, or when adding the ViT canvas encoder), **or**
- A.4 multimodal training (definitely lab).

### Tomorrow

- Download CROHME and IAM Online stroke data.
- Run the same architecture on CROHME, evaluate by sample quality.
- If both pass, plan the combined-dataset pretraining run.

---

## 2026-05-10 — A.2 speech-stroke alignment lands

### Motivation

A.1 gives us clean stroke trajectories for OCT-style content. The
training-shaped data format we ultimately want is an interleaved
event stream: words and pen events together, sorted by time. A.2 is
the bridge — run speech-to-text on the audio, get word-level
timestamps, and merge them with the stroke timestamps from A.1.

### Decisions

- **STT backend**: `faster-whisper` (CTranslate2-based). Apple Silicon
  friendly, supports word-level timestamps natively, fast on CPU at
  int8 quantization. The original `openai-whisper` package would also
  work but is slower.
- **Model size**: `small` (244 MB) as the default. The OCT tutor
  speaks clearly and at moderate pace; small is sufficient for word
  boundaries within ~100–200 ms. We can swap in `medium` if systematic
  alignment errors show up later.
- **Event schema**: a single time-ordered JSONL of `word`, `pen`, and
  `stroke_end` events. `pen` events carry `(x, y, t, stroke_id, first/
  last_in_stroke)`. This is the shape the downstream training data
  takes — the same schema feeds whichever token-granularity variant we
  train (stroke-anchored A.4a, word-anchored A.4b, or symbol-anchored
  A.4c). Format decisions stay consistent across the variants.
- **Layout**: new directory `part-a/experiments/a2-alignment/` with
  three small scripts: `stt.py` (audio → words.jsonl), `merge.py`
  (a1 strokes + a2 words → events.jsonl), `viz_events.py` (animated
  strokes plus a scrolling caption highlighting the active word).

### Work

Walked through the pipeline end-to-end on the existing OCT 3-minute
pilot:

1. Installed `faster-whisper`. Whisper-small auto-downloads the model
   (no API keys, fully local). Total install time on top of the
   existing a1 venv: ~30 seconds.
2. `stt.py oct --duration 180` produced 425 word events in 53 seconds
   of CPU inference. Detected language `en` with probability 1.00.
   ~142 words per minute — believable for a tutor speaking at a
   teaching pace.
3. `merge.py oct` produced a 1442-event time-ordered JSONL
   interleaving 425 word events, 880 pen events, and 137 stroke_end
   events.
4. `viz_events.py oct` renders an mp4 that combines the smoothed
   stroke reconstruction from A.1 with a scrolling caption bar at the
   bottom showing recently spoken words. The current spoken word is
   highlighted in cyan above the caption.

### Eyeball validation

When the caption reads "It's A squared plus B squared is equal to C
squared", the canvas shows the equation `a² + b² = c²` being written
in real time. Small symbols and superscripts land at the moments the
corresponding words are spoken. Long-running silent periods correctly
have no word events. Stretches with both speaking and writing have
both event types interleaved smoothly.

### Where this leaves us

A.1 + A.2 together produce, per video, a clean training-shaped event
stream:

```
videos/<tag>.mp4
  │
  ├─ a1: extract.py  → strokes.jsonl   (137 strokes / 3 min)
  ├─ a2: stt.py      → words.jsonl     (425 words / 3 min)
  └─ a2: merge.py    → events.jsonl    (1442 events, time-ordered)
```

Pipeline is ready to scale. The next bottleneck is corpus size — we
have one OCT video; we want at least a few hours.

### Tomorrow

Two parallel tracks:

- **Corpus growth**: pick 3–5 more OCT-style videos, run them through
  a1+a2, validate by reconstruction + caption alignment, accept those
  that pass. Aim for ~3 hours of clean event data.
- **A.3 baseline preparation**: download CROHME and IAM Online
  datasets, set up a small Sketch-RNN training script. This is the
  modeling-stack validation step — train an unconditional stroke
  generator on public data while the corpus grows in parallel.

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
