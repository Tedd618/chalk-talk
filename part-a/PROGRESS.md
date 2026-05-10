# Part A — Progress Log

A running diary of what I worked on, what decisions were made, and where
things ended up. Newest entries on top.

---

## 2026-05-09 (Saturday) — A.3: training the stroke model

### Motivation

Yesterday ended with a working data pipeline (a1 + a2). The natural
next move would have been to keep growing the corpus, but inspecting
the algebra reconstruction surfaced a nagging worry: the strokes we
extract from video have a hard quality ceiling. No amount of more
videos fixes that. The plan in PLAN.md already had the answer — pretrain
on clean public stroke data, then fine-tune on extracted teacher data.
Today was the day to actually do that pretraining step (A.3).

The goal: take the architecture committed in PLAN.md (a transformer
with MDN and pen-state heads), train it on real handwritten math, and
produce a checkpoint that A.4 can inherit as its backbone.

### Decisions made along the way

A few things were decided as I went:

- **Validate first, scale later.** Before training on math, run a tiny
  smoke test on something easy. QuickDraw cat (Sketch-RNN's original
  testbed) is the obvious choice — single class, well-known, and you
  can tell visually whether the model is working.
- **Drop QuickDraw and IAM Online from the broader pretraining plan.**
  QuickDraw cats are unrelated to math; IAM Online is English text the
  OCT teacher doesn't actually write. CROHME-equivalent on its own
  should give the math-symbol stroke priors we need.
- **CROHME → MathWriting.** CROHME's official hosting is dead in 2026.
  Every accessible mirror is image-only or auth-gated. Found Google's
  MathWriting (2024) instead — a publicly hosted, CC-licensed dataset
  of 229k human-handwritten math expressions. Larger and cleaner than
  CROHME would have been. Pivoted.
- **Skip SketchAgent synthetic data permanently.** Frontier-VLM
  sketching of math is wonky; training on it would teach the model to
  imitate Claude's bad drawings, not real teachers. Defeats the
  research premise.
- **No conditioning yet.** A.3 is unconditional generation only. The
  conditioning (canvas image, topic, speech) is A.4's job.

### Work

Built the modeling stack in `part-a/experiments/a3-sketchrnn/`:

- `download.py` to fetch QuickDraw .npz files
- `data.py` for QuickDraw, `data_mathwriting.py` for MathWriting's InkML
- `model.py` — the transformer with MDN and pen-state heads (~3.3M
  params), implementing the MDN loss from the Sketch-RNN paper
- `train.py` — AdamW + cosine LR, picks MPS / CUDA / CPU automatically
- `sample.py` — temperature-controlled generation + matplotlib grid
- `viz_data.py` — renders raw training samples for sanity inspection

**Cat smoke test (50 min on Mac MPS).** 10,000 steps on QuickDraw cat.
Loss dropped from 4.07 to about -0.02. Generated samples at T=0.4
look clearly like cats — heads, ears, whiskers, tails. Not
professional-looking, but unmistakably cats. The modeling stack is
working.

**MathWriting pretraining (70 min on Mac MPS).** 10,000 steps on the
229k human-written math expressions. Loss dropped from 3.58 to
-2.85. Pen-state classification reached about 98% accuracy. Training
sped past the cat numbers because math handwriting has more
structure than freehand cat drawings, and longer sequences give more
learning signal per sample.

**Sample inspection — and a useful surprise.** When I sampled
unconditional outputs at three temperatures (0.2, 0.4, 0.7), they all
came out as math-flavored gibberish. Strokes that have the cadence
and density of human math handwriting, but no recognizable
expressions.

That was confusing at first — cat samples looked like cats; why
don't math samples look like math? The reason clicked when I thought
about prototype shapes. Cats have one (head + body + ears). Math
doesn't — every training sample is a *different* unique expression.
An unconditional model on diverse data can only learn the *aggregate*
appearance, which for math handwriting is "small loops, varying
baseline, occasional structural elements." That's exactly what came
out.

So this is the expected behavior of a healthy unconditional model on
diverse data. The model has learned **how** to write math (stroke
dynamics, pen-up timing, math aesthetics), but not **what** to write
— that's the job of conditioning, which A.4 adds.

The numbers confirm the model is healthy: tight per-step prediction,
high pen-state accuracy, strokes terminate cleanly, no infinite
loops. A.3's role — be the stroke-knowledgeable backbone for A.4 —
is fulfilled.

### Where this leaves us

- ✅ A.3 checkpoint: `mathwriting.final.pt` (~3.3M params, 13 MB).
- → This file is the input to A.4 — its weights become the backbone
  of the multimodal fine-tuning.
- ⚠️ Before A.4 is meaningful, the OCT corpus needs to grow from
  ~37 minutes to several hours.
- ⚠️ A.4 itself probably needs a school-lab GPU.

### Tomorrow

- Decide on school-lab access logistics.
- Scale OCT corpus by adding more tablet-tutor videos.
- Plan A.4 architecture concretely: which weights load from A.3,
  which layers (canvas encoder, cross-attention, word head,
  page-break head) start fresh, and how `events.jsonl` from a1+a2
  becomes training batches.

---

## 2026-05-08 (Friday) — A.1 stroke extraction and A.2 speech-stroke alignment

### Motivation

Part A is the research arm of the project: a trained model that
generates math and text strokes on a virtual blackboard, alongside
spoken explanations, learned from real teacher videos. Today was about
(a) absorbing two recent papers that bear directly on this problem,
(b) getting the architecture and training plan crisp enough to act on,
and (c) actually starting the data pipeline so we have something to
train on later.

### Reading

Two papers, cover to cover:

- **Sketch-RNN** (Ha & Eck, 2017). A sequence-to-sequence VAE that
  generates vector sketches as pen-action sequences with a Mixture
  Density Network output. The representation is well-tested and small
  enough to train on a laptop. This paper gives us our **output
  representation**.
- **SketchAgent** (Vinker et al., MIT/Stanford 2024). A frontier
  multimodal LLM running in a closed loop, sketching on a numbered
  grid via a string-based stroke language with Bézier post-processing.
  No training. This paper gives us the **closed-loop mechanic** and a
  zero-training baseline that any trained model should beat.

### Decisions made along the way

- **Token granularity (the unit the model emits per autoregressive
  step).** Three options exist: stroke-anchored, word-anchored, and
  symbol-anchored. We will pursue all three eventually. The order is
  **stroke-anchored first** (the harder, more novel research path),
  **word-anchored next** (the practical path on top of Qwen-Math),
  **symbol-anchored last** (gated on building an HMR labeling
  pipeline).
- **Architecture: a single causal transformer** with multiple per-
  position output heads (action-type categorical, word softmax, MDN
  over offsets, pen-state categorical). No LSTM sub-decoder.
- **Pretraining strategy.** Training from scratch on KA + OCT data
  alone would underfit. Plan is staged: warm-start vision and text
  encoders from pretrained models, pretrain the decoder on public
  stroke datasets, then fine-tune on extracted teacher-video data.
- **Data curation policy: OCT-style only for v1.** After running A.1
  on both Khan Academy and Organic Chemistry Tutor pilots,
  reconstruction validation showed OCT extracts cleanly while KA
  cursive flow does not. Defer KA-style content until extraction is
  better.
- **Repo: `chalk-talk`** (private) on GitHub.

### Work — A.1 stroke-extraction pipeline

Most of the day went into the stroke-extraction pipeline. The plan
was: pick one Khan Academy video and one Organic Chemistry Tutor
video on the same topic (Pythagorean theorem), build a stroke
extractor, and validate by **reconstruction** — replay the extracted
strokes alone on a blank canvas, with no original ink underneath, and
see if the writing is still recognizable. If yes, the data is usable
for training; if not, the pipeline lost information.

This took many iterations. Each one taught me something about a
different failure mode:

- **v0** — Frame-diff plus centroid plus gap/jump segmentation. The
  cursor sprite dominated; KA collapsed to a single 30-second
  "stroke" because the cursor was always present.
- **v1** — Persistence check (a pixel only counts as ink if it's
  still bright LOOKAHEAD frames later). Cursor problem solved. But
  segmentation went wrong in opposite directions for the two videos:
  KA cursive flow was under-cut, OCT slow line-drawing was over-cut.
  No threshold combination worked for both.
- **v2** — Replaced gap/jump segmentation with greedy cubic-Bézier-fit
  segmentation. Better, but it cuts whenever one cubic can't cover a
  curve, fragmenting smooth strokes that simply needed multiple
  cubics chained together.
- **v3** — Replaced Bézier-fit-error with **cusp detection**: cut a
  pen-down run only where the trajectory turns sharply, not where the
  math says one cubic isn't enough. Combined with sticky pen-down,
  v3 gave 137 strokes on 3 minutes of OCT — believable.
- **Reconstruction validation.** Built `reconstruct.py` to replay
  strokes on a blank canvas — the honest test. OCT reconstructions
  were recognizable. KA reconstructions were not. Curated v1 down to
  OCT-style content only.
- **v4–v8** — Several creative attempts to clean up residual phantom
  strokes: net pixel-growth detection, shape-based component
  filtering, size-band line filtering, dot/cursor-blob filter on
  `new_persist`. None matched v3 quality.
- **The diagnostic moment.** Inspecting v3 at higher zoom revealed
  that residual phantom long lines were not coming from the cursor.
  They "popped" between distant points — the algorithm bridging two
  separate physical strokes inside one sticky window.
- **The principled fix — path verification.** Before appending point
  B to a run that ended at A, sample the straight line A→B against
  the current persistent ink mask. If at least 40% of the sampled
  pixels lie on existing ink, the line is a real continuous stroke;
  if it goes through blank canvas, it is a fake bridge. This used
  ground-truth ink rather than a distance heuristic. Cleaned up
  phantom long lines decisively while leaving real strokes intact.
- **Tiny-letter capture.** Dropped `MIN_NEW_INK_PIXELS` from 2 to 1
  to capture small careful strokes (superscripts, decimal-like marks).
- **Reconstruction smoothing.** `reconstruct.py` now applies
  Chaikin's algorithm (two iterations) per stroke. Output looks like
  flowing handwriting rather than jagged polylines. Cosmetic only —
  does not change the data.

### Work — corpus growth and page-break detection

Once A.1 was working on the pilot, scaled to four videos:

- `oct` (Pythagorean, 3 min cap)
- `oct-algebra` (Algebra For Beginners, 10 min cap)
- `oct-fractions` (Adding Fractions, full ~10 min)
- `oct-geometry` (Lines/Rays/Angles, full ~14 min)

Built `corpus.json` and `batch.py` to run download → frames →
extract → STT → merge → reconstruct for each video and produce a
summary table.

The algebra reconstruction surfaced a new problem that I had ignored:
the tutor doesn't only write — he also *clears the canvas* between
problems. Without modeling this, the reconstruction accumulated
every stroke ever drawn into one overlapping mess.

Wrote `pages.py` to detect canvas-clear events from sudden drops in
inked-pixel count (current ink less than 30% of previous ink over
~1 second). Updated `merge.py` to emit `page_break` events into the
events stream. Updated `reconstruct.py` to clear the canvas at each
page break and only render strokes from the current page.

Detected page-breaks per video:

- `oct` (3 min): 0 (he never cleared in the pilot window — correct)
- `oct-algebra`: 6
- `oct-fractions`: 8
- `oct-geometry`: 10

After this fix, the algebra reconstruction shows seven distinct
problems in sequence, each on a fresh canvas, each readable.

### Work — A.2 speech-stroke alignment

A.1 gives clean stroke trajectories per video. A.2 brings in the
audio side: run Whisper STT on each video, get word-level
timestamps, and merge them with the stroke timestamps from A.1 into
a single time-ordered event stream.

Built `part-a/experiments/a2-alignment/`:

- `stt.py` — `faster-whisper` (CTranslate2-based) at int8 on CPU,
  word-level timestamps. Default model: `small` (244 MB), enough for
  clear teacher speech.
- `merge.py` — combines a1 strokes + a2 words + page breaks into a
  single time-ordered `events.jsonl`.
- `viz_events.py` — animated stroke reconstruction with a scrolling
  caption bar; the currently-spoken word is highlighted.

Pilot result on the OCT 3-minute Pythagorean theorem video: 425
words, 880 pen events, 137 stroke ends — 1442 events total. The
eyeball test passed: when the caption reads "It's A squared plus B
squared is equal to C squared", the canvas shows the equation
`a² + b² = c²` being written in the same time window.

### Where this leaves us (end of day Friday)

- ✅ Working a1 + a2 pipeline that produces, per video, a clean
  training-shaped event stream.
- ✅ Corpus of 4 OCT-style videos, ~37 minutes total, ~15k events.
- ✅ Page-break detection so the canvas resets at problem boundaries.
- ⚠️ Quality of extracted strokes has a hard ceiling — the model
  trained on this alone would inherit the noise. Need clean public
  data (CROHME-equivalent) to pretrain on first.

### Tomorrow

- Pretrain a stroke model on clean public math-stroke data (Track 2 /
  A.3) before scaling the OCT corpus further.
