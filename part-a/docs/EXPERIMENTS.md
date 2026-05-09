# Part A — Experiments

Stroke-level Method A. The critical path is data extraction (A.1, A.2)
because we cannot train a stroke-level model without clean stroke
sequences. Modeling experiments (A.3, A.4) are gated on data quality.
A.0 runs in parallel as a no-training baseline.

---

## Experiment A.0 — Frontier-VLM baseline (SketchAgent-style)

### Question
**How well does an off-the-shelf multimodal LLM, prompted with a
SketchAgent-style numbered grid and run in a closed loop, do at
producing a math lesson with strokes for everything?**

This is now a **baseline measurement**, not a project gate.
SketchAgent's published results already establish that the loop is
viable for sketches. Our run extends this to math content (formulas,
algebra writing, geometry) and gives us a quantitative target a
trained model must beat.

### Setup
- Reuse Part B's renderer, but feed it one stroke at a time.
- Numbered grid (50×50) overlaid on the canvas image fed to the model.
- Prompt: SketchAgent's stroke-language adapted for math content.
- Backbone: Claude Sonnet (latest) with vision, called via API.
- Loop: snapshot canvas → feed to model with prior actions → parse one
  action → render → repeat. Stop on `</s>` token.

### Metrics (10 runs on a fixed topic, e.g. Pythagorean theorem)
- Coherence rating (1–5, manual)
- Math symbols are recognizable (yes/no per formula attempt)
- Drift events (model contradicts itself)
- Latency p50, p95
- Cost per lesson

### Pass criteria
This is a baseline; it doesn't pass or fail. We record the numbers
and use them as the bar for A.4.

### Deliverable
`experiments/a0-frontier-baseline/` with driver code, prompt, 10 runs,
RESULTS.md with the baseline numbers.

---

## Experiment A.1 — Stroke extraction pilot

### Question
**Can we extract stroke trajectories — sequences of (x, y, t) points,
grouped into pen-down strokes — from Khan Academy and Organic
Chemistry Tutor videos with high enough fidelity to train on?**

This is the critical path. Without clean strokes, no Method A.

### Setup
Pick 1 Khan Academy video (e.g. "Intro to the Pythagorean theorem")
and 1 Organic Chemistry Tutor video (e.g. an algebra word-problem
walkthrough). Build:

```
video → ffmpeg frames (30 fps)
      → board ROI crop (handle UI/borders)
      → per-frame foreground delta (current vs. previous frame)
      → mask out hand/cursor (KA: small cursor; OCT: hand)
      → vectorize new ink → (x, y, t) points
      → group into strokes (pen-up = gap > threshold ms or position jump)
      → per-stroke smoothing (Savitzky-Golay or RDP)
      → export JSONL: one stroke per line, list of (x, y, t)
```

Hand/cursor masking strategy: SAM2 (Segment Anything 2) per frame on
the cursor/hand class, expand mask, exclude masked region from delta.

### Metrics
On a 30-second segment of each video, manually annotate ground-truth
stroke timestamps and compare:

| Metric | Target |
|---|---|
| **Stroke recall** (extracted strokes that match ground truth) | ≥ 85% on KA, ≥ 70% on OCT |
| **Stroke precision** (extracted strokes that are real, not noise) | ≥ 90% on both |
| **Endpoint error** (px between extracted and true stroke endpoints) | ≤ 5 px median |
| **Temporal error** (ms between extracted and true stroke start) | ≤ 100 ms median |

### Pass criteria
Hit metrics above on at least one of the two sources at the higher
target. Identify failure modes for the harder source. If neither
hits target, redesign before scaling up.

### Deliverable
`experiments/a1-stroke-extraction/`
- `extract.py` — pipeline
- `samples/` — extracted JSONL for the pilot videos
- `eval/` — ground-truth annotations + comparison
- `RESULTS.md` — metrics, failure modes, decision

---

## Experiment A.2 — Speech-stroke alignment

### Question
**Can we produce an interleaved sequence of (speech token, pen event)
that accurately represents what the teacher said while writing?**

### Setup
- Whisper-large-v3 → word-level timestamps for the same pilot videos.
- Take A.1's stroke output (with `t` per pen event).
- Merge into a single time-ordered sequence:
  ```
  PEN(Δx, Δy, p_down) at t=0.000
  PEN(Δx, Δy, p_down) at t=0.033
  ...
  SPEECH("c") at t=1.234
  SPEECH("squared") at t=1.467
  PEN(Δx, Δy, p_down) at t=1.500
  ...
  ```
- Drop the explicit `t` after merging — order encodes timing.

### Metrics
- **Word-stroke pairing accuracy**: when the teacher says "the
  hypotenuse" and writes `c`, do the words land within ±200 ms of
  the corresponding strokes? Hand-judge 50 events per video.
- Sequence length distribution (informs decoder context length).

### Pass criteria
≥ 80% of word-stroke pairs correctly co-located. Sequence length p95
≤ 16k tokens for a 5-min segment.

### Deliverable
`experiments/a2-alignment/` with merged JSONL, evaluation, RESULTS.md.

---

## Experiment A.3 — Sketch-RNN baseline (strokes only, no speech)

### Question
**Can a Sketch-RNN-style model learn to generate math/text strokes
from the extracted stroke data alone?**

This is the sanity check on the data — strip away conditioning and
ask if the strokes themselves form a learnable distribution.

### Setup
- Take stroke-only sequences from A.1 across ~50 videos (KA + OCT).
- Sketch-RNN architecture: bidirectional LSTM encoder, autoregressive
  LSTM decoder, MDN over (Δx, Δy), categorical over pen-state. Mostly
  unmodified from the original paper.
- Train unconditional (no encoder, decoder only — predict next pen
  action from prior pen actions).
- Sample from the trained model with varying τ.

### Metrics
- **NLL** on held-out sequences.
- **Visual quality** (manual): do generated samples look like
  recognizable math/text strokes? Even nonsense math is fine — we
  want to see strokes that look like *somebody writing on a board*.
- **Stroke statistics** match training (avg stroke length, pen-up
  rate, etc.).

### Pass criteria
Generated samples are visually plausible as on-board writing
(handwritten letters/digits/symbols/lines, not random noise). NLL
clearly improves over a uniform baseline.

### Deliverable
`experiments/a3-sketch-rnn-baseline/` with training code, checkpoints,
sample grid, RESULTS.md.

---

## Experiment A.4 — Multimodal Method A model (three variants)

### Question
**Can a stroke-level autoregressive model conditioned on (canvas
image, topic, prior speech, prior strokes) generate a coherent math
lesson — beating the A.0 frontier-VLM baseline on coherence and
matching it on speed/cost?**

This is the project goal. We train **three variants in parallel**,
one per token-granularity option from
[PLAN.md](PLAN.md#token-granularity-options):

- **A.4a — Stroke-anchored** (token = stroke + words spoken during it)
- **A.4b — Word-anchored** (token = word + strokes drawn while said)
- **A.4c — Symbol-anchored** (token = symbol + words at that moment;
  requires HMR labeling, see "Prerequisite for A.4c" below)

A.4a is the recommended starting variant — minimal data preprocessing
beyond A.1/A.2, cleanest match to Sketch-RNN.

### Common architecture

Shared across all three variants (target ~50–500M params total).
See [PLAN.md](PLAN.md#common-across-all-three) for the full spec; in
short:

- **Canvas encoder**: pretrained ViT-small (DINOv2 or similar),
  frozen for first runs, LoRA-tuned later. Projects to 256-dim.
- **Topic encoder**: pretrained sentence transformer (frozen).
- **Single causal transformer**, ~12 layers / ~512 hidden, cross-
  attention to canvas+topic embeddings. No sub-decoder.
- **Output heads** dispatched per position by predicted action type:
  - Action-type categorical: `{word, pen, separator, end}`
  - Word head: softmax over BPE vocab (~16k–32k)
  - MDN head: GMM over `(Δx, Δy)`, M=20 components
  - Pen-state head: categorical over `{down, up, end}`
- **Variant-specific structural tokens** mark the hierarchy
  (e.g. A.4a places `<stroke_start>` / `<stroke_end>` separators;
  A.4b uses word boundaries; A.4c uses symbol-unit separators).
  Same model, different organization of the supervision signal.
- Re-render canvas every K pen events during training so the visual
  conditioning stays current. K ≈ 30 (~1 second of writing).
- **Pretraining required**: don't train from scratch on KA+OCT only —
  use the staged strategy in [PLAN.md](PLAN.md#pretraining-strategy)
  (warm-start components → stroke-only pretrain on QuickDraw +
  CROHME + IAM → multimodal fine-tune on KA + OCT).

### Per-variant data preparation
- **A.4a**: directly consumes A.2's interleaved (stroke, word)
  sequences.
- **A.4b**: same data, regrouped — words become the top-level units,
  strokes become attached payloads.
- **A.4c**: requires an additional preprocessing pass over A.2's
  output to label which strokes group into which symbol. Pipeline:
  segment the stroke sequence at probable symbol boundaries (large
  spatial jumps + pen-up duration), classify each group with an HMR
  model. **A.4c does not start until this labeling pipeline is built**
  (see "A.3.5 Symbol labeling pipeline" below).

### Metrics (each variant)
Same coherence rating + drift events as A.0, plus:
- **Match rate vs. ground truth** on held-out lessons (next-token
  accuracy on the actions a real teacher took, in that variant's
  token type).
- **Latency** per top-level token (target: < 100 ms p50 on a single GPU,
  tighter for A.4a since its tokens are most frequent).
- **Cost** at inference (zero — local model, vs. A.0's API cost).
- **Sequence length distribution** — sanity check that our context
  budget is enough.

### Pass criteria (per variant)
- Coherence rating ≥ A.0's mean − 0.5.
- Recognizable handwritten math symbols on at least one held-out topic.
- No infinite loops; ends gracefully.

A variant is "shipped" if it hits these. We compare across variants
to identify which approach generalizes best.

### Deliverables
- `experiments/a4a-stroke-anchored/`
- `experiments/a4b-word-anchored/`
- `experiments/a4c-symbol-anchored/`

Each contains training code, checkpoints, eval, sample recordings,
`RESULTS.md`. A top-level `experiments/a4-comparison.md` compares all
three against A.0.

---

## Experiment A.3.5 — Symbol labeling pipeline (prerequisite for A.4c only)

### Question
**Can extracted stroke sequences be reliably grouped and labeled as
math symbols / letters / shapes for symbol-anchored training?**

Only required for the symbol-anchored variant (A.4c). Skip if only
running A.4a / A.4b.

### Setup
- Segment A.2's stroke sequences at candidate symbol boundaries
  (spatial gaps + pen-up duration thresholds).
- Classify each segment with an HMR model. Candidates to evaluate:
  open-source HMR models (e.g. Pix2Tex, im2markup), MathPix API
  (commercial), or train a small classifier on QuickDraw + math symbol
  datasets.
- Output: per-stroke-group label (symbol, letter, digit, operator, or
  shape).

### Metrics
- **Symbol classification accuracy** on a hand-labeled subset (50 events).
- **Boundary recall** — fraction of true symbol boundaries detected.
- **Coverage** — fraction of strokes successfully assigned to a symbol.

### Pass criteria
- Classification accuracy ≥ 70% on common symbols.
- Boundary recall ≥ 80%.
- Coverage ≥ 90%.

### Deliverable
`experiments/a35-symbol-labeling/` with pipeline, evaluation, RESULTS.md.

---

## Experiment A.5 — Closed-loop deployment

Wire A.4's model into Part B's player. Replace JSON-script playback
with a live token stream. Demonstrate end-to-end on 3 topics.
Engineering, not research; spec written when A.4 lands.

---

## Run order summary

```
A.0 (parallel) ─────────────────────── baseline numbers
                                          │
                                          ▼
A.1 ──→ A.2 ──→ A.3 ──→  A.4a  ──→ A.5
extract  align  sanity   stroke    deploy
                         (recommended start)
                  ┌─────► A.4b
                  │       word
                  │
                  └─► A.3.5 ─► A.4c
                      label    symbol
```

- A.0 runs anytime.
- A.1 must pass before A.2.
- A.3 can start as soon as A.1 has any data.
- **A.4a** needs A.2 and A.3 done. Recommended first variant.
- **A.4b** needs the same; can be trained in parallel with A.4a.
- **A.4c** additionally needs A.3.5 (symbol labeling pipeline).
- A.5 deploys whichever A.4 variant we want to demo. Can deploy
  multiple.
