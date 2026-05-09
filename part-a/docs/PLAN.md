# Part A — Plan

## What Part A is

A **trained** multimodal autoregressive model that, given the current
state of the blackboard and what's been said so far, generates the
next pen movement or the next spoken word — at **stroke-level
granularity for everything on the board**, including math symbols
and English text. No primitives, no LaTeX shortcut for formulas. The
model writes math the way a human teacher does: stroke by stroke.

```
       ┌─────────────────────────────────────────┐
       ▼                                         │
  [canvas image]                                 │
  [prior speech tokens]   ──→  [model]  ──→ next token
  [prior pen events]                             │
                                                 │
              token ∈ {                          │
                SPEECH(word),                    │
                PEN(Δx, Δy, p_down|p_up|p_end)   │
              }                                  │
                                                 │
            execute (TTS or render) ─────────────┘
```

## Output representation

The final rendered output is always **pen events**
`(Δx, Δy, p_down|p_up|p_end)` following Sketch-RNN (Ha & Eck 2017;
see [RELATED_WORK.md](RELATED_WORK.md)) — math and text are written
stroke by stroke. Whatever the top-level token is, a sub-decoder
emits pen events to produce actual writing.

What is **not yet decided** is the **token granularity** that the
top-level autoregressive model operates on. Three approaches will be
explored as parallel variants — see "Token granularity options"
below. The choice affects the model's structure, sequence length, and
data preprocessing requirements, but not the final rendered output
(always strokes).

### Token granularity options

We will train one A.4 variant per option (see [EXPERIMENTS.md](EXPERIMENTS.md)).
All three are stroke-generative at render time. They differ in the
*unit* the top-level model emits per autoregressive step.

#### Option 1 — Stroke-anchored
**Token = one stroke + words spoken during that stroke.**

Each top-level step emits one full stroke (a pen-down → trajectory →
pen-up sequence) together with any words the teacher spoke while
drawing it (zero or more). Strokes themselves are produced by a
Sketch-RNN-style sub-decoder (MDN over Δ, categorical over pen-state).

- Most natural to the physics of writing.
- Sequence length per 5-min lesson: ~200–500 strokes.
- Data extraction unit matches A.1's natural output — no extra
  semantic labeling.
- Word-stroke pairing is approximate: one symbol is often several
  strokes; a long verbal explanation with no writing has no stroke to
  attach to (we'd add a no-stroke "speech-only" token type to handle
  that).

#### Option 2 — Word-anchored
**Token = one spoken word + strokes drawn while saying that word.**

Each top-level step emits one word together with any pen events that
happened during the word's audio duration.

- Shortest sequences (~600 words / lesson).
- Speech-led — natural if we think of the tutor as primarily
  speaking, illustrating as they go.
- Most pen events have a co-occurring word (KA / OCT empirically have
  little drawing-without-speech), so "empty stroke list" tokens are the
  exception, not the rule. See "Analysis" below.
- **Stroke-segmentation robust**: the model trains on raw timestamped
  pen events grouped by word. How those events are "segmented into
  strokes" is cosmetic; the model never sees stroke boundaries as
  labels. This is a major practical advantage given the difficulty of
  clean stroke segmentation.
- **Naturally pairs with a pretrained math LM** (Qwen-Math etc.) —
  see "Architectural pairing" below.

#### Option 3 — Symbol-anchored
**Token = one written symbol or shape + words spoken at that moment.**

Each top-level step emits one semantic unit — a letter, a digit, a
math operator (`²`, `=`, `+`), or a geometric shape (triangle,
arrow) — together with any words spoken during it. The symbol's
strokes are produced by a sub-decoder.

- Most semantically meaningful; best fit for math content.
- Shortest meaningful sequences (~50–200 units / lesson).
- Requires **handwritten math recognition (HMR)** during data
  extraction to label which extracted strokes group into which
  symbol. Significant added work in the data pipeline. Tools to
  evaluate: Im2Latex, MathPix, MyScript, open-source HMR models.
- Highest ceiling, highest cost.

### Common across all three

The model is a **single causal transformer over a flat token
sequence** — no separate sub-decoder, no LSTM. Sketch-RNN's
contribution we keep is its **output representation** (MDN over
`(Δx, Δy)`); we drop its RNN backbone in favor of transformer.

- **Single causal transformer**, ~12 layers / ~512 hidden as
  starting size; tuned later. Modern stack: rotary positional
  embeddings, FlashAttention, gradient checkpointing.
- **Cross-attention** to canvas-image embeddings and topic
  embeddings (both pre-encoded and prepended to the sequence).
- **Token vocabulary** = BPE word tokens + structural separators
  (`<stroke_start>`, `<stroke_end>`, `<end>`) + a generic `<pen>`
  marker. The actual `(Δx, Δy)` and pen-state values are emitted by
  output heads at `<pen>` positions, not as discrete tokens.
- **Output heads** dispatched per position by predicted action type:
  - **Action-type categorical** — `{word, pen, separator, end}`
  - **Word head** — softmax over BPE vocab (~16k–32k)
  - **MDN head** — Mixture of M=20 bivariate Gaussians over
    `(Δx, Δy)` (Sketch-RNN's representation, on a transformer
    backbone)
  - **Pen-state head** — categorical over `{down, up, end}`
- **BPE tokenizer** trained on KA + OCT transcripts. Same tokenizer
  across variants for fair comparison.

The hierarchy ("stroke-anchored" / "word-anchored" / "symbol-anchored")
is **implicit, not architectural**. It describes how the data is
organized and where separator tokens are placed; mechanically the
model is the same flat transformer in all three cases.

### Pretraining strategy

Training a transformer for joint stroke + speech generation purely
on KA + OCT data alone would underfit dramatically. Instead we use
a multi-stage strategy. There is **no single pretrained "stroke
transformer" base model to fine-tune from** (the closest, Sketch-RNN
checkpoints, are LSTM-based and not directly transferable), but we
can warm-start most components:

**Stage 1 — Component warm-starts (off-the-shelf):**
- Canvas encoder: pretrained ViT (e.g. DINOv2-small), frozen for
  first runs, LoRA-tuned later.
- Topic / text encoder: pretrained sentence transformer (frozen).
- Word embeddings: initialized from a pretrained tokenizer's
  embedding table (e.g. Llama or Qwen-Math), even though the
  decoder is freshly initialized.

**Stage 2 — Stroke-only pretraining on public datasets:**
Pretrain the transformer (without canvas / topic conditioning) on
publicly available stroke datasets. This gives the model priors for
"what does a hand-drawn stroke look like" before we add multimodal
conditioning:
- **QuickDraw** — 50M sketches, 345 classes (used by Sketch-RNN).
  Generic shape priors.
- **CROHME** — handwritten math expressions with stroke traces.
  Math-symbol priors.
- **IAM Online** — handwritten English text with stroke traces.
  English-letter priors.

**Stage 3 — Multimodal fine-tuning on KA + OCT:**
Add canvas + speech + topic conditioning, fine-tune on extracted
teacher-video data with the full multimodal objective. Most
parameters are warm-started by Stages 1–2; only the conditioning
attention layers and possibly the action-type head start fresh.

This staged approach is the standard recipe for "from scratch in a
new modality" research (cf. LLaVA's vision-language alignment via
CLIP + Llama; Whisper's pretraining hierarchy). The word-anchored
variant gets even more help — its entire backbone can be warm-started
from Qwen-Math (see the architectural-pairing table above).

### Analysis (from A.1 pilot inspection)

Two observations from inspecting Khan Academy and OCT pilot videos
sharpen the trade-offs between options:

**1. Speech and drawing co-occurrence.** Tutors speak almost
continuously, including throughout drawing periods. Drawing-without-
speech stretches are rare; speaking-without-drawing stretches are
common (transitions, setup, problem statements). This means:
- Word-anchored isn't burdened by long "empty word" tokens (drawing
  silently is rare).
- Stroke-anchored has to handle plenty of "speech-only" intervals,
  which need a separate non-stroke token type.

**2. Stroke-segmentation difficulty.** Even with cursor-resistant
extraction, stroke boundaries are hard to detect cleanly: KA cursive
flows letters together (under-cuts), OCT slow line-drawing fragments
single lines (over-cuts), and a single threshold combo doesn't fit
both. The implications differ by option:
- **Stroke-anchored**: stroke boundaries are the *labels* we train
  on. Bad segmentation = bad supervision signal. This is fragile.
- **Word-anchored**: stroke boundaries are not labels. The model
  trains on raw pen events grouped by word. Segmentation noise is
  invisible to the loss.
- **Symbol-anchored**: even worse than stroke-anchored — symbol
  boundaries are even harder to recover than stroke boundaries.

### Architectural pairing

Each option has a natural model-family fit:

| Option | Natural foundation |
|---|---|
| Stroke-anchored | **From-scratch** Sketch-RNN-derived autoregressive model with Sketch-RNN sub-decoder. Learns language and math reasoning together with stroke dynamics. |
| Word-anchored | **Pretrained math LLM** (Qwen-Math 7B or similar) with a vision encoder for canvas + a stroke decoder head for `<pen>` tokens. Math reasoning + language come for free; the network only needs to learn joint stroke production. |
| Symbol-anchored | Either, with HMR labeling pipeline first. |

The word-anchored path is *much* more likely to produce a working
system in the short term — it inherits Qwen-Math's reasoning. The
stroke-anchored path is the harder, purer research bet — it forces
the model to learn teacher dynamics from data, with no language or
math priors borrowed in.

### Decision

**We will pursue both eventually. Order: stroke-anchored first, then
word-anchored.**

Rationale:
- Stroke-anchored is the harder path but the more novel research
  contribution. Doing it first establishes the dataset and the joint
  generation infrastructure on the most-restrictive case; word-
  anchored on Qwen-Math reuses the same data and pipeline.
- If stroke-anchored fails or hits a ceiling, word-anchored is the
  fallback — and it's likely to succeed because of the Qwen-Math
  prior.
- If stroke-anchored succeeds, comparing it to word-anchored becomes
  a clean ablation about whether language priors help or hurt
  teacher-style joint generation.

A.4a (stroke-anchored) is the active build target; A.4b (word-
anchored on Qwen-Math) is the next-up variant. A.4c (symbol-anchored)
remains gated on A.3.5 and is the longest-tail variant.

## Conditioning inputs

| Input | Encoder |
|---|---|
| Current canvas snapshot | Small CNN (or ViT-tiny) → fixed-dim embedding |
| Topic / lesson title | Text encoder (frozen pretrained or trained from scratch) |
| Prior speech transcript | The autoregressive context itself (the same sequence) |
| Prior pen actions | Same as above — they are tokens in the sequence |

Since prior speech and prior strokes are part of the autoregressive
context, the only "external" conditioning we need is the topic and
the canvas image. The canvas embedding is prepended (or
cross-attended to) once per step.

## Data sources (training)

### v1 curation policy: OCT-style only

After the A.1 pilot we found that the two source styles have
dramatically different extraction quality with our current pipeline:

- **OCT-style** (slow, deliberate writing with frequent pen-lifts —
  e.g. The Organic Chemistry Tutor): reconstructs cleanly. Strokes
  alone, replayed on a blank canvas, are recognizable as the original
  writing.
- **KA-style** (cursive flow with continuous pen contact across many
  letters — e.g. Sal Khan): reconstruction is unreadable. Cursive
  flow defeats centroid-based pen-tip estimation and Bézier
  segmentation alike.

**Decision**: For v1 of the model, train only on OCT-style content.
Defer KA-style until we have a substantially better cursive-aware
extractor (probably requires explicit cursor tracking — which our
v2 attempt failed at — or a fundamentally different approach like
trajectory recovery from inked region skeletons).

This is a curation-driven strategy, not a permanent abandonment of
KA. If A.4a (stroke-anchored from-scratch) succeeds on OCT-style
data, we'll know the modeling approach works and can then invest in
KA-style extraction.

### v1 corpus (OCT-style)

1. **The Organic Chemistry Tutor** — YouTube, the canonical example.
   Hundreds of hours of math, chemistry, physics with discrete
   strokes.
2. **PatrickJMT, Professor Leonard, Mr. H Tutoring** and similar
   tablet-tutor channels — to be selected per video for clean
   stroke recording.
3. **(Possibly) SketchAgent-style synthetic data** — frontier VLM
   in the loop generates additional samples on math topics. Useful
   for filling gaps but secondary to real video extraction.

### v2+ corpus (when cursive-aware extraction is ready)

1. **Khan Academy** — CC BY-NC-SA, research-friendly. ~1000s of
   hours. The biggest single math corpus, blocked on extraction
   quality.
2. **3Blue1Brown** — animation-heavy, only the handwritten segments
   are stroke-applicable.

## Why this is hard

The architecture doc and SketchAgent's discussion together
identify these challenges. We inherit them all:

1. **Stroke extraction from video** — frame-diff + segmentation +
   vectorization. Hand occlusion (worst on OCT, mild on KA). Variable
   resolutions, codecs, marker styles.
2. **Speech-stroke alignment** — Whisper gives word timestamps in ms.
   Strokes have video-frame timestamps. We need to interleave them
   accurately enough that "c squared" is followed by the strokes that
   write `c²`, not the strokes for the next thing.
3. **Sequence length explosion** — a 5-minute lesson at 30 Hz pen
   sampling is ~9000 pen events, plus ~600 spoken words. Decoder must
   handle ~10k-token sequences. Transformer with long context, or
   hierarchical encoding (one "stroke" = one continuous pen-down arc
   = one mid-level token expanded into pen events).
4. **Long-horizon coherence** — the model must remember that it
   already drew a triangle 5 minutes ago, and not redraw it. Canvas
   image input helps; long context helps.
5. **Inference latency** — same as the original doc. Sketch-RNN-style
   sampling is fast per step but the sequence is long. Speculative
   execution + streaming render is still relevant.
6. **Compute budget for training** — far smaller than fine-tuning a
   VLM, but non-trivial. Aim for a model in the 50M–500M parameter
   range, trainable on a single high-end GPU (or rented A100/H100
   for the larger end).

## Reuse from Part B

| Reused | How |
|---|---|
| The browser player's animation engine | Render pen events in real time; same `animatePolyline` logic, fed one event at a time instead of one shape at a time |
| Canvas dimensions and color | Identical 800×500 canvas |
| TTS layer | Web Speech API consumes generated word stream |

What is not reused: Part B's JSON script schema, the primitive shapes,
KaTeX. Method A doesn't produce structured shapes; it produces pen
events.

## Milestones

| # | Milestone | Decides |
|---|---|---|
| **A.0** | **Frontier-VLM baseline (SketchAgent-style)** | Quantitative baseline for "no-training" performance — the bar a trained model has to beat |
| **A.1** | **Stroke extraction pilot** (1 KA video + 1 OCT video) | Whether stroke recall is high enough to train on the full corpus |
| **A.2** | **Speech-stroke alignment** | Whether Whisper word timestamps can be aligned with extracted strokes accurately |
| **A.3** | **Sketch-RNN baseline** trained on extracted strokes only (no speech, no canvas image) | Whether the data is learnable at the most basic level — sanity check |
| **A.4a / A.4b / A.4c** | **Multimodal Method A model — three variants in parallel, one per token-granularity option** (stroke-anchored / word-anchored / symbol-anchored) | Whether Method A actually works, and which token granularity is best |
| **A.5** | **Closed-loop deployment** — model wired into Part B's player, real-time generation | Whether latency/coherence is good enough for use |
| **A.6** | **(Stretch) RL with student-feedback signal** | Future research |

A.1 and A.2 are the **critical path**. A.3 validates that the data is
learnable in isolation. A.4 is the actual goal. A.0 is run in parallel
to A.1–A.2 as a baseline measurement.

Detailed experiment specs in [EXPERIMENTS.md](EXPERIMENTS.md).

## Non-goals

- Pre-baked LaTeX/KaTeX rendering of formulas. The model must produce
  formula strokes, not formula strings.
- Real student-facing deployment. Research prototype only.
- Audio input from a real student. Optional later via text input.
- Full coverage of all KA / OCT topics. Pilot = math (KA arithmetic +
  algebra + geometry + intro calc), then expand.
