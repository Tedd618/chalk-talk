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
- Speech is sparse and bursty — long silent writing stretches force
  either "silent-word" tokens or one word stretched over many
  strokes. Likely the worst fit to the data of the three.

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

- **Action-type head** at each top-level step: decides what kind of
  token is being emitted (per-option type vocabulary) plus an `END`.
- **Sub-decoder** producing pen events, conditioned on the top-level
  token. Sketch-RNN style: MDN over `(Δx, Δy)`, categorical over
  pen-state.
- **Word vocabulary**: subword tokenizer (BPE) trained on the
  KA + OCT transcripts, ~16k–32k tokens. Same tokenizer across all
  three variants for fair comparison.

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

1. **Khan Academy** — CC BY-NC-SA, research-friendly. Sal Khan's
   board work: digital pen on a black canvas, mostly no hand
   occlusion (pure stroke trace), ~1000s of hours, good speech-board
   sync. Best signal-to-noise for stroke extraction.
2. **The Organic Chemistry Tutor** — YouTube, more diverse content,
   includes molecule diagrams and longer derivations. Some videos
   show physical paper with hand occlusion; others are tablet
   recordings. Provides style diversity.
3. **(Possibly) 3Blue1Brown** — animation-heavy, less directly usable
   for stroke-level training, but the speech is high-quality. Listed
   for future consideration.
4. **(Possibly) SketchAgent-style synthetic data** — frontier VLM in
   the loop generates additional training samples on math topics.
   Useful for filling gaps but secondary to real video extraction.

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
