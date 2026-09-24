# Related Work — Part A

> **2026-09-24:** every arXiv ID in this file was re-verified against arxiv.org
> (titles, authors, dates all correct; DiffInk was subsequently accepted at ICLR 2026).
> The full, verified bibliography used by the Method 1 report — 36 entries covering
> handwriting/sketch generation, lecture-video analysis, multimodal transformers,
> context/attention behaviour, measurement, and LLM drawing agents — is
> `report/references.bib`.

Two papers anchor the design. Each addresses one half of the problem.

---

## Sketch-RNN — Ha & Eck, 2017 (arXiv:1704.03477)

**Summary.** A sequence-to-sequence Variational Autoencoder for vector
sketches. Sketches are represented as a sequence of pen actions:

```
each step → (Δx, Δy, p_down, p_up, p_end)
```

`(Δx, Δy)` is the offset from the previous pen position. The pen state
is a one-hot over three categories: pen touching paper (line will be
drawn to next point), pen lifted (no line), and end-of-sketch.

The decoder outputs, at each step:
- A **Mixture Density Network (GMM)** over `(Δx, Δy)` — `M` bivariate
  Gaussians with means, std devs, correlation, and mixture weights.
- A **categorical** over `(p_down, p_up, p_end)`.

Trained on QuickDraw — 70K samples per class. Conditional generation
encodes a sketch into a latent vector `z`; unconditional drops the
encoder and trains the decoder alone.

**What we adopt for Method A**:
- The pen-action representation `(Δx, Δy, p_down, p_up, p_end)` —
  this is our native output format for *all* on-board content,
  including math symbols and English words.
- The MDN-over-Δ + categorical-over-pen-state head pattern.
- Temperature-controlled sampling (`τ`) for varying determinism at
  inference.

**What we have to add (Sketch-RNN doesn't do)**:
- Multimodal conditioning. Sketch-RNN conditions on at most a class
  label or a latent. We need conditioning on the **current canvas
  image, prior speech, and prior strokes**.
- Heterogeneous output. Sketch-RNN outputs only strokes. We need
  interleaved **speech tokens + pen actions** in one sequence.
- Long-horizon scale. A QuickDraw sketch is ~50–200 pen events. A
  5-minute lesson is thousands. We need a transformer-based decoder,
  not the original RNN, and probably hierarchical structure.

---

## SketchAgent — Vinker et al., MIT/Stanford 2024

**Summary.** A frontier multimodal LLM (Claude 3.5 Sonnet) sketches
**without any training** by being given a custom string-based
"sketching language" via in-context learning. The agent draws on a
50×50 numbered grid (cells named `x5y20` etc.), produces sparse
coordinate samples per stroke, which are post-processed into smooth
cubic Bézier curves.

Their pipeline runs the model in a loop with the canvas re-fed as an
image after each batch of strokes — exactly the closed-loop structure
we want for Method A. Includes a stop-token convention and a
human-in-the-loop pause-resume mechanism for collaborative sketching.

**What we adopt for Method A**:
- The **closed-loop mechanic** itself. SketchAgent already validates
  that re-feeding the canvas works for sketching with frontier VLMs.
- The **numbered grid prompt** as a way to make off-the-shelf VLMs
  spatially competent — their Fig. 6 explicitly shows GPT-4o failing
  with raw pixel coordinates and succeeding with grid cells. Useful
  for **synthetic data generation**.
- The stop-token / pause-resume convention.

**What we don't adopt**:
- The output representation. SketchAgent's sparse-points-plus-Bézier
  is great when you only have a frontier VLM and want smooth curves
  cheaply, but it bypasses the dynamics of writing. We want pen
  trajectories at the temporal resolution of an actual hand, learned
  from data — Sketch-RNN's regime.
- The "no training" stance. SketchAgent is an inference-time
  technique. Method A's research goal is a **trained** model.

**Role in our project**: SketchAgent is the **data bootstrap** —
running a frontier VLM in this loop on math topics gives us synthetic
training samples to mix with the real video extractions while we
build the extraction pipeline.

---

## Speech-Synchronized Whiteboard Generation — arXiv:2603.25870 (March 2026)

**Summary.** The closest prior art to this project. Given a topic prompt,
a VLM generates a structured drawing representation (whiteboard content
plan), which is then rendered as synchronized whiteboard video with speech.
The output is a whiteboard lecture video with coordinated speech.

**Key limitation**: trained and demonstrated on only **24 demos**. No
large-scale real teaching video used. The strokes are synthetically
planned by a VLM, not learned from a real teacher's actual hand movements
and speech patterns.

**Why our approach is different**:
- We train from **hundreds of hours of real teacher video** — actual
  hand movements, actual timing, actual teaching cadence learned from data.
- We generate **learned handwriting style** (OCT's specific stroke shapes
  and flow), not VLM-planned geometry.
- We interleave speech and strokes as a joint sequence prediction problem,
  not a plan-then-render pipeline.
- Scale: ~20 hours of OCT vs. their 24 demos.

**The genuine research gap**: No prior work trains a generative model on
real teaching video at scale to jointly generate speech tokens and pen
strokes as a single learned sequence.

---

## VideoSketcher — arXiv:2602.15819 (February 2026)

**Summary.** Generates sketching videos from text prompts. Focuses on
the process of drawing (showing the hand sketching) rather than
educational content. Art/illustration domain.

**Relevance**: Shows the field is moving toward video-as-output for
generative sketch models. But it doesn't address the math education
domain, speech-stroke alignment, or learning from real teacher video.

---

## Where this stands (September 2026)

The plan above was written in May 2026 around a trained, stroke-level,
speech-conditioned model. That model was built and measured
(`METHOD1_CONTEXT_EXPERIMENT.md`, `report/report.pdf`): strokes carry real
information about the next spoken word (+0.125 nats, three seeds), but
carrying pen tokens in the same sequence costs more than that (−0.42 nats),
and speech carries no usable information about the next pen move at this
scale. The joint stroke+speech generation goal was not reached.

Consequences for this reading list:

- **Sketch-RNN** stays the output representation for pen data and the reason
  the model has a mixture-density head. Its VAE/LSTM parts were never used.
- **SketchAgent**, **DiagrammerGPT**, **LayoutGPT** and **AutomaTikZ** — LLMs
  that emit a drawing *plan* with coordinates — are now the model for the next
  step: an agent that writes the lecture as `say`/`draw` steps on Part B's
  script schema, rendered by the player.
- **arXiv:2603.25870** (speech-synchronized whiteboard generation from 24
  demos) remains the closest prior work; the difference is that we have the
  1,191-video corpus to learn pacing and layout conventions from.
- The token-granularity survey (ScribeTokens, DiffInk, StrokeFusion) is
  relevant again only if primitives are later rendered in the teacher's own
  handwriting; DiffInk (ICLR 2026) would be the tool.

Full citations for all of the above: `report/references.bib` (36 entries, each
verified against its arXiv/publisher page on 2026-09-24).
