# Related Work — Part A

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

## How they combine for Method A

| Phase of Method A | Inheritance |
|---|---|
| A.0 baseline (frontier VLM in a loop, no training) | SketchAgent for the loop mechanic and grid prompt |
| A.1 stroke extraction from videos | Neither — original work, see [EXPERIMENTS.md](EXPERIMENTS.md) |
| A.2 speech-stroke alignment | Whisper + frame timing — neither paper |
| A.3 stroke-only generative model (Sketch-RNN baseline) | Sketch-RNN architecture directly |
| A.4 multimodal stroke + speech model | Sketch-RNN backbone + new conditioning layers |

The two papers together cover roughly: "how do we run the loop?"
(SketchAgent) and "how do we generate strokes?" (Sketch-RNN). The
work specific to Method A — math/text strokes, multimodal
conditioning, KA + Organic Chemistry Tutor extraction, speech-stroke
interleaving — is what fills the gap between them.

---

## Token Granularity — What Recent Work Uses (May 2026)

Three approaches have emerged for tokenizing continuous pen trajectories:

**Discretized point tokens (ScribeTokens, arXiv:2603.02805, 2026)**
Converts stroke offsets to unit steps (Bresenham decomposition), then compresses with BPE into a fixed vocabulary. Makes handwriting compatible with standard LM machinery. Achieves 17.33% CER vs 70.29% for raw vectors on sentence generation. Still point-level but discrete.

**Stroke-level latent tokens (DiffInk, arXiv:2509.23624, 2025)**
Trains a stroke VAE (InkVAE) with OCR + style losses, then runs a latent diffusion transformer (DiT) over the compact latent codes. Current SotA for text-to-online handwriting. 10-30x shorter sequences than point-level. StrokeFusion (arXiv:2503.23752, 2025) does the same non-autoregressively.

**Hierarchical: word planner + stroke diffusion decoder (Option D)**
Generate word-level bounding box plan autoregressively, then decode each word's strokes via a small stroke diffusion model. Natural fit for speech-stroke interleaving — word boundaries align speech tokens and stroke groups naturally.

**Recommendation for A.4a**: Option D — word-level autoregressive planning + per-word stroke diffusion using a VAE latent pretrained on MathWriting. Keeps sequences tractable, aligns with speech at word boundaries, and matches DiffInk's architecture.

---

## Novelty summary (as of May 2026)

The closest prior work (arXiv:2603.25870) uses a VLM to *plan* whiteboard
content and renders it — 24 demos, no training from real video. VideoSketcher
(2602.15819) generates art-domain sketching video, not educational content.

**What exists**: VLM-planned whiteboard rendering, art sketch generation,
speech-to-text, text-to-speech.

**What does not exist**: A model trained end-to-end on real teaching video
that *jointly* generates speech tokens and pen strokes as a single learned
sequence, in the style of a specific teacher.

This is the gap our work fills.
