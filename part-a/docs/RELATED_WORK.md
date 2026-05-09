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
