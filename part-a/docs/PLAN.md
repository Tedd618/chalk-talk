# Part A — Plan

## The core research question

> Can handwriting strokes and spoken words act as context for each
> other — so that a model trained on real teacher videos can generate
> a full handwritten lecture on its own?

This is the central bet of Part A. We train a model by showing it
hundreds of hours of a real teacher (The Organic Chemistry Tutor)
writing and speaking simultaneously. Then we give it a single
sentence — "Today we are going to learn the Pythagorean theorem" —
and ask it to generate the rest of the lesson: the words, the
diagrams, the equations, written stroke by stroke in that teacher's
hand.

Nobody has trained a model this way before. The novelty is treating
*the live act of teaching* — words and pen movements woven together
in time — as a learnable sequence.

---

## How it works

A lesson is a long sequence of two kinds of events, interleaved
in time:

```
"today"  "we"  "have"  [draws a²]  "a"  "squared"  [draws +]  "plus" ...
```

We train a model to predict the next event given everything before it.
At inference, we give it the opening sentence and let it run.

The model writes the way OCT writes — stroke by stroke, character by
character, no shortcuts. Every letter and symbol is handwritten, just
as it appears in the training videos.

---

## The three methods we will build and compare

This comparison IS the research paper.

---

### Method 1 — Pure OCT model

> Train one model entirely on OCT data. It learns to speak and write
> by watching OCT teach.

```
"Today we will learn the Pythagorean theorem."
                    │
                    ▼
           ┌─────────────────┐
           │  OCT model      │
           │  (trained on    │
           │  OCT lectures)  │
           └────────┬────────┘
                    │
                    ▼
    words + handwritten strokes, generated together
    (full lecture, in OCT's voice and handwriting)
```

**Training data:** events.jsonl from OCT videos — words and strokes
interleaved in time order. The model predicts the next word or the
next stroke given all previous words and strokes.

**What it learns from data alone:**
- OCT's handwriting style and stroke timing
- Which strokes go with which spoken words
- How to structure a lesson (problem → explanation → working → answer)
- When to draw a diagram vs. write an equation
- Spatial layout — where things go on the board

**The research question it answers:**
Can a model become a teacher just by watching one? No math knowledge
built in — everything must come from the data.

**Data needed:** ~20 hours of diverse OCT math videos, covering
algebra, geometry, fractions, calculus. Broad topic coverage matters
more than total hours.

---

### Method 2 — Qwen speaks, OCT draws

> Use a powerful AI (Qwen) to generate the words. Use a trained OCT
> stroke model only to render the handwriting.

```
"Explain the Pythagorean theorem"
                    │
                    ▼
           ┌─────────────────┐
           │  Qwen           │
           │  (knows math,   │
           │  generates text)│
           └────────┬────────┘
                    │ spoken explanation, word by word
                    ▼
           ┌─────────────────┐
           │  OCT stroke     │
           │  model          │
           │  (renders each  │
           │  word as strokes│
           │  in OCT's hand) │
           └────────┬────────┘
                    │
                    ▼
    Qwen's explanation, written in OCT's handwriting
```

**Training data for the OCT stroke model:** same events.jsonl, but
the model is only trained on the stroke-generation task — given the
words being spoken and what is already on the board, draw the next
stroke.

**What Qwen contributes:** mathematical correctness and the ability
to handle novel problems OCT never covered.

**What the OCT stroke model contributes:** handwriting style, spatial
layout, timing.

**The research question it answers:**
Does separating thinking (Qwen) from writing (OCT model) produce
better lectures than a single model doing both?

**Data needed:** same OCT corpus for the stroke model; Qwen used
as-is with no training.

---

### Method 3 — Qwen fine-tuned to teach like OCT

> Teach Qwen OCT's teaching style, then use the OCT stroke model
> to render what Qwen says.

```
"Explain the Pythagorean theorem"
                    │
                    ▼
           ┌─────────────────┐
           │  Qwen           │
           │  (fine-tuned on │
           │  OCT transcripts│
           │  to teach like  │
           │  OCT)           │
           └────────┬────────┘
                    │ explanation in OCT's teaching style
                    ▼
           ┌─────────────────┐
           │  OCT stroke     │
           │  model          │
           │  (same as       │
           │  Method 2)      │
           └────────┬────────┘
                    │
                    ▼
    OCT-style explanation, written in OCT's handwriting
    with Qwen's mathematical depth
```

**Training data for Qwen fine-tune:** OCT speech transcripts — what
he says, how he structures explanations, his phrasing and pacing.

**The research question it answers:**
Can we get the best of both — Qwen's mathematical power AND OCT's
teaching personality? And how does this compare to training from
scratch on video data?

**Data needed:** OCT transcripts (text only) for fine-tuning Qwen;
OCT corpus for stroke model.

---

## What we are comparing

| | Math accuracy | OCT style | Novel topics | Data needed |
|---|---|---|---|---|
| Method 1 (pure OCT) | OCT's level | ⭐⭐⭐ high | Only seen topics | 20+ hrs video |
| Method 2 (Qwen + OCT draw) | Qwen's level | ⭐⭐ medium | Any topic | 20+ hrs video |
| Method 3 (Qwen fine-tune + OCT draw) | Qwen's level | ⭐⭐⭐ high | Any topic | 20+ hrs video + transcripts |

The goal is to see which approach produces a lecture that a real
student would want to watch.

---

## Data pipeline

```
YouTube video (OCT math lecture)
        │
        ▼  download.py
  video file
        │
        ▼  frames.py  (extract frames at 30fps)
  frame images
        │
        ▼  extract_v4.py  (skeleton-based stroke extraction)
  strokes.jsonl  (each stroke: x,y,t points + page number)
        │
        │          audio track
        │               │
        │               ▼  stt.py  (Whisper speech-to-text)
        │          words.jsonl  (each word + timestamp)
        │               │
        └───────────────┘
                        │
                        ▼  merge.py
                  events.jsonl  (words and strokes interleaved in time)
                        │
                        ▼  [to be built] align.py
                  training.jsonl  (each stroke tagged with surrounding words;
                                   ready for sequence model training)
```

One command runs everything:
```bash
python pipeline.py "https://youtube.com/watch?v=..." --tag my-video
```

---

## Training sequence format

Each training example is one full lesson, represented as an ordered
sequence of events:

```
[LESSON: "Pythagorean theorem"]
  → word("today")
  → word("we")
  → word("have")
  → stroke(points=[[120,80],[180,80]], page=0)   ← draws "a²"
  → word("a")
  → word("squared")
  → stroke(points=[[200,80],[200,80]], page=0)   ← draws "+"
  → word("plus")
  ...
  → page_break
  → word("now")
  → stroke(...)
  ...
  → [END]
```

The model is trained to predict the next item in the sequence given
all previous items. At inference, the trigger sentence starts the
sequence and the model generates the rest.

---

## Milestones

### Done ✅
- **A.1** Stroke extraction from OCT video (v4 skeleton pipeline,
  per-page, colored annotations, 10fps subsampling)
- **A.2** Speech-stroke alignment (Whisper + merge into events.jsonl)
- **A.3** Stroke model pretrained on MathWriting (229k expressions,
  3.3M params — gives the model a feel for how math strokes flow)
- **pipeline.py** — one command, URL to training data

### Next — corpus and data format
- **A.4-data** Scale OCT corpus to ~20 hours across diverse math topics.
  Run `pipeline.py` on ~40-60 OCT videos on the lab machine.
- **A.4-format** Build `align.py` to add word context windows to each
  stroke, producing the final training sequence format.

### Core training — Method 1 first
- **A.4a** Train the pure OCT model. This is the main research claim.
  Success = give it a trigger sentence, it generates a visible,
  coherent handwritten lecture.

### Comparison methods
- **A.4b** Train/connect the OCT stroke model for Method 2 (Qwen as
  script generator, OCT model renders strokes).
- **A.4c** Fine-tune Qwen on OCT transcripts + connect OCT stroke
  model (Method 3).

### Evaluation
- **A.5** Compare all three methods. What makes a good lecture?
  Can a student follow it? Does the model write what it says?

### Stretch
- **A.6** Close the loop — wire the best model into a real browser
  player. Student types a question, model generates the lecture live.

---

## Why OCT specifically

The Organic Chemistry Tutor writes slowly and deliberately — each
letter and symbol is a separate stroke, clearly separated. This makes
extraction clean and reliable. KA-style cursive writing is much harder
to extract accurately and is deferred.

OCT also has hundreds of hours of math content on YouTube, all in
a consistent style, which is exactly the kind of large homogeneous
dataset that lets a model learn a specific person's teaching behaviour.

---

## Why this is hard

1. **Long sequences** — a 10-minute lesson is ~300 strokes and ~800
   words interleaved. The model needs to hold context over the whole
   thing.
2. **Two modalities at once** — predicting the next word and predicting
   the next stroke are very different tasks. The model must learn to
   switch between them naturally.
3. **Spatial coherence** — the model must remember what is already on
   the board and not redraw it, put new things in sensible places,
   and keep equations aligned.
4. **Style consistency** — the handwriting should look like the same
   person throughout, not drift into random shapes.
5. **Data scale** — 20 hours sounds like a lot but for a from-scratch
   model it is modest. Quality and diversity of topics matter enormously.
