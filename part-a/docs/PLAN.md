# Part A — Plan (September 2026)

The May 2026 plan is preserved in
[archive/PLAN_2026-05.md](archive/PLAN_2026-05.md). This file is what
replaced it after the Method 1 experiment.

## The core question, unchanged

> Can handwriting strokes and spoken words act as context for each other —
> so that a model trained on real teacher videos can generate a full
> handwritten lecture on its own?

## What we learned

Method 1 — one flat transformer over the interleaved word/pen sequence,
trained from scratch on 1,191 Organic Chemistry Tutor lectures — was
built, debugged and measured with a controlled ablation
([EXPERIMENTS.md](EXPERIMENTS.md), [report/report.pdf](report/report.pdf)).

1. **The board does tell you the next word.** Real pen coordinates reduce
   word loss by 0.125 nats, in every seed. The relationship is real.
2. **But carrying pen tokens in the same stream costs more than that.**
   Attention spreads over everything it can see, and 78% of the sequence
   is pen; that costs 0.27 nats before any content is even read, and the
   positional spreading of words costs another 0.15. A word-only model
   predicts words better than the full model.
3. **Speech does not tell you the next pen move.** The pen's next move is
   decided by its last few moves.
4. **The model cannot draw**, and not because the data is small — pen loss
   was still falling — but because nothing in the data says *which symbol*
   is being drawn, and the speech is not a usable substitute at this size.

The information is there, in one direction. Mixing the two modalities into
one flat stream is the wrong way to carry it.

## Decision

Stop pursuing stroke-level joint generation at this scale. The next system
separates the two jobs the flat model was forced to do at once: a language
model decides *what to say and what to draw, where*; a renderer puts it on
the board. Findings 1–4 all point this way — keep the language reasoning
in a stream that is all language, and give the drawing side an explicit
instruction instead of asking it to guess from the words.

Part B already has the format for this. Its script schema is a list of
`say` / `draw` / `say_draw` steps whose draws are shapes with board
coordinates, and its player animates them stroke by stroke. The plan
builds up from there.

## Next: an agent that writes the lecture

**Output.** A script in Part B's JSON: for each step, the sentence to say
and the elements to put on the board with coordinates — e.g.
`fraction 2/5 at (0.15, 0.20)`, `arrow from (0.17, 0.26) to (0.20, 0.38)`,
`text "LCD = 20" at (0.15, 0.40)`.

**M1 — extend the schema and renderer.** Add the primitives a math lecture
actually needs: fraction, equation/text line, arrow, underline/box,
labelled axes and a curve. Measure the teacher's layout conventions from
the corpus — where a problem starts on the board, line spacing, when he
moves right or down — and encode them as defaults.

**M2 — the agent.** A language model run as a loop (plan a step → render →
look at the board state → plan the next), in the style of SketchAgent /
DiagrammerGPT / LayoutGPT. Few-shot examples come from the aligned
transcripts: how this teacher opens a topic, how many words per drawing,
where the `<silent>` writing pauses fall. Target: three topics end to end
in the player.

**M3 — evaluation.** Two things the flat model could never be asked:
(a) does the drawing match what is being said at that step (judged per
step), and (b) does the layout stay readable over a whole lecture (no
overlaps, nothing off-board). Compare against Part B's existing
hand-prompted scripts and against a real teacher segment.

**M4 — stretch: the teacher's handwriting.** Render the primitives in his
hand instead of a font, using a text-to-ink model fine-tuned on the a1
strokes. This is where the stroke corpus comes back in.

## Not doing

- Methods 2 and 3 from the May plan (Qwen fine-tuned end-to-end on stroke
  tokens; Qwen fine-tuned on transcripts + a learned stroke renderer). The
  agent above is the practical descendant of Method 3; the stroke-token
  variant has the same problem finding 2 measured.
- A bigger flat model, for now. It might shrink the attention-dilution
  cost until finding 1 wins; that is a scaling question, not the product
  path. Two cheap checks are noted at the end of EXPERIMENTS.md if there
  is time.

## Data

`output.zip` (1,191 videos, word-anchored `training.jsonl`, 52 MB) is on
the [`data-v1` release](https://github.com/Tedd618/chalk-talk/releases/tag/data-v1).
The extraction and alignment pipeline is in `experiments/a1-*` and
`a2-*`; the full a1/a2 intermediates for all 1,191 videos are on the lab
filesystem only.
