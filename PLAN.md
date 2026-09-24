# Plan — after the Method 1 experiment

*September 2026. The May 2026 plans are kept as written in
`research/docs/archive/` and `tutor/docs/archive/`.*

## Where this starts from

The research half of the project asked whether one model can learn a
teacher's board and voice together from video. It was built, measured and
written up (`research/docs/CONCLUSIONS.md`, `research/docs/report/`). In
short: the board carries real information about the next word; carrying
pen tokens in the same sequence costs more than that information is worth;
the speech carries nothing usable about the next pen move; and the model
cannot draw because the data never says which symbol is being drawn.

## The decision

Stop pursuing stroke-level joint generation at this scale. Separate the two
jobs the flat model was forced to do at once:

- **deciding** what to say and what to put on the board, where — a language
  problem, handled by a language model;
- **rendering** it on the board — a drawing problem, handled by the player.

The script format in `tutor/` already is that separation: a lesson is a
list of `say` / `draw` / `say_draw` steps whose draws are shapes with board
coordinates (`tutor/docs/SCHEMA.md`). The May-2026 roadmap for that half
(Khan Academy pipeline → Qwen fine-tune → Qwen-VL real-time agent) is
retired along with the stroke-level plan; what replaces both is below.

## Next

**1. Extend the script primitives.** The schema has `line`, `rect`,
`circle`, `triangle`, `text`, `formula`. A math lecture also needs a
fraction, an underline/box for emphasis, an arrow, labelled axes with a
curve, and a "clear the board" step. Measure the teacher's layout habits
from the corpus — where a problem starts, line spacing, when he moves
right or down — and turn them into defaults the writer can follow.

**2. Write the lesson with an agent instead of a pasted prompt.** A language
model runs as a loop: plan a step, render it, look at the board state, plan
the next. It emits the same JSON the player already runs. Examples for
pacing come from the aligned transcripts — how this teacher opens a topic,
how many words per drawing, where the silent writing pauses fall. Target:
three topics end to end in the player without hand edits.

**3. Evaluate what the flat model could never be asked.** Per step: does the
drawing match what is being said? Per lesson: does the board stay readable
(no overlaps, nothing off-canvas)? Compare with the existing hand-prompted
scripts and with a real teacher segment.

**4. Later, the teacher's handwriting.** Render the primitives in his hand
instead of a font, with a text-to-ink model fine-tuned on the extracted
strokes. This is where the stroke corpus comes back in.

## Not doing

- Fine-tuning a math LLM end-to-end on stroke tokens, or fine-tuning one on
  transcripts and bolting on a learned stroke renderer (the May plan's
  Methods 2 and 3). The agent above is the practical descendant of the
  second; the first has the same problem the ablation measured.
- A bigger flat model for its own sake. It might shrink the
  attention-dilution cost until the board's information wins; that is a
  scaling question worth a few GPU-hours some day, not the product path.
  Two cheap checks are noted at the end of `research/docs/EXPERIMENTS.md`.

## Data

`output.zip` (1,191 videos, word-anchored `training.jsonl`, 52 MB):
[`data-v1` release](https://github.com/Tedd618/chalk-talk/releases/tag/data-v1).
The extraction and alignment pipeline is in `research/experiments/`; the
per-video intermediates for the full corpus exist only on the lab
filesystem.
