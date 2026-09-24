# AI Blackboard Tutor

An AI tutor that teaches math by drawing on a virtual blackboard while
explaining out loud, in sync — like a teacher on YouTube.

The project has two parts. Part A is the research: can a model learn a
teacher's board and voice *together* from real lecture video? Part B is the
player and script format that a lesson runs on. After the Part A
experiment, the two meet: the next step is an agent that writes lessons in
Part B's format.

## Part A — research: learning from real teacher video ([part-a/](part-a/))

1,191 Organic Chemistry Tutor videos were turned into an aligned sequence
of spoken words and pen movements (12.5M tokens), and one transformer was
trained on that sequence to answer one question: **does what is on the
board help predict the next word, and does what is said help predict the
next pen move?**

The answer, from a controlled ablation (7 context conditions × 3 seeds):

| | result |
|---|---|
| Board → next word | **yes**, +0.125 nats, robust across seeds |
| Cost of carrying pen tokens in the same sequence | −0.42 nats — more than the gain |
| Speech → next pen move | no usable signal |

So the two signals are related, but mixing them into one flat stream is the
wrong way to use that relationship at this scale. Full write-up:
[part-a/docs/report/report.pdf](part-a/docs/report/report.pdf)
(LaTeX source and verified bibliography alongside it).

## Part B — script format and browser player ([part-b/](part-b/))

A lesson is a JSON script of `say` / `draw` / `say_draw` steps; draws are
shapes with board coordinates (`line`, `rect`, `triangle`, `circle`,
`text`). The browser player speaks the text and animates the drawing
stroke by stroke. Working end to end.

This is the foundation of the **next step**: a language-model agent that
writes the script — what to say, what to draw, where — using the Part A
corpus for the teacher's pacing and layout conventions. See
[part-a/docs/PLAN.md](part-a/docs/PLAN.md).

## Data

The training data (1,191 videos → `training.jsonl`, 52 MB zipped) is not in
git. It is attached to the
[`data-v1` release](https://github.com/Tedd618/chalk-talk/releases/tag/data-v1)
as `output.zip`. The pipeline that produced it is in
`part-a/experiments/a1-stroke-extraction/` and `a2-alignment/`.

## Layout

```
part-a/
  docs/            PLAN.md · EXPERIMENTS.md · METHOD1_CONTEXT_EXPERIMENT.md
                   RELATED_WORK.md · report/ (tex, bib, pdf) · archive/ (May plans)
  PROGRESS.md      dated diary of the work
  experiments/
    a1-stroke-extraction/   video → strokes (skeleton extraction, queue workers)
    a2-alignment/           Whisper words + strokes → events.jsonl
    a3-sketchrnn/           stroke-only transformer pretraining (standalone)
    a4-train/               flat stroke+speech transformer, context ablation, results
part-b/
  docs/            SCHEMA.md · PROMPT.md · PLAN.md
  player/          index.html — open in a browser
  scripts/         example lesson
```
