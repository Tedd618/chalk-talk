# AI Blackboard Tutor

An AI tutor that teaches math by drawing on a virtual blackboard while
explaining out loud, in sync — the way a teacher on YouTube does.

## What happened so far

**May 2026 — two ways to make a lesson.** The quick one: a language model
writes the whole lesson up front as a script of `say` / `draw` steps with
board coordinates, and a browser player speaks and animates it. That was
built first and works end to end (`tutor/`). The hard one — the research
question — was whether a model could learn a teacher's *board and voice
together* from real lecture video, and then generate both stroke by
stroke, in that teacher's hand.

**June–September 2026 — the experiment.** 1,191 Organic Chemistry Tutor
videos were turned into an aligned sequence of spoken words and pen
movements (12.5M tokens). One transformer was trained on that sequence,
and a controlled ablation — seven context conditions on identical data,
three training seeds each — measured whether each modality actually helps
predict the other (`research/`, full write-up in
[research/docs/report/report.pdf](research/docs/report/report.pdf)).

**What it found.**

| question | answer |
|---|---|
| Does the board help predict the next word? | **Yes.** +0.125 nats, robust across seeds. |
| Is it worth carrying pen tokens in the same sequence to get that? | **No.** Doing so costs −0.42 nats — more than the gain — through attention dilution and positional spreading. |
| Does the speech help predict the next pen move? | **No** usable signal. |
| Can the model draw? | No. Scatter in the right place on the board, never a character — and not for lack of data, but because nothing in the data says *which symbol* is being drawn. |

**What changed because of it.** The two signals are related, but one flat
stream is the wrong way to carry the relationship at this scale. The
decision of *what to say and what to draw* is a language problem; *putting
it on the board* is a rendering problem; the experiment says to keep them
apart. That is exactly the shape the script + player already has. So the
path that started as the quick MVP becomes the main line — with the lesson
written by an agent rather than pasted from a chat — and the stroke corpus
becomes reference material for how this teacher paces and lays out a
lesson, and later for rendering in his handwriting, instead of the thing a
model learns to emit point by point. What comes next is in
[PLAN.md](PLAN.md).

## Repository

```
research/   the finished experiment — data pipeline (video → strokes → aligned
            events), the flat stroke+speech model, the ablation, results,
            report; PROGRESS.md is the dated diary; docs/archive/ has the
            May plans as written
tutor/      lesson script schema (say / draw / say_draw with coordinates),
            browser player, the prompt that produces scripts; what the next
            step builds on
PLAN.md     the direction after the experiment
```

## Data

The training data (1,191 videos → `training.jsonl`, 52 MB zipped) is not
in git. It is attached to the
[`data-v1` release](https://github.com/Tedd618/chalk-talk/releases/tag/data-v1)
as `output.zip`; the pipeline that produced it is
`research/experiments/a1-stroke-extraction/` → `a2-alignment/` →
`a4-train/align.py`.
