# Part A — Real-time stroke-level reactive tutor

A vision-language model that runs in a closed loop: at each step, it
looks at the current canvas image + speech history and decides the
**next single action** (one spoken phrase or one drawing primitive).
The canvas updates and the model is called again.

This is the research target — Method A in the architecture doc. Where
Part B commits to a full script, Part A reads the board every step
and is structurally able to react (to its own past output, to student
input, to mistakes).

## Layout
- [docs/PLAN.md](docs/PLAN.md) — architecture, output representation, milestones
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — concrete experiments to run, in order
- [docs/RELATED_WORK.md](docs/RELATED_WORK.md) — Sketch-RNN and SketchAgent: what we adopt, what we don't
- [experiments/](experiments/) — code per experiment as we go

## Status
Planning. No code yet. Critical path is **stroke extraction from
Khan Academy and Organic Chemistry Tutor videos** (Experiment A.1) —
without clean stroke data we can't train. Frontier-VLM baseline
(A.0) runs in parallel as a comparison target.
