# Part A — learning a teacher's board and voice from video

## Question

> Can handwriting strokes and spoken words act as context for each other?

Measured on 1,191 Organic Chemistry Tutor lectures with one flat
interleaved transformer and a 7-condition, 3-seed ablation. Answer: the
board helps predict the next word (+0.125 nats, robust); carrying pen
tokens in the same sequence costs more than that (−0.42 nats); speech does
not help predict the pen. The model does not draw legibly. Details and the
reasoning are in the report.

## Documents

- [docs/report/report.pdf](docs/report/report.pdf) — the report (source: `report.tex`, `references.bib`)
- [docs/METHOD1_CONTEXT_EXPERIMENT.md](docs/METHOD1_CONTEXT_EXPERIMENT.md) — experiment design, every bug found, v1 and v2 results
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — what was actually run, A.1 → A.4
- [docs/PLAN.md](docs/PLAN.md) — what was learned and what comes next
- [docs/RELATED_WORK.md](docs/RELATED_WORK.md) — the papers this builds on
- [docs/archive/](docs/archive/) — the May 2026 plans, kept as written
- [PROGRESS.md](PROGRESS.md) — dated diary

## Experiments

| dir | what it does | status |
|---|---|---|
| `experiments/a1-stroke-extraction/` | YouTube video → per-stroke `(x, y, t)` via skeleton extraction; queue workers for 1,191 videos | done |
| `experiments/a2-alignment/` | Whisper word timestamps + strokes → time-ordered `events.jsonl` | done |
| `experiments/a3-sketchrnn/` | stroke-only transformer on QuickDraw / MathWriting (standalone; not used by a4) | done |
| `experiments/a4-train/` | `align.py` (events → word-anchored training data), the flat stroke+speech model, the context-ablation notebook, results | done — see its README |

Data for a4 (52 MB zip) is on the
[`data-v1` release](https://github.com/Tedd618/chalk-talk/releases/tag/data-v1).
