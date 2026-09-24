# Research — Conclusions

What the Method 1 experiment established. The experiment itself is in
[EXPERIMENTS.md](EXPERIMENTS.md), the design and every bug in
[METHOD1_CONTEXT_EXPERIMENT.md](METHOD1_CONTEXT_EXPERIMENT.md), the
write-up in [report/report.pdf](report/report.pdf). What the project does
because of these conclusions is the repository-level [PLAN.md](../../PLAN.md).

## The question

> Can handwriting strokes and spoken words act as context for each other —
> so that a model trained on real teacher videos can generate a full
> handwritten lecture on its own?

Tested with one flat transformer over the interleaved word/pen sequence of
1,191 Organic Chemistry Tutor lectures, and a 7-condition, 3-seed ablation
that removes one kind of context at a time on identical validation windows.

## What we learned

1. **The board does tell you the next word.** Real pen coordinates reduce
   word loss by 0.125 nats, in every seed, every interval. The relationship
   between what is drawn and what is said is real and a model can use it.
2. **Carrying pen tokens in the same stream costs more than that.** With
   78% of the sequence being pen, attention spread over the empty pen slots
   costs 0.27 nats before any content is read, and the positional spreading
   of the words costs another 0.15. A word-only model predicts words better
   than the full model (perplexity 47 vs 63).
3. **The speech does not tell you the next pen move.** Real word identities
   make pen prediction slightly worse than a placeholder, in all seeds. The
   pen's next move is decided by its last few moves.
4. **The model cannot draw** — scatter in the right region of the board,
   never a character — and not for lack of data: pen loss was still falling
   at the last epoch and the corpus is larger than what classic handwriting
   synthesis was trained on. What is missing is a label for *which symbol*
   is being drawn; the speech is not a usable substitute at this size.

Together: the information is there, in one direction, and mixing the two
modalities into one flat stream is the wrong way to carry it at this scale.
"More context is better" stops holding when the extra context takes 78% of
what the model is looking at.

## What this does not settle

- All of it is at 4.6M parameters and 2,700 steps. A larger model could
  learn to ignore the pen tokens, shrinking cost 2 until benefit 1 wins.
- The stroke ladder's bottom rung leaves `<silent>` visible in two
  conditions and removes it in the third, so its length effect includes
  that marker.
- Magnitudes vary 2–3× across seeds even where signs and intervals agree.
- Strokes are attached to words by time overlap, which is approximate; that
  makes finding 1 look smaller and finding 3 closer to zero than they are.
- Every video is one page; board clears did not survive the pipeline.

## What follows

The decision — stop stroke-level joint generation at this scale, keep the
*deciding* (language) and the *rendering* (drawing) apart, build the next
step on the tutor's script format — and the plan for it are in
[PLAN.md](../../PLAN.md).
