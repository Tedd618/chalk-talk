# AI Blackboard Tutor

An AI tutor that teaches math concepts by drawing on a virtual blackboard
while explaining out loud, in sync.

The architecture doc lays out two methods. Both target the same blackboard
format and player; they differ in *how* the script is produced.

## Two parts

### [part-b/](part-b/) — Method B: pre-generated script (MVP, working)
A text LLM produces the entire lesson up-front as JSON. The browser
player reads the JSON and animates speech + drawings stroke-by-stroke.
No multimodal model needed. Currently runs via Claude web (manual paste).

> Status: working end-to-end. See [part-b/README.md](part-b/README.md).

### [part-a/](part-a/) — Method A: real-time stroke-level reactive tutor (research)
A vision-language model that, at each step, looks at the current
canvas + history and decides the **next single action** (one phrase
or one stroke). True closed loop — reactive, no pre-baked script.

> Status: planning + experiment design. See [part-a/README.md](part-a/README.md).

## Why both

The architecture doc treats them as a sequence: B is a working product
and A's data foundation; A is the research target where the canvas
becomes the model's state. Same player, same primitives — A swaps
"generate the whole script" for "loop one step at a time looking at
the canvas".
