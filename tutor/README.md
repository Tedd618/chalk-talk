# Tutor — lesson script and browser player

A lesson is a JSON script of steps: `speech`, `draw`, `say_draw` (speak
while drawing — the step that gives the "teacher writing as he talks"
feel) and `wait`. Draws are shapes with board coordinates — `line`,
`rect`, `circle`, `triangle`, `text`, `formula` (LaTeX via KaTeX). The
browser player speaks the text with the Web Speech API and animates each
shape stroke by stroke on a Canvas.

Built in May 2026 as the quick "pre-generated script" MVP while the
research half of the project ([../research/](../research/)) pursued
stroke-level generation. After that experiment concluded that deciding
*what to say and draw* and *rendering it* should be kept apart, this
format and player became the base of the next step: a lesson written by
an agent rather than pasted from a chat. See [../PLAN.md](../PLAN.md).

## Layout

- [docs/SCHEMA.md](docs/SCHEMA.md) — the script JSON schema and primitives
- [docs/PROMPT.md](docs/PROMPT.md) — the system prompt that makes an LLM chat emit a valid script (written for Claude web)
- [player/index.html](player/index.html) — the player; open in Chrome or Edge
- [scripts/example-pythagoras.json](scripts/example-pythagoras.json) — sample script
- [docs/archive/](docs/archive/) — the May 2026 roadmap, kept as written

## Status

End to end working: speech and animated drawings, synced. Known limitation:
`text` renders as a typewriter effect and `formula` fades in — neither is
drawn stroke by stroke. That is a player-side gap, not a format issue.

## Quick start

1. Open `player/index.html` in a browser.
2. Click "Load example" → "Play".
3. For a new topic: paste the system prompt from
   [docs/PROMPT.md](docs/PROMPT.md) into an LLM chat, send a topic, paste
   the JSON back into the player.
