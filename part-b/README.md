# Part B — Pre-generated script (MVP)

A text LLM produces the entire lesson as a JSON script up-front. The
browser player runs the script: speech blocks until TTS finishes,
drawings animate stroke-by-stroke, and `say_draw` steps run them in
parallel for the "tutor speaks while writing" feel.

## Layout
- [docs/PLAN.md](docs/PLAN.md) — roadmap and Stage 1 detailed plan
- [docs/PROMPT.md](docs/PROMPT.md) — master prompt for Claude web
- [docs/SCHEMA.md](docs/SCHEMA.md) — script JSON schema and primitives
- [player/index.html](player/index.html) — browser player (open in Chrome/Edge)
- [scripts/example-pythagoras.json](scripts/example-pythagoras.json) — sample script

## Status
End-to-end working. Speech + animated drawings synced.
**Known limitation**: text labels and formulas don't render stroke-by-stroke
(text uses typewriter, formula fades in). This is a player-side rendering
gap, not a format issue. See [docs/PLAN.md](docs/PLAN.md).

## Quick start
1. Open `player/index.html` in a browser.
2. Click "Load example" → "Play".
3. To try a new topic: paste [docs/PROMPT.md](docs/PROMPT.md)'s system prompt
   into Claude web, send a topic, paste the JSON back into the player.
