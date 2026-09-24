# Plan

## Big picture (4-stage roadmap)

The original architecture document presents two methods:
- **Method A** — At each step, the model looks at the current canvas and
  decides the next action (multimodal, improvisational, can react to the student).
- **Method B** — Given a topic, generate the entire script up-front
  (text-only LLM, pre-generated, no improvisation).

A is the ultimate goal, but the B pipeline becomes the data foundation for A,
so B → A is the rational order.

| Stage | Goal | Deliverable |
|---|---|---|
| **1. MVP (current)** | Validate Method B works, stabilize the format | Claude-web prompt + browser player |
| 2. Data pipeline | Khan Academy videos → (script) extraction | STT + stroke extraction + interleaving refiner |
| 3. B fine-tuning | Fine-tune Qwen2.5-Math-7B | Self-hosted model + API |
| 4. A extension | Add canvas input, multimodal, streaming | Qwen2-VL-based real-time agent |

Core principles:
- **Skeleton (timing/structure) from Khan Academy, flesh (responsiveness)
  from synthetic data**
- Stages 1–3 all use the same script format → pipeline/player are reusable
- When moving to A, just add canvas images to the same data

---

## Stage 1 MVP — detailed plan

### Goal
"Without fine-tuning, validate that prompt engineering alone can make
Method B produce a working format."

Success criteria:
1. Claude web reliably emits a valid script for **arbitrary math topics**
   using the same prompt
2. The player can take a script and play it back with **synced
   speech + drawing**
3. One full cycle (topic input → generation → playback) completes in
   **under 5 minutes**

### Non-goals (out of scope)
- Fine-tuning, self-hosting → stages 2–3
- Automated script generation (CLI/API) → user manually copies from Claude web
- Student responsiveness, interactivity → Method A
- Complex shapes (curves, Bézier, automatic graph axes) → later
- Precise timestamps, simultaneous playback → MVP is sequential
- Handwriting-style rendering → clean vector is enough

### Architecture

```
[user]
   │ topic
   ▼
[Claude web + PROMPT.md]
   │ JSON script
   ▼
[browser player (player/index.html)]
   │
   ├─ JSON parser
   ├─ sequential runner
   │     ├─ speech → Web Speech API (TTS, blocking)
   │     ├─ draw   → Canvas 2D immediate render
   │     └─ wait   → setTimeout
   └─ KaTeX (renders formula primitives)
```

### Tech decisions

| Item | Choice | Reason |
|---|---|---|
| Script generation | Claude web (manual) | No API key, fast iteration, sufficient for MVP |
| Script format | JSON | LLMs emit it reliably, simple to parse |
| Player | Single HTML in browser | Zero install, free TTS via Web Speech API |
| Graphics | Canvas 2D | Sufficient, no dependencies |
| Math typesetting | KaTeX (CDN) | LaTeX → clean render, faster than MathJax |
| TTS | Web Speech API | Free, browser-native |
| Timing | Runtime sequential (blocking) | Model doesn't need to emit timestamps; stable |

### Tasks

- [x] Project structure
- [x] Define script JSON schema ([SCHEMA.md](SCHEMA.md))
- [x] Master prompt for Claude web ([PROMPT.md](PROMPT.md))
- [x] Browser player ([player/index.html](../player/index.html))
- [x] End-to-end validation with sample script (Pythagorean theorem)
- [ ] **User validation**: try new topics in Claude web, collect failure cases
- [ ] Iterate the prompt based on validation results

### Validation scenarios

Generate and play back a script for each of these topics:
1. Pythagorean theorem (sample exists — baseline)
2. Quadratic formula
3. π and area of a circle
4. Definition and properties of exponents

Check points:
- JSON parsing failure rate
- How often coordinates fall outside the canvas (800×500)
- Whether speech-drawing sync feels natural
- LaTeX error rate

### Conditions to advance to next stage

- All 4 topics above play back **without manual edits**
- Prompt is stable enough that 5 consecutive runs of the same topic
  produce comparable quality
- → Begin Stage 2 (Khan Academy data pipeline)
