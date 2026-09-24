# Script Schema

## Top-level structure

```json
{
  "title": "Pythagorean Theorem",
  "level": "Grade 8",
  "canvas": { "width": 800, "height": 500 },
  "steps": [ ... ]
}
```

| Field | Type | Description |
|---|---|---|
| `title` | string | Lesson title |
| `level` | string | Student level (e.g. "Grade 8", "High school 1") |
| `canvas` | object | Coordinate system. MVP always uses `{800, 500}` |
| `steps` | array | Steps executed sequentially |

## Steps

Each step is one of four types, distinguished by the `type` field.

### 1. `say_draw` — speak and draw together (preferred)

Speaks `text` while concurrently animating the listed draws in order.
Step ends when **both** the speech and all the draws finish.

```json
{
  "type": "say_draw",
  "text": "Let's draw a triangle",
  "draws": [
    { "shape": "triangle", "x1": 200, "y1": 350, "x2": 500, "y2": 350, "x3": 500, "y3": 150 }
  ]
}
```

This is what produces the "tutor explains while writing on the board"
feel. Use it whenever the speech describes what is being drawn.

### 2. `speech` — spoken explanation only

Blocks the next step until TTS finishes. Use for narration without
new drawings.

```json
{ "type": "speech", "text": "Notice that c is the longest side" }
```

### 3. `draw` — drawing only

Animates the shape (handwriting style). Blocks until the animation
finishes. Use sparingly — prefer `say_draw`.

```json
{ "type": "draw", "shape": "triangle", "x1": 200, "y1": 350, "x2": 500, "y2": 350, "x3": 500, "y3": 150 }
```

### 4. `wait` — pause

Waits the given number of milliseconds.

```json
{ "type": "wait", "ms": 500 }
```

## Animation semantics

All draws now animate stroke-by-stroke (handwriting feel) instead of
appearing instantly:

| Shape | How it animates |
|---|---|
| `line` | Pen travels from start to end |
| `rect` | Four edges drawn in sequence |
| `triangle` | Three edges drawn in sequence |
| `circle` | Arc traced from top, going clockwise |
| `text` | Typewriter, one character at a time |
| `formula` | KaTeX renders, then fades in over 500ms |

Default duration is computed from shape length (~0.9 ms/px) or character
count (~70 ms/char). Override with an optional `duration` field (ms) on
any draw.

## Primitives (draw.shape)

Coordinate system: top-left (0,0), bottom-right (800,500). Units are pixels.

### `line`
```json
{ "type": "draw", "shape": "line", "x1": 100, "y1": 100, "x2": 300, "y2": 100 }
```

### `rect`
```json
{ "type": "draw", "shape": "rect", "x": 100, "y": 100, "w": 200, "h": 150 }
```

### `circle`
```json
{ "type": "draw", "shape": "circle", "cx": 400, "cy": 250, "r": 80 }
```

### `triangle`
Three vertex coordinates.
```json
{ "type": "draw", "shape": "triangle", "x1": 200, "y1": 350, "x2": 500, "y2": 350, "x3": 500, "y3": 150 }
```

### `text`
Plain text (labels, variable names, etc.).
```json
{ "type": "draw", "shape": "text", "x": 350, "y": 370, "text": "a", "size": 24 }
```
- `size` is optional, defaults to 20.

### `formula`
Math — LaTeX string rendered with KaTeX.
```json
{ "type": "draw", "shape": "formula", "x": 250, "y": 420, "latex": "c^2 = a^2 + b^2" }
```
- `(x, y)` is the top-left of the rendered box.

## Optional style fields (any draw step)

| Field | Default | Description |
|---|---|---|
| `color` | `#000` | Stroke / text color |
| `width` | `2` | Stroke width in px |

Example:
```json
{ "type": "draw", "shape": "line", "x1": 0, "y1": 0, "x2": 100, "y2": 100, "color": "#d33", "width": 3 }
```

## Coordinate guidelines (for the Claude prompt)

- Canvas is 800×500
- Place the main figure in the center region (x: 200–600, y: 100–400)
- Formulas usually go at the bottom (y: 400+) or to the right (x: 500+)
- Labels offset 10–20 px from the corresponding edge/vertex
