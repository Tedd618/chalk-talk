"""Dataset for flat interleaved OCT word+stroke sequences (v2 — scaled delta-xy).

Loads training.jsonl files (word-anchored format from align.py) and
flattens them into interleaved sequences at load time:

    [W:"today"] [W:"we"] [S:dx,dy,MID] [S:dx,dy,UP] [W:"a"] ...

Stroke coordinates are stored as **scaled deltas** (offsets from the
previous stroke point, multiplied by DELTA_SCALE). Raw deltas have
overall std ~0.07; scaling by 20 makes std ~1.25 — right at the MDN's
initial sigma of 1.

Each position in the flat sequence has:
    token_type: 0=WORD, 1=STROKE
    word_id:    vocab index (0 for stroke positions)
    xy:         scaled (dx, dy) delta coords (0,0 for word positions)
    pen:        pen state MID=0 / UP=1 (0 for word positions)

Vocabulary
----------
Built from all word tokens in the training files.

Special tokens:
  <pad>        index 0   — padding
  <unk>        index 1   — unknown / rare word
  <bos>        index 2   — beginning of sequence
  <eos>        index 3   — end of sequence
  <silent>     index 4   — draws without speaking
  <page_break> index 5   — board cleared between problems
"""
from __future__ import annotations
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

# ── token types ──────────────────────────────────────────────────────────────

WORD_TYPE   = 0
STROKE_TYPE = 1

# ── pen states (no P_STOP — type transition handles it) ──────────────────────

PEN_MID = 0
PEN_UP  = 1

# ── vocabulary ───────────────────────────────────────────────────────────────

PAD       = "<pad>"
UNK       = "<unk>"
BOS       = "<bos>"
EOS       = "<eos>"
SILENT    = "<silent>"
PGBREAK   = "<page_break>"
SPECIALS  = [PAD, UNK, BOS, EOS, SILENT, PGBREAK]

# ── stroke encoding caps ─────────────────────────────────────────────────────

MAX_STROKES_PER_WORD = 8       # max strokes kept per word
MAX_POINTS_PER_WORD  = 150     # max total points per word

# ── delta scaling ────────────────────────────────────────────────────────────

DELTA_SCALE = 20.0  # raw deltas overall std ~0.07 → scaled std ~1.25


# ─── vocabulary helpers ──────────────────────────────────────────────────────

def build_vocab(training_dir: Path, min_freq: int = 2) -> dict[str, int]:
    """Count word frequencies and build index mapping."""
    counter: Counter = Counter()
    for path in sorted(training_dir.glob("*.training.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            tok = json.loads(line)
            if tok["type"] == "word":
                counter[tok["word"]] += 1

    vocab: dict[str, int] = {s: i for i, s in enumerate(SPECIALS)}
    for word, freq in counter.most_common():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def save_vocab(vocab: dict[str, int], path: Path) -> None:
    path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2))


def load_vocab(path: Path) -> dict[str, int]:
    return json.loads(path.read_text())


# ─── augmentation ────────────────────────────────────────────────────────────

def augment_delta_xy(
    dxy: torch.Tensor,
    scale_lo: float = 0.85,
    scale_hi: float = 1.15,
    jitter:   float = 0.1,
) -> torch.Tensor:
    """Scale + jitter SCALED delta stroke coordinates. No clamping.

    jitter=0.1 on scaled deltas ≈ 0.005 on raw coords (~30% of delta std).
    """
    if dxy.size(0) == 0:
        return dxy
    scale = random.uniform(scale_lo, scale_hi)
    return dxy * scale + torch.randn_like(dxy) * jitter


# ─── absolute to scaled delta conversion ──────────────────────────────────────

def abs_to_delta_scaled(
    types: list[int],
    xys:   list[list[float]],
    scale: float = DELTA_SCALE,
) -> list[list[float]]:
    """Convert absolute xy to SCALED delta-xy for stroke tokens.

    For each STROKE token, delta = (current_abs - last_stroke_abs) * scale.
    For WORD tokens, delta = (0, 0).
    last_stroke_abs starts at (0, 0) and is NOT reset at word boundaries,
    so jumps between stroke groups are captured as larger deltas.
    """
    delta_xys: list[list[float]] = []
    last_x, last_y = 0.0, 0.0
    for t, xy in zip(types, xys):
        if t == STROKE_TYPE:
            dx = (xy[0] - last_x) * scale
            dy = (xy[1] - last_y) * scale
            delta_xys.append([dx, dy])
            last_x, last_y = xy[0], xy[1]
        else:
            delta_xys.append([0.0, 0.0])
    return delta_xys


# ─── flatten page ────────────────────────────────────────────────────────────

def flatten_page(
    tokens: list[dict],
    vocab:  dict[str, int],
    max_points_per_word: int = MAX_POINTS_PER_WORD,
    max_strokes_per_word: int = MAX_STROKES_PER_WORD,
) -> dict:
    """Convert a list of word-anchored tokens into flat interleaved arrays.

    Returns dict with parallel lists. XY values are converted to
    SCALED delta-xy (multiplied by DELTA_SCALE).
    """
    unk_id    = vocab[UNK]
    silent_id = vocab[SILENT]

    token_types: list[int]         = []
    word_ids:    list[int]         = []
    abs_xys:     list[list[float]] = []
    pens:        list[int]         = []

    for tok in tokens:
        t = tok["type"]

        if t == "word":
            wid = vocab.get(tok["word"], unk_id)
            token_types.append(WORD_TYPE)
            word_ids.append(wid)
            abs_xys.append([0.0, 0.0])
            pens.append(0)

        elif t == "silent":
            token_types.append(WORD_TYPE)
            word_ids.append(silent_id)
            abs_xys.append([0.0, 0.0])
            pens.append(0)

        else:
            continue  # skip lesson_start, page_break, end

        # Emit STROKE tokens for this word's strokes
        strokes = tok.get("strokes", [])
        n_pts = 0
        for stroke in strokes[:max_strokes_per_word]:
            for i, pt in enumerate(stroke):
                if n_pts >= max_points_per_word:
                    break
                x, y = float(pt[0]), float(pt[1])
                pen = PEN_UP if (i == len(stroke) - 1) else PEN_MID

                token_types.append(STROKE_TYPE)
                word_ids.append(0)
                abs_xys.append([x, y])
                pens.append(pen)
                n_pts += 1
            if n_pts >= max_points_per_word:
                break

    # Convert absolute xy to SCALED delta-xy
    scaled_delta_xys = abs_to_delta_scaled(token_types, abs_xys)

    return {
        "token_types": token_types,
        "word_ids":    word_ids,
        "xys":         scaled_delta_xys,
        "pens":        pens,
    }


# ─── dataset ─────────────────────────────────────────────────────────────────

class FlatOCTDataset(Dataset):
    """One episode = one page from one video, flattened into interleaved
    WORD+STROKE sequence with scaled delta-xy coordinates.

    Returns a window of seq_len tokens from each page.
    BOS is prepended when the window starts at position 0.
    EOS is appended when the window ends at the last position.

    Window sampling (v3): pages average ~9,300 tokens, so a uniformly
    random window starts at position 0 with probability ~0.01% — the
    model would never see page starts, yet generation always seeds from
    [BOS + intro words]. Now start_frac of windows begin at position 0
    and end_frac end at the last token; the rest are uniform.
    """

    def __init__(
        self,
        training_dir: Path,
        vocab:        dict[str, int],
        seq_len:      int   = 512,
        augment:      bool  = False,
        unk_rate:     float = 0.05,
        min_tokens:   int   = 20,
        start_frac:   float = 0.2,
        end_frac:     float = 0.1,
    ):
        self.vocab    = vocab
        self.seq_len  = seq_len
        self.augment  = augment
        self.unk_rate = unk_rate
        self.min_tokens = min_tokens
        self.start_frac = start_frac
        self.end_frac   = end_frac

        self.pad_id    = vocab[PAD]
        self.unk_id    = vocab[UNK]
        self.bos_id    = vocab[BOS]
        self.eos_id    = vocab[EOS]
        self.silent_id = vocab[SILENT]
        self.pgbrk_id  = vocab[PGBREAK]

        self.pages: list[dict] = []
        self._load(training_dir)

    def _load(self, training_dir: Path) -> None:
        paths = sorted(training_dir.glob("*.training.jsonl"))
        for path in paths:
            self._load_file(path)
        total_tokens = sum(len(p["token_types"]) for p in self.pages)
        print(
            f"[dataset] {len(self.pages)} pages from {len(paths)} videos  "
            f"({total_tokens} total tokens, augment={self.augment})"
        )

    def _load_file(self, path: Path) -> None:
        lines = [
            json.loads(l)
            for l in path.read_text().splitlines()
            if l.strip()
        ]
        current: list[dict] = []
        meta: dict = {}

        for tok in lines:
            if tok["type"] == "lesson_start":
                current = []
                meta = tok
            elif tok["type"] in ("page_break", "end"):
                if current:
                    self._process_page(current, meta)
                current = []
            else:
                current.append(tok)

        if current:
            self._process_page(current, meta)

    def _process_page(self, tokens: list[dict], meta: dict) -> None:
        flat = flatten_page(tokens, self.vocab)

        if len(flat["token_types"]) < self.min_tokens:
            return

        flat["topic"] = meta.get("topic", "")
        flat["tag"]   = meta.get("tag", "")
        self.pages.append(flat)

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, idx: int) -> dict:
        page = self.pages[idx]
        total_len = len(page["token_types"])

        # Window selection
        effective_len = self.seq_len - 2  # room for BOS + EOS

        if total_len <= effective_len:
            start, end = 0, total_len
            add_bos, add_eos = True, True
        else:
            r = random.random()
            if r < self.start_frac:
                start = 0                                  # page start
            elif r < self.start_frac + self.end_frac:
                start = total_len - effective_len          # page end
            else:
                start = random.randint(0, total_len - effective_len)
            end   = start + effective_len
            add_bos = (start == 0)
            add_eos = (end == total_len)

        # Extract window (already in scaled delta-xy form)
        types = list(page["token_types"][start:end])
        wids  = list(page["word_ids"][start:end])
        xys_raw = [list(xy) for xy in page["xys"][start:end]]
        ps    = list(page["pens"][start:end])

        # Prepend BOS
        if add_bos:
            types.insert(0, WORD_TYPE)
            wids.insert(0, self.bos_id)
            xys_raw.insert(0, [0.0, 0.0])
            ps.insert(0, 0)

        # Append EOS
        if add_eos:
            types.append(WORD_TYPE)
            wids.append(self.eos_id)
            xys_raw.append([0.0, 0.0])
            ps.append(0)

        # Convert to tensors
        token_types = torch.tensor(types, dtype=torch.long)
        word_ids    = torch.tensor(wids, dtype=torch.long)
        xy          = torch.tensor(xys_raw, dtype=torch.float32)
        pen         = torch.tensor(ps, dtype=torch.long)

        # Augmentation
        if self.augment:
            # UNK masking on word tokens (skip specials)
            specials = {self.pad_id, self.bos_id, self.eos_id,
                        self.silent_id, self.pgbrk_id}
            for i in range(len(word_ids)):
                if token_types[i] == WORD_TYPE and word_ids[i].item() not in specials:
                    if random.random() < self.unk_rate:
                        word_ids[i] = self.unk_id

            # Scaled delta-xy augmentation (scale + jitter, no clamping)
            stroke_mask = (token_types == STROKE_TYPE)
            if stroke_mask.any():
                xy[stroke_mask] = augment_delta_xy(xy[stroke_mask])

        return {
            "token_types": token_types,
            "word_ids":    word_ids,
            "xy":          xy,
            "pen":         pen,
        }


# ─── collation ───────────────────────────────────────────────────────────────

def flat_collate_fn(batch: list[dict], pad_id: int = 0) -> dict:
    """Collate flat interleaved sequences into padded tensors."""
    B = len(batch)
    T_max = max(b["token_types"].size(0) for b in batch)

    token_types = torch.zeros(B, T_max, dtype=torch.long)
    word_ids    = torch.full((B, T_max), pad_id, dtype=torch.long)
    xy          = torch.zeros(B, T_max, 2)
    pen         = torch.zeros(B, T_max, dtype=torch.long)
    pad_mask    = torch.ones(B, T_max, dtype=torch.bool)   # True = pad

    for i, b in enumerate(batch):
        L = b["token_types"].size(0)
        token_types[i, :L] = b["token_types"]
        word_ids[i, :L]    = b["word_ids"]
        xy[i, :L]          = b["xy"]
        pen[i, :L]         = b["pen"]
        pad_mask[i, :L]    = False

    return {
        "token_types": token_types,
        "word_ids":    word_ids,
        "xy":          xy,
        "pen":         pen,
        "pad_mask":    pad_mask,
    }
