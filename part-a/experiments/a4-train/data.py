"""Dataset and vocabulary for OCT word-anchored training sequences.

Loads training.jsonl files produced by align.py and provides
(word_sequence, stroke_contexts) pairs for the three training phases.

Vocabulary
----------
Built from all word tokens in the training files.  Words appearing
fewer than min_freq times are mapped to <unk>.

Special tokens:
  <pad>        index 0   — padding (also used as padding_idx in embedding)
  <unk>        index 1   — unknown / rare word
  <bos>        index 2   — beginning of sequence
  <eos>        index 3   — end of sequence
  <silent>     index 4   — draws without speaking
  <page_break> index 5   — board cleared between problems

Stroke encoding
---------------
Each word's strokes are flattened into one (x,y,p) sequence:
  p = 0  mid-stroke point
  p = 1  last point of a stroke (pen lifts; next point begins new stroke)
  p = 2  stop token (no more strokes for this word)

The sequence always ends with the stop token.  Cap per word:
  MAX_STROKES_PER_WORD = 8   strokes kept per word
  MAX_STROKE_SEQ       = 150 total points (including stop)
"""
from __future__ import annotations
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

# ── vocabulary ────────────────────────────────────────────────────────────────

PAD       = "<pad>"
UNK       = "<unk>"
BOS       = "<bos>"
EOS       = "<eos>"
SILENT    = "<silent>"
PGBREAK   = "<page_break>"
SPECIALS  = [PAD, UNK, BOS, EOS, SILENT, PGBREAK]

# ── stroke encoding caps ──────────────────────────────────────────────────────

MAX_STROKES_PER_WORD = 8
MAX_STROKE_SEQ       = 150   # total points per word incl. stop token

P_MID  = 0
P_UP   = 1
P_STOP = 2


# ─── vocabulary helpers ───────────────────────────────────────────────────────

def build_vocab(training_dir: Path, min_freq: int = 2) -> dict[str, int]:
    """Count word frequencies and build index mapping.

    Only tokens with type == 'word' are counted; special tokens are
    prepended at fixed indices regardless of frequency.
    """
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


# ─── stroke encoding ──────────────────────────────────────────────────────────

def encode_strokes(
    strokes: list,
    max_total: int = MAX_STROKE_SEQ,
    max_per_word: int = MAX_STROKES_PER_WORD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a list of strokes into (xy, p) tensors.

    Adds the stop token at the end.  Caps total length and stroke count.

    Args:
        strokes: list of stroke sequences, each [[x,y,p], ...]
                 (p in data = 0 mid, 1 last-in-stroke)

    Returns:
        xy: FloatTensor(S, 2)  — normalized coordinates
        p:  LongTensor(S)      — pen states (P_MID / P_UP / P_STOP)
    """
    xy_list: list[list[float]] = []
    p_list:  list[int]         = []

    for stroke in strokes[:max_per_word]:
        for i, pt in enumerate(stroke):
            x, y = float(pt[0]), float(pt[1])
            # last point in a stroke → P_UP; others → P_MID
            pen = P_UP if (i == len(stroke) - 1) else P_MID
            xy_list.append([x, y])
            p_list.append(pen)
            if len(xy_list) >= max_total - 1:
                break
        if len(xy_list) >= max_total - 1:
            break

    if not xy_list:
        return torch.zeros(0, 2), torch.zeros(0, dtype=torch.long)

    # Append stop token
    xy_list.append([0.0, 0.0])
    p_list.append(P_STOP)

    return (
        torch.tensor(xy_list, dtype=torch.float32),
        torch.tensor(p_list,  dtype=torch.long),
    )


# ─── augmentation ─────────────────────────────────────────────────────────────

def augment_xy(
    xy: torch.Tensor,
    scale_lo: float = 0.85,
    scale_hi: float = 1.15,
    jitter:   float = 0.008,
) -> torch.Tensor:
    """Scale + jitter stroke coordinates; clamp to [0,1].

    Conservative ranges based on research findings for OCT-style writing.
    """
    if xy.size(0) == 0:
        return xy
    scale = random.uniform(scale_lo, scale_hi)
    xy    = xy * scale + torch.randn_like(xy) * jitter
    return xy.clamp(0.0, 1.0)


def unk_mask(word_id: int, unk_id: int, rate: float) -> int:
    """Randomly replace a word token with <unk> (token-level masking)."""
    return unk_id if random.random() < rate else word_id


# ─── dataset ──────────────────────────────────────────────────────────────────

class OCTDataset(Dataset):
    """One episode = one page from one video.

    Each episode is a dict:
        word_ids:       list[int]            word token indices
        stroke_pos:     list[int]            word positions that have strokes
        stroke_xys:     list[Tensor(S,2)]    stroke coord sequences
        stroke_ps:      list[Tensor(S)]      stroke pen-state sequences
        topic:          str
        tag:            str
    """

    def __init__(
        self,
        training_dir: Path,
        vocab:        dict[str, int],
        augment:      bool  = False,
        max_seq_len:  int   = 2048,
        unk_rate:     float = 0.05,
        min_words:    int   = 5,
    ):
        self.vocab       = vocab
        self.augment     = augment
        self.max_seq_len = max_seq_len
        self.unk_rate    = unk_rate
        self.min_words   = min_words

        self.pad_id    = vocab[PAD]
        self.unk_id    = vocab[UNK]
        self.bos_id    = vocab[BOS]
        self.eos_id    = vocab[EOS]
        self.silent_id = vocab[SILENT]
        self.pgbrk_id  = vocab[PGBREAK]

        self.episodes: list[dict] = []
        self._load(training_dir)

    # ── loading ───────────────────────────────────────────────────────────────

    def _load(self, training_dir: Path) -> None:
        paths = sorted(training_dir.glob("*.training.jsonl"))
        for path in paths:
            self._load_file(path)
        print(
            f"[dataset] {len(self.episodes)} episodes "
            f"from {len(paths)} videos  "
            f"(augment={self.augment})"
        )

    def _load_file(self, path: Path) -> None:
        lines = [
            json.loads(l)
            for l in path.read_text().splitlines()
            if l.strip()
        ]
        current: list[dict] = []
        for tok in lines:
            if tok["type"] == "lesson_start":
                current = [tok]
            elif tok["type"] in ("page_break", "end"):
                if current:
                    self._process_page(current)
                    current = []
            else:
                if current:
                    current.append(tok)
        if current:
            self._process_page(current)

    def _process_page(self, tokens: list[dict]) -> None:
        meta   = tokens[0] if tokens[0]["type"] == "lesson_start" else {}
        topic  = meta.get("topic", "")
        tag    = meta.get("tag",   "")
        body   = tokens[1:] if tokens[0]["type"] == "lesson_start" else tokens

        if not body:
            return

        word_ids:   list[int]           = [self.bos_id]
        stroke_pos: list[int]           = []
        stroke_xys: list[torch.Tensor]  = []
        stroke_ps:  list[torch.Tensor]  = []

        for tok in body:
            if len(word_ids) >= self.max_seq_len - 1:
                break

            t = tok["type"]

            if t == "word":
                wid = self.vocab.get(tok["word"], self.unk_id)
                pos = len(word_ids)
                word_ids.append(wid)
            elif t == "silent":
                pos = len(word_ids)
                word_ids.append(self.silent_id)
            else:
                continue   # page_break / end handled by caller; skip others

            strokes = tok.get("strokes", [])
            if strokes:
                xy, p = encode_strokes(strokes)
                if xy.size(0) > 1:   # at least one real point + stop
                    stroke_pos.append(pos)
                    stroke_xys.append(xy)
                    stroke_ps.append(p)

        word_ids.append(self.eos_id)

        if len(word_ids) - 2 < self.min_words:   # too short (excl. bos/eos)
            return

        self.episodes.append({
            "word_ids":   word_ids,
            "stroke_pos": stroke_pos,
            "stroke_xys": stroke_xys,
            "stroke_ps":  stroke_ps,
            "topic":      topic,
            "tag":        tag,
        })

    # ── dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        ep = self.episodes[idx]

        # Apply token-level UNK masking to word sequence
        if self.augment:
            wids = [
                unk_mask(w, self.unk_id, self.unk_rate)
                if w not in (self.bos_id, self.eos_id,
                             self.silent_id, self.pgbrk_id, self.pad_id)
                else w
                for w in ep["word_ids"]
            ]
        else:
            wids = list(ep["word_ids"])

        # Apply stroke augmentation
        xys = [xy.clone() for xy in ep["stroke_xys"]]
        if self.augment:
            xys = [augment_xy(xy) for xy in xys]

        return {
            "word_ids":   wids,
            "stroke_pos": list(ep["stroke_pos"]),
            "stroke_xys": xys,
            "stroke_ps":  [p.clone() for p in ep["stroke_ps"]],
            "topic":      ep["topic"],
            "tag":        ep["tag"],
        }


# ─── collation ────────────────────────────────────────────────────────────────

MAX_STROKE_CTX = 256   # max stroke contexts per batch (randomly subsampled)


def collate_fn(batch: list[dict], pad_id: int = 0) -> dict:
    """Collate a list of episodes into padded tensors.

    Word sequences are padded to the longest in the batch.
    Stroke sequences are collected across all batch items and padded
    to the longest stroke sequence in the batch.

    Returns a dict with:
        word_ids:      (B, T_max)  long
        word_pad_mask: (B, T_max)  bool  True = pad
        strokes: dict or None
            batch_idx: (N,)         long  — which batch item
            word_pos:  (N,)         long  — word position within that item
            xy:        (N, S_max, 2) float
            p:         (N, S_max)    long
            pad_mask:  (N, S_max)    bool  True = pad
    where N = total words-with-strokes across the batch.
    """
    B = len(batch)

    # ── word sequences ────────────────────────────────────────────────────────
    T_max = max(len(b["word_ids"]) for b in batch)
    word_ids      = torch.full((B, T_max), pad_id, dtype=torch.long)
    word_pad_mask = torch.ones(B, T_max, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = len(b["word_ids"])
        word_ids[i, :L]      = torch.tensor(b["word_ids"], dtype=torch.long)
        word_pad_mask[i, :L] = False

    # ── stroke contexts ───────────────────────────────────────────────────────
    all_bidx, all_wpos, all_xy, all_p = [], [], [], []
    for i, b in enumerate(batch):
        for pos, xy, p in zip(b["stroke_pos"], b["stroke_xys"], b["stroke_ps"]):
            all_bidx.append(i)
            all_wpos.append(pos)
            all_xy.append(xy)
            all_p.append(p)

    # Randomly subsample stroke contexts if too many
    if len(all_xy) > MAX_STROKE_CTX:
        idx = random.sample(range(len(all_xy)), MAX_STROKE_CTX)
        all_bidx = [all_bidx[i] for i in idx]
        all_wpos = [all_wpos[i] for i in idx]
        all_xy   = [all_xy[i]   for i in idx]
        all_p    = [all_p[i]    for i in idx]

    strokes = None
    if all_xy:
        S_max    = max(xy.size(0) for xy in all_xy)
        N        = len(all_xy)
        s_xy     = torch.zeros(N, S_max, 2)
        s_p      = torch.zeros(N, S_max, dtype=torch.long)
        s_pad    = torch.ones(N, S_max, dtype=torch.bool)

        for i, (xy, p) in enumerate(zip(all_xy, all_p)):
            S            = xy.size(0)
            s_xy[i, :S]  = xy
            s_p[i,  :S]  = p
            s_pad[i, :S] = False

        strokes = {
            "batch_idx": torch.tensor(all_bidx, dtype=torch.long),
            "word_pos":  torch.tensor(all_wpos,  dtype=torch.long),
            "xy":        s_xy,
            "p":         s_p,
            "pad_mask":  s_pad,
        }

    return {
        "word_ids":      word_ids,
        "word_pad_mask": word_pad_mask,
        "strokes":       strokes,
    }
