"""Builds context_ablation_colab.ipynb from raw-string cells and a byte-identical
smoke_test.py so the shipped code is exactly what was tested locally."""
import json, sys, ast
from pathlib import Path

OUT_NB = Path(sys.argv[1])
OUT_SMOKE = Path(sys.argv[2])

# ═══════════════════════════════════════════════════════════════════════════
CELL_HEADER = r"""# Method 1 — Cross-Modal Context Ablation (v2, length-controlled)

**Question (PLAN.md):** can handwriting strokes and spoken words act as context for each other?

**v2 fixes two issues found by inspecting the v1 run's results (see design doc):**
1. **Length confound.** v1 compared `drop_strokes` (tokens physically removed, window
   23.5% as long) against `mask_strokes` (tokens kept, features zeroed, full length) and
   called the difference "value of stroke timing." That conflates two effects: knowing an
   event happened, and the fact that removing tokens changes word-to-word attention
   distance. v2 adds `nostroke_hidden` / `noword_hidden` — same length as `full`, but the
   other modality's positions are masked out of attention entirely (indistinguishable from
   ordinary batch padding) — splitting the confound into three orthogonal pieces:
   **content** (real values vs zeroed), **marker** (visible-empty vs hidden, length fixed),
   **length** (hidden vs dropped, info fixed at zero).
2. **Statistical rigor.** v1's "|Δ|≥0.05 nats is real" was a token-count heuristic assuming
   independence; tokens within one lecture are correlated. v2 bootstraps over the 363
   validation *windows* instead of over tokens, and reports a 95% CI per delta.

Seven conditions, all transforms of the *same* sampled windows:

| condition | words | strokes | read |
|---|---|---|---|
| `full` | real | real | word CE, stroke NLL |
| `mask_strokes` | real | kept, features zeroed | word CE |
| `nostroke_hidden` | real | kept, features zeroed AND hidden from attention | word CE |
| `drop_strokes` | real | removed (shorter window) | word CE |
| `mask_words` | `<mask>` | real | stroke NLL |
| `noword_hidden` | `<mask>`, hidden from attention | real | stroke NLL |
| `drop_words` | removed (shorter window) | real | stroke NLL |

Design doc: `part-a/docs/METHOD1_CONTEXT_EXPERIMENT.md`

**Before running:**
1. Runtime → Change runtime type → GPU (T4)
2. Upload `output.zip` (also in your local `~/Downloads/`) to **exactly**
   `My Drive/chalk-talk/a4-train/output.zip` — directly inside `a4-train/`,
   *not* inside an `output/` or other subfolder. Cell 1 unzips it, checks the
   result, and self-heals if it ends up nested one level too deep either way.
3. Run all cells top to bottom.

**Seeds.** Two runs of the same condition with the same seed differed by 0.136 nats in
word CE (fp16 attention kernels are nondeterministic) — bigger than most of the deltas.
So every condition is trained once per seed in `SEEDS` (default 3) and each delta is
reported per seed plus mean ± std across seeds; a delta is only called ROBUST if its
window-bootstrap CI excludes zero in every seed *and* all seeds agree in sign.

Runtime: ≈ 40 min per seed (7 conditions, 2 short) → ≈ 2 h for 3 seeds on a T4.
Every (condition, seed) run is saved on completion, so a disconnect resumes where it
stopped; the three seed-42 conditions finished in the first v2 attempt are reused.
Set `SEEDS = [42]` in cell 4 for the quick single-seed version (≈ 25 min).
Results go to `ctx_ablation_v2/` — the v1 `ctx_ablation/` run is untouched.
"""

# ═══════════════════════════════════════════════════════════════════════════
CELL_SETUP = r"""# ── 1. Drive + data + GPU ────────────────────────────────────────────────────
# Expected layout on Drive (upload output.zip here, not inside a subfolder):
#   My Drive/chalk-talk/a4-train/output.zip
# This cell extracts it to My Drive/chalk-talk/a4-train/output/*.training.jsonl
# and self-heals if the zip's internal layout differs from what we expect.
from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/MyDrive/chalk-talk/a4-train'
DATA_DIR   = f'{DRIVE_ROOT}/output'
RUN_DIR    = f'{DRIVE_ROOT}/ctx_ablation_v2'   # v2: length-controlled -- v1's ctx_ablation/ untouched
ZIP_PATH   = f'{DRIVE_ROOT}/output.zip'

import os, zipfile, shutil
os.makedirs(RUN_DIR, exist_ok=True)
print(f'Expecting zip at:  {ZIP_PATH}')
print(f'Expecting data at: {DATA_DIR}/*.training.jsonl')


def _extract_output_zip(zip_path, drive_root, data_dir):
    # Extract regardless of whether entries are 'output/<file>' or flat
    # '<file>' -- inspect the zip instead of assuming one layout.
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = [n for n in zf.namelist() if n.endswith('.training.jsonl')]
        if not names:
            raise RuntimeError(f'{zip_path} contains no .training.jsonl files')
        prefixes = {n.split('/', 1)[0] for n in names if '/' in n}
        if len(prefixes) == 1 and all(n.startswith(next(iter(prefixes)) + '/') for n in names):
            prefix = next(iter(prefixes))
            print(f"  zip entries prefixed with '{prefix}/' -> extracting to {drive_root}")
            zf.extractall(drive_root)
        else:
            print(f'  zip entries are flat -> extracting to {data_dir}')
            os.makedirs(data_dir, exist_ok=True)
            zf.extractall(data_dir)


def _heal_nesting(data_dir):
    # If files ended up one level too deep (e.g. someone extracted the zip
    # INTO data_dir instead of its parent), find them in whichever subfolder
    # and move them up. Not hardcoded to a folder literally named 'output'.
    if not os.path.isdir(data_dir):
        return []
    direct = [f for f in os.listdir(data_dir) if f.endswith('.training.jsonl')]
    if len(direct) >= 10:
        return direct
    for sub in os.listdir(data_dir):
        subpath = os.path.join(data_dir, sub)
        if os.path.isdir(subpath):
            nested = [f for f in os.listdir(subpath) if f.endswith('.training.jsonl')]
            if len(nested) > 10:
                print(f'  found {len(nested)} files nested in {sub}/, moving up...')
                for f in nested:
                    shutil.move(os.path.join(subpath, f), os.path.join(data_dir, f))
                if not os.listdir(subpath):
                    os.rmdir(subpath)
                break
    return [f for f in os.listdir(data_dir) if f.endswith('.training.jsonl')]


if os.path.exists(ZIP_PATH) and not os.path.isdir(DATA_DIR):
    print('Unzipping output.zip ...')
    _extract_output_zip(ZIP_PATH, DRIVE_ROOT, DATA_DIR)
    print('Done.')

files = _heal_nesting(DATA_DIR)
print(f'Found {len(files)} training files in {DATA_DIR}')
assert len(files) > 10, (
    f'No training files found.\n'
    f'  Checked for files at: {DATA_DIR}/*.training.jsonl\n'
    f'  Checked for zip at:   {ZIP_PATH}\n'
    f'Upload output.zip directly into the a4-train/ folder on Drive '
    f'(not inside output/ or any other subfolder), then re-run this cell.'
)

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)
if device == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0),
          f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"""

# ═══════════════════════════════════════════════════════════════════════════
CELL_DATA = r"""# ── 2. Data: vocab, flattening, windowed dataset, conditions ─────────────────
import json, math, random, re, time, glob
from collections import Counter
from pathlib import Path
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

WORD_TYPE, STROKE_TYPE = 0, 1
PEN_MID, PEN_UP = 0, 1
DELTA_SCALE = 20.0          # raw delta std ~0.07 → scaled ~1.25 ≈ MDN initial sigma

PAD, UNK, BOS, EOS, SILENT, MASK = '<pad>', '<unk>', '<bos>', '<eos>', '<silent>', '<mask>'
SPECIALS = [PAD, UNK, BOS, EOS, SILENT, MASK]
# v2: added nostroke_hidden / noword_hidden to decompose the confounded
# drop_* vs mask_* comparison into three orthogonal pieces (see CELL_RESULTS):
#   content  = mask_* vs full            (same length, only feature values differ)
#   marker   = nostroke/noword_hidden vs mask_*   (same length, event-visibility differs)
#   length   = drop_* vs nostroke/noword_hidden   (same zero-info content, length differs)
CONDITIONS = ['full', 'mask_strokes', 'nostroke_hidden', 'drop_strokes',
             'mask_words', 'noword_hidden', 'drop_words']

_punct = re.compile(r"^[^\w']+|[^\w']+$")

def norm_word(w):
    # Whisper emits 'video.' 'video,' 'Video' as distinct tokens; collapse them.
    return _punct.sub('', w.lower().strip())


def build_vocab(data_dir, min_freq=2):
    counter = Counter()
    for path in sorted(Path(data_dir).glob('*.training.jsonl')):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            tok = json.loads(line)
            if tok['type'] == 'word':
                w = norm_word(tok['word'])
                if w:
                    counter[w] += 1
    vocab = {s: i for i, s in enumerate(SPECIALS)}
    for w, f in counter.most_common():
        if f >= min_freq and w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def flatten_page(tokens, vocab):
    # No per-token stroke caps: the flat model doesn't need them and they only
    # discard 0.3% of points anyway.
    unk, sil = vocab[UNK], vocab[SILENT]
    types, wids, xs, ys, pens = [], [], [], [], []
    for tok in tokens:
        t = tok['type']
        if t == 'word':
            w = norm_word(tok['word'])
            types.append(WORD_TYPE); wids.append(vocab.get(w, unk) if w else unk)
        elif t == 'silent':
            types.append(WORD_TYPE); wids.append(sil)
        else:
            continue
        xs.append(0.0); ys.append(0.0); pens.append(0)
        for stroke in tok.get('strokes', []):
            n = len(stroke)
            for i, pt in enumerate(stroke):
                types.append(STROKE_TYPE); wids.append(0)
                xs.append(float(pt[0])); ys.append(float(pt[1]))
                pens.append(PEN_UP if i == n - 1 else PEN_MID)
    types = torch.tensor(types, dtype=torch.long)
    wids  = torch.tensor(wids, dtype=torch.long)
    absxy = torch.stack([torch.tensor(xs), torch.tensor(ys)], -1).float()
    pens  = torch.tensor(pens, dtype=torch.long)
    delta = torch.zeros_like(absxy)
    sm = types == STROKE_TYPE
    if sm.any():
        spts = absxy[sm]
        prev = torch.cat([torch.zeros(1, 2), spts[:-1]], 0)
        delta[sm] = (spts - prev) * DELTA_SCALE       # stroke-to-stroke, words skipped
    return {'types': types, 'wids': wids, 'delta': delta, 'abs': absxy, 'pens': pens}


def load_pages(data_dir, vocab, min_tokens=64, limit=None):
    pages = []
    paths = sorted(Path(data_dir).glob('*.training.jsonl'))
    if limit:
        paths = paths[:limit]
    for fi, path in enumerate(paths):
        if fi % 100 == 0:
            print(f'\r  loading {fi+1}/{len(paths)}', end='', flush=True)
        cur, meta = [], {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            tok = json.loads(line)
            if tok['type'] == 'lesson_start':
                cur, meta = [], tok
            elif tok['type'] in ('page_break', 'end'):
                if cur:
                    p = flatten_page(cur, vocab)
                    if p['types'].size(0) >= min_tokens:
                        p['tag'] = meta.get('tag', path.stem); p['topic'] = meta.get('topic', '')
                        pages.append(p)
                cur = []
            else:
                cur.append(tok)
        if cur:
            p = flatten_page(cur, vocab)
            if p['types'].size(0) >= min_tokens:
                p['tag'] = meta.get('tag', path.stem); p['topic'] = meta.get('topic', '')
                pages.append(p)
    n_tok = sum(p['types'].size(0) for p in pages)
    n_w = sum(int((p['types'] == WORD_TYPE).sum()) for p in pages)
    print(f'\r[data] {len(pages)} pages, {n_tok:,} tokens ({n_w:,} word, {n_tok-n_w:,} stroke)')
    return pages


class FlatOCTDataset(Dataset):
    # One item = one window of one page. Windows: start_frac begin at the page
    # start (with BOS), end_frac end at the page end (with EOS), rest uniform.
    # deterministic=True (validation) seeds the window choice by index so every
    # condition is evaluated on identical windows.
    def __init__(self, pages, vocab, seq_len=1024, condition='full',
                 deterministic=False, n_windows=1, start_frac=0.2, end_frac=0.1):
        assert condition in CONDITIONS, condition
        self.pages, self.vocab, self.seq_len = pages, vocab, seq_len
        self.condition, self.deterministic, self.n_windows = condition, deterministic, n_windows
        self.start_frac, self.end_frac = start_frac, end_frac
        self.pad_id, self.bos_id, self.eos_id = vocab[PAD], vocab[BOS], vocab[EOS]
        self.mask_id = vocab[MASK]
        self.special_ids = torch.tensor([vocab[s] for s in SPECIALS])

    def __len__(self):
        return len(self.pages) * self.n_windows

    def __getitem__(self, idx):
        page = self.pages[idx // self.n_windows]
        rng = random.Random(idx * 7919 + 17) if self.deterministic else random
        for _attempt in range(20):
            item = self._window(page, rng)
            # Training: an empty window (drop_strokes on an all-stroke span)
            # would be a fully-masked batch row -> NaN risk in backward, and a
            # wasted sample. Resample. Validation windows are never resampled
            # so index i is the same underlying span in every condition; the
            # metric code skips a window with no valid targets.
            if item['types'].numel() > 0 or self.deterministic:
                return item
        # Deterministic fallback: a window at the page start gets BOS
        # prepended, and BOS is kept by every condition -> never empty.
        return self._window(page, rng, start=0)

    def _window(self, page, rng, start=None):
        total, eff = page['types'].size(0), self.seq_len - 2
        if total <= eff:
            start, end = 0, total
        else:
            if start is None:
                r = rng.random()
                if r < self.start_frac:
                    start = 0
                elif r < self.start_frac + self.end_frac:
                    start = total - eff
                else:
                    start = rng.randint(0, total - eff)
            end = start + eff
        tt = page['types'][start:end]; wi = page['wids'][start:end]
        de = page['delta'][start:end]; ab = page['abs'][start:end]; pn = page['pens'][start:end]
        if start == 0:
            tt = torch.cat([torch.tensor([WORD_TYPE]), tt]); wi = torch.cat([torch.tensor([self.bos_id]), wi])
            de = torch.cat([torch.zeros(1, 2), de]);        ab = torch.cat([torch.zeros(1, 2), ab])
            pn = torch.cat([torch.tensor([0]), pn])
        if end == total:
            tt = torch.cat([tt, torch.tensor([WORD_TYPE])]); wi = torch.cat([wi, torch.tensor([self.eos_id])])
            de = torch.cat([de, torch.zeros(1, 2)]);        ab = torch.cat([ab, torch.zeros(1, 2)])
            pn = torch.cat([pn, torch.tensor([0])])
        return self._apply_condition(tt.clone(), wi.clone(), de.clone(), ab.clone(), pn.clone())

    def _apply_condition(self, tt, wi, de, ab, pn):
        c = self.condition
        if c in ('drop_strokes', 'drop_words'):
            if c == 'drop_strokes':
                keep = tt == WORD_TYPE
            else:   # drop_words: keep strokes and the BOS/EOS sequence markers
                keep = (tt == STROKE_TYPE) | (wi == self.bos_id) | (wi == self.eos_id)
            tt, wi, de, ab, pn = tt[keep], wi[keep], de[keep], ab[keep], pn[keep]
        wi_in, de_in, ab_in, pn_in = wi.clone(), de.clone(), ab.clone(), pn.clone()
        # hide: positions invisible to attention AND to the loss (folded into
        # the pad mask downstream) while staying IN the sequence at their
        # original index -- indistinguishable from ordinary end-of-batch
        # padding. Used by nostroke_hidden/noword_hidden to hold sequence
        # length and word/stroke attention-distance fixed while removing ALL
        # trace of the other modality (not even "an event happened here").
        hide = torch.zeros_like(tt, dtype=torch.bool)
        if c == 'mask_strokes':
            sm = tt == STROKE_TYPE
            de_in[sm] = 0.0; ab_in[sm] = 0.0; pn_in[sm] = 0
        elif c == 'mask_words':
            wm = (tt == WORD_TYPE) & ~torch.isin(wi, self.special_ids)
            wi_in[wm] = self.mask_id
        elif c == 'nostroke_hidden':
            sm = tt == STROKE_TYPE
            de_in[sm] = 0.0; ab_in[sm] = 0.0; pn_in[sm] = 0
            hide[sm] = True
        elif c == 'noword_hidden':
            wm = (tt == WORD_TYPE) & ~torch.isin(wi, self.special_ids)
            wi_in[wm] = self.mask_id
            hide[wm] = True
        # Under causal attention, position 0 is the ONLY causally-valid key
        # for query 0. If position 0 is also hidden, that row has zero valid
        # keys -> softmax over an all -inf row -> NaN, propagating through the
        # whole batch. This isn't rare: ~78% of tokens are strokes, so a
        # mid-page window (no prepended BOS) starts with a stroke ~78% of the
        # time. Position 0 being visible also guarantees every later query
        # has at least one fallback causal key, so this one line fixes the
        # whole sequence, not just position 0. Negligible one-token leak.
        # (numel guard: a drop_strokes window can be EMPTY when the 1,022-token
        # span is one long silent drawing run with no word/<silent> token in
        # it -- happened at epoch 14 of the first v2 run. __getitem__ resamples
        # such windows for training; validation keeps them and the metrics
        # skip them, so the window index stays paired across conditions.)
        if hide.numel():
            hide[0] = False
        return {'types': tt, 'wids_in': wi_in, 'wids': wi, 'delta_in': de_in, 'abs_in': ab_in,
                'pens_in': pn_in, 'delta': de, 'pens': pn, 'hide': hide}


def collate(batch, pad_id=0):
    B = len(batch); T = max(b['types'].size(0) for b in batch)
    out = {
        'types':    torch.zeros(B, T, dtype=torch.long),
        'wids_in':  torch.full((B, T), pad_id, dtype=torch.long),
        'wids':     torch.full((B, T), pad_id, dtype=torch.long),
        'delta_in': torch.zeros(B, T, 2), 'abs_in': torch.zeros(B, T, 2),
        'pens_in':  torch.zeros(B, T, dtype=torch.long),
        'delta':    torch.zeros(B, T, 2), 'pens': torch.zeros(B, T, dtype=torch.long),
        'pad':      torch.ones(B, T, dtype=torch.bool),      # batch padding OR hidden-from-attention
    }
    for i, b in enumerate(batch):
        L = b['types'].size(0)
        for k in ('types', 'wids_in', 'wids', 'delta_in', 'abs_in', 'pens_in', 'delta', 'pens'):
            out[k][i, :L] = b[k]
        out['pad'][i, :L] = b['hide']    # False except where this condition hides a position
    return out

print('data code loaded')
"""

# ═══════════════════════════════════════════════════════════════════════════
CELL_MODEL = r"""# ── 3. Model: flat causal transformer + MDN stroke head ──────────────────────
MDN_K = 20   # bivariate Gaussians (Graves 2013)


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


def mdn_split(params):
    # Always float32: fp16 logsumexp/exp under autocast can NaN.
    params = params.float(); K = MDN_K
    log_pi  = F.log_softmax(params[..., :K], dim=-1)
    mu      = params[..., K:3*K].reshape(*params.shape[:-1], K, 2)
    log_sig = params[..., 3*K:5*K].reshape(*params.shape[:-1], K, 2).clamp(-4.0, 3.0)
    rho     = 0.95 * torch.tanh(params[..., 5*K:6*K])
    return log_pi, mu, log_sig.exp(), rho


def mdn_nll(params, target):
    # target (N,2), params (N,6K) → (N,) negative log-likelihood (nats)
    log_pi, mu, sigma, rho = mdn_split(params)
    t  = target.float().unsqueeze(-2)
    zx = (t[..., 0] - mu[..., 0]) / sigma[..., 0]
    zy = (t[..., 1] - mu[..., 1]) / sigma[..., 1]
    omr = (1 - rho ** 2).clamp(min=1e-6)
    z = zx ** 2 + zy ** 2 - 2 * rho * zx * zy
    log_gauss = (-z / (2 * omr) - torch.log(sigma[..., 0]) - torch.log(sigma[..., 1])
                 - 0.5 * torch.log(omr) - math.log(2 * math.pi))
    return -torch.logsumexp(log_pi + log_gauss, dim=-1)


def mdn_sample(params, pi_temp=1.0, sigma_temp=1.0):
    log_pi, mu, sigma, rho = mdn_split(params.unsqueeze(0))
    log_pi, mu, sigma, rho = log_pi[0], mu[0], sigma[0], rho[0]
    k  = int(torch.multinomial(F.softmax(log_pi / max(pi_temp, 1e-6), dim=-1), 1))
    sx, sy, r = sigma[k, 0] * sigma_temp, sigma[k, 1] * sigma_temp, rho[k]
    z1, z2 = torch.randn(2, device=params.device)
    dx = mu[k, 0] + sx * z1
    dy = mu[k, 1] + sy * (r * z1 + torch.sqrt((1 - r ** 2).clamp(min=1e-6)) * z2)
    return float(dx), float(dy)


class FlatOCTModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=8, d_ff=1024,
                 max_seq_len=2048, dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model, self.vocab_size, self.pad_idx = d_model, vocab_size, pad_idx
        self.word_embed  = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.type_embed  = nn.Embedding(2, d_model)
        self.stroke_proj = nn.Linear(4, d_model)        # (dx, dy, x, y): delta + absolute board position
        self.pen_embed   = nn.Embedding(2, d_model)
        self.pos_enc     = SinusoidalPE(d_model, max_seq_len, dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.norm_out  = nn.LayerNorm(d_model)
        self.type_head = nn.Linear(d_model, 2)
        self.word_head = nn.Linear(d_model, vocab_size, bias=False)
        self.word_head.weight = self.word_embed.weight
        self.xy_head   = nn.Linear(d_model, 6 * MDN_K)
        self.pen_head  = nn.Linear(d_model, 2)
        self.register_buffer('pen_weight', torch.tensor([1.0, 2.0]))
        self._init()

    def _init(self):
        for e in (self.word_embed, self.type_embed, self.pen_embed):
            nn.init.normal_(e.weight, std=0.02)
        for name, p in self.named_parameters():
            if 'embed' in name:
                continue
            if p.dim() > 1 and 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        nn.init.normal_(self.xy_head.weight, std=0.001)   # sigma starts at exp(0)=1 ≈ data std
        nn.init.zeros_(self.xy_head.bias)

    def forward(self, types, wids_in, feat_in, pens_in, pad):
        h = self.type_embed(types)
        wm = (types == WORD_TYPE).unsqueeze(-1).float()
        sm = (types == STROKE_TYPE).unsqueeze(-1).float()
        h = h + self.word_embed(wids_in) * wm
        h = h + (self.stroke_proj(feat_in) + self.pen_embed(pens_in)) * sm
        h = self.pos_enc(h)
        T = types.size(1)
        cmask = torch.triu(torch.ones(T, T, device=types.device, dtype=torch.bool), 1)
        h = self.transformer(h, mask=cmask, src_key_padding_mask=pad, is_causal=True)
        h = self.norm_out(h)
        return {'type': self.type_head(h), 'word': self.word_head(h),
                'xy': self.xy_head(h), 'pen': self.pen_head(h)}

    def loss(self, out, b):
        tgt_t, tgt_w = b['types'][:, 1:], b['wids'][:, 1:]
        tgt_d, tgt_p = b['delta'][:, 1:], b['pens'][:, 1:]
        valid = ~b['pad'][:, 1:]
        wp = valid & (tgt_t == WORD_TYPE); sp = valid & (tgt_t == STROKE_TYPE)
        z = out['type'].new_tensor(0.0)
        l_type = F.cross_entropy(out['type'][:, :-1][valid], tgt_t[valid]) if valid.any() else z
        l_word = F.cross_entropy(out['word'][:, :-1][wp], tgt_w[wp], ignore_index=self.pad_idx,
                                 label_smoothing=0.1) if wp.any() else z
        l_xy   = mdn_nll(out['xy'][:, :-1][sp], tgt_d[sp]).mean() if sp.any() else z
        l_pen  = F.cross_entropy(out['pen'][:, :-1][sp], tgt_p[sp], weight=self.pen_weight) if sp.any() else z
        return {'total': l_type + l_word + l_xy + l_pen, 'type': l_type, 'word': l_word, 'xy': l_xy, 'pen': l_pen}

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def stroke_feat(delta, absxy):
    return torch.cat([delta, absxy], -1)


@torch.no_grad()
def raw_metrics(out, b):
    # Unsmoothed, unweighted per-token sums for the reported metrics.
    tgt_t, tgt_w = b['types'][:, 1:], b['wids'][:, 1:]
    tgt_d, tgt_p = b['delta'][:, 1:], b['pens'][:, 1:]
    valid = ~b['pad'][:, 1:]
    wp = valid & (tgt_t == WORD_TYPE); sp = valid & (tgt_t == STROKE_TYPE)
    m = {'type_sum': F.cross_entropy(out['type'][:, :-1][valid].float(), tgt_t[valid], reduction='sum').item(),
         'type_n': int(valid.sum())}
    if wp.any():
        m['word_sum'] = F.cross_entropy(out['word'][:, :-1][wp].float(), tgt_w[wp], reduction='sum').item()
        m['word_n'] = int(wp.sum())
    if sp.any():
        m['stroke_sum'] = mdn_nll(out['xy'][:, :-1][sp], tgt_d[sp]).sum().item()
        m['stroke_n'] = int(sp.sum())
        m['pen_sum'] = F.cross_entropy(out['pen'][:, :-1][sp].float(), tgt_p[sp], reduction='sum').item()
        m['pen_n'] = int(sp.sum())
    return m

print('model code loaded')
"""

# ═══════════════════════════════════════════════════════════════════════════
CELL_CONFIG = r"""# ── 4. Config, vocab, pages, split ───────────────────────────────────────────
CFG = dict(
    d_model=256, n_layers=4, n_heads=8, d_ff=1024, dropout=0.1,
    seq_len=1024, batch_size=8, lr=3e-4, weight_decay=0.01,
    epochs=20, warmup=200, grad_clip=1.0,
    val_frac=0.1, val_windows=3, min_freq=2, seed=42,
)
# Training seeds. Two runs of the SAME condition with the same seed differed
# by 0.136 nats in word CE (fp16 attention kernels are nondeterministic and a
# 4.6M model at lr 3e-4 amplifies it) -- larger than most of the deltas being
# measured. The window bootstrap only captures evaluation noise, not this
# training-run noise, so every condition is trained once per seed and deltas
# are reported per seed + mean/std across seeds. Seed 42 reuses results from
# an earlier partial run if present. SEEDS = [42] gives the quick single-seed
# version (~25 min if the 3 finished seed-42 conditions are on Drive).
SEEDS = [42, 43, 44]
print('CFG:', CFG)
print('SEEDS:', SEEDS)

vocab_path = f'{RUN_DIR}/vocab.json'
if os.path.exists(vocab_path):
    vocab = json.load(open(vocab_path))
    print(f'loaded vocab: {len(vocab)}')
else:
    print('building vocab (normalized: lowercase, outer punctuation stripped)...')
    vocab = build_vocab(DATA_DIR, min_freq=CFG['min_freq'])
    json.dump(vocab, open(vocab_path, 'w'), ensure_ascii=False)
    print(f'built vocab: {len(vocab)}')
pad_id = vocab[PAD]

pages = load_pages(DATA_DIR, vocab)
rng = random.Random(CFG['seed'])
order = list(range(len(pages))); rng.shuffle(order)
n_val = max(1, int(len(pages) * CFG['val_frac']))
val_pages   = [pages[i] for i in order[:n_val]]
train_pages = [pages[i] for i in order[n_val:]]
print(f'train pages: {len(train_pages)}  val pages: {len(val_pages)}  '
      f'val windows: {len(val_pages) * CFG["val_windows"]} (deterministic, shared by all conditions)')
"""

# ═══════════════════════════════════════════════════════════════════════════
CELL_TRAIN = r"""# ── 5. Train + evaluate one condition ────────────────────────────────────────
from tqdm.auto import tqdm
# 0 workers: __getitem__ is sub-millisecond tensor slicing, so workers buy
# nothing here, and on Colab's Python 3.13 the worker teardown spams
# "can only test a child process" tracebacks every epoch (harmless, but it
# buried the real error in the first v2 run).
NUM_WORKERS = globals().get('NUM_WORKERS', 0)


def make_loaders(condition, seed):
    tr = FlatOCTDataset(train_pages, vocab, CFG['seq_len'], condition)
    va = FlatOCTDataset(val_pages, vocab, CFG['seq_len'], condition,
                        deterministic=True, n_windows=CFG['val_windows'])
    g = torch.Generator().manual_seed(seed)
    coll = partial(collate, pad_id=pad_id)
    tr_dl = DataLoader(tr, CFG['batch_size'], shuffle=True, collate_fn=coll,
                       num_workers=NUM_WORKERS, drop_last=True, generator=g)
    va_dl = DataLoader(va, CFG['batch_size'], shuffle=False, collate_fn=coll, num_workers=NUM_WORKERS)
    return tr_dl, va_dl


def to_dev(b):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def forward_batch(model, b):
    return model(b['types'], b['wids_in'], stroke_feat(b['delta_in'], b['abs_in']), b['pens_in'], b['pad'])


@torch.no_grad()
def evaluate(model, dl):
    model.eval(); acc = {}
    for b in dl:
        b = to_dev(b)
        with torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
            out = forward_batch(model, b)
        for k, v in raw_metrics(out, b).items():
            acc[k] = acc.get(k, 0.0) + v
    m = {}
    for name in ('type', 'word', 'stroke', 'pen'):
        n = acc.get(f'{name}_n', 0)
        m[name] = (acc[f'{name}_sum'] / n) if n else None
        m[f'{name}_n'] = int(n)
    if m['word'] is not None:
        m['word_ppl'] = math.exp(m['word'])
    return m


def result_path(condition, seed):
    # Seed-tagged names. The first v2 run (crashed in drop_strokes) saved
    # three conditions under the untagged name with seed 42 -- reuse them.
    tagged = f'{RUN_DIR}/result_{condition}_s{seed}.json'
    legacy = f'{RUN_DIR}/result_{condition}.json'
    if not os.path.exists(tagged) and seed == 42 and os.path.exists(legacy):
        return legacy
    return tagged


def ckpt_path(condition, seed):
    tagged = f'{RUN_DIR}/{condition}_s{seed}.best.pt'
    legacy = f'{RUN_DIR}/{condition}.best.pt'
    if not os.path.exists(tagged) and seed == 42 and os.path.exists(legacy):
        return legacy
    return tagged


def run_condition(condition, seed):
    res_path = result_path(condition, seed)
    if os.path.exists(res_path):
        print(f'[{condition} s{seed}] already done — loading {os.path.basename(res_path)}')
        return json.load(open(res_path))
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    tr_dl, va_dl = make_loaders(condition, seed)
    model = FlatOCTModel(len(vocab), CFG['d_model'], CFG['n_layers'], CFG['n_heads'], CFG['d_ff'],
                         max(2 * CFG['seq_len'], 2048), CFG['dropout'], pad_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    total_steps = CFG['epochs'] * len(tr_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / CFG['warmup'] if s < CFG['warmup']
        else 0.5 * (1 + math.cos(math.pi * (s - CFG['warmup']) / max(total_steps - CFG['warmup'], 1))))
    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f'[{condition} s{seed}] {model.n_params/1e6:.1f}M params, {len(tr_dl)} batches/epoch, {CFG["epochs"]} epochs')

    history, best = [], None
    for ep in range(1, CFG['epochs'] + 1):
        t0 = time.time(); model.train(); tr_sum = {}; nb = 0
        for b in tqdm(tr_dl, desc=f'{condition} s{seed} ep{ep:02d}', leave=False):
            b = to_dev(b); opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                out = forward_batch(model, b)
                L = model.loss(out, b)
            scaler.scale(L['total']).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            scaler.step(opt); scaler.update(); sched.step()
            for k, v in L.items():
                tr_sum[k] = tr_sum.get(k, 0.0) + float(v)
            nb += 1
        va = evaluate(model, va_dl)
        rec = {'epoch': ep, 'train': {k: v / nb for k, v in tr_sum.items()}, 'val': va,
               'elapsed': round(time.time() - t0, 1)}
        history.append(rec)
        fmt = lambda x: 'n/a' if x is None else f'{x:.4f}'
        print(f'  ep{ep:02d}  train={rec["train"]["total"]:.3f}  val: type={fmt(va["type"])} '
              f'word={fmt(va["word"])} stroke={fmt(va["stroke"])} pen={fmt(va["pen"])}  ({rec["elapsed"]:.0f}s)')
        # model selection on the metric this condition is read for
        key = 'stroke' if condition in ('mask_words', 'noword_hidden', 'drop_words') else 'word'
        score = va[key] if va[key] is not None else va['type']
        if best is None or score < best['score']:
            best = {'score': score, 'epoch': ep, 'val': va}
            # save every condition's checkpoint -- the bootstrap-CI pass (cell
            # 7) needs to re-evaluate all of them, not just 'full'
            torch.save({'model_state': model.state_dict(), 'cfg': CFG, 'epoch': ep, 'val': va,
                        'seed': seed}, f'{RUN_DIR}/{condition}_s{seed}.best.pt')
    result = {'condition': condition, 'seed': seed, 'cfg': CFG, 'best': best,
              'final': history[-1]['val'], 'history': history}
    json.dump(result, open(res_path, 'w'), indent=1)
    print(f'[{condition} s{seed}] best epoch {best["epoch"]}: {best["val"]}')
    return result
"""

# ═══════════════════════════════════════════════════════════════════════════
CELL_RUN = r"""# ── 6. Run all conditions x seeds (resumable per run) ────────────────────────
results = {c: {} for c in CONDITIONS}
for seed in SEEDS:
    for cond in CONDITIONS:
        results[cond][seed] = run_condition(cond, seed)
print(f'\nall {len(CONDITIONS)} conditions x {len(SEEDS)} seeds done')
"""

# ═══════════════════════════════════════════════════════════════════════════
CELL_BOOTSTRAP = r"""# ── 7. Per-window bootstrap: fixes the "|Δ|>=0.05 is real" hand-wave ────────
# Tokens within one lecture are correlated (if the model is confused about a
# page, most of that page's tokens get correlated extra loss) -- a token-count
# standard error massively overstates precision. This evaluates every
# condition's best checkpoint per VALIDATION WINDOW (363 of them, matched by
# list position across conditions since the val loader is unshuffled and
# every condition sees the same window at the same index) and bootstraps
# over windows instead of tokens.

# >>> BOOTSTRAP_FUNCS
@torch.no_grad()
def evaluate_per_window(model, dl):
    model.eval()
    n_win = len(dl.dataset)
    out = {'type': [None] * n_win, 'word': [None] * n_win,
          'stroke': [None] * n_win, 'pen': [None] * n_win}
    pos = 0
    for b in dl:
        b = to_dev(b)
        with torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
            o = forward_batch(model, b)
        tgt_t, tgt_w = b['types'][:, 1:], b['wids'][:, 1:]
        tgt_d, tgt_p = b['delta'][:, 1:], b['pens'][:, 1:]
        valid = ~b['pad'][:, 1:]
        for i in range(tgt_t.size(0)):
            v = valid[i]
            if v.any():
                out['type'][pos] = F.cross_entropy(o['type'][i, :-1][v].float(), tgt_t[i][v],
                                                   reduction='mean').item()
            wp = v & (tgt_t[i] == WORD_TYPE)
            if wp.any():
                out['word'][pos] = F.cross_entropy(o['word'][i, :-1][wp].float(), tgt_w[i][wp],
                                                   reduction='mean').item()
            sp = v & (tgt_t[i] == STROKE_TYPE)
            if sp.any():
                out['stroke'][pos] = mdn_nll(o['xy'][i, :-1][sp], tgt_d[i][sp]).mean().item()
                out['pen'][pos] = F.cross_entropy(o['pen'][i, :-1][sp].float(), tgt_p[i][sp],
                                                  reduction='mean').item()
            pos += 1
    assert pos == n_win
    return out


def bootstrap_delta(list_lo, list_hi, n_boot=3000, seed=0):
    # delta = mean(lo) - mean(hi); positive = the "hi" condition (more info)
    # scored lower loss, i.e. that information helped. Paired by position
    # (same validation window in both lists); positions missing in either
    # (rare: a window with zero targets of this kind) are dropped from both.
    pairs = [(a, b) for a, b in zip(list_lo, list_hi) if a is not None and b is not None]
    if not pairs:
        return None
    diffs = np.array([a - b for a, b in pairs])   # lo_loss - hi_loss: positive = hi is better
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'point': float(diffs.mean()), 'ci_lo': float(lo), 'ci_hi': float(hi),
           'n_windows': n, 'significant': bool(lo > 0 or hi < 0)}
# <<< BOOTSTRAP_FUNCS


def pw_key(cond, seed):
    return f'{cond}_s{seed}'


per_window = {}
pw_path = f'{RUN_DIR}/per_window.json'
if os.path.exists(pw_path):
    per_window = json.load(open(pw_path))
    print(f'loaded cached per-window metrics for {len(per_window)} runs')
for seed in SEEDS:
    for cond in CONDITIONS:
        k = pw_key(cond, seed)
        if k in per_window:
            continue
        ck = torch.load(ckpt_path(cond, seed), map_location=device)
        m = FlatOCTModel(len(vocab), CFG['d_model'], CFG['n_layers'], CFG['n_heads'], CFG['d_ff'],
                         max(2 * CFG['seq_len'], 2048), CFG['dropout'], pad_id).to(device)
        m.load_state_dict(ck['model_state'])
        _, va_dl = make_loaders(cond, seed)
        per_window[k] = evaluate_per_window(m, va_dl)
        print(f'[{k}] per-window metrics collected '
              f"(word: {sum(1 for x in per_window[k]['word'] if x is not None)} windows, "
              f"stroke: {sum(1 for x in per_window[k]['stroke'] if x is not None)} windows)")
        json.dump(per_window, open(pw_path, 'w'))   # cache after every run

print('\nper-window evaluation done')
"""

# ═══════════════════════════════════════════════════════════════════════════
CELL_RESULTS = r"""# ── 8. Results: per-seed table, length-controlled deltas, seed spread ────────
# Every delta is computed WITHIN a seed (paired by validation window), then
# summarised ACROSS seeds. A delta is only claimed if (a) its window-bootstrap
# CI excludes zero in every seed AND (b) all seeds agree in sign -- the
# across-seed spread is the honest uncertainty for a training-run effect.

def g(cond, seed, key):
    return results[cond][seed]['best']['val'].get(key)


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    s = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return m, s


print(f'{"condition":16s} {"word CE (mean±std)":>22s} {"ppl":>7s} {"stroke NLL (mean±std)":>24s} {"seeds":>6s}')
for c in CONDITIONS:
    wm, ws = mean_std([g(c, s, 'word') for s in SEEDS])
    sm, ss = mean_std([g(c, s, 'stroke') for s in SEEDS])
    w = '         n/a' if wm is None else f'{wm:8.4f} ± {ws:.4f}'
    st = '         n/a' if sm is None else f'{sm:8.4f} ± {ss:.4f}'
    ppl = '    n/a' if wm is None else f'{math.exp(wm):7.1f}'
    print(f'{c:16s} {w:>22s} {ppl:>7s} {st:>24s} {len(SEEDS):>6d}')

print()
print('per-seed word CE:  ' + '  '.join(f'{c}=' + '/'.join(f'{g(c,s,"word"):.3f}' for s in SEEDS)
                                         for c in ('full', 'mask_strokes', 'nostroke_hidden', 'drop_strokes')))
print('per-seed strk NLL: ' + '  '.join(f'{c}=' + '/'.join(f'{g(c,s,"stroke"):.3f}' for s in SEEDS)
                                         for c in ('full', 'mask_words', 'noword_hidden', 'drop_words')))


def delta_over_seeds(lo_cond, hi_cond, key):
    per_seed = {s: bootstrap_delta(per_window[pw_key(lo_cond, s)][key],
                                   per_window[pw_key(hi_cond, s)][key]) for s in SEEDS}
    pts = [d['point'] for d in per_seed.values() if d is not None]
    m, sd = mean_std(pts)
    all_sig = all(d is not None and d['significant'] for d in per_seed.values())
    same_sign = len({p > 0 for p in pts}) == 1
    return {'per_seed': per_seed, 'mean': m, 'std': sd, 'all_sig': all_sig, 'same_sign': same_sign}


def show(name, what, d):
    verdict = 'ROBUST' if (d['all_sig'] and d['same_sign']) else ('same sign, not all sig' if d['same_sign'] else 'seeds DISAGREE')
    seeds_txt = '  '.join(f"s{s}:{v['point']:+.3f}[{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}]"
                          for s, v in d['per_seed'].items() if v is not None)
    print(f'  {name:11s} {what:44s} mean {d["mean"]:+.4f} ± {d["std"]:.4f}   {verdict}')
    print(f'  {"":11s} {seeds_txt}')


print()
print('=' * 96)
print('CORRECTED: length-controlled 3-way split  (point = loss reduction from adding that info; + helps)')
print('=' * 96)
print('WORDS   chain: drop_strokes -> nostroke_hidden -> mask_strokes -> full')
dW_length  = delta_over_seeds('drop_strokes',    'nostroke_hidden', 'word')
dW_marker  = delta_over_seeds('nostroke_hidden', 'mask_strokes',    'word')
dW_content = delta_over_seeds('mask_strokes',    'full',            'word')
show('ΔW_length',  'same zero-info, only sequence length differs', dW_length)
show('ΔW_marker',  'same length, stroke-event marker becomes visible', dW_marker)
show('ΔW_content', 'same length, stroke features become real', dW_content)
print()
print('STROKES chain: drop_words -> noword_hidden -> mask_words -> full')
dS_length  = delta_over_seeds('drop_words',    'noword_hidden', 'stroke')
dS_marker  = delta_over_seeds('noword_hidden', 'mask_words',    'stroke')
dS_content = delta_over_seeds('mask_words',    'full',          'stroke')
show('ΔS_length',  'same zero-info, only sequence length differs', dS_length)
show('ΔS_marker',  'same length, word-event marker becomes visible', dS_marker)
show('ΔS_content', 'same length, word identities become real', dS_content)

print()
print('=' * 96)
print('v1-style confounded 2-way split (mean over seeds) -- kept for the record')
print('=' * 96)
def v1_delta(lo, hi, key):
    m, sd = mean_std([g(lo, s, key) - g(hi, s, key) for s in SEEDS]); return f'{m:+.4f} ± {sd:.4f}'
print(f'  ΔW_content (mask_strokes − full)        = {v1_delta("mask_strokes","full","word")}')
print(f'  ΔW_events  (drop_strokes − mask_strokes) = {v1_delta("drop_strokes","mask_strokes","word")}   '
      f'<- confounded: drop_strokes windows are {g("drop_strokes",SEEDS[0],"type_n")/g("full",SEEDS[0],"type_n")*100:.1f}% as long')
print(f'  ΔS_content (mask_words − full)          = {v1_delta("mask_words","full","stroke")}')
print(f'  ΔS_events  (drop_words − mask_words)     = {v1_delta("drop_words","mask_words","stroke")}   '
      f'<- confounded: drop_words windows are {g("drop_words",SEEDS[0],"type_n")/g("full",SEEDS[0],"type_n")*100:.1f}% as long')
print()
print('CI = 95% paired bootstrap over validation windows within one seed (3000 resamples).')
print('ROBUST = CI excludes zero in every seed AND all seeds agree in sign.')

json.dump({'seeds': SEEDS,
           'v2_corrected': {'W_length': dW_length, 'W_marker': dW_marker, 'W_content': dW_content,
                            'S_length': dS_length, 'S_marker': dS_marker, 'S_content': dS_content},
           'table': {c: {s: results[c][s]['best']['val'] for s in SEEDS} for c in CONDITIONS}},
          open(f'{RUN_DIR}/summary.json', 'w'), indent=1, default=str)
"""

# ═══════════════════════════════════════════════════════════════════════════
CELL_GEN = r"""# ── 8. Qualitative: generation from the `full` model ─────────────────────────
import matplotlib.pyplot as plt
id2word = {v: k for k, v in vocab.items()}

ck = torch.load(ckpt_path('full', SEEDS[0]), map_location=device)
gen_model = FlatOCTModel(len(vocab), CFG['d_model'], CFG['n_layers'], CFG['n_heads'], CFG['d_ff'],
                         max(2 * CFG['seq_len'], 2048), CFG['dropout'], pad_id).to(device)
gen_model.load_state_dict(ck['model_state']); gen_model.eval()
print(f'full model from epoch {ck["epoch"]}')

# >>> GEN_FUNCS
@torch.no_grad()
def generate(model, seed, max_new=800, temperature=0.8, top_k=40,
             sigma_temp=0.65, pi_temp=1.0, pen_temp=1.0, ctx=1024):
    # seed: dict of 1-D tensors types/wids/delta/abs/pens (real content).
    # <silent> is ALLOWED (49% of drawing starts from it); <eos> stops.
    dev = device
    types = seed['types'].clone().to(dev); wids = seed['wids'].clone().to(dev)
    delta = seed['delta'].clone().to(dev); absxy = seed['abs'].clone().to(dev); pens = seed['pens'].clone().to(dev)
    sm = types == STROKE_TYPE
    ax, ay = (float(absxy[sm][-1, 0]), float(absxy[sm][-1, 1])) if sm.any() else (0.0, 0.0)
    forbid = [vocab[PAD], vocab[UNK], vocab[MASK], vocab[BOS]]
    out_tokens = []
    for _ in range(max_new):
        sl = slice(-ctx, None)
        pad = torch.zeros(1, types[sl].size(0), dtype=torch.bool, device=dev)
        o = model(types[sl][None], wids[sl][None], stroke_feat(delta[sl], absxy[sl])[None], pens[sl][None], pad)
        nt = int(torch.multinomial(F.softmax(o['type'][0, -1].float(), -1), 1))
        if nt == WORD_TYPE:
            wl = o['word'][0, -1].float().clone(); wl[forbid] = -float('inf')
            wl = wl / temperature
            if top_k:
                v, i = torch.topk(wl, top_k); m = torch.full_like(wl, -float('inf')); m[i] = v; wl = m
            w = int(torch.multinomial(F.softmax(wl, -1), 1))
            if w == vocab[EOS]:
                break
            out_tokens.append({'type': 'word', 'word': id2word[w]})
            types = torch.cat([types, torch.tensor([WORD_TYPE], device=dev)])
            wids = torch.cat([wids, torch.tensor([w], device=dev)])
            delta = torch.cat([delta, torch.zeros(1, 2, device=dev)]); absxy = torch.cat([absxy, torch.zeros(1, 2, device=dev)])
            pens = torch.cat([pens, torch.tensor([0], device=dev)])
        else:
            sdx, sdy = mdn_sample(o['xy'][0, -1], pi_temp, sigma_temp)
            nx = min(1.0, max(0.0, ax + sdx / DELTA_SCALE)); ny = min(1.0, max(0.0, ay + sdy / DELTA_SCALE))
            sd = torch.tensor([[(nx - ax) * DELTA_SCALE, (ny - ay) * DELTA_SCALE]], device=dev)
            ax, ay = nx, ny
            p = int(torch.multinomial(F.softmax(o['pen'][0, -1].float() / pen_temp, -1), 1))
            out_tokens.append({'type': 'stroke', 'x': nx, 'y': ny, 'pen': p})
            types = torch.cat([types, torch.tensor([STROKE_TYPE], device=dev)])
            wids = torch.cat([wids, torch.tensor([0], device=dev)])
            delta = torch.cat([delta, sd]); absxy = torch.cat([absxy, torch.tensor([[nx, ny]], device=dev)])
            pens = torch.cat([pens, torch.tensor([p], device=dev)])
    return out_tokens


def page_tokens(page, a, b):
    # ground-truth slice → same token-dict format as generate()
    toks = []
    for i in range(a, min(b, page['types'].size(0))):
        if int(page['types'][i]) == STROKE_TYPE:
            toks.append({'type': 'stroke', 'x': float(page['abs'][i, 0]), 'y': float(page['abs'][i, 1]),
                         'pen': int(page['pens'][i])})
        else:
            toks.append({'type': 'word', 'word': id2word[int(page['wids'][i])]})
    return toks


def draw(ax, toks, color, alpha=0.9, lw=1.2):
    cur = []
    for t in toks:
        if t['type'] != 'stroke':
            continue
        cur.append((t['x'], 1 - t['y']))
        if t['pen'] == PEN_UP:
            if len(cur) > 1:
                ax.plot(*zip(*cur), color=color, lw=lw, alpha=alpha, solid_capstyle='round')
            cur = []
    if len(cur) > 1:
        ax.plot(*zip(*cur), color=color, lw=lw, alpha=alpha)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect(1280 / 720); ax.axis('off')


def words_of(toks, n=40):
    ws = [('·' if t['word'] == SILENT else t['word']) for t in toks if t['type'] == 'word']
    return ' '.join(ws[:n]) + (' …' if len(ws) > n else '')


def seed_from_page(page, n):
    return {k: page[k][:n] for k in ('types', 'wids', 'delta', 'abs', 'pens')}


def stats(toks):
    n_s = sum(1 for t in toks if t['type'] == 'stroke'); n_w = len(toks) - n_s
    return f'{n_w} words / {n_s} points ({100 * n_s / max(len(toks), 1):.0f}% strokes)'
# <<< GEN_FUNCS

random.seed(0)
for trial in range(2):
    page = random.choice(val_pages)
    # BOS + page so the seed matches training windows that start at 0
    bos = {'types': torch.tensor([WORD_TYPE]), 'wids': torch.tensor([vocab[BOS]]),
           'delta': torch.zeros(1, 2), 'abs': torch.zeros(1, 2), 'pens': torch.tensor([0])}
    full = {k: torch.cat([bos[k], page[k]]) for k in bos}

    # ── cold start: BOS + first 5 words ──
    idx = torch.nonzero(full['types'] == WORD_TYPE)[:6, 0]         # BOS + 5 words
    cold_seed = {k: full[k][idx] for k in full}
    cold = generate(gen_model, cold_seed, max_new=800)

    # ── warm start: first 400 real tokens as context ──
    N_CTX, N_GEN = 400, 600
    warm_seed = seed_from_page(full, N_CTX)
    warm = generate(gen_model, warm_seed, max_new=N_GEN)
    gt_ctx, gt_cont = page_tokens(full, 0, N_CTX), page_tokens(full, N_CTX, N_CTX + N_GEN)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
    fig.suptitle(f"{page['tag']} | {page['topic'][:60]}", fontsize=11)
    draw(axes[0], gt_ctx, '#1a3a8c'); draw(axes[0], gt_cont, '#999999', alpha=0.7)
    axes[0].set_title(f'ground truth: context (blue) + true continuation (grey)\n{stats(gt_cont)}', fontsize=9)
    draw(axes[1], gt_ctx, '#1a3a8c'); draw(axes[1], warm, '#b0202a')
    axes[1].set_title(f'WARM: real context (blue) + generated (red)\n{stats(warm)}', fontsize=9)
    draw(axes[2], cold, '#b0202a')
    axes[2].set_title(f'COLD: BOS + 5 words → generated\n{stats(cold)}', fontsize=9)
    plt.tight_layout(); plt.savefig(f'{RUN_DIR}/gen_{page["tag"]}.png', dpi=130, bbox_inches='tight'); plt.show()
    print('true continuation words:', words_of(gt_cont))
    print('warm generated words:   ', words_of(warm))
    print('cold generated words:   ', words_of(cold))
    print()
"""

# ═══════════════════════════════════════════════════════════════════════════
cells = [
    ("markdown", CELL_HEADER),
    ("code", CELL_SETUP),
    ("code", CELL_DATA),
    ("code", CELL_MODEL),
    ("code", CELL_CONFIG),
    ("code", CELL_TRAIN),
    ("code", CELL_RUN),
    ("code", CELL_BOOTSTRAP),
    ("code", CELL_RESULTS),
    ("code", CELL_GEN),
]

nb = {
    "cells": [],
    "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}, "accelerator": "GPU"},
    "nbformat": 4, "nbformat_minor": 0,
}
for kind, src in cells:
    lines = src.splitlines(keepends=True)
    cell = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        cell["outputs"] = []; cell["execution_count"] = None
        ast.parse(src)   # syntax check
    nb["cells"].append(cell)
OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {OUT_NB} ({len(cells)} cells, all code cells parse)")

# Smoke test = the library cells verbatim + a tiny local driver.
# CELL_BOOTSTRAP has an executable tail (loads checkpoints, needs CFG/vocab/
# pages to already exist) -- only its function defs go in the shared lib;
# smoke_test.py drives the loop itself, same as CELL_RUN does for training.
bootstrap_funcs = CELL_BOOTSTRAP.split("# >>> BOOTSTRAP_FUNCS")[1].split("# <<< BOOTSTRAP_FUNCS")[0]
smoke = "\n\n".join([CELL_DATA, CELL_MODEL,
                     CELL_TRAIN.replace("from tqdm.auto import tqdm", "tqdm = lambda x, **k: x"),
                     bootstrap_funcs])
OUT_SMOKE.write_text(smoke)
print(f"wrote {OUT_SMOKE}")
gen_funcs = CELL_GEN.split("# >>> GEN_FUNCS")[1].split("# <<< GEN_FUNCS")[0]
(OUT_SMOKE.parent / "gen_lib.py").write_text(gen_funcs)
print("wrote gen_lib.py")
