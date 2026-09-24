"""Flat interleaved OCT lecture generation model (v3 — MDN stroke head).

Architecture
------------
Single causal transformer operating on a flat interleaved sequence
of WORD and STROKE tokens:

    [W:"today"] [W:"we"] [S:dx,dy,MID] [S:dx,dy,UP] [W:"a"] ...

Stroke coordinates are scaled deltas (offsets from the previous stroke
point, multiplied by DELTA_SCALE).

v3: the xy head is a Mixture Density Network (Graves 2013). Handwriting
deltas are multimodal — the pen can go any direction next depending on
the letter being drawn. A deterministic regression head predicts the
MEAN of those options (~0) and generated strokes collapse to dots (the
v2 failure mode). The MDN learns the actual distribution and generation
SAMPLES from it.

At each position t, the model predicts what comes at t+1:
  - Type head:  {WORD=0, STROKE=1}
  - Word head:  word ID from vocab  (active when next is WORD)
  - XY head:    MDN params — K bivariate Gaussians (active when next is STROKE)
  - Pen head:   {MID=0, UP=1}       (active when next is STROKE)

Loss:
    L = L_type + L_word + xy_weight * L_xy_nll + L_pen

    - L_word uses label_smoothing=0.1
    - L_xy_nll is the MDN negative log-likelihood (can go NEGATIVE as
      the density sharpens; starts ~3, good models reach < 0)
    - L_pen uses mild class weights [1.0, 2.0] (4.0 over-predicted UP
      at sampling time and shredded strokes into dashes)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Token types
WORD_TYPE   = 0
STROKE_TYPE = 1

# Pen states (no P_STOP — type transition handles it)
PEN_MID = 0
PEN_UP  = 1

# Delta scaling: raw deltas have overall std ~0.07. Scale by 20 → std ~1.25,
# which means the MDN's initial sigma=exp(0)=1 starts near the data scale.
DELTA_SCALE = 20.0

# MDN mixture components (Graves 2013 used 20)
MDN_K = 20


# ─── helpers ─────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


def causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular boolean mask for causal attention."""
    return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)


# ─── MDN helpers (bivariate Gaussian mixture, Graves 2013 eq. 23-25) ─────────

def mdn_split(params: torch.Tensor):
    """Split raw MDN params (..., 6K) into mixture components.

    Returns log_pi (..., K), mu (..., K, 2), sigma (..., K, 2), rho (..., K).
    Always computed in float32 — fp16 logsumexp/exp under AMP can NaN.
    """
    params = params.float()
    K = MDN_K
    log_pi  = F.log_softmax(params[..., :K], dim=-1)
    mu      = params[..., K:3 * K].reshape(*params.shape[:-1], K, 2)
    log_sig = params[..., 3 * K:5 * K].reshape(*params.shape[:-1], K, 2).clamp(-4.0, 3.0)
    rho     = 0.95 * torch.tanh(params[..., 5 * K:6 * K])
    return log_pi, mu, log_sig.exp(), rho


def mdn_nll(params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of target (N, 2) under the GMM (N, 6K)."""
    log_pi, mu, sigma, rho = mdn_split(params)
    t  = target.float().unsqueeze(-2)                    # (N, 1, 2)
    zx = (t[..., 0] - mu[..., 0]) / sigma[..., 0]        # (N, K)
    zy = (t[..., 1] - mu[..., 1]) / sigma[..., 1]
    one_m_rho2 = (1 - rho ** 2).clamp(min=1e-6)
    z = zx ** 2 + zy ** 2 - 2 * rho * zx * zy
    log_gauss = (-z / (2 * one_m_rho2)
                 - torch.log(sigma[..., 0]) - torch.log(sigma[..., 1])
                 - 0.5 * torch.log(one_m_rho2) - math.log(2 * math.pi))
    return -torch.logsumexp(log_pi + log_gauss, dim=-1)  # (N,)


def mdn_sample(params: torch.Tensor, pi_temp: float = 1.0,
               sigma_temp: float = 1.0) -> tuple[float, float]:
    """Sample one (dx, dy) from the mixture. params: flat (6K,) tensor.

    sigma_temp < 1 reduces noise (cleaner strokes); pi_temp < 1 sharpens
    the component choice.
    """
    log_pi, mu, sigma, rho = mdn_split(params.unsqueeze(0))
    log_pi, mu, sigma, rho = log_pi[0], mu[0], sigma[0], rho[0]
    k  = int(torch.multinomial(F.softmax(log_pi / max(pi_temp, 1e-6), dim=-1), 1))
    sx = sigma[k, 0] * sigma_temp
    sy = sigma[k, 1] * sigma_temp
    r  = rho[k]
    z1, z2 = torch.randn(2, device=params.device)
    dx = mu[k, 0] + sx * z1
    dy = mu[k, 1] + sy * (r * z1 + torch.sqrt((1 - r ** 2).clamp(min=1e-6)) * z2)
    return float(dx), float(dy)


# ─── flat interleaved model ──────────────────────────────────────────────────

class FlatOCTModel(nn.Module):
    """Flat interleaved OCT lecture generation model (v3 — MDN).

    Words and stroke points share one causal transformer context.
    Each position is either WORD (type=0) or STROKE (type=1).
    Stroke coordinates are scaled delta-xy (offsets * DELTA_SCALE).
    """

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int = 384,
        n_layers:    int = 6,
        n_heads:     int = 8,
        d_ff:        int = 1536,
        max_seq_len: int = 2048,
        dropout:     float = 0.05,
        pad_idx:     int = 0,
    ):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx

        # ── input embeddings ──────────────────────────────────────────────────
        self.word_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.type_embed = nn.Embedding(2, d_model)          # WORD=0, STROKE=1
        self.stroke_proj = nn.Linear(2, d_model)            # scaled (dx,dy) → d_model
        self.pen_embed  = nn.Embedding(2, d_model)          # MID=0, UP=1
        self.pos_enc    = SinusoidalPE(d_model, max_len=max_seq_len, dropout=dropout)

        # ── causal transformer ────────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm_out    = nn.LayerNorm(d_model)

        # ── output heads ──────────────────────────────────────────────────────
        self.type_head = nn.Linear(d_model, 2)
        self.word_head = nn.Linear(d_model, vocab_size, bias=False)
        self.word_head.weight = self.word_embed.weight
        self.xy_head  = nn.Linear(d_model, 6 * MDN_K)   # MDN: pi, mu_xy, sig_xy, rho
        self.pen_head = nn.Linear(d_model, 2)

        # Pen class weights: PEN_UP is ~17.5% of stroke points. Mild 2x weight.
        self.register_buffer("pen_weight", torch.tensor([1.0, 2.0]))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.word_embed.weight, std=0.02)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        nn.init.normal_(self.pen_embed.weight, std=0.02)
        for name, p in self.named_parameters():
            if any(s in name for s in ("word_embed", "type_embed", "pen_embed")):
                continue
            if p.dim() > 1 and "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        # Near-zero init for the MDN head → log_sigma starts at 0 (sigma=1),
        # right at the scaled data std (~1.25). Stable from step one.
        nn.init.normal_(self.xy_head.weight, std=0.001)
        nn.init.zeros_(self.xy_head.bias)

    def _embed(self, token_types, word_ids, xy, pen):
        h = self.type_embed(token_types)
        word_mask   = (token_types == WORD_TYPE).unsqueeze(-1).float()
        stroke_mask = (token_types == STROKE_TYPE).unsqueeze(-1).float()
        h = h + self.word_embed(word_ids) * word_mask
        h = h + (self.stroke_proj(xy) + self.pen_embed(pen)) * stroke_mask
        return self.pos_enc(h)

    def forward(self, token_types, word_ids, xy, pen, pad_mask):
        B, T = token_types.shape
        h = self._embed(token_types, word_ids, xy, pen)
        cmask  = causal_mask(T, h.device)
        hidden = self.transformer(h, mask=cmask, src_key_padding_mask=pad_mask,
                                  is_causal=True)
        hidden = self.norm_out(hidden)
        return {
            "type_logits": self.type_head(hidden),
            "word_logits": self.word_head(hidden),
            "xy_params":   self.xy_head(hidden),     # MDN params (B, T, 6K)
            "pen_logits":  self.pen_head(hidden),
        }

    def compute_loss(self, outputs, token_types, word_ids, xy, pen,
                     pad_mask, xy_weight=1.0):
        pred_type = outputs["type_logits"][:, :-1]
        pred_word = outputs["word_logits"][:, :-1]
        pred_xy   = outputs["xy_params"][:, :-1]
        pred_pen  = outputs["pen_logits"][:, :-1]

        tgt_type = token_types[:, 1:]
        tgt_word = word_ids[:, 1:]
        tgt_xy   = xy[:, 1:]
        tgt_pen  = pen[:, 1:]
        tgt_pad  = pad_mask[:, 1:]
        valid    = ~tgt_pad

        # Type loss
        l_type = F.cross_entropy(
            pred_type[valid], tgt_type[valid]
        ) if valid.any() else pred_type.new_tensor(0.0)

        # Word loss with label smoothing (helps generalization with large vocab)
        word_pos = valid & (tgt_type == WORD_TYPE)
        l_word = F.cross_entropy(
            pred_word[word_pos], tgt_word[word_pos],
            ignore_index=self.pad_idx,
            label_smoothing=0.1,
        ) if word_pos.any() else pred_word.new_tensor(0.0)

        # XY loss: MDN negative log-likelihood. Can go NEGATIVE as the
        # density sharpens — starts ~3.0, good models reach < 0.
        stroke_pos = valid & (tgt_type == STROKE_TYPE)
        l_xy = mdn_nll(pred_xy[stroke_pos], tgt_xy[stroke_pos]).mean() \
               if stroke_pos.any() else pred_xy.new_tensor(0.0)

        # Pen loss with mild class weighting
        l_pen = F.cross_entropy(
            pred_pen[stroke_pos], tgt_pen[stroke_pos],
            weight=self.pen_weight,
        ) if stroke_pos.any() else pred_pen.new_tensor(0.0)

        total = l_type + l_word + xy_weight * l_xy + l_pen

        return {
            "total": total,
            "type":  l_type,
            "word":  l_word,
            "xy":    l_xy,
            "pen":   l_pen,
        }

    # ── generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        seed_types:  torch.Tensor,   # (1, S) long
        seed_words:  torch.Tensor,   # (1, S) long
        seed_xy:     torch.Tensor,   # (1, S, 2) float — scaled delta-xy
        seed_pen:    torch.Tensor,   # (1, S) long
        vocab:       dict[str, int],
        max_len:     int   = 1000,
        temperature: float = 0.8,
        top_k:       int   = 50,
        stroke_bias: float = 0.0,
        pen_temperature: float = 1.0,
        pi_temp:     float = 1.0,
        sigma_temp:  float = 0.65,
    ) -> list[dict]:
        """Autoregressive generation. Strokes are SAMPLED from the MDN.

        sigma_temp=0.65 (Graves-style) gives cleaner strokes; raise toward
        1.0 for more variety. Returns tokens with ABSOLUTE coordinates:
            {'type': 'word', 'word_id': int, 'word': str}
            {'type': 'stroke', 'x': float, 'y': float, 'pen': int}
        """
        self.eval()
        device = seed_types.device
        id2word = {v: k for k, v in vocab.items()}
        forbidden = {vocab.get("<pad>", -1), vocab.get("<unk>", -1),
                     vocab.get("<silent>", -1), vocab.get("<page_break>", -1)}
        forbidden.discard(-1)

        types = seed_types.clone()
        words = seed_words.clone()
        xys   = seed_xy.clone()       # scaled deltas
        pens  = seed_pen.clone()
        generated = []

        # Track absolute position (accumulate unscaled seed deltas)
        abs_x, abs_y = 0.0, 0.0
        for i in range(seed_types.size(1)):
            if seed_types[0, i] == STROKE_TYPE:
                abs_x += float(seed_xy[0, i, 0]) / DELTA_SCALE
                abs_y += float(seed_xy[0, i, 1]) / DELTA_SCALE

        for _ in range(max_len):
            T = types.size(1)
            if T > 1024:
                types = types[:, -1024:]
                words = words[:, -1024:]
                xys   = xys[:, -1024:]
                pens  = pens[:, -1024:]
                T = 1024

            pad = torch.zeros(1, T, dtype=torch.bool, device=device)
            out = self.forward(types, words, xys, pens, pad)

            # Sample type
            type_logits = out["type_logits"][0, -1].clone()
            type_logits[STROKE_TYPE] += stroke_bias
            type_probs = F.softmax(type_logits, dim=-1)
            next_type = int(torch.multinomial(type_probs, 1))

            if next_type == WORD_TYPE:
                word_logits = out["word_logits"][0, -1].clone()
                for fid in forbidden:
                    word_logits[fid] = -float("inf")
                word_logits = word_logits / max(temperature, 1e-6)
                if top_k > 0:
                    topk_vals, topk_idx = torch.topk(word_logits, min(top_k, word_logits.size(0)))
                    mask = torch.full_like(word_logits, -float("inf"))
                    mask.scatter_(0, topk_idx, topk_vals)
                    word_logits = mask
                probs = F.softmax(word_logits, dim=-1)
                word_id = int(torch.multinomial(probs, 1))
                if id2word.get(word_id) == "<eos>":
                    break

                generated.append({"type": "word", "word_id": word_id,
                                  "word": id2word.get(word_id, "?")})
                types = torch.cat([types, torch.tensor([[WORD_TYPE]], device=device)], 1)
                words = torch.cat([words, torch.tensor([[word_id]], device=device)], 1)
                xys   = torch.cat([xys, torch.zeros(1, 1, 2, device=device)], 1)
                pens  = torch.cat([pens, torch.zeros(1, 1, dtype=torch.long, device=device)], 1)

            else:
                # SAMPLE a scaled delta from the mixture, unscale, accumulate
                sdx, sdy = mdn_sample(out["xy_params"][0, -1],
                                      pi_temp=pi_temp, sigma_temp=sigma_temp)
                dx = sdx / DELTA_SCALE
                dy = sdy / DELTA_SCALE

                # Accumulate to absolute, clamp to canvas
                new_x = max(0.0, min(1.0, abs_x + dx))
                new_y = max(0.0, min(1.0, abs_y + dy))

                # Recompute actual scaled delta after clamping
                actual_sdx = (new_x - abs_x) * DELTA_SCALE
                actual_sdy = (new_y - abs_y) * DELTA_SCALE
                abs_x, abs_y = new_x, new_y

                # Sample pen
                pen_logits = out["pen_logits"][0, -1].clone()
                pen_logits = pen_logits / max(pen_temperature, 1e-6)
                pen_probs  = F.softmax(pen_logits, dim=-1)
                pen_state  = int(torch.multinomial(pen_probs, 1))

                generated.append({"type": "stroke", "x": new_x,
                                  "y": new_y, "pen": pen_state})

                # Feed back SCALED delta (model works in scaled delta space)
                types = torch.cat([types, torch.tensor([[STROKE_TYPE]], device=device)], 1)
                words = torch.cat([words, torch.zeros(1, 1, dtype=torch.long, device=device)], 1)
                xys   = torch.cat([xys, torch.tensor([[[actual_sdx, actual_sdy]]], device=device)], 1)
                pens  = torch.cat([pens, torch.tensor([[pen_state]], device=device)], 1)

        return generated

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
