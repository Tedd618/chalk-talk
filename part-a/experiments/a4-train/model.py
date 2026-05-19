"""Word-anchored OCT lecture generation model.

Architecture
------------
Outer transformer (causal LM, GPT-style):
  Predicts the next word token given all prior words.
  Hidden states at each word position condition the stroke decoder.

Stroke decoder (per word):
  Given a word's context vector, autoregressively generates the pen
  movements the teacher makes while saying that word.
  Strokes are absolute normalized coordinates (x,y) ∈ [0,1]².

Style embedding:
  A single learned vector added to every outer position.
  Captures OCT's consistent global writing style.

Pen states used by the stroke decoder:
  0  P_MID  — mid-stroke (pen down, more points follow)
  1  P_UP   — end of this stroke (pen lifts; next begins a new stroke)
  2  P_STOP — all strokes for this word are done (stop token)

Training phases:
  Phase 1 — stroke decoder only  (word embeddings + stroke decoder)
  Phase 2 — outer LM only        (outer transformer)
  Phase 3 — joint, lower LR      (everything end-to-end)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

P_MID  = 0
P_UP   = 1
P_STOP = 2


# ─── helpers ─────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
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
        return self.dropout(x + self.pe[:, : x.size(1)])


def causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular boolean mask for causal (autoregressive) attention."""
    return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)


# ─── stroke decoder ───────────────────────────────────────────────────────────

class StrokeDecoder(nn.Module):
    """Small autoregressive transformer decoder for stroke generation.

    At each step it takes the previous (x, y, p) point and predicts the
    next one.  Cross-attention to the word context vector conditions the
    generation on what word is being written.

    Teacher-forced during training; autoregressive at inference.
    """

    def __init__(
        self,
        d_ctx: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 200,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Embed pen state (3 categories) and concatenate with raw (x, y)
        self.p_embed = nn.Embedding(3, 16)
        self.pt_proj = nn.Linear(2 + 16, d_model)      # (x, y, p_emb) → d
        self.pos_emb = nn.Embedding(max_len + 2, d_model)  # learned pos emb

        # Project word context to d_model for cross-attention memory
        self.ctx_proj = nn.Linear(d_ctx, d_model)

        # Start-of-sequence learnable token
        self.start_tok = nn.Parameter(torch.randn(d_model) * 0.02)

        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Output heads
        self.xy_head = nn.Linear(d_model, 2)   # predict absolute (x, y)
        self.p_head  = nn.Linear(d_model, 3)   # predict pen-state logits

    # ── internal helpers ──────────────────────────────────────────────────────

    def _encode_pts(
        self, xy: torch.Tensor, p: torch.Tensor, offset: int = 1
    ) -> torch.Tensor:
        """Encode (x,y,p) tensors → d_model with positional embedding.

        Args:
            xy:     (B, T, 2)
            p:      (B, T) long
            offset: positional index of the first token (default 1, after start)
        Returns:
            (B, T, d_model)
        """
        T = xy.size(1)
        p_emb = self.p_embed(p)                          # (B, T, 16)
        enc   = self.pt_proj(torch.cat([xy, p_emb], -1)) # (B, T, d_model)
        pos   = torch.arange(offset, offset + T, device=xy.device)
        enc   = enc + self.pos_emb(pos).unsqueeze(0)
        return enc

    def _start(self, B: int, device: torch.device) -> torch.Tensor:
        """Start token: (B, 1, d_model)."""
        return (
            self.start_tok.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            + self.pos_emb(torch.zeros(1, dtype=torch.long, device=device))
        )

    # ── forward (teacher-forced) ──────────────────────────────────────────────

    def forward(
        self,
        ctx:      torch.Tensor,           # (B, d_ctx)
        tgt_xy:   torch.Tensor,           # (B, S, 2)  full target incl. stop
        tgt_p:    torch.Tensor,           # (B, S)     full target pen states
        pad_mask: torch.Tensor | None = None,  # (B, S) bool, True = pad
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced forward.

        Input to decoder:  [start, tgt[:S-1]]  — length S
        Prediction target: tgt[:S]              — length S

        Prediction at position i corresponds to target at position i,
        so computing loss is simply: loss(pred, tgt) with no extra shift.

        Returns:
            xy_pred:  (B, S, 2)
            p_logits: (B, S, 3)
        """
        B, S, _ = tgt_xy.shape

        # Decoder input: start token + first S-1 ground-truth points
        inp_pts = self._encode_pts(tgt_xy[:, :-1], tgt_p[:, :-1])  # (B, S-1, d)
        inp     = torch.cat([self._start(B, ctx.device), inp_pts], dim=1)  # (B, S, d)

        # Memory: word context as a length-1 sequence
        mem  = self.ctx_proj(ctx).unsqueeze(1)  # (B, 1, d_model)

        # Causal self-attention mask
        cmask = causal_mask(S, ctx.device)

        out  = self.transformer(tgt=inp, memory=mem,
                                tgt_mask=cmask,
                                tgt_key_padding_mask=pad_mask)
        out  = self.norm(out)

        return self.xy_head(out), self.p_head(out)   # (B,S,2), (B,S,3)

    # ── generation (autoregressive) ───────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        ctx:         torch.Tensor,   # (1, d_ctx)
        max_len:     int | None = None,
        temperature: float = 1.0,
    ) -> list[list[float]]:
        """Autoregressively generate stroke points for one word.

        Returns list of [x, y, p] values.
        The last entry has p == P_STOP.
        """
        max_len = max_len or self.max_len
        device  = ctx.device
        mem     = self.ctx_proj(ctx).unsqueeze(1)   # (1, 1, d_model)
        seq     = self._start(1, device)             # (1, 1, d_model)
        pts: list[list[float]] = []

        for step in range(max_len):
            L     = seq.size(1)
            cmask = causal_mask(L, device)
            out   = self.transformer(tgt=seq, memory=mem, tgt_mask=cmask)
            out   = self.norm(out)

            xy_p  = self.xy_head(out[0, -1])       # (2,)
            p_log = self.p_head(out[0, -1])         # (3,)

            x = float(xy_p[0].clamp(0, 1))
            y = float(xy_p[1].clamp(0, 1))
            p = int(torch.distributions.Categorical(
                logits=p_log / max(temperature, 1e-6)
            ).sample())

            pts.append([x, y, p])
            if p == P_STOP:
                break

            # Encode new point and append
            xy_t = torch.tensor([[[x, y]]], dtype=torch.float32, device=device)
            p_t  = torch.tensor([[p]], dtype=torch.long, device=device)
            new  = self._encode_pts(xy_t, p_t, offset=step + 1)  # (1,1,d)
            seq  = torch.cat([seq, new], dim=1)
            if seq.size(1) > max_len:
                seq = seq[:, -max_len:]

        return pts


# ─── outer model ─────────────────────────────────────────────────────────────

class OCTModel(nn.Module):
    """Word-anchored OCT lecture generation model.

    Trained from scratch on OCT teaching videos only.
    Generates word sequences and, for each word, the pen strokes
    the teacher draws while saying that word.
    """

    def __init__(
        self,
        vocab_size:     int,
        d_model:        int = 256,
        n_layers:       int = 4,
        n_heads:        int = 8,
        max_seq_len:    int = 2048,
        d_stroke:       int = 128,
        n_stroke_layers:int = 2,
        dropout:        float = 0.3,
        pad_idx:        int = 0,
    ):
        super().__init__()
        self.d_model   = d_model
        self.vocab_size = vocab_size
        self.pad_idx   = pad_idx

        # ── outer causal transformer ──────────────────────────────────────────
        self.word_embed  = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc     = SinusoidalPE(d_model, max_len=max_seq_len, dropout=dropout)
        # Single learnable OCT style vector — one teacher, one style
        self.style_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm_out    = nn.LayerNorm(d_model)

        # Word prediction head — weight-tied with embedding (saves params,
        # standard in LM literature)
        self.word_head = nn.Linear(d_model, vocab_size, bias=False)
        self.word_head.weight = self.word_embed.weight

        # ── stroke decoder ────────────────────────────────────────────────────
        self.stroke_decoder = StrokeDecoder(
            d_ctx=d_model, d_model=d_stroke,
            n_layers=n_stroke_layers, n_heads=4,
            dropout=dropout,
        )

        self._init_weights()

    # ── weight init ───────────────────────────────────────────────────────────

    def _init_weights(self):
        nn.init.normal_(self.word_embed.weight, std=0.02)
        for name, p in self.named_parameters():
            if "word_embed" in name or "style_embed" in name:
                continue
            if p.dim() > 1 and "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    # ── outer transformer forward ─────────────────────────────────────────────

    def encode_words(
        self,
        word_ids: torch.Tensor,                  # (B, T)
        pad_mask: torch.Tensor | None = None,    # (B, T) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the outer transformer.

        Returns:
            hidden: (B, T, d_model) — context vectors for stroke conditioning
            logits: (B, T, vocab_size) — next-word prediction logits
        """
        B, T = word_ids.shape
        x = self.word_embed(word_ids)        # (B, T, d_model)
        x = x + self.style_embed            # add OCT style bias
        x = self.pos_enc(x)

        cmask  = causal_mask(T, word_ids.device)
        hidden = self.transformer(
            x, mask=cmask,
            src_key_padding_mask=pad_mask,
            is_causal=True,
        )
        hidden = self.norm_out(hidden)
        logits = self.word_head(hidden)     # (B, T, vocab_size)
        return hidden, logits

    # ── stroke decoder forward ────────────────────────────────────────────────

    def decode_strokes(
        self,
        ctx:        torch.Tensor,            # (N, d_model)
        stroke_xy:  torch.Tensor,            # (N, S, 2)
        stroke_p:   torch.Tensor,            # (N, S)
        stroke_pad: torch.Tensor | None = None,  # (N, S) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run stroke decoder for N (word context, stroke sequence) pairs.

        N is the total number of words-with-strokes across the batch.

        Returns:
            xy_pred:  (N, S, 2)
            p_logits: (N, S, 3)
        """
        return self.stroke_decoder(ctx, stroke_xy, stroke_p, stroke_pad)

    # ── full forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        word_ids:   torch.Tensor,                    # (B, T)
        word_pad:   torch.Tensor | None = None,      # (B, T) bool
        stroke_ctx: torch.Tensor | None = None,      # (N, d_model) pre-extracted
        stroke_xy:  torch.Tensor | None = None,      # (N, S, 2)
        stroke_p:   torch.Tensor | None = None,      # (N, S)
        stroke_pad: torch.Tensor | None = None,      # (N, S) bool
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Full model forward.

        If stroke_ctx is None, only the word forward pass runs.
        stroke_ctx should be pre-extracted from hidden states by the
        training loop using (batch_idx, word_pos) indices.

        Returns:
            word_logits: (B, T, vocab_size)
            xy_pred:     (N, S, 2) or None
            p_logits:    (N, S, 3) or None
        """
        hidden, word_logits = self.encode_words(word_ids, word_pad)

        xy_pred, p_logits = None, None
        if stroke_ctx is not None and stroke_xy is not None:
            xy_pred, p_logits = self.decode_strokes(
                stroke_ctx, stroke_xy, stroke_p, stroke_pad
            )

        return word_logits, xy_pred, p_logits

    # ── utils ─────────────────────────────────────────────────────────────────

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extract_ctx(
        self,
        hidden:     torch.Tensor,   # (B, T, d_model)
        batch_idx:  torch.Tensor,   # (N,) long
        word_pos:   torch.Tensor,   # (N,) long
    ) -> torch.Tensor:
        """Extract word context vectors for stroke decoding.

        Args:
            hidden:    outer transformer hidden states
            batch_idx: which batch item each stroke belongs to
            word_pos:  which word position in that batch item

        Returns: (N, d_model)
        """
        return hidden[batch_idx, word_pos]
