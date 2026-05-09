"""Transformer-based stroke generator with MDN and pen-state outputs.

This is the architecture committed in part-a/docs/PLAN.md: a single
causal transformer over flat token sequence, with multiple per-position
output heads (MDN over Δx,Δy + categorical over pen-state). For Track 2
we use it unconditionally — train on QuickDraw cat data, no canvas or
text conditioning yet, just validate that the modeling stack can learn
stroke dynamics.

Per-step prediction targets (from the Sketch-RNN paper):
  - mixture of M bivariate Gaussians over (Δx, Δy)
  - 3-way categorical over pen-state {down, up, end}

Total per-step output dim = 6M + 3 (5M Gaussian params + M mixture
weights + 3 pen logits).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StrokeTransformer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 1024,
                 num_components: int = 20, max_len: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.M = num_components
        self.max_len = max_len

        # 5D input → d_model
        self.input_proj = nn.Linear(5, d_model)
        # learned positional embeddings (simple, fine at this scale)
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm_out = nn.LayerNorm(d_model)

        # output head: 6M + 3 dims
        self.out_head = nn.Linear(d_model, 6 * num_components + 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, T, 5) input sequence. Returns (gmm_params, mix_logits,
        pen_logits)."""
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.input_proj(x) + self.pos_emb(pos)[None]

        # causal mask: position i may attend to 0..i
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool),
                          diagonal=1)
        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.norm_out(h)

        out = self.out_head(h)  # (B, T, 6M+3)
        M = self.M
        # split: first 5M are Gaussian params (mu_x, mu_y, log_sigma_x,
        # log_sigma_y, raw_rho) per component; next M are mixture logits;
        # last 3 are pen-state logits.
        gauss = out[..., : 5 * M].reshape(B, T, M, 5)
        mix_logits = out[..., 5 * M : 6 * M]
        pen_logits = out[..., 6 * M :]
        return gauss, mix_logits, pen_logits


def gmm_neg_log_likelihood(
    gauss: torch.Tensor,         # (B, T, M, 5)
    mix_logits: torch.Tensor,    # (B, T, M)
    target_xy: torch.Tensor,     # (B, T, 2)
    mask: torch.Tensor,          # (B, T) — 1 where loss applies
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mixture density log-likelihood for bivariate Gaussian targets.

    Standard formula (cf. Bishop 1994; Sketch-RNN 2017 sec. 3.2)."""
    mu_x = gauss[..., 0]
    mu_y = gauss[..., 1]
    sigma_x = gauss[..., 2].exp() + eps
    sigma_y = gauss[..., 3].exp() + eps
    rho = torch.tanh(gauss[..., 4]) * (1 - eps)

    tx = target_xy[..., 0:1]   # (B, T, 1)
    ty = target_xy[..., 1:2]
    z_x = (tx - mu_x) / sigma_x
    z_y = (ty - mu_y) / sigma_y
    z = z_x ** 2 + z_y ** 2 - 2 * rho * z_x * z_y
    one_minus_rho2 = 1 - rho ** 2
    log_norm = math.log(2 * math.pi) + sigma_x.log() + sigma_y.log() + 0.5 * one_minus_rho2.log()
    log_p_per_comp = -z / (2 * one_minus_rho2) - log_norm

    log_mix = F.log_softmax(mix_logits, dim=-1)  # (B, T, M)
    log_p = torch.logsumexp(log_mix + log_p_per_comp, dim=-1)  # (B, T)

    return -(log_p * mask).sum() / mask.sum().clamp_min(1.0)


def pen_cross_entropy(
    pen_logits: torch.Tensor,   # (B, T, 3)
    target_pen: torch.Tensor,   # (B, T, 3) one-hot
    mask: torch.Tensor,         # (B, T)
) -> torch.Tensor:
    log_p = F.log_softmax(pen_logits, dim=-1)
    target_idx = target_pen.argmax(dim=-1)
    nll = F.nll_loss(
        log_p.reshape(-1, 3), target_idx.reshape(-1),
        reduction="none",
    ).reshape_as(mask)
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


def loss_fn(model: StrokeTransformer,
            seqs: torch.Tensor,        # (B, T+1, 5)
            lens: torch.Tensor,        # (B,)
            ) -> tuple[torch.Tensor, dict]:
    """Teacher-forced loss. Input = seqs[:, :-1], target = seqs[:, 1:].
    The loss includes positions 0..n (the trailing end-of-sketch token).
    Padding past n+1 is masked out."""
    inp = seqs[:, :-1]              # (B, T, 5)
    tgt = seqs[:, 1:]               # (B, T, 5)
    B, T, _ = inp.shape

    gauss, mix_logits, pen_logits = model(inp)

    target_xy = tgt[..., :2]
    target_pen = tgt[..., 2:5]

    # Loss mask: positions where the target is a real step OR the
    # synthesized end token. = positions [0..n] inclusive (n is len).
    # In tgt indexing, position i corresponds to the (i+1)-th step in
    # the original seq. We want loss where i <= n  (so positions 0..n).
    arange = torch.arange(T, device=seqs.device).unsqueeze(0)   # (1, T)
    mask = (arange <= lens.unsqueeze(1)).float()                # (B, T)

    # GMM loss only on pen-down/up steps (not on the end-of-sketch
    # position, which has no real (Δx, Δy)).
    is_real = (target_pen[..., 2] < 0.5).float()
    gmm_mask = mask * is_real
    l_gmm = gmm_neg_log_likelihood(gauss, mix_logits, target_xy, gmm_mask)

    l_pen = pen_cross_entropy(pen_logits, target_pen, mask)

    loss = l_gmm + l_pen
    return loss, {"loss": loss.item(), "gmm": l_gmm.item(), "pen": l_pen.item()}


@torch.no_grad()
def sample(model: StrokeTransformer,
           max_steps: int = 250,
           temperature: float = 0.4,
           device: str = "cpu") -> list[tuple[float, float, int]]:
    """Generate a single stroke sequence. Returns list of
    (Δx, Δy, pen_state) — pen_state ∈ {0=down, 1=up, 2=end}."""
    model.eval()
    seq = torch.zeros((1, 1, 5), device=device)
    seq[0, 0, 2] = 1.0  # start with pen-down marker
    out_pts: list[tuple[float, float, int]] = []

    for _ in range(max_steps):
        gauss, mix_logits, pen_logits = model(seq)
        last_g = gauss[0, -1]               # (M, 5)
        last_m = mix_logits[0, -1]          # (M,)
        last_p = pen_logits[0, -1]          # (3,)

        # sample a mixture component
        log_mix = F.log_softmax(last_m / max(temperature, 1e-6), dim=-1)
        comp = torch.distributions.Categorical(logits=log_mix).sample().item()
        mu_x, mu_y = last_g[comp, 0].item(), last_g[comp, 1].item()
        sigma_x = last_g[comp, 2].exp().item() * (temperature ** 0.5)
        sigma_y = last_g[comp, 3].exp().item() * (temperature ** 0.5)
        rho = torch.tanh(last_g[comp, 4]).item()

        # bivariate normal sample
        z1, z2 = torch.randn(2).tolist()
        dx = mu_x + sigma_x * z1
        dy = mu_y + sigma_y * (rho * z1 + (1 - rho ** 2) ** 0.5 * z2)

        # pen-state sample
        pen_logits_t = last_p / max(temperature, 1e-6)
        pen = torch.distributions.Categorical(logits=pen_logits_t).sample().item()

        out_pts.append((float(dx), float(dy), int(pen)))
        if pen == 2:
            break

        # build next input row
        next_row = torch.zeros((1, 1, 5), device=device)
        next_row[0, 0, 0] = dx
        next_row[0, 0, 1] = dy
        next_row[0, 0, 2 + pen] = 1.0
        seq = torch.cat([seq, next_row], dim=1)
        # cap context length
        if seq.shape[1] >= model.max_len:
            seq = seq[:, -model.max_len :]

    return out_pts
