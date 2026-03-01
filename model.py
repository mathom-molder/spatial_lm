"""
Spatial Language Model — Direction 3 (v2: per-head learned distance scales).

Each attention head has a single learned scalar (distance_scale) that controls
how much it penalises attending to distant sequence positions:

    logit(i→j) = (Q_i · K_j) / sqrt(d) - distance_scale_h * |i - j|

The distances are fixed (|i - j|, the integer sequence gap) — no learned 3D
positions, no repulsion term. The model learns *how local* each head should be
purely from the language modelling objective.

If distance_scale_h is large and positive:  head attends locally.
If distance_scale_h is near zero:           head behaves like standard attention.
If distance_scale_h is negative:            head prefers long-range attention.

The key diagnostic is the learned scale heatmap (n_layers × n_heads) after
training. If the idea works, heads will spread across a range of scales — some
local, some global — and this specialisation should improve val loss over a
standard transformer.

Set energy_weight=0, flop_weight=0 (defaults) to let scales learn from CE alone.
Set energy_weight > 0 to additionally penalise long-range attention in the loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Multi-head attention with per-head learned distance scaling.

    The attention logit from position i to position j is:
        logit(i→j) = (Q_i · K_j) / sqrt(d) - distance_scale_h * |i - j|

    distance_scale_h is a learned scalar per head, initialized to
    distance_scale_init. Gradient descent will push it up (more local) or
    down toward zero/negative (more global) based on what helps the task.

    Distances are fixed: dist[i,j] = |i - j|, stored as a non-trainable buffer.
    """

    def __init__(self, d_model, n_heads, seq_len, distance_scale_init=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_entropy = math.log(seq_len)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # Per-head learned distance scale — one scalar per head.
        # Initialized small so training starts close to a standard transformer.
        self.distance_scales = nn.Parameter(
            torch.full((n_heads,), distance_scale_init)
        )

        # Fixed distance matrix: dist[i,j] = |i - j|
        i = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)
        self.register_buffer('seq_dists', torch.abs(i - j))  # (seq_len, seq_len)

        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        )

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, dh)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product logits
        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, T, T)

        # Fixed sequence distances
        dists = self.seq_dists[:T, :T]  # (T, T)

        # Per-head distance penalty: subtract scale_h * |i-j| from each head's logits
        scales = self.distance_scales.view(1, self.n_heads, 1, 1)  # broadcast over B, T, T
        attn_logits = attn_logits - scales * dists

        # Causal mask
        attn_logits = attn_logits.masked_fill(self.causal_mask[:T, :T], float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, H, T, T)

        # ── Distance energy ────────────────────────────────────────────────
        # Average sequence distance of the attended token per query.
        # Low = local attention; high = global attention.
        dist_energy = (attn_weights * dists).sum(dim=-1).mean()

        # ── FLOP energy (attention entropy) ───────────────────────────────
        # H = -Σ_j w(i→j)·log w(i→j). Higher = more diffuse = more FLOPs.
        flop_energy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=-1).mean()

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), dist_energy, flop_energy, attn_weights.detach(), dists.detach()


class SpatialBlock(nn.Module):
    """Transformer block: spatial attention + feedforward."""

    def __init__(self, d_model, n_heads, seq_len, distance_scale_init=0.1, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SpatialAttention(d_model, n_heads, seq_len, distance_scale_init)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, dist_energy, flop_energy, weights, dists = self.attn(self.ln1(x))
        x = x + self.drop(attn_out)
        x = x + self.ffn(self.ln2(x))
        return x, dist_energy, flop_energy, weights, dists


class SpatialLanguageModel(nn.Module):
    """
    Character-level autoregressive language model with per-head distance scaling.

    Each head in each layer learns a scalar that controls its locality preference.
    After training, read out model.get_distance_scales() to see what the model
    learned: which heads went local, which stayed global.

    Set energy_weight=0, flop_weight=0 to train with no explicit regularisation
    (scales learn purely from cross-entropy). This is the cleanest experiment.
    """

    def __init__(
        self,
        vocab_size,
        seq_len=128,
        d_model=128,
        n_heads=4,
        n_layers=4,
        distance_scale_init=0.1,
        energy_weight=0.0,
        flop_weight=0.0,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.energy_weight = energy_weight
        self.flop_weight = flop_weight

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SpatialBlock(d_model, n_heads, seq_len, distance_scale_init, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.token_emb.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)

        total_dist_energy  = torch.tensor(0.0, device=idx.device)
        total_flop_energy  = torch.tensor(0.0, device=idx.device)
        all_weights = []
        all_dists = []

        for block in self.blocks:
            x, dist_energy, flop_energy, weights, dists = block(x)
            total_dist_energy = total_dist_energy + dist_energy
            total_flop_energy = total_flop_energy + flop_energy
            all_weights.append(weights)
            all_dists.append(dists)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = (ce
                    + self.energy_weight * total_dist_energy
                    + self.flop_weight   * total_flop_energy)

        return logits, loss, total_dist_energy, total_flop_energy, all_weights, all_dists

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len:]
            logits, _, _, _, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def get_distance_scales(self):
        """
        Return learned distance scales as a (n_layers, n_heads) numpy array.
        Positive = local preference, near-zero = global, negative = anti-local.
        """
        import numpy as np
        scales = []
        for block in self.blocks:
            scales.append(block.attn.distance_scales.detach().cpu().numpy())
        return np.stack(scales)  # (n_layers, n_heads)

    def mean_attention_distance(self, all_weights, all_dists):
        """Average sequence distance of attended tokens across all layers/heads."""
        total = 0.0
        for weights, dists in zip(all_weights, all_dists):
            total += (weights * dists).sum(dim=-1).mean().item()
        return total / len(all_weights)

    def mean_attention_entropy(self, all_weights):
        """Average entropy of attention distributions."""
        total = 0.0
        for weights in all_weights:
            total += -(weights * torch.log(weights + 1e-9)).sum(dim=-1).mean().item()
        return total / len(all_weights)
