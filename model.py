"""
Spatial Language Model — Direction 3.

A transformer where every sequence position has a learnable 3D coordinate.
Attention has two independent energy costs, both carried over from the
spiking network's energy tracker:

  DISTANCE COST  (was: transmission cost per spike × distance)
    Each unit of attention weight paid to position j costs ||pos_i - pos_j||.
    Penalises *where* you attend — long-range connections are expensive.
    → energy_weight controls this term.

  FLOP COST  (was: charge_synapse_maintenance(n_out), charge_fire(n_targets))
    In the spiking network, a neuron with 50 outgoing synapses cost 50×
    more compute than one with 1. Here the equivalent is attention entropy:
      H = -Σ_j  w(i→j) · log w(i→j)
    High entropy = attention spread across many positions = many FLOPs.
    Low entropy = concentrated on 1-2 positions = sparse and cheap.
    Penalises *how many* things you attend to, regardless of distance.
    → flop_weight controls this term.

  REPULSION  (was: neurons cannot physically occupy the same location)
    Two neurons in 3D space cannot coincide. Without this, the model's
    degenerate solution is to collapse all positions to a single point,
    making every pairwise distance → 0 and eliminating the distance penalty
    entirely. Coulomb-style repulsion 1/d diverges as any two positions
    approach each other, preventing collapse.
    → repulsion_weight controls this term.

Combined loss = cross_entropy
              + energy_weight    × dist_energy
              + flop_weight      × flop_energy
              + repulsion_weight × repulsion_energy

Set energy/flop/repulsion weights all to 0.0 to recover a standard transformer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Multi-head self-attention with spatial distance penalty.

    Attention logit from position i to position j:
        logit(i→j) = (Q_i · K_j) / sqrt(d) - distance_penalty * ||pos_i - pos_j||

    This means:
      - Attending to a nearby position is cheap (small distance penalty)
      - Attending far away requires a strong Q·K signal to overcome the penalty
      - The network learns whether a long-range dependency is worth the cost

    Spatial positions are learned parameters, initialized in a small random ball.
    Gradient descent will push frequently-attending pairs closer together.
    """

    def __init__(self, d_model, n_heads, seq_len, d_space=3, distance_penalty=1.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.distance_penalty = distance_penalty
        # Maximum possible entropy for normalisation: log(T).
        # Stored so callers can interpret flop_energy as a fraction of worst-case.
        self.max_entropy = math.log(seq_len)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # Spatial positions: each of the seq_len positions gets a 3D coordinate.
        # Initialized tightly; gradient descent spreads them as needed.
        self.positions = nn.Parameter(torch.randn(seq_len, d_space) * 0.3)

        # Causal mask — positions can only attend to earlier positions (autoregressive)
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

        # Pairwise spatial distances between positions
        pos = self.positions[:T]                              # (T, d_space)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)           # (T, T, d_space)
        dists = torch.norm(diff, dim=-1)                     # (T, T)

        # Subtract distance penalty — long-range attention is harder to win
        attn_logits = attn_logits - self.distance_penalty * dists

        # Causal mask
        attn_logits = attn_logits.masked_fill(self.causal_mask[:T, :T], float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)        # (B, H, T, T)

        # ── Distance energy ───────────────────────────────────────────────
        # Expected distance of the attended position per query.
        # Penalises *where* you attend — long-range connections cost more.
        dist_energy = (attn_weights * dists).sum(dim=-1).mean()

        # ── FLOP energy (attention entropy) ───────────────────────────────
        # H = -Σ_j w(i→j)·log w(i→j)  ∈ [0, log(T)]
        # Uniform attention (all positions equally) → H = log(T) → max FLOPs.
        # Delta attention (one position only)       → H = 0      → min FLOPs.
        # This is the direct analog of charge_synapse_maintenance(n_out):
        # a neuron with many active synapses pays more than one with few.
        flop_energy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=-1).mean()

        # ── Repulsion energy ──────────────────────────────────────────────
        # Two neurons cannot physically occupy the same location.
        # Coulomb-style 1/d repulsion diverges as any two positions coincide,
        # preventing the degenerate solution of collapsing all positions to a
        # single point (which would zero out the distance penalty entirely).
        # Only off-diagonal pairs — a position doesn't repel itself.
        eye_mask = torch.eye(T, dtype=torch.bool, device=dists.device)
        off_diag_dists = dists[~eye_mask]                         # (T*(T-1),)
        repulsion_energy = (1.0 / (off_diag_dists + 1e-4)).mean()

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return (self.proj(out), dist_energy, flop_energy, repulsion_energy,
                attn_weights.detach(), dists.detach())


class SpatialBlock(nn.Module):
    """Transformer block: spatial attention + feedforward."""

    def __init__(self, d_model, n_heads, seq_len, d_space=3,
                 distance_penalty=1.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SpatialAttention(d_model, n_heads, seq_len, d_space, distance_penalty)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, dist_energy, flop_energy, repulsion_energy, weights, dists = self.attn(self.ln1(x))
        x = x + self.drop(attn_out)
        x = x + self.ffn(self.ln2(x))
        return x, dist_energy, flop_energy, repulsion_energy, weights, dists


class SpatialLanguageModel(nn.Module):
    """
    Character-level autoregressive language model with spatial attention.

    Architecture: token embedding + positional embedding → N spatial blocks → LM head.

    The novel part is in SpatialAttention: each sequence position has a learned
    3D coordinate, and attention logits are penalized by the distance between
    positions. The energy_weight hyperparameter controls how hard the model
    is pushed toward local attention.

    Set energy_weight=0.0 to get a standard transformer (ablation baseline).
    Set distance_penalty=0.0 to disable spatial structure (another ablation).
    """

    def __init__(
        self,
        vocab_size,
        seq_len=128,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_space=3,
        distance_penalty=1.0,
        energy_weight=0.01,
        flop_weight=0.1,
        repulsion_weight=0.01,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.energy_weight = energy_weight
        self.flop_weight = flop_weight
        self.repulsion_weight = repulsion_weight

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SpatialBlock(d_model, n_heads, seq_len, d_space, distance_penalty, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: input and output embeddings share weights (standard practice)
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

        total_dist_energy       = torch.tensor(0.0, device=idx.device)
        total_flop_energy       = torch.tensor(0.0, device=idx.device)
        total_repulsion_energy  = torch.tensor(0.0, device=idx.device)
        all_weights = []
        all_dists = []

        for block in self.blocks:
            x, dist_energy, flop_energy, repulsion_energy, weights, dists = block(x)
            total_dist_energy      = total_dist_energy      + dist_energy
            total_flop_energy      = total_flop_energy      + flop_energy
            total_repulsion_energy = total_repulsion_energy + repulsion_energy
            all_weights.append(weights)
            all_dists.append(dists)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = (ce
                    + self.energy_weight    * total_dist_energy
                    + self.flop_weight      * total_flop_energy
                    + self.repulsion_weight * total_repulsion_energy)

        return logits, loss, total_dist_energy, total_flop_energy, total_repulsion_energy, all_weights, all_dists

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len:]
            logits, _, _, _, _, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def get_spatial_positions(self, layer=0):
        """Return learned 3D positions for a given layer (numpy, detached)."""
        return self.blocks[layer].attn.positions.detach().cpu().numpy()

    def mean_attention_distance(self, all_weights, all_dists):
        """Avg distance of attended tokens — lower means more local attention."""
        total = 0.0
        for weights, dists in zip(all_weights, all_dists):
            total += (weights * dists).sum(dim=-1).mean().item()
        return total / len(all_weights)

    def mean_attention_entropy(self, all_weights):
        """
        Avg entropy of attention distributions — lower means sparser (fewer FLOPs).
        Ranges from 0 (delta, 1 connection) to log(T) (uniform, all connections).
        """
        total = 0.0
        for weights in all_weights:
            total += -(weights * torch.log(weights + 1e-9)).sum(dim=-1).mean().item()
        return total / len(all_weights)
