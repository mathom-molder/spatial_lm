"""
Spatial Language Model — Direction 3 (v3: token-space positions).

Each vocabulary token has a learnable 3D coordinate. Attention is penalised
by the Euclidean distance between the tokens being attended to/from, not by
sequence position.

    logit(i→j) = (Q_i · K_j) / sqrt(d) - distance_penalty * ||pos[tok_i] - pos[tok_j]||

Key difference from v1 (sequence positions) and v2 (per-head scales):

  v1 — positions indexed by sequence slot. Sequence slot 20 means different
       things in every batch example → noisy gradient → positions collapse.

  v2 — fixed |i-j| distances, learned per-head scale. Works, but purely
       positional — no information about which tokens are attending.

  v3 — positions indexed by vocabulary token. 'e' is always 'e'.
       Tokens that frequently attend to each other get pulled spatially
       close. This is the original spiking network intuition: neurons with
       stable identities cluster by co-activation.

The model learns a 3D map of its vocabulary. Characters with similar
roles in the language (vowels, consonants, punctuation, capitals) should
cluster together if the idea works.

Repulsion (Coulomb 1/d over all vocab pairs) prevents collapse.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Multi-head attention with token-space distance penalty.

    Receives pre-computed pairwise token distances (B, T, T) and subtracts
    distance_penalty * dist[i,j] from the logit for attending from position i
    (containing some token a) to position j (containing some token b).
    """

    def __init__(self, d_model, n_heads, seq_len, distance_penalty=1.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.distance_penalty = distance_penalty
        self.max_entropy = math.log(seq_len)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        )

    def forward(self, x, token_dists):
        """
        x:           (B, T, C)
        token_dists: (B, T, T) pairwise 3D distances between tokens
        """
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, dh)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, T, T)

        # Token-space distance penalty, broadcast over heads
        attn_logits = attn_logits - self.distance_penalty * token_dists.unsqueeze(1)

        attn_logits = attn_logits.masked_fill(self.causal_mask[:T, :T], float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, H, T, T)

        # Distance energy: expected token-space distance of attended token
        dist_energy = (attn_weights * token_dists.unsqueeze(1)).sum(dim=-1).mean()

        # FLOP energy (attention entropy)
        flop_energy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=-1).mean()

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), dist_energy, flop_energy, attn_weights.detach(), token_dists.detach()


class SpatialBlock(nn.Module):
    """Transformer block: token-space spatial attention + feedforward."""

    def __init__(self, d_model, n_heads, seq_len, distance_penalty=1.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SpatialAttention(d_model, n_heads, seq_len, distance_penalty)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, token_dists):
        attn_out, dist_energy, flop_energy, weights, dists = self.attn(self.ln1(x), token_dists)
        x = x + self.drop(attn_out)
        x = x + self.ffn(self.ln2(x))
        return x, dist_energy, flop_energy, weights, dists


class SpatialLanguageModel(nn.Module):
    """
    Character-level language model where each vocabulary token has a
    learnable 3D coordinate. Attention logits are penalised by the
    Euclidean distance between the attending and attended token's positions.

    After training, call get_token_positions() to see the learned 3D map
    of the vocabulary. Characters that frequently attend to each other
    should cluster together.
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
        energy_weight=0.0,
        flop_weight=0.0,
        repulsion_weight=0.01,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.energy_weight = energy_weight
        self.flop_weight = flop_weight
        self.repulsion_weight = repulsion_weight

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Learnable 3D position for each vocabulary token.
        # Initialized in a small ball; gradient descent will spread them
        # into a map where frequently co-attending tokens cluster together.
        self.token_positions = nn.Parameter(
            torch.randn(vocab_size, d_space) * 0.3
        )

        # Projects 3D token positions into embedding space so the model can
        # use spatial neighbourhood as part of its representation, not just
        # as an attention penalty.
        self.pos_proj = nn.Linear(d_space, d_model, bias=False)

        self.blocks = nn.ModuleList([
            SpatialBlock(d_model, n_heads, seq_len, distance_penalty, dropout)
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

    def _repulsion_energy(self):
        """Coulomb-style repulsion over all vocab pairs to prevent collapse."""
        pos = self.token_positions  # (V, d_space)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (V, V, d_space)
        dists = torch.norm(diff, dim=-1)  # (V, V)
        eye = torch.eye(self.vocab_size, dtype=torch.bool, device=pos.device)
        return (1.0 / (dists[~eye] + 1e-4)).mean()

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)

        # Look up 3D positions for each token in the sequence,
        # then compute all pairwise distances in one shot.
        seq_pos = self.token_positions[idx]                       # (B, T, d_space)
        diff = seq_pos.unsqueeze(2) - seq_pos.unsqueeze(1)        # (B, T, T, d_space)
        token_dists = torch.norm(diff, dim=-1)                    # (B, T, T)

        # Add projected positions to the embedding so the model can use
        # spatial neighbourhood as part of its representation.
        x = x + self.pos_proj(seq_pos)                           # (B, T, d_model)

        repulsion_energy = self._repulsion_energy()

        total_dist_energy = torch.tensor(0.0, device=idx.device)
        total_flop_energy = torch.tensor(0.0, device=idx.device)
        all_weights = []
        all_dists = []

        for block in self.blocks:
            x, dist_energy, flop_energy, weights, dists = block(x, token_dists)
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
                    + self.energy_weight    * total_dist_energy
                    + self.flop_weight      * total_flop_energy
                    + self.repulsion_weight * repulsion_energy)

        return logits, loss, total_dist_energy, total_flop_energy, repulsion_energy, all_weights, all_dists

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

    def get_token_positions(self):
        """Return learned 3D positions as (vocab_size, d_space) numpy array."""
        return self.token_positions.detach().cpu().numpy()

    def mean_attention_distance(self, all_weights, all_dists):
        total = 0.0
        for weights, dists in zip(all_weights, all_dists):
            total += (weights * dists.unsqueeze(1)).sum(dim=-1).mean().item()
        return total / len(all_weights)

    def mean_attention_entropy(self, all_weights):
        total = 0.0
        for weights in all_weights:
            total += -(weights * torch.log(weights + 1e-9)).sum(dim=-1).mean().item()
        return total / len(all_weights)
