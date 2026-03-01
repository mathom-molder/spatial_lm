# Spatial Language Model — Direction 3

A character-level transformer where each sequence position has a **learnable 3D
spatial coordinate**. Attention between positions is penalized by their Euclidean
distance, so long-range dependencies must overcome a metabolic cost to form.

This directly continues the ideas from `spiking_assembly/`:

| Spiking network | Spatial LM |
|---|---|
| `distance_penalty` on initial weights | `distance_penalty` subtracted from attention logits |
| FLOP budget / `compete_resources` | `energy_weight` regularization in training loss |
| STDP (local, unsupervised) | Backpropagation (global, supervised) |
| Static topology after initialization | Learned spatial positions (gradient-driven) |
| ~100 neurons | ~500K parameters |

## Quick start

```bash
# 1. Get training text (tiny Shakespeare, ~1MB, trains in ~20 min on CPU)
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 2. Train
python train.py

# 3. Generate text
python generate.py --prompt "HAMLET:" --temperature 0.8

# 4. Visualize what the network learned spatially
python generate.py --visualize
```

## What to watch during training

The training output shows four columns:

```
    Step  Train loss   Val loss    Energy  Avg dist
────────────────────────────────────────────────────
     500      2.1234     2.1891    0.8234    0.9201
    1000      1.9812     2.0344    0.6123    0.7810
```

- **Val loss** going down = the model is learning to predict text
- **Energy** going down = the model is learning to use shorter-range attention
- **Avg dist** going down = spatial positions are clustering for locality

If energy stays high, the model is paying a metabolic cost for long-range
attention — which means those long-range dependencies genuinely matter for
the task.

## Ablations to try

```bash
# Standard transformer (no spatial penalty) — baseline comparison
python train.py --distance_penalty 0.0 --energy_weight 0.0

# Strong locality pressure — forces short-range attention
python train.py --distance_penalty 3.0 --energy_weight 0.05

# Weaker energy constraint — allows long-range but still tracks cost
python train.py --distance_penalty 1.0 --energy_weight 0.001

# Higher-dimensional spatial embedding
python train.py --d_space 8
```

## What the visualizations show

**`spatial_positions.png`** — the learned 3D positions for each sequence position in
each layer. Positions that frequently attend to each other should cluster. If the
model learns mostly local attention, you'll see a smooth path through 3D space
(nearby sequence positions near each other). If it learns to attend globally,
positions spread out.

**`attention_maps.png`** — standard attention weight matrices (top row) and
attention × distance maps (bottom row). The energy maps show *where* the
metabolic cost is being spent. Bright cells in the energy map are long-range
dependencies the model decided were worth it.

**`distance_dist.png`** — histogram of pairwise distances between spatial positions
in each layer. Watch this shift over training: early on, distances are random
(~N(0, 0.3)). After training, you should see bimodal structure — a cluster of
nearby positions (local circuit) and some farther ones (global context).

## Architecture overview

```
Input tokens
    ↓
Token embedding + positional embedding
    ↓
[SpatialBlock × N_LAYERS]
    │  LayerNorm
    │  SpatialAttention:
    │    QKV projection
    │    logit(i→j) = QK/√d − distance_penalty × ||pos_i − pos_j||
    │    softmax → attention weights
    │    energy = Σ (weight × distance)
    │  Residual
    │  LayerNorm
    │  FFN (4× expansion, GELU)
    │  Residual
    ↓
LayerNorm → LM head → logits
    ↓
Loss = cross_entropy + energy_weight × total_energy
```

## Interesting questions to investigate

1. **Does spatial structure actually help?** Compare val loss with and without
   `distance_penalty`. Does the spatial prior act as a useful inductive bias?

2. **Where does long-range attention survive?** Look at the energy maps — which
   layer and head is paying the highest metabolic cost? What linguistic structure
   might it be tracking?

3. **Do layers specialize?** In standard transformers, early layers attend locally
   and late layers attend globally. Does the spatial penalty reinforce this, or
   does it collapse all layers to local?

4. **What happens at the boundary?** Position 0 can only attend to itself.
   Position 127 can attend to 127 others. Does this asymmetry show up in the
   learned spatial positions?
