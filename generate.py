"""
Generate text and visualize learned spatial structure.

Usage:
    python generate.py                             # generate 200 chars
    python generate.py --prompt "KING:"            # seed with a prompt
    python generate.py --temperature 0.8           # less random
    python generate.py --visualize                 # show spatial positions + attention
    python generate.py --prompt "To be" --visualize --temperature 0.7
"""

import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from model import SpatialLanguageModel


def load_model(checkpoint='checkpoint.pt', vocab_file='vocab.pkl'):
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = SpatialLanguageModel(**ckpt['hparams'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, vocab


def visualize_spatial(model, save='spatial_positions.png'):
    """
    Plot the learned 3D spatial positions for each sequence position in each layer.

    Positions that frequently attend to each other get pulled close by the gradient.
    You should see local structure emerge: adjacent positions cluster together,
    while positions that need long-range context sit farther apart.
    """
    n_layers = len(model.blocks)
    fig = plt.figure(figsize=(6 * n_layers, 5))

    for layer in range(n_layers):
        pos = model.get_spatial_positions(layer)  # (seq_len, 3)
        seq_len = pos.shape[0]

        ax = fig.add_subplot(1, n_layers, layer + 1, projection='3d')
        colors = plt.cm.plasma(np.linspace(0, 1, seq_len))
        sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=25, alpha=0.8)

        # Label a few positions so you can see the ordering
        step = max(1, seq_len // 10)
        for i in range(0, seq_len, step):
            ax.text(pos[i, 0], pos[i, 1], pos[i, 2], str(i), fontsize=6, alpha=0.7)

        # Draw lines between consecutive positions to show the sequence path
        for i in range(seq_len - 1):
            ax.plot([pos[i, 0], pos[i+1, 0]],
                    [pos[i, 1], pos[i+1, 1]],
                    [pos[i, 2], pos[i+1, 2]],
                    'gray', alpha=0.15, linewidth=0.5)

        ax.set_title(f'Layer {layer} — learned positions\n(plasma: early→late)')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    print(f"Saved → {save}")
    plt.show()


def visualize_attention(model, text, vocab, save='attention_maps.png'):
    """
    Show attention weight matrices and distance-weighted attention for a text snippet.

    Top row: raw attention weights (what position attends to what)
    Bottom row: attention * distance (the energy spent per attention step)
    """
    stoi = vocab['stoi']
    itos = vocab['itos']
    T = min(len(text), model.seq_len)

    tokens = [stoi.get(c, 0) for c in text[:T]]
    idx = torch.tensor(tokens).unsqueeze(0)

    with torch.no_grad():
        _, _, _, _, _, all_weights, all_dists = model(idx)

    n_layers = len(all_weights)
    n_heads = all_weights[0].shape[1]
    labels = [itos.get(t, '?') for t in tokens]

    fig, axes = plt.subplots(2, n_layers * n_heads,
                              figsize=(3 * n_layers * n_heads, 6))
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    col = 0
    for l in range(n_layers):
        dists = all_dists[l].cpu().numpy()[:T, :T]
        for h in range(n_heads):
            w = all_weights[l][0, h, :T, :T].cpu().numpy()
            energy_map = w * dists

            # Raw attention
            axes[0, col].imshow(w, cmap='Blues', vmin=0, vmax=w.max())
            axes[0, col].set_title(f'L{l}H{h}\nattention', fontsize=7)
            axes[0, col].set_xticks(range(T)); axes[0, col].set_xticklabels(labels, fontsize=5, rotation=90)
            axes[0, col].set_yticks(range(T)); axes[0, col].set_yticklabels(labels, fontsize=5)

            # Energy (attention * distance) — shows where metabolic cost is paid
            axes[1, col].imshow(energy_map, cmap='Reds', vmin=0)
            axes[1, col].set_title(f'L{l}H{h}\nenergy (w×dist)', fontsize=7)
            axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

            col += 1

    fig.suptitle(f'Attention and energy maps for: "{text[:30]}..."')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    print(f"Saved → {save}")
    plt.show()


def visualize_distance_distribution(model, save='distance_dist.png'):
    """
    Plot the distribution of learned pairwise distances in each layer.

    If distance_penalty is working, you should see the distribution shift
    over training — positions cluster for local attention, separate for global.
    """
    n_layers = len(model.blocks)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for layer in range(n_layers):
        pos = model.get_spatial_positions(layer)
        # All pairwise distances
        diff = pos[:, None, :] - pos[None, :, :]
        dists = np.linalg.norm(diff, axis=-1).flatten()
        dists = dists[dists > 0]  # remove self-distances

        axes[layer].hist(dists, bins=40, color='steelblue', alpha=0.8)
        axes[layer].axvline(dists.mean(), color='red', linestyle='--',
                            label=f'mean={dists.mean():.2f}')
        axes[layer].set_title(f'Layer {layer} pairwise distances')
        axes[layer].set_xlabel('distance'); axes[layer].set_ylabel('count')
        axes[layer].legend()

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    print(f"Saved → {save}")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prompt',      type=str,   default='\n')
    p.add_argument('--max_tokens',  type=int,   default=200)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--visualize',   action='store_true')
    p.add_argument('--checkpoint',  type=str,   default='checkpoint.pt')
    args = p.parse_args()

    model, vocab = load_model(args.checkpoint)
    stoi = vocab['stoi']
    itos = vocab['itos']

    prompt_tokens = torch.tensor([[stoi.get(c, 0) for c in args.prompt]])
    generated = model.generate(prompt_tokens, args.max_tokens, args.temperature)
    text = ''.join(itos[i.item()] for i in generated[0])

    print('\n' + '─' * 60)
    print(text)
    print('─' * 60)

    if args.visualize:
        visualize_spatial(model)
        visualize_attention(model, text, vocab)
        visualize_distance_distribution(model)


if __name__ == '__main__':
    main()
