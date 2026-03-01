"""
Generate text and visualize learned distance scales.

Usage:
    python generate.py                             # generate 200 chars
    python generate.py --prompt "KING:"            # seed with a prompt
    python generate.py --temperature 0.8           # less random
    python generate.py --visualize                 # show scale heatmap + attention
    python generate.py --prompt "To be" --visualize --temperature 0.7
"""

import argparse
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # headless-safe; plt.show() is a no-op
import matplotlib.pyplot as plt
from model import SpatialLanguageModel


def load_model(checkpoint='checkpoint.pt', vocab_file='vocab.pkl'):
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = SpatialLanguageModel(**ckpt['hparams'])
    # Strip _orig_mod. prefix added by torch.compile when saving
    state_dict = ckpt['model_state']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, vocab


def visualize_distance_scales(model, save='distance_scales.png'):
    """
    Plot the learned per-head distance scales as a heatmap.

    Rows = layers, columns = heads.
    Warm colours (positive) = head prefers local attention.
    Cool colours (negative) = head prefers long-range attention.
    Near-zero = head behaves like standard attention.

    This is the key diagnostic: did the model learn to specialise heads?
    """
    scales = model.get_distance_scales()  # (n_layers, n_heads)
    n_layers, n_heads = scales.shape

    fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.8), max(4, n_layers * 0.8)))
    vmax = max(abs(scales.max()), abs(scales.min()), 0.01)
    im = ax.imshow(scales, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    # Annotate each cell with its value
    for i in range(n_layers):
        for j in range(n_heads):
            ax.text(j, i, f'{scales[i, j]:.3f}', ha='center', va='center',
                    fontsize=8, color='black')

    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels([f'H{h}' for h in range(n_heads)])
    ax.set_yticklabels([f'L{l}' for l in range(n_layers)])
    plt.colorbar(im, ax=ax, label='distance_scale (+ = local, − = global)')
    ax.set_title('Learned per-head distance scales\n'
                 'Red = local preference  |  Blue = global preference  |  White = standard')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    print(f"Saved → {save}")
    plt.show()

    # Print summary
    print(f"\nDistance scale summary:")
    print(f"  min={scales.min():.4f}  max={scales.max():.4f}  mean={scales.mean():.4f}")
    print(f"  Local heads  (scale > 0.2): {(scales > 0.2).sum()}/{scales.size}")
    print(f"  Global heads (scale < 0.0): {(scales < 0.0).sum()}/{scales.size}")


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
        _, _, _, _, all_weights, all_dists = model(idx)

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
            axes[0, col].set_title(f'L{l}H{h}\nscale={model.blocks[l].attn.distance_scales[h].item():.3f}',
                                   fontsize=7)
            axes[0, col].set_xticks(range(T)); axes[0, col].set_xticklabels(labels, fontsize=5, rotation=90)
            axes[0, col].set_yticks(range(T)); axes[0, col].set_yticklabels(labels, fontsize=5)

            # Energy (attention * distance)
            axes[1, col].imshow(energy_map, cmap='Reds', vmin=0)
            axes[1, col].set_title(f'L{l}H{h}\nenergy (w×dist)', fontsize=7)
            axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

            col += 1

    fig.suptitle(f'Attention and energy maps for: "{text[:30]}..."')
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
        visualize_distance_scales(model)
        visualize_attention(model, text, vocab)


if __name__ == '__main__':
    main()
