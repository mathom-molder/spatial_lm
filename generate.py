"""
Generate text and visualize the learned token-space positions.

Usage:
    python generate.py                             # generate 200 chars
    python generate.py --prompt "KING:"            # seed with a prompt
    python generate.py --temperature 0.8           # less random
    python generate.py --visualize                 # show token map + attention
    python generate.py --prompt "To be" --visualize --temperature 0.7
"""

import argparse
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from model import SpatialLanguageModel


def load_model(checkpoint='checkpoint.pt', vocab_file='vocab.pkl'):
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = SpatialLanguageModel(**ckpt['hparams'])
    state_dict = ckpt['model_state']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, vocab


def visualize_token_positions(model, vocab, save='token_positions.png'):
    """
    Plot the learned 3D positions for each vocabulary token.

    Characters that frequently attend to each other should cluster together.
    If the model learned meaningful linguistic structure, expect:
      - Vowels to cluster (they follow consonants predictably)
      - Uppercase letters to cluster (they follow newlines/spaces)
      - Punctuation to form its own group
      - Digits to cluster

    Colour key: blue=uppercase, orange=lowercase, green=digit, red=other
    """
    itos = vocab['itos']
    pos = model.get_token_positions()  # (vocab_size, 3)
    vocab_size = pos.shape[0]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    color_map = {'upper': 'royalblue', 'lower': 'darkorange',
                 'digit': 'green',     'other': 'crimson'}
    colors, labels = [], []
    for i in range(vocab_size):
        c = itos[i]
        if c.isupper():
            colors.append(color_map['upper'])
        elif c.islower():
            colors.append(color_map['lower'])
        elif c.isdigit():
            colors.append(color_map['digit'])
        else:
            colors.append(color_map['other'])
        labels.append(repr(c) if c in (' ', '\n', '\t', '\r') else c)

    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=80, alpha=0.8, depthshade=True)

    for i in range(vocab_size):
        ax.text(pos[i, 0], pos[i, 1], pos[i, 2], labels[i], fontsize=7, alpha=0.85)

    ax.set_title('Learned 3D token positions\n'
                 'Blue=uppercase  Orange=lowercase  Green=digit  Red=punctuation/space')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    print(f"Saved → {save}")

    # Report pairwise distances for interesting character pairs
    stoi = vocab['stoi']
    print("\nPairwise token-space distances:")
    pairs = [
        ('t', 'h'), ('i', 'n'), ('e', 'r'), ('a', 'n'),  # common bigrams
        ('e', 'a'), ('e', 'i'), ('a', 'o'),               # vowel pairs
        (' ', '\n'), ('.', ','), ('!', '?'),               # punctuation pairs
    ]
    for a, b in pairs:
        if a in stoi and b in stoi:
            ia, ib = stoi[a], stoi[b]
            d = np.linalg.norm(pos[ia] - pos[ib])
            print(f"  {a!r} ↔ {b!r}: {d:.3f}")


def visualize_attention(model, text, vocab, save='attention_maps.png'):
    """
    Show attention weight matrices and token-distance-weighted attention.

    Top row: raw attention weights
    Bottom row: attention × token-distance (metabolic cost per step)
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
        # all_dists[l] is (B, T, T); B=1 here
        dists = all_dists[l][0].cpu().numpy()[:T, :T]
        for h in range(n_heads):
            w = all_weights[l][0, h, :T, :T].cpu().numpy()
            energy_map = w * dists

            axes[0, col].imshow(w, cmap='Blues', vmin=0, vmax=w.max())
            axes[0, col].set_title(f'L{l}H{h}\nattention', fontsize=7)
            axes[0, col].set_xticks(range(T))
            axes[0, col].set_xticklabels(labels, fontsize=5, rotation=90)
            axes[0, col].set_yticks(range(T))
            axes[0, col].set_yticklabels(labels, fontsize=5)

            axes[1, col].imshow(energy_map, cmap='Reds', vmin=0)
            axes[1, col].set_title(f'L{l}H{h}\nenergy (w×dist)', fontsize=7)
            axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

            col += 1

    fig.suptitle(f'Attention and energy maps for: "{text[:30]}..."')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    print(f"Saved → {save}")


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
        visualize_token_positions(model, vocab)
        visualize_attention(model, text, vocab)


if __name__ == '__main__':
    main()
