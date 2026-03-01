"""
Train the Spatial Language Model on a text corpus.

Quick start (CPU, small model — for testing):
    python train.py

GPU run (3090 or similar — recommended settings):
    python train.py --d_model 512 --n_heads 8 --n_layers 8 --seq_len 256 \
                    --batch_size 128 --max_steps 50000

Ablations:
    # Standard transformer baseline (no spatial costs)
    python train.py --distance_penalty 0 --energy_weight 0 --flop_weight 0 --repulsion_weight 0

    # Distance + repulsion only (no FLOP pressure)
    python train.py --flop_weight 0

    # See effect of repulsion alone
    python train.py --energy_weight 0 --flop_weight 0 --repulsion_weight 0.1
"""

import os
import math
import time
import pickle
import argparse
import numpy as np
import torch
from model import SpatialLanguageModel


# ── CPU defaults (safe on any machine) ──────────────────────────────────────
SEQ_LEN          = 128
BATCH_SIZE       = 32
D_MODEL          = 128
N_HEADS          = 4
N_LAYERS         = 4
D_SPACE          = 3
DISTANCE_PENALTY = 1.0
ENERGY_WEIGHT    = 0.01
FLOP_WEIGHT      = 0.1
REPULSION_WEIGHT = 0.01   # Coulomb-style 1/d repulsion — prevents position collapse
DROPOUT          = 0.1
LEARNING_RATE    = 3e-4
MAX_STEPS        = 10_000
EVAL_INTERVAL    = 500
EVAL_BATCHES     = 20
DATA_FILE        = 'data/input.txt'
CHECKPOINT_FILE  = 'checkpoint.pt'
VOCAB_FILE       = 'vocab.pkl'

# ── Recommended GPU settings (uncomment and paste as --args, or edit above) ─
# --d_model 512 --n_heads 8 --n_layers 8 --seq_len 256
# --batch_size 128 --max_steps 50000 --eval_interval 1000
# (trains in ~40 min on a 3090, ~25M params)
# ────────────────────────────────────────────────────────────────────────────


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    return data[:split], data[split:], len(chars), stoi, itos


def get_batch(data, seq_len, batch_size, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, seq_len, batch_size, device, dtype):
    model.eval()
    results = {}
    for name, data in [('train', train_data), ('val', val_data)]:
        losses, dist_es, flop_es, rep_es = [], [], [], []
        for _ in range(EVAL_BATCHES):
            x, y = get_batch(data, seq_len, batch_size, device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == 'cuda')):
                _, loss, dist_e, flop_e, rep_e, _, _ = model(x, y)
            losses.append(loss.item())
            dist_es.append(dist_e.item())
            flop_es.append(flop_e.item())
            rep_es.append(rep_e.item())
        results[name] = (np.mean(losses), np.mean(dist_es), np.mean(flop_es), np.mean(rep_es))
    model.train()
    return results


def cosine_schedule(step, max_steps, warmup=200, lr_min_ratio=0.1):
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / (max_steps - warmup)
    return lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # bfloat16 is numerically stable and fast on Ampere+ (3090 is Ampere)
    dtype  = torch.bfloat16 if device == 'cuda' else torch.float32
    print(f"Device: {device}  |  dtype: {dtype}")

    if not os.path.exists(args.data):
        print(f"\nNo data file at '{args.data}'.")
        print("  curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        return

    train_data, val_data, vocab_size, stoi, itos = load_data(args.data)
    print(f"Vocab: {vocab_size} chars | Train: {len(train_data):,} | Val: {len(val_data):,}")

    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump({'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}, f)

    model = SpatialLanguageModel(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_space=args.d_space,
        distance_penalty=args.distance_penalty,
        energy_weight=args.energy_weight,
        flop_weight=args.flop_weight,
        repulsion_weight=args.repulsion_weight,
        dropout=args.dropout,
    ).to(device)

    # torch.compile — big free speedup on PyTorch 2+ with CUDA
    if device == 'cuda' and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"distance_penalty={args.distance_penalty}  energy_weight={args.energy_weight}  "
          f"flop_weight={args.flop_weight}  repulsion_weight={args.repulsion_weight}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_schedule(step, args.max_steps)
    )
    # GradScaler only needed for float16, not bfloat16
    scaler = torch.cuda.GradScaler(enabled=(device == 'cuda' and dtype == torch.float16))

    start_step = 0
    if args.resume and os.path.exists(CHECKPOINT_FILE):
        ckpt = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_step = ckpt['step'] + 1
        print(f"Resumed from step {start_step}")

    print(f"{'Step':>8}  {'Train':>8}  {'Val':>8}  {'DistE':>7}  {'FlopE':>7}  {'RepE':>8}  {'AvgDist':>8}  {'Entropy':>8}")
    print('─' * 82)

    t0 = time.time()
    for step in range(start_step, args.max_steps):
        x, y = get_batch(train_data, args.seq_len, args.batch_size, device)

        with torch.autocast(device_type=device, dtype=dtype, enabled=(device == 'cuda')):
            _, loss, dist_e, flop_e, rep_e, all_weights, all_dists = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if (step + 1) % args.eval_interval == 0:
            metrics    = estimate_loss(model, train_data, val_data,
                                       args.seq_len, args.batch_size, device, dtype)
            avg_dist    = model.mean_attention_distance(all_weights, all_dists)
            avg_entropy = model.mean_attention_entropy(all_weights)
            t_loss, t_dist_e, t_flop_e, t_rep_e = metrics['train']
            v_loss = metrics['val'][0]
            elapsed = time.time() - t0
            print(f"{step+1:>8}  {t_loss:>8.4f}  {v_loss:>8.4f}  "
                  f"{t_dist_e:>7.4f}  {t_flop_e:>7.4f}  {t_rep_e:>8.2f}  "
                  f"{avg_dist:>8.4f}  {avg_entropy:>8.4f}  ({elapsed:.0f}s)")

            torch.save({
                'step': step,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'hparams': {
                    'vocab_size': vocab_size,
                    'seq_len': args.seq_len,
                    'd_model': args.d_model,
                    'n_heads': args.n_heads,
                    'n_layers': args.n_layers,
                    'd_space': args.d_space,
                    'distance_penalty': args.distance_penalty,
                    'energy_weight': args.energy_weight,
                    'flop_weight': args.flop_weight,
                    'repulsion_weight': args.repulsion_weight,
                    'dropout': args.dropout,
                },
            }, CHECKPOINT_FILE)

    print(f"\nDone. Checkpoint saved to {CHECKPOINT_FILE}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',             default=DATA_FILE)
    p.add_argument('--seq_len',          type=int,   default=SEQ_LEN)
    p.add_argument('--batch_size',       type=int,   default=BATCH_SIZE)
    p.add_argument('--d_model',          type=int,   default=D_MODEL)
    p.add_argument('--n_heads',          type=int,   default=N_HEADS)
    p.add_argument('--n_layers',         type=int,   default=N_LAYERS)
    p.add_argument('--d_space',          type=int,   default=D_SPACE)
    p.add_argument('--distance_penalty', type=float, default=DISTANCE_PENALTY)
    p.add_argument('--energy_weight',    type=float, default=ENERGY_WEIGHT)
    p.add_argument('--flop_weight',      type=float, default=FLOP_WEIGHT)
    p.add_argument('--repulsion_weight', type=float, default=REPULSION_WEIGHT)
    p.add_argument('--dropout',          type=float, default=DROPOUT)
    p.add_argument('--lr',               type=float, default=LEARNING_RATE)
    p.add_argument('--max_steps',        type=int,   default=MAX_STEPS)
    p.add_argument('--eval_interval',    type=int,   default=EVAL_INTERVAL)
    p.add_argument('--resume',           action='store_true')
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
