"""
Plot the Pareto frontier: val loss vs average attention distance.

After running pareto_sweep.sh, call:
    python pareto_plot.py

Reads pareto_p{penalty}.log files, extracts the final evaluation row,
and plots val_loss (y) vs avg_attention_distance (x).

Lower-left is better — if spatial models form a frontier below the
ablation (penalty=0) point, the spatial prior is genuinely buying
efficiency without sacrificing accuracy.
"""

import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PENALTIES = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]


def parse_log(path):
    """Return (val_loss, avg_dist) from the last evaluation line in path."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    # Data lines start with a step number (digits then whitespace)
    data_lines = [l.strip() for l in lines if re.match(r'^\s*\d+\s+\d', l)]
    if not data_lines:
        return None

    cols = data_lines[-1].split()
    # Log format: Step | Train | Val | DistE | RepE | AvgDist | Entropy | (Ns)
    try:
        val_loss = float(cols[2])
        avg_dist = float(cols[5])
        return val_loss, avg_dist
    except (IndexError, ValueError):
        return None


def main():
    results = []
    print(f"{'Penalty':>8}  {'Val loss':>9}  {'Avg dist':>9}")
    print('─' * 32)
    for p in PENALTIES:
        fname = f'pareto_p{p}.log'
        r = parse_log(fname)
        if r is None:
            print(f"{p:>8}  {'(missing)':>9}")
            continue
        val, dist = r
        results.append({'penalty': p, 'val': val, 'avg_dist': dist})
        print(f"{p:>8.1f}  {val:>9.4f}  {dist:>9.4f}")

    if not results:
        print("\nNo log files found. Run pareto_sweep.sh first.")
        sys.exit(1)

    vals  = [r['val']      for r in results]
    dists = [r['avg_dist'] for r in results]
    pens  = [r['penalty']  for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Colour-code by penalty: blue (low) → red (high)
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=min(pens), vmax=max(pens))
    sc = ax.scatter(dists, vals, c=pens, cmap=cmap, norm=norm,
                    s=120, zorder=5, edgecolors='k', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label='distance_penalty λ')

    for r in results:
        ax.annotate(
            f"λ={r['penalty']}",
            (r['avg_dist'], r['val']),
            textcoords='offset points', xytext=(7, 4), fontsize=9
        )

    # Draw the convex hull of the lower-left frontier (Pareto-efficient points)
    pts = np.array([[r['avg_dist'], r['val']] for r in results])
    # A point is Pareto-efficient if no other point dominates it
    # (lower val AND lower avg_dist simultaneously)
    pareto = []
    for i, p in enumerate(pts):
        dominated = any(
            (pts[j, 0] <= p[0] and pts[j, 1] <= p[1] and
             (pts[j, 0] < p[0] or pts[j, 1] < p[1]))
            for j in range(len(pts)) if j != i
        )
        if not dominated:
            pareto.append(p)
    if len(pareto) > 1:
        pareto = sorted(pareto, key=lambda x: x[0])
        px, py = zip(*pareto)
        ax.plot(px, py, '--', color='gray', alpha=0.6, linewidth=1.5,
                label='Pareto frontier')
        ax.legend(fontsize=9)

    ax.set_xlabel('Average attention distance (token-space)', fontsize=11)
    ax.set_ylabel('Validation loss', fontsize=11)
    ax.set_title('Pareto frontier: accuracy vs. attention locality\n'
                 'lower-left = more efficient', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = 'pareto_frontier.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {out}")

    # Text summary
    ablation = next((r for r in results if r['penalty'] == 0.0), None)
    if ablation:
        print("\nEfficiency gains over ablation (penalty=0):")
        for r in results:
            if r['penalty'] == 0.0:
                continue
            delta_val  = r['val']      - ablation['val']
            delta_dist = r['avg_dist'] - ablation['avg_dist']
            print(f"  λ={r['penalty']:.1f}:  val {delta_val:+.4f}  "
                  f"avg_dist {delta_dist:+.4f}")


if __name__ == '__main__':
    main()
