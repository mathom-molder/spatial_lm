"""
Plot the effect of energy_weight on the accuracy/efficiency tradeoff.

After running energy_sweep.sh, call:
    python energy_plot.py

Reads energy_ew{weight}.log files plus the pareto baseline (pareto_p0.1.log),
and plots val_loss vs avg_attention_distance for each energy_weight value.

The baseline point (energy_weight=0, penalty=0.1) already beats the standard
transformer. Adding energy_weight should push avg_dist lower — the question is
how much accuracy we sacrifice per unit of efficiency gained.
"""

import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ENERGY_WEIGHTS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]


def parse_log(path):
    """Return (val_loss, avg_dist, dist_energy) from the last eval line."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    data_lines = [l.strip() for l in lines if re.match(r'^\s*\d+\s+\d', l)]
    if not data_lines:
        return None

    cols = data_lines[-1].split()
    # Log format: Step | Train | Val | DistE | RepE | AvgDist | Entropy | (Ns)
    try:
        val_loss   = float(cols[2])
        dist_energy = float(cols[3])
        avg_dist   = float(cols[5])
        return val_loss, avg_dist, dist_energy
    except (IndexError, ValueError):
        return None


def main():
    results = []

    print(f"{'EnergyW':>8}  {'Val loss':>9}  {'Avg dist':>9}  {'DistE':>7}")
    print('─' * 40)

    for ew in ENERGY_WEIGHTS:
        if ew == 0.0:
            path = 'pareto_p0.1.log'   # baseline from pareto sweep
        else:
            path = f'energy_ew{ew}.log'

        r = parse_log(path)
        if r is None:
            print(f"{ew:>8}  {'(missing)':>9}")
            continue

        val, dist, de = r
        results.append({'ew': ew, 'val': val, 'avg_dist': dist, 'dist_energy': de})
        print(f"{ew:>8.3f}  {val:>9.4f}  {dist:>9.4f}  {de:>7.4f}")

    if not results:
        print("\nNo log files found. Run energy_sweep.sh first.")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: Pareto scatter (val_loss vs avg_dist) ──────────────────────────
    ax = axes[0]
    vals  = [r['val']      for r in results]
    dists = [r['avg_dist'] for r in results]
    ews   = [r['ew']       for r in results]

    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0, vmax=max(ews) if max(ews) > 0 else 1)
    sc = ax.scatter(dists, vals, c=ews, cmap=cmap, norm=norm,
                    s=120, zorder=5, edgecolors='k', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label='energy_weight')

    for r in results:
        ax.annotate(
            f"ew={r['ew']}",
            (r['avg_dist'], r['val']),
            textcoords='offset points', xytext=(7, 4), fontsize=9
        )

    # Pareto-efficient hull
    pts = np.array([[r['avg_dist'], r['val']] for r in results])
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
    ax.set_title('Accuracy vs. attention locality\n(λ=0.1, varying energy_weight)', fontsize=11)
    ax.grid(True, alpha=0.3)

    # ── Right: val_loss and avg_dist vs energy_weight ────────────────────────
    ax2 = axes[1]
    ews_sorted = sorted(results, key=lambda r: r['ew'])
    x  = [r['ew']      for r in ews_sorted]
    y1 = [r['val']     for r in ews_sorted]
    y2 = [r['avg_dist'] for r in ews_sorted]

    color1, color2 = 'royalblue', 'darkorange'
    ax2.plot(x, y1, 'o-', color=color1, label='Val loss', linewidth=2)
    ax2.set_xlabel('energy_weight', fontsize=11)
    ax2.set_ylabel('Val loss', color=color1, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color1)

    ax3 = ax2.twinx()
    ax3.plot(x, y2, 's--', color=color2, label='Avg dist', linewidth=2)
    ax3.set_ylabel('Avg attention distance', color=color2, fontsize=11)
    ax3.tick_params(axis='y', labelcolor=color2)

    ax2.set_title('Effect of metabolic pressure\n(higher energy_weight = more pressure)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    plt.tight_layout()
    out = 'energy_frontier.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {out}")

    # Summary
    baseline = next((r for r in results if r['ew'] == 0.0), None)
    if baseline:
        print("\nEffect of metabolic pressure vs baseline (ew=0):")
        for r in results:
            if r['ew'] == 0.0:
                continue
            dv = r['val']      - baseline['val']
            dd = r['avg_dist'] - baseline['avg_dist']
            print(f"  ew={r['ew']:.3f}:  val {dv:+.4f}  avg_dist {dd:+.4f}")


if __name__ == '__main__':
    main()
