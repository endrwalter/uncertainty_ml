"""8_figure_8.py
=============================================================
Drop-in replacement for the fig8_ccaugrc_real_table function
in paper_figures.py.

    python 6_plot_figure_8.py \
        --ccaugrc ../results/real_world_results/ccaugrc_real_world.csv \
        --out_dir ../figures/paper/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PALETTE_METHOD = {
    'h_total':    '#C0392B',
    'margin':     '#E67E22',
    'h_tau':      '#2980B9',
    'h_tau_ccrc': '#27AE60',
}

LABELS = {
    'h_total':    'H_Total (Global)',
    'margin':     'Margin (Global)',
    'h_tau':      'H_tau (Global)',
    'h_tau_ccrc': 'H_tau + CCRC (Proposed)',
}

DISEASES_ORDER = ['MS', 'PD', 'AD']
METHODS_ORDER  = ['h_total', 'margin', 'h_tau', 'h_tau_ccrc']


def fig8_ccaugrc_real_table(ccaugrc_data: pd.DataFrame, out_dir: str):
    """
    Grouped bar chart version of the ccAUGRC real-world decomposition.

    Three groups of bars (Global | Progressor | Stable ccAUGRC),
    one cluster per disease × method combination.

    """
    plt.rcParams.update({
        'font.family':       'serif',
        'font.size':         10,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.alpha':        0.25,
        'grid.linestyle':    '--',
        'axes.grid.axis':    'y',
        'figure.dpi':        150,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
    })

    metrics = [
        ('global_augrc',       'Global AUGRC'),
        ('ccaugrc_progressor', 'Progressor ccAUGRC'),
        ('ccaugrc_stable',     'Stable ccAUGRC'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    fig.suptitle(
        'ccAUGRC Decomposition — Real-World Neurodegenerative Cohorts\n'
        'Global AUGRC Rewards Minority Abandonment; '
        'Class-Conditioned Decomposition Reveals True Risk  (lower = safer)',
        fontsize=12, fontweight='bold'
    )

    n_methods  = len(METHODS_ORDER)
    n_diseases = len(DISEASES_ORDER)
    group_width = 0.8
    bar_width   = group_width / n_methods
    disease_gap = 1.5   # gap between disease clusters

    for col, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[col]

        # x positions: one cluster per disease, bars within each cluster per method
        x_centers = np.arange(n_diseases) * disease_gap

        for m_idx, method in enumerate(METHODS_ORDER):
            offset = (m_idx - (n_methods - 1) / 2) * bar_width
            vals   = []
            xpos   = []

            for d_idx, disease in enumerate(DISEASES_ORDER):
                sub = ccaugrc_data[
                    (ccaugrc_data['disease'] == disease) &
                    (ccaugrc_data['method']  == method)
                ]
                val = sub[metric_col].values[0] if len(sub) > 0 else np.nan
                vals.append(val)
                xpos.append(x_centers[d_idx] + offset)

            bars = ax.bar(
                xpos, vals,
                width=bar_width * 0.9,
                color=PALETTE_METHOD[method],
                alpha=0.85,
                label=LABELS[method],
                edgecolor='white',
                linewidth=0.8,
            )

            # Value labels on bars
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f'{val:.3f}',
                        ha='center', va='bottom',
                        fontsize=7.5, color='#2C3E50'
                    )

        # Disease labels as x-ticks
        tau_labels = {'MS': 'MS\n(τ=0.13)', 'PD': 'PD\n(τ=0.39)', 'AD': 'AD\n(τ=0.54)'}
        ax.set_xticks(x_centers)
        ax.set_xticklabels([tau_labels[d] for d in DISEASES_ORDER], fontsize=10)
        ax.set_ylabel('AUGRC Score  (lower = safer)', fontsize=10)
        ax.set_title(metric_label, fontsize=11, fontweight='bold', pad=8)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)

        # Annotations only on the middle panel (progressor ccAUGRC)
        if col == 1:
            ax.annotate(
                'H_Total achieves high\nprogressor risk by\ndeferring all progressors',
                xy=(x_centers[0],
                    ccaugrc_data[(ccaugrc_data['disease']=='MS') &
                                 (ccaugrc_data['method']=='h_total')][metric_col].values[0]),
                xytext=(x_centers[0] + 0.5, ax.get_ylim()[1] * 0.75),
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5),
                fontsize=8, color='#C0392B',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', alpha=0.9)
            )

        if col == 0:
            ax.legend(fontsize=8, loc='upper right')

        # Vertical separators between diseases
        for sep in x_centers[:-1] + disease_gap / 2:
            ax.axvline(sep, color='#BDC3C7', lw=0.8, ls='--', alpha=0.6)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig8_ccaugrc_real_table.pdf')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig8_ccaugrc_table_only(ccaugrc_data: pd.DataFrame, out_dir: str):
    """
    Alternative: clean matplotlib table (no bar chart).
    Rows = disease × method. Cols = Global | Progressor | Stable.
    Color-coded cells.
    """
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9,
                         'figure.dpi': 150, 'savefig.dpi': 300,
                         'savefig.bbox': 'tight'})

    col_labels = [
        'Disease', 'τ', 'Method',
        'Global\nAUGRC',
        'Progressor\nccAUGRC',
        'Stable\nccAUGRC',
    ]

    rows = []
    row_colors = []

    for disease in DISEASES_ORDER:
        tau_val = ccaugrc_data[ccaugrc_data['disease'] == disease]['tau'].iloc[0] \
                  if len(ccaugrc_data[ccaugrc_data['disease'] == disease]) > 0 else '—'
        first = True
        for method in METHODS_ORDER:
            sub = ccaugrc_data[
                (ccaugrc_data['disease'] == disease) &
                (ccaugrc_data['method']  == method)
            ]
            if sub.empty:
                continue

            r = sub.iloc[0]

            def fmt(v):
                return f'{v:.4f}' if not np.isnan(v) else '—'

            rows.append([
                disease if first else '',
                f'{tau_val:.2f}' if first else '',
                LABELS[method],
                fmt(r['global_augrc']),
                fmt(r['ccaugrc_progressor']),
                fmt(r['ccaugrc_stable']),
            ])
            first = False

            # Row background: highlight proposed method
            base_color = '#FFFFFF'
            if method == 'h_tau_ccrc':
                base_color = '#EAFAF1'
            elif method == 'h_total':
                base_color = '#FEF9E7'
            row_colors.append([base_color] * 6)

    n_rows = len(rows)
    n_cols = len(col_labels)

    fig_height = max(4, n_rows * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('off')

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=row_colors,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(n_cols)))

    # Header styling
    for col in range(n_cols):
        cell = table[0, col]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_height(0.12)

    # Bold the proposed method rows
    for row_idx, row in enumerate(rows):
        if 'H_tau + CCRC' in row[2]:
            for col in range(n_cols):
                table[row_idx + 1, col].set_text_props(fontweight='bold')

    ax.set_title(
        'ccAUGRC Decomposition — Real-World Cohorts\n'
        'Global AUGRC rewards minority abandonment; '
        'Progressor ccAUGRC reveals true risk',
        fontsize=11, fontweight='bold', pad=15, y=1.02
    )

    path = os.path.join(out_dir, 'fig8_ccaugrc_table.pdf')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ccaugrc', required=True,
                        help='ccaugrc_real_world.csv')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--style',   default='bars',
                        choices=['bars', 'table'],
                        help='bars = grouped bar chart, table = formatted table')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = pd.read_csv(args.ccaugrc)

    if args.style == 'bars':
        fig8_ccaugrc_real_table(data, args.out_dir)
    else:
        fig8_ccaugrc_table_only(data, args.out_dir)