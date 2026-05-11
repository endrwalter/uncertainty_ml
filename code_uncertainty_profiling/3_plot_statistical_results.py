"""
plot_statistical_results.py
=============================================================
Loads all_statistical_tests.csv and produces:

1. A printed summary table for both Progressor and Stable ccAUGRC
2. A publication-ready figure: grouped bar chart with 95% CI error bars
   and significance annotations for all three diseases

Usage
-----
    python plot_statistical_results.py \
        --tests   ../results/statistical_tests/all_statistical_tests.csv \
        --distrib ../results/statistical_tests/ \
        --out_dir ../figures/paper/
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

METHODS_ORDER = ['h_total', 'margin', 'h_tau', 'h_tau_ccrc']

METHOD_LABELS = {
    'h_total':    'H_Total\n(Global)',
    'margin':     'Margin\n(Global)',
    'h_tau':      'H_tau\n(Global)',
    'h_tau_ccrc': 'H_tau+CCRC\n(Proposed)',
}

METHOD_LABELS_SHORT = {
    'h_total':    'H_Total',
    'margin':     'Margin',
    'h_tau':      'H_tau',
    'h_tau_ccrc': 'H_tau+CCRC',
}

PALETTE = {
    'h_total':    '#C0392B',
    'margin':     '#E67E22',
    'h_tau':      '#2980B9',
    'h_tau_ccrc': '#27AE60',
}

DISEASES_ORDER = ['MS', 'PD', 'AD']
DISEASE_LABELS = {
    'MS': 'MS  (τ=0.11)',
    'PD': 'PD  (τ=0.39)',
    'AD': 'AD  (τ=0.54)',
}

REFERENCE = 'h_tau_ccrc'


# ─────────────────────────────────────────────
# PRINTED TABLE
# ─────────────────────────────────────────────

def sig_marker(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def print_full_table(tests: pd.DataFrame, distributions: dict):
    """
    Print a complete table with point estimates [95% CI] and
    significance markers for both Progressor and Stable ccAUGRC.
    """
    metrics = [
        ('ccaugrc_progressor', 'Progressor ccAUGRC'),
        ('ccaugrc_stable',     'Stable ccAUGRC'),
        ('global_augrc',       'Global AUGRC'),
    ]

    for disease in DISEASES_ORDER:
        print(f"\n{'═'*90}")
        print(f"  {DISEASE_LABELS[disease]}")
        print(f"{'═'*90}")

        # Get distributions for this disease to compute point estimates + CI
        dist_df = distributions.get(disease)

        header = (
            f"  {'Method':<30} "
            f"{'Progressor ccAUGRC':>22} "
            f"{'sig vs ref':>12} "
            f"{'Stable ccAUGRC':>18} "
            f"{'sig vs ref':>12} "
            f"{'Global AUGRC':>14}"
        )
        print(header)
        print(f"  {'─'*88}")

        for method in METHODS_ORDER:
            # Point estimates and CIs from distributions
            row_parts = [f"  {METHOD_LABELS_SHORT[method]:<30}"]

            for metric, _ in metrics:
                if dist_df is not None and method in dist_df['method'].values:
                    vals     = dist_df[dist_df['method'] == method][metric].values
                    mean_val = vals.mean()
                    ci_low   = np.percentile(vals, 2.5)
                    ci_high  = np.percentile(vals, 97.5)
                    est_str  = f"{mean_val:.4f} [{ci_low:.4f},{ci_high:.4f}]"
                else:
                    est_str = '—'

                # Significance vs reference (only for non-reference methods)
                if method != REFERENCE:
                    sub = tests[
                        (tests['disease'] == disease) &
                        (tests['metric'] == metric) &
                        (tests['comparison_method'] == method)
                    ]
                    if len(sub) > 0:
                        row = sub.iloc[0]
                        direction = '↑worse' if row['mean_difference'] > 0 else '↓better'
                        sig_str   = f"{sig_marker(row['p_value'])} {direction}"
                    else:
                        sig_str = '—'
                else:
                    sig_str = '[reference]'

                if metric == 'global_augrc':
                    row_parts.append(f"{est_str:>30}")
                elif metric == 'ccaugrc_progressor':
                    row_parts.append(f"{est_str:>30} {sig_str:>12}")
                else:
                    row_parts.append(f"{est_str:>26} {sig_str:>12}")

            print(''.join(row_parts))

        print(f"\n  Reference method: {METHOD_LABELS_SHORT[REFERENCE]}")
        print(f"  *** p<0.001  ** p<0.01  * p<0.05  ns not significant")
        print(f"  ↑worse = comparison method has higher (worse) ccAUGRC than reference")
        print(f"  ↓better = comparison method has lower (better) ccAUGRC than reference")


# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────

def plot_results(tests: pd.DataFrame, distributions: dict, out_dir: str):
    """
    Two-row figure:
    Row 1: Progressor ccAUGRC — grouped bars with CI + significance
    Row 2: Stable ccAUGRC — same layout

    Three columns: MS, PD, AD
    """
    plt.rcParams.update({
        'font.family':       'serif',
        'font.size':         10,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.alpha':        0.25,
        'grid.linestyle':    '--',
        'axes.grid.axis':         'y',
        'figure.dpi':        150,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
    })

    metrics = [
        ('ccaugrc_progressor', 'Progressor ccAUGRC\n(lower = safer)', 'minority protection'),
        ('ccaugrc_stable',     'Stable ccAUGRC\n(lower = safer)',     'majority protection'),
    ]

    n_methods  = len(METHODS_ORDER)
    bar_width  = 0.18
    x_center   = 0.0
    offsets    = np.linspace(
        -(n_methods-1)/2 * bar_width,
         (n_methods-1)/2 * bar_width,
        n_methods
    )

    fig, axes = plt.subplots(2, 3, figsize=(20, 11), sharey=False)
    fig.suptitle(
        'ccAUGRC Decomposition — Real-World Cohorts\n'
        '95% CI from 1,000-iteration patient bootstrap  |  '
        'Significance vs H_tau+CCRC (Proposed)',
        fontsize=13, fontweight='bold'
    )

    for row_idx, (metric, ylabel, subtitle) in enumerate(metrics):
        for col_idx, disease in enumerate(DISEASES_ORDER):
            ax       = axes[row_idx][col_idx]
            dist_df  = distributions.get(disease)

            means, ci_lows, ci_highs = [], [], []

            for method in METHODS_ORDER:
                if dist_df is not None and method in dist_df['method'].values:
                    vals = dist_df[dist_df['method'] == method][metric].values
                    means.append(vals.mean())
                    ci_lows.append(np.percentile(vals, 2.5))
                    ci_highs.append(np.percentile(vals, 97.5))
                else:
                    means.append(np.nan)
                    ci_lows.append(np.nan)
                    ci_highs.append(np.nan)

            # Draw bars
            for m_idx, method in enumerate(METHODS_ORDER):
                xpos  = x_center + offsets[m_idx]
                mean  = means[m_idx]
                ci_lo = ci_lows[m_idx]
                ci_hi = ci_highs[m_idx]

                if np.isnan(mean):
                    continue

                bar = ax.bar(
                    xpos, mean,
                    width=bar_width * 0.9,
                    color=PALETTE[method],
                    alpha=0.85 if method != REFERENCE else 1.0,
                    edgecolor='black' if method == REFERENCE else 'white',
                    linewidth=1.5 if method == REFERENCE else 0.5,
                    zorder=3,
                    label=METHOD_LABELS_SHORT[method],
                )

                # Error bars (95% CI)
                ax.errorbar(
                    xpos, mean,
                    yerr=[[mean - ci_lo], [ci_hi - mean]],
                    fmt='none',
                    color='#2C3E50',
                    capsize=4,
                    capthick=1.5,
                    elinewidth=1.5,
                    zorder=4,
                )

                # Value label on bar
                ax.text(
                    xpos, ci_hi + 0.003,
                    f'{mean:.3f}',
                    ha='center', va='bottom',
                    fontsize=7.5, color='#2C3E50',
                    zorder=5,
                )

            # Significance annotations vs reference
            ref_mean = means[METHODS_ORDER.index(REFERENCE)]
            ref_ci_hi = ci_highs[METHODS_ORDER.index(REFERENCE)]
            y_max = max([h for h in ci_highs if not np.isnan(h)]) + 0.015

            bracket_y_start = y_max + 0.005
            bracket_step    = 0.022

            for m_idx, method in enumerate(METHODS_ORDER):
                if method == REFERENCE:
                    continue

                sub = tests[
                    (tests['disease'] == disease) &
                    (tests['metric'] == metric) &
                    (tests['comparison_method'] == method)
                ]
                if sub.empty:
                    continue

                row = sub.iloc[0]
                sig = sig_marker(row['p_value'])
                if sig == 'ns':
                    continue

                xpos_comp = x_center + offsets[m_idx]
                xpos_ref  = x_center + offsets[METHODS_ORDER.index(REFERENCE)]
                brac_y    = bracket_y_start + (m_idx * bracket_step)

                # Bracket
                ax.plot(
                    [xpos_comp, xpos_comp, xpos_ref, xpos_ref],
                    [brac_y - 0.003, brac_y, brac_y, brac_y - 0.003],
                    color='#2C3E50', linewidth=1.0, zorder=6
                )
                ax.text(
                    (xpos_comp + xpos_ref) / 2, brac_y + 0.001,
                    sig,
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#2C3E50',
                    zorder=7,
                )

            # Axis formatting
            all_means = [m for m in means if not np.isnan(m)]
            y_ceiling = bracket_y_start + (n_methods * bracket_step) + 0.01
            ax.set_ylim(0, max(y_ceiling, max(all_means) * 1.4) if all_means else 0.4)
            ax.set_xticks([x_center])
            ax.set_xticklabels([''])
            ax.set_xlim(-0.5, 0.5)

            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row_idx == 0:
                ax.set_title(
                    f'{DISEASE_LABELS[disease]}',
                    fontsize=11, fontweight='bold', pad=8
                )

            # Shade reference bar lightly
            ax.axhline(ref_mean, color=PALETTE[REFERENCE],
                       ls='--', lw=1.0, alpha=0.4, zorder=2)

    # Legend
    legend_patches = [
        mpatches.Patch(
            color=PALETTE[m], alpha=0.85,
            label=METHOD_LABELS_SHORT[m] + (' (Proposed)' if m == REFERENCE else '')
        )
        for m in METHODS_ORDER
    ]
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=4,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    path = os.path.join(out_dir, 'fig_statistical_ccaugrc.pdf')
    plt.savefig(path)
    plt.close()
    print(f"\nSaved figure: {path}")


# ─────────────────────────────────────────────
# LOAD DISTRIBUTIONS
# ─────────────────────────────────────────────

def load_distributions(distrib_dir: str) -> dict:
    """
    Load per-disease bootstrap distribution CSVs.
    Returns dict: {disease: DataFrame}
    """
    distributions = {}
    for disease in DISEASES_ORDER:
        path = os.path.join(distrib_dir, f'{disease.lower()}_distributions.csv')
        if os.path.exists(path):
            distributions[disease] = pd.read_csv(path)
            print(f"  Loaded distributions: {path}")
        else:
            print(f"  [SKIP] distributions not found: {path}")
    return distributions


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests',   required=True,
                        help='all_statistical_tests.csv')
    parser.add_argument('--distrib', required=True,
                        help='Directory containing {disease}_distributions.csv files')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data...")
    tests         = pd.read_csv(args.tests)
    distributions = load_distributions(args.distrib)

    print("\n" + "═"*90)
    print("STATISTICAL SUMMARY TABLE")
    print("═"*90)
    print_full_table(tests, distributions)

    print("\nGenerating figure...")
    plot_results(tests, distributions, args.out_dir)
