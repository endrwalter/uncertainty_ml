"""
Stage 3: visualize_phase_diagram.py
=============================================================
Generates all synthetic experiment figures for the paper.

Figures produced
----------------
Fig A: Phase diagram heatmaps (tau × d grid)
    - Minority sensitivity at 30% rejection: H_total vs H_tau+CCRC vs difference
    - Sensitivity stability score
    - Minority coverage at 30% rejection

Fig B: Scaling law — scalar metrics vs tau (collapsed over d)

Fig C: Separability interaction — scalar metrics vs d at fixed tau values

Fig D: ccAUGRC decomposition across conditions (the "algorithmic cowardice" figure)

Fig E: Example rejection curves for four representative conditions
    (one per quadrant of the phase diagram)

Usage
-----
    python visualize_phase_diagram.py \
        --scalars  ../data/synthetic_results/scalar_summaries.csv \
        --curves   ../data/synthetic_results/rejection_curves.csv \
        --out_dir  ../figures/synthetic/
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')


# ── Consistent style ──
PALETTE = {
    'h_total':    '#E74C3C',   # red
    'margin':     '#F39C12',   # orange
    'h_tau':      '#3498DB',   # blue
    'h_tau_ccrc': '#2ECC71',   # green (proposed)
}
METHOD_LABELS = {
    'h_total':    'H_Total (Global)',
    'margin':     'Margin (Global)',
    'h_tau':      'H_tau (Global)',
    'h_tau_ccrc': 'H_tau + CCRC (Proposed)',
}
FIXED_REJECTION = 0.30


def load_data(scalars_path: str, curves_path: str):
    scalars = pd.read_csv(scalars_path)
    curves  = pd.read_csv(curves_path)

    # Separate by N for parallel analysis
    scalars_300  = scalars[scalars['n'] == 300].copy()
    scalars_1000 = scalars[scalars['n'] == 1000].copy()

    return scalars, curves, scalars_300, scalars_1000


# ─────────────────────────────────────────────
# FIG A: PHASE DIAGRAM HEATMAPS
# ─────────────────────────────────────────────

def plot_phase_diagram(scalars: pd.DataFrame, out_dir: str, n: int = 300):
    """
    3-panel heatmap: H_total | H_tau+CCRC | Difference
    Color = minority sensitivity at fixed rejection rate.
    """
    df = scalars[scalars['n'] == n].copy()
    metric = f'sensitivity_at_{int(FIXED_REJECTION*100)}pct'

    tau_vals = sorted(df['tau'].unique())
    d_vals   = sorted(df['d'].unique())

    def make_grid(method):
        pivot = (
            df[df['method'] == method]
            .pivot(index='tau', columns='d', values=metric)
            .reindex(index=tau_vals, columns=d_vals)
        )
        return pivot.values

    grid_htotal = make_grid('h_total')
    grid_ccrc   = make_grid('h_tau_ccrc')
    grid_diff   = grid_ccrc - grid_htotal   # positive = CCRC better

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Phase Diagram: Minority Sensitivity at {int(FIXED_REJECTION*100)}% Rejection  '
        f'(N={n})',
        fontsize=14, fontweight='bold'
    )

    kw_shared = dict(
        xticklabels=[str(d) for d in d_vals],
        yticklabels=[f'{t:.2f}' for t in tau_vals],
        linewidths=0.5,
    )

    # Panel 1: H_total
    sns.heatmap(
        grid_htotal, ax=axes[0],
        vmin=0, vmax=1, cmap='RdYlGn', annot=True, fmt='.2f',
        **kw_shared
    )
    axes[0].set_title('H_Total (Baseline)', fontsize=12)
    axes[0].set_xlabel('Separability (d)')
    axes[0].set_ylabel('Class Prior (τ)')

    # Panel 2: H_tau + CCRC
    sns.heatmap(
        grid_ccrc, ax=axes[1],
        vmin=0, vmax=1, cmap='RdYlGn', annot=True, fmt='.2f',
        **kw_shared
    )
    axes[1].set_title('H_tau + CCRC (Proposed)', fontsize=12)
    axes[1].set_xlabel('Separability (d)')
    axes[1].set_ylabel('')

    # Panel 3: Difference (CCRC - H_total)
    vmax_diff = np.nanmax(np.abs(grid_diff))
    sns.heatmap(
        grid_diff, ax=axes[2],
        vmin=-vmax_diff, vmax=vmax_diff,
        cmap='RdBu_r', annot=True, fmt='+.2f',
        center=0,
        **kw_shared
    )
    axes[2].set_title('Difference (Proposed − Baseline)', fontsize=12)
    axes[2].set_xlabel('Separability (d)')
    axes[2].set_ylabel('')

    # Mark real disease points on difference panel
    real_diseases = {
        'MS\n(τ=0.11)': (0.11, 0.5),
        'PD\n(τ=0.29)': (0.29, 1.0),
        'AD\n(τ=0.54)': (0.54, 2.0),
    }
    for label, (tau_real, d_real) in real_diseases.items():
        # Find closest grid position
        tau_idx = np.argmin(np.abs(np.array(tau_vals) - tau_real))
        d_idx   = np.argmin(np.abs(np.array(d_vals)   - d_real))
        axes[2].add_patch(plt.Rectangle(
            (d_idx, tau_idx), 1, 1,
            fill=False, edgecolor='black', lw=2.5, label=label
        ))
        axes[2].text(
            d_idx + 0.5, tau_idx + 0.5, label,
            ha='center', va='center', fontsize=7,
            fontweight='bold', color='black'
        )

    plt.tight_layout()
    path = os.path.join(out_dir, f'fig_A_phase_diagram_n{n}.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# FIG B: SCALING LAW (tau axis)
# ─────────────────────────────────────────────

def plot_scaling_law(scalars: pd.DataFrame, out_dir: str, n: int = 300):
    """
    Minority sensitivity at 30% rejection vs tau,
    collapsed across d (mean ± std shading).
    """
    df = scalars[scalars['n'] == n].copy()
    metric = f'sensitivity_at_{int(FIXED_REJECTION*100)}pct'

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Scaling Law: Performance vs Class Prior τ  (N={n})', fontsize=14)

    metrics_to_plot = [
        (f'sensitivity_at_{int(FIXED_REJECTION*100)}pct',
         f'Minority Sensitivity @ {int(FIXED_REJECTION*100)}% Rejection'),
        ('sensitivity_stability',
         'Sensitivity Stability (slope over 0–50% rejection)'),
        (f'minority_coverage_at_{int(FIXED_REJECTION*100)}pct',
         f'Minority Coverage @ {int(FIXED_REJECTION*100)}% Rejection'),
    ]

    for ax, (met, title) in zip(axes, metrics_to_plot):
        for method, label in METHOD_LABELS.items():
            sub = df[df['method'] == method].groupby('tau')[met]
            mean = sub.mean()
            std  = sub.std()

            ax.plot(mean.index, mean.values,
                    color=PALETTE[method], label=label,
                    marker='o', linewidth=2.0)
            ax.fill_between(
                mean.index,
                mean.values - std.values,
                mean.values + std.values,
                color=PALETTE[method], alpha=0.15
            )

        ax.set_xlabel('Class Prior (τ)', fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='τ=0.5')
        ax.grid(True, alpha=0.3)

        # Mark real disease taus
        for disease, tau_val, color in [('MS', 0.11, '#8E44AD'),
                                         ('PD', 0.29, '#1ABC9C'),
                                         ('AD', 0.54, '#E67E22')]:
            ax.axvline(x=tau_val, color=color, linestyle=':', alpha=0.7, linewidth=1.5)
            ax.text(tau_val, ax.get_ylim()[0] if ax.get_ylim()[0] else 0,
                    f' {disease}', color=color, fontsize=8, va='bottom')

    plt.tight_layout()
    path = os.path.join(out_dir, f'fig_B_scaling_law_n{n}.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# FIG C: SEPARABILITY INTERACTION
# ─────────────────────────────────────────────

def plot_separability_interaction(scalars: pd.DataFrame, out_dir: str, n: int = 300):
    """
    Minority sensitivity vs d at fixed tau values.
    Shows how separability modulates pathology severity.
    """
    df  = scalars[scalars['n'] == n].copy()
    met = f'sensitivity_at_{int(FIXED_REJECTION*100)}pct'

    # Pick representative tau values
    tau_to_show = [0.10, 0.20, 0.30, 0.50]
    tau_available = sorted(df['tau'].unique())
    tau_to_show = [t for t in tau_to_show if t in tau_available]

    fig, axes = plt.subplots(1, len(tau_to_show), figsize=(5 * len(tau_to_show), 5))
    if len(tau_to_show) == 1:
        axes = [axes]
    fig.suptitle(f'Separability Interaction: Sensitivity vs d at Fixed τ  (N={n})',
                 fontsize=13)

    for ax, tau_val in zip(axes, tau_to_show):
        sub = df[df['tau'] == tau_val]
        for method, label in METHOD_LABELS.items():
            m_sub = sub[sub['method'] == method].sort_values('d')
            ax.plot(m_sub['d'], m_sub[met],
                    color=PALETTE[method], label=label,
                    marker='s', linewidth=2.0)

        ax.set_title(f'τ = {tau_val:.2f}', fontsize=11)
        ax.set_xlabel('Separability (d)', fontsize=10)
        ax.set_ylabel(f'Sensitivity @ {int(FIXED_REJECTION*100)}%', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(out_dir, f'fig_C_separability_interaction_n{n}.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# FIG D: ccAUGRC DECOMPOSITION
# ─────────────────────────────────────────────

def plot_ccaugrc_decomposition(scalars: pd.DataFrame, out_dir: str, n: int = 300):
    """
    ccAUGRC decomposition heatmaps showing the 'algorithmic cowardice' effect.
    3 rows (global, progressor, stable) × 2 columns (H_total, H_tau+CCRC).
    """
    df = scalars[scalars['n'] == n].copy()

    tau_vals = sorted(df['tau'].unique())
    d_vals   = sorted(df['d'].unique())

    metrics = [
        ('global_augrc',       'Global AUGRC'),
        ('ccaugrc_progressor', 'Progressor ccAUGRC'),
        ('ccaugrc_stable',     'Stable ccAUGRC'),
    ]
    methods_to_show = ['h_total', 'h_tau_ccrc', 'margin']

    fig, axes = plt.subplots(3, 3, figsize=(14, 16))
    fig.suptitle(f'ccAUGRC Decomposition (N={n}, lower=better)',
                 fontsize=14, fontweight='bold')

    for row, (metric_col, metric_label) in enumerate(metrics):
        for col, method in enumerate(methods_to_show):
            ax = axes[row][col]
            pivot = (
                df[df['method'] == method]
                .pivot(index='tau', columns='d', values=metric_col)
                .reindex(index=tau_vals, columns=d_vals)
            )

            sns.heatmap(
                pivot.values, ax=ax,
                vmin=0, vmax=0.3,
                cmap='RdYlGn_r',
                annot=True, fmt='.3f',
                xticklabels=[str(d) for d in d_vals],
                yticklabels=[f'{t:.2f}' for t in tau_vals],
                linewidths=0.5,
                cbar=(col == 1),
            )
            ax.set_title(f'{METHOD_LABELS[method]}\n{metric_label}', fontsize=10)
            ax.set_xlabel('Separability (d)' if row == 2 else '')
            ax.set_ylabel('Class Prior (τ)' if col == 0 else '')

    plt.tight_layout()
    path = os.path.join(out_dir, f'fig_D_ccaugrc_decomposition_n{n}.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# FIG E: REPRESENTATIVE REJECTION CURVES
# ─────────────────────────────────────────────

def plot_representative_curves(curves: pd.DataFrame, out_dir: str, n: int = 300):
    """
    Four representative conditions, one per quadrant of tau × d space.
    Shows full rejection curves for all four methods.
    """
    quadrants = [
        (0.10, 0.5,  'Low τ, Low d\n(Hard: rare + ambiguous)'),
        (0.10, 3.0,  'Low τ, High d\n(Rare + distinct)'),
        (0.50, 0.5,  'High τ, Low d\n(Balanced + ambiguous)'),
        (0.50, 3.0,  'High τ, High d\n(Control: balanced + distinct)'),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle(
        f'Representative Rejection Curves — Four Phase Diagram Quadrants  (N={n})',
        fontsize=13, fontweight='bold'
    )

    row_labels = ['Sensitivity vs Rejection', 'Specificity vs Rejection',
                  'AUPRC vs Rejection']
    metrics    = ['sensitivity', 'specificity', 'auprc']

    for col, (tau_q, d_q, title) in enumerate(quadrants):
        sub = curves[
            (curves['n'] == n) &
            (np.isclose(curves['tau'], tau_q)) &
            (np.isclose(curves['d'],   d_q))
        ]

        for row, (metric, ylabel) in enumerate(zip(metrics, row_labels)):
            ax = axes[row][col]

            for method, label in METHOD_LABELS.items():
                m_sub = sub[sub['method'] == method].sort_values('rejection_rate')
                ax.plot(
                    m_sub['rejection_rate'],
                    m_sub[metric],
                    color=PALETTE[method],
                    label=label,
                    linewidth=2.0,
                    linestyle='--' if method == 'h_total' else '-',
                )

            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 2:
                ax.set_xlabel('Rejection Rate', fontsize=9)

            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, 0.8)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=FIXED_REJECTION, color='gray',
                       linestyle=':', alpha=0.5, linewidth=1)

            if row == 0 and col == 3:
                ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    path = os.path.join(out_dir, f'fig_E_representative_curves_n{n}.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# FIG F: N=300 vs N=1000 COMPARISON
# ─────────────────────────────────────────────

def plot_n_comparison(scalars: pd.DataFrame, out_dir: str):
    """
    Difference heatmap side-by-side for N=300 and N=1000.
    If results look the same, one goes to appendix.
    """
    metric = f'sensitivity_at_{int(FIXED_REJECTION*100)}pct'

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Effect of Sample Size: Difference Heatmap (H_tau+CCRC − H_total)',
        fontsize=13
    )

    for ax, n in zip(axes, [300, 1000]):
        df = scalars[scalars['n'] == n]
        tau_vals = sorted(df['tau'].unique())
        d_vals   = sorted(df['d'].unique())

        def make_grid(method):
            return (
                df[df['method'] == method]
                .pivot(index='tau', columns='d', values=metric)
                .reindex(index=tau_vals, columns=d_vals)
                .values
            )

        diff = make_grid('h_tau_ccrc') - make_grid('h_total')
        vmax = np.nanmax(np.abs(diff))

        sns.heatmap(
            diff, ax=ax,
            vmin=-vmax, vmax=vmax, center=0,
            cmap='RdBu_r', annot=True, fmt='+.2f',
            xticklabels=[str(d) for d in d_vals],
            yticklabels=[f'{t:.2f}' for t in tau_vals],
            linewidths=0.5,
        )
        ax.set_title(f'N = {n}', fontsize=12)
        ax.set_xlabel('Separability (d)')
        ax.set_ylabel('Class Prior (τ)')

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_F_n_comparison.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main(scalars_path: str, curves_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data...")
    scalars, curves, _, _ = load_data(scalars_path, curves_path)

    print("\nGenerating figures...")

    for n in [300, 1000]:
        if scalars[scalars['n'] == n].empty:
            print(f"  [SKIP] No data for n={n}")
            continue
        plot_phase_diagram(scalars, out_dir, n=n)
        plot_scaling_law(scalars, out_dir, n=n)
        plot_separability_interaction(scalars, out_dir, n=n)
        plot_ccaugrc_decomposition(scalars, out_dir, n=n)
        plot_representative_curves(curves, out_dir, n=n)

    if not scalars[scalars['n'] == 300].empty and \
       not scalars[scalars['n'] == 1000].empty:
        plot_n_comparison(scalars, out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scalars',  required=True)
    parser.add_argument('--curves',   required=True)
    parser.add_argument('--out_dir',  required=True)
    args = parser.parse_args()

    main(args.scalars, args.curves, args.out_dir)
