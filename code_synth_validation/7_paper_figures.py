"""
7_paper_figures.py
=============================================================
Generates all publication-ready figures for the synthetic validation
experiments.

Inputs
------
--scalars: ../data/synthetic_results/scalar_summaries.csv 
           (Produced by 5_compute_rejection_metrics.py)
--curves:  ../data/synthetic_results/rejection_curves.csv 
           (Produced by 5_compute_rejection_metrics.py)

Outputs
-------
--out_dir: Directory where outputs are saved (e.g., ../figures/paper/)
           Produces:
           1. fig3_sensitivity_stability_n<N>.pdf
           2. fig2_failure_map_n<N>.pdf
           4. figC3_rejection_curves_n<N>_<METRIC>.pdf

Usage
-----
    python 7_paper_figures.py \
        --scalars   ../data/synthetic_results/scalar_summaries.csv \
        --curves    ../data/synthetic_results/rejection_curves.csv \
        --out_dir   ../figures/paper/
"""
import matplotlib.patches as patches
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────

PALETTE = {
    'h_total':      '#C0392B',   # deep red
    'margin':       '#E67E22',   # amber
    'h_tau':        '#2980B9',   # steel blue
    'h_tau_ccrc':   '#27AE60',   # forest green — proposed
    'h_total_ccrc': '#8E44AD',   # purple
    'margin_ccrc':  '#16A085',   # teal
}
LABELS = {
    'h_total':      'H_Total (Global)',
    'margin':       'Margin (Global)',
    'h_tau':        'H_tau (Global)',
    'h_tau_ccrc':   'H_tau + CCR (Proposed)',
    'h_total_ccrc': 'H_Total + CCR',
    'margin_ccrc':  'Margin + CCR',
}
DISEASE_MARKERS = {
    'MS': {'tau': 0.11, 'color': '#8E44AD', 'marker': 'D'},
    'PD': {'tau': 0.39, 'color': '#16A085', 'marker': 's'},
    'AD': {'tau': 0.54, 'color': '#E67E22', 'marker': '^'},
}

def set_style():
    plt.rcParams.update({
        'font.family':        'serif',
        'font.size':          11,
        'axes.titlesize':     12,
        'axes.titleweight':   'bold',
        'axes.labelsize':     11,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.grid':          True,
        'grid.alpha':         0.25,
        'grid.linestyle':     '--',
        'legend.framealpha':  0.9,
        'legend.fontsize':    9,
        'figure.dpi':         150,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
    })



# ─────────────────────────────────────────────
# FIG 3 — SENSITIVITY STABILITY (PRIMARY)
# ─────────────────────────────────────────────

def fig_sensitivity_stability(scalars: pd.DataFrame, out_dir: str, n: int = 1000):
    """
    Print-optimized primary figure for Sensitivity Stability.
    """
    set_style()
    df = scalars[scalars['n'] == n].copy()
    methods_to_show = ['h_total', 'h_tau', 'h_tau_ccrc']

    # 1. Print-Optimized Dimensions (10 inches wide fits standard journal column/page)
    fig = plt.figure(figsize=(10, 7))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[1.5, 1],
                            hspace=0.45, wspace=0.3)


    # ── Main panel: sensitivity stability ──
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.margins(x=0)

    for method in methods_to_show:
        sub  = df[df['method'] == method].groupby('tau')['sensitivity_stability']
        mean = sub.mean()
        std  = sub.std()

        ax_main.plot(
            mean.index, mean.values,
            color=PALETTE[method], label=LABELS[method],
            marker='o', linewidth=2.0, markersize=5, zorder=3
        )
        ax_main.fill_between(
            mean.index, mean.values - std.values, mean.values + std.values,
            color=PALETTE[method], alpha=0.15
        )

    ax_main.axhline(0, color='#2C3E50', ls='--', lw=1.2, alpha=0.7, label='Reference (slope = 0)')
    
    # Annotation
    min_val = df[df['method']=='h_total']['sensitivity_stability'].min()
    #ax_main.axhspan(min_val * 1.05, -0.3, alpha=0.06, color='#C0392B')


    ax_main.set_xlabel('Class Prior  π', fontsize=10)
    ax_main.set_ylabel('Sensitivity Stability\n(slope over 0–50% rej.)', fontsize=10)
    ax_main.set_title('A', loc='left', fontsize=12, fontweight='bold', pad=8)
    ax_main.tick_params(labelsize=9)

    # ── Supporting panel left: sensitivity at 30% ──
    ax_s = fig.add_subplot(gs[1, 0])
    ax_s.margins(x=0)

    for method in methods_to_show:
        sub = df[df['method'] == method].groupby('tau')['sensitivity_at_30pct']
        ax_s.plot(sub.mean().index, sub.mean().values, color=PALETTE[method], marker='o', lw=1.5, ms=4)
        ax_s.fill_between(sub.mean().index, sub.mean()-sub.std(), sub.mean()+sub.std(), 
                          color=PALETTE[method], alpha=0.15)

    ax_s.set_xlabel('Class Prior  π', fontsize=10)
    ax_s.set_ylabel('Sensitivity @ 30%', fontsize=10)
    ax_s.set_title('B', loc='left', fontsize=12, fontweight='bold', pad=8)
    ax_s.tick_params(labelsize=9)
    ax_s.set_ylim(0, 1.05)

    # ── Supporting panel right: specificity at 30% ──
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.margins(x=0)
    for method in methods_to_show:
        sub = df[df['method'] == method].groupby('tau')['specificity_at_30pct']
        ax_c.plot(sub.mean().index, sub.mean().values, color=PALETTE[method], marker='o', lw=1.5, ms=4)
        ax_c.fill_between(sub.mean().index, sub.mean()-sub.std(), sub.mean()+sub.std(), 
                          color=PALETTE[method], alpha=0.15)

    ax_c.set_xlabel('Class Prior  π', fontsize=10)
    ax_c.set_ylabel('Specificity @ 30%', fontsize=10)
    ax_c.set_title('C', loc='left', fontsize=12, fontweight='bold', pad=8)
    ax_c.tick_params(labelsize=9)
    ax_c.set_ylim(0, 1.05)

    # Reserve space for the legend
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # Global single-line legend
    handles, labels = ax_main.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=len(by_label), fontsize=10, frameon=False)

    path = os.path.join(out_dir, f'fig3_sensitivity_stability_n{n}.pdf')
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()

    # ── Comprehensive Verification Print Block ──
    if n == 1000:
        print("\n" + "="*70)
        print("VERIFICATION")
        print("="*70)

        # Helper function to safely extract means matching a specific prior
        def get_val(method_name, target_tau, metric):
            # Round tau to 2 decimal places to avoid floating-point mismatch
            subset = df[(df['method'] == method_name) & (df['tau'].round(2) == target_tau)]
            if subset.empty:
                return float('nan')
            return subset[metric].mean()

        # Claim 1: H_Total neutral slope at tau=0.50 (+0.12)
        h_total_50 = get_val('h_total', 0.50, 'sensitivity_stability')
        print(f"1. H_Total slope at tau=0.50:            {h_total_50:+.2f}")

        # Claim 2: H_Total extreme deletion at tau=0.10 (-2.08)
        h_total_10 = get_val('h_total', 0.10, 'sensitivity_stability')
        print(f"2. H_Total slope at tau=0.10:            {h_total_10:+.2f}")

        # Claim 3: H_tau partial fix at tau=0.10 (-0.26)
        h_tau_10 = get_val('h_tau', 0.10, 'sensitivity_stability')
        print(f"3. H_tau slope at tau=0.10:              {h_tau_10:+.2f}")

        # Claim 4: H_tau+CCR bounds across all priors (+0.13 to +0.24)
        if 'h_tau_ccrc' in df['method'].values:
            ccrc_means = df[df['method'] == 'h_tau_ccrc'].groupby('tau')['sensitivity_stability'].mean()
            print(f"4. H_tau+CCR slope bounds (All Taus):    {ccrc_means.min():+.2f} to {ccrc_means.max():+.2f}")

        # Claim 5: Baseline Specificity at tau=0.10 (~0.95)
        spec_baseline_10 = get_val('h_total', 0.10, 'specificity_at_30pct')
        print(f"5. Baseline Specificity at tau=0.10:     {spec_baseline_10:.2f}")

        print("="*70 + "\n")


# ─────────────────────────────────────────────
# FIG 2 — STANDARD UQ FAILURE MAP
# ─────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fig_failure_map(scalars: pd.DataFrame, out_dir: str, n: int = 1000):
    """
    Print-optimized Failure Map (Figure 2) with Hatched N/A handling.
    """
    # set_style()
    df = scalars[scalars['n'] == n].copy()
    metric = 'sensitivity_at_30pct'

    tau_vals = sorted(df['tau'].unique())
    d_vals   = sorted(df['d'].unique())

    def make_grid(method):
        return (df[df['method'] == method]
                .pivot(index='tau', columns='d', values=metric)
                .reindex(index=tau_vals, columns=d_vals).values)

    grid_htotal = make_grid('h_total')
    grid_ccrc   = make_grid('h_tau_ccrc')
    grid_diff   = grid_ccrc - grid_htotal

    # --- 1. Custom Annotation Arrays ---
    def build_annot_matrix(grid, fmt_string):
        annot = np.empty_like(grid, dtype=object)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isnan(grid[i, j]):
                    annot[i, j] = 'N/A'
                else:
                    annot[i, j] = fmt_string.format(grid[i, j])
        return annot

    annot_htotal = build_annot_matrix(grid_htotal, "{:.1f}")
    annot_diff = build_annot_matrix(grid_diff, "{:+.1f}")

    # --- 2. Set NaN background to White ---
    cmap_htotal = plt.get_cmap('RdYlGn').copy()
    cmap_htotal.set_bad(color='white') 

    cmap_diff = plt.get_cmap('RdBu_r').copy()
    cmap_diff.set_bad(color='white')

    # --- 3. Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    kw = dict(
        xticklabels=[f'{d:.1f}' for d in d_vals],
        yticklabels=[f'{t:.2f}' for t in tau_vals],
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.8}
    )

    # Panel A
    sns.heatmap(grid_htotal, ax=axes[0], vmin=0, vmax=1, cmap=cmap_htotal,
                annot=annot_htotal, fmt='', annot_kws={'size': 7}, **kw)
    axes[0].set_title('A', loc='left', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Separability (d)', fontsize=10)
    axes[0].set_ylabel('Class prior  π', fontsize=10)
    axes[0].collections[0].colorbar.set_label('Sensitivity', fontsize=9)

    # Panel B
    vmax_diff = max(0.01, np.nanmax(np.abs(grid_diff)))
    sns.heatmap(grid_diff, ax=axes[1], vmin=-vmax_diff, vmax=vmax_diff,
                cmap=cmap_diff, center=0,
                annot=annot_diff, fmt='', annot_kws={'size': 7}, **kw)
    axes[1].set_title('B', loc='left', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Separability (d)', fontsize=10)
    axes[1].set_ylabel('')
    axes[1].collections[0].colorbar.set_label('Gain in Sensitivity', fontsize=9)

# --- 4. Apply Hatching Pattern to NaN Cells ---
    def add_hatching(ax, grid):
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isnan(grid[i, j]):
                    rect = patches.Rectangle(
                        (j, i), 1, 1,          
                        fill=False,            
                        hatch='....',          # Dense dotted pattern instead of lines
                        edgecolor='#B0B0B0',   # A softer, light gray (#B0B0B0 or #CCCCCC)
                        lw=0,                  
                        zorder=2               
                    )
                    ax.add_patch(rect)

    add_hatching(axes[0], grid_htotal)
    add_hatching(axes[1], grid_diff)

    plt.tight_layout()
    path = os.path.join(out_dir, f'fig2_failure_map_n{n}.pdf')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
# ─────────────────────────────────────────────
# FIG — SEPARABILITY INTERACTION (τ=0.10)
# ─────────────────────────────────────────────

def fig_separability(scalars: pd.DataFrame, out_dir: str, n: int = 1000):
    """
    Single panel: sensitivity at 30% rejection vs d, at tau=0.10.
    Main message: H_Total cannot exploit separability. H_tau can.
    """
    set_style()
    df = scalars[(scalars['n'] == n) & (np.isclose(scalars['tau'], 0.10))].copy()

    fig, ax = plt.subplots(figsize=(9, 6))

    methods_to_show = ['h_total', 'margin', 'h_tau', 'h_tau_ccrc']

    for method in methods_to_show:
        sub = df[df['method'] == method].sort_values('d')
        ax.plot(
            sub['d'], sub['sensitivity_at_30pct'],
            color=PALETTE[method], label=LABELS[method],
            marker='o', linewidth=2.5, markersize=9, zorder=3
        )

    # Annotate the key message at d=1.0
    d1_htotal = df[(df['method']=='h_total') & (np.isclose(df['d'], 1.0))]['sensitivity_at_30pct'].values
    d1_ccrc   = df[(df['method']=='h_tau_ccrc') & (np.isclose(df['d'], 1.0))]['sensitivity_at_30pct'].values

    if len(d1_htotal) > 0 and len(d1_ccrc) > 0:
        ax.annotate(
            f'At identical separability (d=1.0):\n'
            f'H_Total = {d1_htotal[0]:.2f}  |  H_tau+CCR = {d1_ccrc[0]:.2f}\n'
            f'Standard UQ cannot exploit class structure',
            xy=(1.0, d1_htotal[0]),
            xytext=(1.5, 0.30),
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5),
            fontsize=9, color='#C0392B',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FDEDEC', alpha=0.9)
        )

    ax.set_xlabel('Class Separability  (d — mean distance in std units)', fontsize=12)
    ax.set_ylabel('Minority Sensitivity at 30% Rejection', fontsize=12)
    ax.set_title(
        'Standard UQ Cannot Exploit Class Separability at Low τ\n'
        'Even When Progressors Are Biologically Distinct, They Are Rejected  (τ = 0.10,  N=1000)',
        fontsize=12, pad=10
    )
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xticks(sorted(df['d'].unique()))

    ax.fill_between(
        sorted(df['d'].unique()),
        [0]*len(df['d'].unique()), [0.5]*len(df['d'].unique()),
        alpha=0.06, color='#C0392B',
        label='Clinical risk zone (sensitivity < 0.5)'
    )

    path = os.path.join(out_dir, 'fig_separability_interaction.pdf')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ────────────────────────────────────────────
# FIG — REJECTION CURVES (APPENDIX)
# ────────────────────────────────────────────
def fig_rejection_curves(curves: pd.DataFrame, out_dir: str, n: int = 1000, metric='auprc'):
    """
    Print-optimized full rejection curves grid for the Appendix.
    Sized to cleanly occupy an entire standard manuscript page.
    """
    set_style()
    df = curves[curves['n'] == n].copy()

    tau_vals = sorted(df['tau'].unique())
    d_vals   = sorted(df['d'].unique())

    # Map line styles to ensure accessibility (matching main text figures)
    style_map = {
        'h_tau_ccrc': {'ls': '-',  'lw': 2.0, 'zorder': 4},
        'h_tau_ccr':  {'ls': '-',  'lw': 2.0, 'zorder': 4}, # fallback naming
        'h_tau':      {'ls': '--', 'lw': 1.5, 'zorder': 3},
        'margin':     {'ls': '-.', 'lw': 1.5, 'zorder': 2},
        'h_total':    {'ls': ':',  'lw': 1.5, 'zorder': 1}
    }

    # 10x11.5 mimics a full page aspect ratio, keeping fonts proportional
    fig, axes = plt.subplots(len(tau_vals), len(d_vals), figsize=(10, 11.5),
                             sharex=True, sharey=True)

    # Safety catch in case the data only has 1 dimension
    if len(tau_vals) == 1 and len(d_vals) == 1:
        axes = np.array([[axes]])
    elif len(tau_vals) == 1:
        axes = axes[np.newaxis, :]
    elif len(d_vals) == 1:
        axes = axes[:, np.newaxis]

    for i, tau in enumerate(tau_vals):
        for j, d in enumerate(d_vals):
            ax = axes[i, j]
            subset = df[(df['tau'] == tau) & (df['d'] == d)]

            for method in subset['method'].unique():
                method_data = subset[subset['method'] == method]
                
                ls = style_map.get(method, {}).get('ls', '-')
                lw = style_map.get(method, {}).get('lw', 1.5)
                zo = style_map.get(method, {}).get('zorder', 2)

                ax.plot(method_data['rejection_rate'], method_data[metric],
                        label=LABELS.get(method, method), 
                        color=PALETTE.get(method, '#333333'), 
                        linewidth=lw, linestyle=ls, zorder=zo)

            # Context line matching the main figures
            ax.axvline(0.30, color='#bdc3c7', ls='--', lw=1.0, alpha=0.6, zorder=0)

            # Clean subplot titles
            ax.set_title(f'τ = {tau:.2f}  |  d = {d:.1f}', fontsize=10, pad=6)
            
            # Standardized ticks and limits
            ax.tick_params(labelsize=9)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, 0.8) # Keeps the x-axis consistent with earlier plots

            # Only place labels on the outer edges of the grid
            if i == len(tau_vals) - 1:
                ax.set_xlabel('Rejection Rate', fontsize=10)
            if j == 0:
                ax.set_ylabel(metric.upper(), fontsize=10)

    # Reserve the bottom 5% of the page for the global legend
    plt.tight_layout(rect=[0, 0.05, 1, 1], h_pad=1.2, w_pad=1.2)

    # ── Global Legend ──
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(),
                   loc='lower center', 
                   bbox_to_anchor=(0.5, 0.01),
                   ncol=min(4, len(by_label)), 
                   fontsize=11, 
                   frameon=False)

    path = os.path.join(out_dir, f'figC3_rejection_curves_n{n}_{metric}.pdf')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main(scalars_path: str, curves_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    set_style()

    print("Loading synthetic data...")
    scalars = pd.read_csv(scalars_path)
    curves  = pd.read_csv(curves_path)

    print("\nGenerating figures...\n")

    print("sensitivity_stability scaling law...")
    fig_sensitivity_stability(scalars, out_dir, n=300)
    fig_sensitivity_stability(scalars, out_dir, n=1000)

    print("Standard UQ Failure Map...")
    fig_failure_map(scalars, out_dir, n=1000)
    fig_failure_map(scalars, out_dir, n=300)

    print("Fig  — Rejection Curves (full curves in Appendix)...")
    fig_rejection_curves(curves, out_dir, n=300, metric='sensitivity')
    fig_rejection_curves(curves, out_dir, n=1000, metric='sensitivity')
    fig_rejection_curves(curves, out_dir, n=300, metric='auprc')
    fig_rejection_curves(curves, out_dir, n=1000, metric='auprc')

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scalars',  required=True,
                        help='scalar_summaries_full.csv')
    parser.add_argument('--curves',   required=True,
                        help='rejection_curves_full.csv')

    parser.add_argument('--out_dir',  required=True)
    args = parser.parse_args()

    main(args.scalars, args.curves, args.out_dir)