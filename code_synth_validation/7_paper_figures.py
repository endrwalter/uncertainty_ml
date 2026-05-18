"""
paper_figures.py
=============================================================
Generates all publication-ready figures for the paper.

Figures produced
----------------
Fig 1 : Conceptual — Three Pathologies (no data needed)
Fig 2 : Synthetic  — Sensitivity Stability Scaling Law (primary)
Fig 3 : Synthetic  — Standard UQ Failure Map (phase diagram)
Fig 4 : Synthetic  — Separability Interaction at tau=0.10
Fig 5 : Synthetic  — ccAUGRC Conceptual Summary (one condition)


Usage
-----
    python 7_paper_figures.py \
        --scalars   ../data/synthetic_results/scalar_summaries.csv \
        --curves    ../data/synthetic_results/rejection_curves.csv \
        --out_dir   ../figures/paper/
"""

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
    'h_tau_ccrc':   'H_tau + CCRC (Proposed)',
    'h_total_ccrc': 'H_Total + CCRC',
    'margin_ccrc':  'Margin + CCRC',
}
DISEASE_MARKERS = {
    'MS': {'tau': 0.11, 'color': '#8E44AD', 'marker': 'D'},
    'PD': {'tau': 0.29, 'color': '#16A085', 'marker': 's'},
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
# FIG 1 — CONCEPTUAL: THREE PATHOLOGIES
# ─────────────────────────────────────────────

def fig1_three_pathologies(out_dir: str):
    """
    Three-panel conceptual figure.
    No data required — all analytically generated.
    """
    set_style()
    fig = plt.figure(figsize=(18, 5.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    tau = 0.11   # MS-like, most extreme case

    # ── Panel 1: Threshold-Entropy Mismatch ──
    ax1 = fig.add_subplot(gs[0])
    p   = np.linspace(0.001, 0.999, 500)
    H   = -p * np.log2(p) - (1-p) * np.log2(1-p)

    ax1.plot(p, H, color='#2C3E50', lw=2.5, label='Shannon Entropy H(p)')
    ax1.axvline(0.5,  color='#7F8C8D', ls='--', lw=1.5, label='H(p) maximum (p=0.5)')
    ax1.axvline(tau,  color=DISEASE_MARKERS['MS']['color'],
                ls='-', lw=2.0, label=f'Decision boundary (τ={tau})')

    # Shade the mismatch zone
    ax1.axvspan(tau, 0.5, alpha=0.12, color='#C0392B',
                label='Mismatch zone')

    # Annotate the patient example
    p_patient = 0.55
    H_patient = -p_patient*np.log2(p_patient) - (1-p_patient)*np.log2(1-p_patient)
    ax1.annotate(
        f'Confident progressor\n(p={p_patient}) flagged as\nmaximally uncertain',
        xy=(p_patient, H_patient), xytext=(0.62, 0.65),
        arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5),
        fontsize=8.5, color='#C0392B',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', alpha=0.9)
    )
    ax1.scatter([p_patient], [H_patient], color='#C0392B', s=60, zorder=5)

    ax1.set_xlabel('Predicted Probability p')
    ax1.set_ylabel('Binary Entropy H(p)')
    ax1.set_title('Pathology 1\nThreshold–Entropy Mismatch')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.text(0.02, 0.97, 'A', transform=ax1.transAxes,
             fontsize=14, fontweight='bold', va='top')

    # ── Panel 2: Spatial Penalty ──
    ax2 = fig.add_subplot(gs[1])

    stable_max    = tau
    prog_max      = 1 - tau
    categories    = ['Stable Class\n(Majority)', 'Progressor Class\n(Minority)']
    maxima        = [stable_max, prog_max]
    colors_bars   = ['#2980B9', '#C0392B']

    bars = ax2.bar(categories, maxima, color=colors_bars, width=0.45,
                   alpha=0.85, edgecolor='white', linewidth=1.5)

    # Global threshold line
    threshold = 0.20
    ax2.axhline(threshold, color='#2C3E50', ls='--', lw=2,
                label=f'Global rejection threshold')

    # Shade rejected zone for stable class
    ax2.bar(['Stable Class\n(Majority)'], [min(threshold, stable_max)],
            color='#C0392B', alpha=0.35, width=0.45, label='Rejected zone')

    ax2.set_ylabel('Maximum Decision Margin |p − τ|')
    ax2.set_title('Pathology 2\nSpatial Penalty')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)

    # Annotate runway values
    for bar, val, label in zip(bars, maxima, [f'Max = {stable_max:.2f}', f'Max = {prog_max:.2f}']):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                 label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.text(0.02, 0.97, 'B', transform=ax2.transAxes,
             fontsize=14, fontweight='bold', va='top')

    # ── Panel 3: Uncertainty Floor / Probability Compression ──
    ax3 = fig.add_subplot(gs[2])

    rng = np.random.default_rng(42)

    # Majority class: tight cluster near 0
    p_majority = rng.beta(2, 18, 3000) * 0.5
    # Minority class: spread across [tau, 0.60] — compressed by glass ceiling
    p_minority = rng.beta(2, 5, 300) * (0.60 - tau) + tau

    x_range = np.linspace(0, 1, 400)

    kde_maj = gaussian_kde(p_majority, bw_method=0.08)
    kde_min = gaussian_kde(p_minority, bw_method=0.12)

    ax3.fill_between(x_range, kde_maj(x_range), alpha=0.55,
                     color='#2980B9', label='Stable (Majority)')
    ax3.fill_between(x_range, kde_min(x_range), alpha=0.55,
                     color='#C0392B', label='Progressor (Minority)')
    ax3.plot(x_range, kde_maj(x_range), color='#2980B9', lw=1.5)
    ax3.plot(x_range, kde_min(x_range), color='#C0392B', lw=1.5)

    ax3.axvline(tau,  color=DISEASE_MARKERS['MS']['color'],
                lw=2.0, ls='-', label=f'τ = {tau}')
    ax3.axvline(0.60, color='#C0392B', lw=1.5, ls=':',
                label='Epistemic glass ceiling (p=0.60)')

    ax3.annotate(
        'Uncertainty\nFloor',
        xy=(0.50, 0.5), xytext=(0.68, 2.5),
        arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5),
        fontsize=8.5, color='#C0392B',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', alpha=0.9)
    )

    ax3.set_xlabel('Predicted Probability p')
    ax3.set_ylabel('Patient Density')
    ax3.set_title('Pathology 3\nAsymmetric Predictive Capacity')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.text(0.02, 0.97, 'C', transform=ax3.transAxes,
             fontsize=14, fontweight='bold', va='top')

    fig.suptitle(
        'Three Structural Pathologies of Standard Uncertainty Quantification\n'
        'under Clinical Class Imbalance  (illustrated at τ = 0.11)',
        fontsize=13, fontweight='bold', y=1.02
    )

    path = os.path.join(out_dir, 'fig1_three_pathologies.pdf')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# FIG 2 — SENSITIVITY STABILITY (PRIMARY)
# ─────────────────────────────────────────────

def fig2_sensitivity_stability(scalars: pd.DataFrame, out_dir: str, n: int = 1000):
    """
    Full-width primary figure.
    Top: sensitivity stability slope vs tau (main panel)
    Bottom strip: sensitivity and coverage at 30% rejection (supporting)
    """
    set_style()
    df = scalars[scalars['n'] == n].copy()

    methods_to_show = ['h_total', 'h_tau', 'h_tau_ccrc']

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[1.8, 1],
                            hspace=0.42, wspace=0.32)

    # ── Main panel: sensitivity stability ──
    ax_main = fig.add_subplot(gs[0, :])

    for method in methods_to_show:
        sub  = df[df['method'] == method].groupby('tau')['sensitivity_stability']
        mean = sub.mean()
        std  = sub.std()

        ax_main.plot(
            mean.index, mean.values,
            color=PALETTE[method], label=LABELS[method],
            marker='o', linewidth=2.5, markersize=7, zorder=3
        )
        ax_main.fill_between(
            mean.index,
            mean.values - std.values,
            mean.values + std.values,
            color=PALETTE[method], alpha=0.15
        )

    # Zero line — the stability reference
    ax_main.axhline(0, color='#2C3E50', ls='--', lw=1.5, alpha=0.7,
                    label='Stability reference (slope = 0)')

    # Danger zone annotation
    ax_main.axhspan(
        df[df['method']=='h_total']['sensitivity_stability'].min() * 1.05,
        -0.3, alpha=0.06, color='#C0392B'
    )
    ax_main.text(
        0.08, -1.8,
        'Active minority\ndeletion zone',
        fontsize=9, color='#C0392B', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', alpha=0.8)
    )

    # Disease markers
    for disease, props in DISEASE_MARKERS.items():
        ax_main.axvline(
            props['tau'], color=props['color'],
            ls=':', lw=2.0, alpha=0.9
        )
        ax_main.text(
            props['tau'] + 0.005,
            ax_main.get_ylim()[0] if ax_main.get_ylim()[0] < -0.5 else -0.5,
            disease,
            color=props['color'], fontsize=9,
            fontweight='bold', rotation=0
        )

    ax_main.set_xlabel('Class Prior  τ', fontsize=12)
    ax_main.set_ylabel('Sensitivity Stability\n(slope of sensitivity over 0–50% rejection)',
                        fontsize=11)
    ax_main.set_title(
        'Standard UQ Actively Destroys Minority Sensitivity Under Rejection\n'
        'H_tau Maintains Stability Across All Class Priors',
        fontsize=13, pad=10
    )
    ax_main.legend(loc='lower right', fontsize=9)
    ax_main.text(0.01, 0.97, 'A', transform=ax_main.transAxes,
                 fontsize=14, fontweight='bold', va='top')

    # ── Supporting panel left: sensitivity at 30% ──
    ax_s = fig.add_subplot(gs[1, 0])
    for method in methods_to_show:
        sub  = df[df['method'] == method].groupby('tau')['sensitivity_at_30pct']
        mean = sub.mean()
        std  = sub.std()
        ax_s.plot(mean.index, mean.values,
                  color=PALETTE[method], marker='o', lw=2.0, ms=5)
        ax_s.fill_between(mean.index,
                          mean.values - std.values,
                          mean.values + std.values,
                          color=PALETTE[method], alpha=0.15)

    for disease, props in DISEASE_MARKERS.items():
        ax_s.axvline(props['tau'], color=props['color'], ls=':', lw=1.5, alpha=0.8)

    ax_s.set_xlabel('Class Prior  τ')
    ax_s.set_ylabel('Minority Sensitivity @ 30%')
    ax_s.set_title('B   Sensitivity at Fixed Rejection Rate', fontweight='bold')
    ax_s.set_ylim(0, 1.05)

    # ── Supporting panel right: minority coverage at 30% ──
    ax_c = fig.add_subplot(gs[1, 1])
    for method in methods_to_show:
        sub  = df[df['method'] == method].groupby('tau')['minority_coverage_at_30pct']
        mean = sub.mean()
        std  = sub.std()
        ax_c.plot(mean.index, mean.values,
                  color=PALETTE[method], label=LABELS[method],
                  marker='o', lw=2.0, ms=5)
        ax_c.fill_between(mean.index,
                          mean.values - std.values,
                          mean.values + std.values,
                          color=PALETTE[method], alpha=0.15)

    for disease, props in DISEASE_MARKERS.items():
        ax_c.axvline(props['tau'], color=props['color'], ls=':', lw=1.5, alpha=0.8)

    ax_c.set_xlabel('Class Prior  τ')
    ax_c.set_ylabel('Minority Coverage @ 30% Rejection')
    ax_c.set_title('C   Minority Class Coverage at Fixed Rejection', fontweight='bold')
    ax_c.set_ylim(0, 1.05)
    ax_c.legend(fontsize=8)

    path = os.path.join(out_dir, 'fig2_sensitivity_stability.pdf')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# FIG 3 — STANDARD UQ FAILURE MAP
# ─────────────────────────────────────────────

def fig3_failure_map(scalars: pd.DataFrame, out_dir: str, n: int = 1000):
    """
    Phase diagram reframed as a failure map.
    Two panels: H_Total | Difference (Proposed - Baseline)
    Phase boundary prominently marked.
    Disease points as primary visual anchors.
    """
    set_style()
    df = scalars[scalars['n'] == n].copy()
    metric = 'sensitivity_at_30pct'

    tau_vals = sorted(df['tau'].unique())
    d_vals   = sorted(df['d'].unique())

    def make_grid(method):
        return (
            df[df['method'] == method]
            .pivot(index='tau', columns='d', values=metric)
            .reindex(index=tau_vals, columns=d_vals)
            .values
        )

    grid_htotal = make_grid('h_total')
    grid_ccrc   = make_grid('h_tau_ccrc')
    grid_diff   = grid_ccrc - grid_htotal

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        'Standard UQ Failure Map: Where Minority Patients Are Abandoned\n'
        f'(Minority Sensitivity at 30% Rejection,  N={n})',
        fontsize=13, fontweight='bold'
    )

    kw = dict(
        xticklabels=[str(d) for d in d_vals],
        yticklabels=[f'{t:.2f}' for t in tau_vals],
        linewidths=0.8, linecolor='white',
    )

    # Panel A: H_Total failure map
    # Custom colormap: red (failure) → yellow → green (safe)
    sns.heatmap(
        grid_htotal, ax=axes[0],
        vmin=0, vmax=1,
        cmap='RdYlGn',
        annot=True, fmt='.2f', annot_kws={'size': 10},
        **kw
    )
    axes[0].set_title('A   Standard UQ (H_Total)\nMinority Sensitivity at 30% Rejection',
                      fontsize=11, pad=8)
    axes[0].set_xlabel('Separability  (d)', fontsize=11)
    axes[0].set_ylabel('Class Prior  τ', fontsize=11)
    """
    # Draw phase boundary: contour where sensitivity < 0.5
    # Approximate by shading cells where value < 0.5
    for ri, row in enumerate(grid_htotal):
        for ci, val in enumerate(row):
            if not np.isnan(val) and val < 0.5:
                axes[0].add_patch(plt.Rectangle(
                    (ci, ri), 1, 1,
                    fill=False, edgecolor='#C0392B',
                    linewidth=2.5, zorder=5
                ))

    # Add "DANGER ZONE" label
    axes[0].text(
        0.5, 0.15,
        'DANGER ZONE\n(sensitivity < 0.5)',
        transform=axes[0].transAxes,
        fontsize=9, color='#C0392B', fontweight='bold',
        ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#C0392B', alpha=0.9)
    )
    """
    # Panel B: Difference map — the recovery
    vmax_diff = max(0.01, np.nanmax(np.abs(grid_diff)))
    im = sns.heatmap(
        grid_diff, ax=axes[1],
        vmin=-vmax_diff, vmax=vmax_diff,
        cmap='RdBu_r', center=0,
        annot=True, fmt='+.2f', annot_kws={'size': 10},
        **kw
    )
    axes[1].set_title('B   Recovery: H_tau + CCRC vs H_Total\n(Positive = framework recovers minority sensitivity)',
                      fontsize=11, pad=8)
    axes[1].set_xlabel('Separability  (d)', fontsize=11)
    axes[1].set_ylabel('')

    # Disease points — primary visual anchors
    disease_tau_d = {
        'MS\n(τ=0.11)': (0.10, 0.5),
        'PD\n(τ=0.29)': (0.30, 1.0),
        'AD\n(τ=0.54)': (0.50, 2.0),
    }
    for label, (tau_r, d_r) in disease_tau_d.items():
        tau_idx = np.argmin(np.abs(np.array(tau_vals) - tau_r))
        d_idx   = np.argmin(np.abs(np.array(d_vals)   - d_r))

        for ax in axes:

            ax.text(
                d_idx + 0.98, tau_idx + 0.02, label,
                ha='right', va='top',
                fontsize=8, fontweight='bold', color='black',
                zorder=7
            )

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_failure_map.pdf')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# FIG 4 — SEPARABILITY INTERACTION (τ=0.10)
# ─────────────────────────────────────────────

def fig4_separability(scalars: pd.DataFrame, out_dir: str, n: int = 1000):
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
            f'H_Total = {d1_htotal[0]:.2f}  |  H_tau+CCRC = {d1_ccrc[0]:.2f}\n'
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

    path = os.path.join(out_dir, 'fig4_separability_interaction.pdf')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# FIG 5 — ccAUGRC CONCEPTUAL SUMMARY
# ─────────────────────────────────────────────

def fig5_ccaugrc_concept(scalars: pd.DataFrame, out_dir: str, n: int = 1000):
    """
    Simple grouped bar chart for one representative condition.
    tau=0.10, d=0.5 — the MS-like worst case.
    Three bars per method: Global | Progressor | Stable ccAUGRC.
    """
    set_style()

    cond = scalars[
        (scalars['n'] == n) &
        (np.isclose(scalars['tau'], 0.10)) &
        (np.isclose(scalars['d'], 0.5))
    ].copy()

    methods_to_show = ['h_total', 'margin', 'h_tau_ccrc']
    metrics = ['global_augrc', 'ccaugrc_progressor', 'ccaugrc_stable']
    metric_labels = ['Global AUGRC', 'Progressor ccAUGRC', 'Stable ccAUGRC']

    x       = np.arange(len(methods_to_show))
    width   = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(12, 6))

    bar_colors = ['#2C3E50', '#C0392B', '#27AE60']

    for offset, metric, mlabel, bcolor in zip(offsets, metrics, metric_labels, bar_colors):
        vals = [
            cond[cond['method'] == m][metric].values[0]
            if len(cond[cond['method'] == m]) > 0 else np.nan
            for m in methods_to_show
        ]
        bars = ax.bar(
            x + offset, vals, width,
            label=mlabel, color=bcolor,
            alpha=0.85, edgecolor='white', linewidth=1.2
        )
        # Value labels on bars
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.004,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8.5
                )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods_to_show], fontsize=10)
    ax.set_ylabel('AUGRC Score  (lower = safer)', fontsize=11)
    ax.set_title(
        'Global AUGRC Misidentifies the Safer Method\n'
        'ccAUGRC Decomposition Exposes the True Picture\n'
        '(τ = 0.10,  d = 0.5,  N = 1000)',
        fontsize=12, pad=10
    )
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    # Annotations
    ax.annotate(
        'Global AUGRC ranks\nH_Total as safest →\nalgorithmic cowardice',
        xy=(0 - width, cond[cond['method']=='h_total']['global_augrc'].values[0]),
        xytext=(-0.4, 0.22),
        arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5),
        fontsize=8.5, color='#C0392B',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', alpha=0.9)
    )
    ax.annotate(
        'Progressor ccAUGRC\nexposes true risk',
        xy=(0, cond[cond['method']=='h_total']['ccaugrc_progressor'].values[0]),
        xytext=(0.5, 0.28),
        arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5),
        fontsize=8.5, color='#27AE60',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAFAF1', alpha=0.9)
    )

    path = os.path.join(out_dir, 'fig5_ccaugrc_concept.pdf')
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

    # ── Synthetic figures (no real data needed) ──
    print("Fig 1 — Three Pathologies (conceptual)...")
    fig1_three_pathologies(out_dir)

    print("Fig 2 — Sensitivity Stability Scaling Law...")
    fig2_sensitivity_stability(scalars, out_dir, n=1000)

    print("Fig 3 — Standard UQ Failure Map...")
    fig3_failure_map(scalars, out_dir, n=1000)

    print("Fig 4 — Separability Interaction at tau=0.10...")
    fig4_separability(scalars, out_dir, n=1000)

    print("Fig 5 — ccAUGRC Conceptual Summary...")
    fig5_ccaugrc_concept(scalars, out_dir, n=1000)

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