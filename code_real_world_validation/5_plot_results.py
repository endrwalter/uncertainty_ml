"""
Fig 6 : Real-world — Rejection Curves MS / PD / AD
Fig 7 : Real-world — Asymmetry Test
Fig 8 : Real-world — ccAUGRC Decomposition Table

python 5_plot_results.py \
    --real_dir  ../results/real_world_results/ \
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
    'h_tau_ccr':   '#27AE60',   # forest green — proposed
    'h_total_ccr': '#8E44AD',   # purple
    'margin_ccr':  '#16A085',   # teal
}
LABELS = {
    'h_total':      'H_Total (Global)',
    'margin':       'Margin (Global)',
    'h_tau':        'H_tau (Global)',
    'h_tau_ccr':   'H_tau + ccr (Proposed)',
    'h_total_ccr': 'H_Total + ccr',
    'margin_ccr':  'Margin + ccr',
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
# FIG 6 — REAL-WORLD REJECTION CURVES
# ─────────────────────────────────────────────

def fig6_real_rejection_curves(real_data: dict, out_dir: str):
    """
    3 rows (MS, PD, AD) × 3 cols (sensitivity, specificity, AUPRC).
    real_data: dict with keys 'MS', 'PD', 'AD', each a DataFrame with
               columns [rejection_rate, method, sensitivity, specificity, auprc]
    """
    set_style()
    diseases = [
        ('MS', 'MS Cohort  (τ=0.13, Severe Imbalance)',  DISEASE_MARKERS['MS']['color']),
        ('PD', 'PD Cohort  (τ=0.39, Moderate Imbalance)', DISEASE_MARKERS['PD']['color']),
        ('AD', 'AD Cohort  (τ=0.54, Near-Balanced)',      DISEASE_MARKERS['AD']['color']),
    ]
    metrics = [
        ('sensitivity', 'Sensitivity\n(Minority Protection)'),
        ('specificity', 'Specificity\n(Majority Protection)'),
        ('auprc',       'AUPRC\n(Overall Clinical Utility)'),
    ]
    methods_to_show = ['h_total', 'margin', 'h_tau', 'h_tau_ccr']

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(
        'Real-World Validation: Rejection Curves Across Neurodegenerative Cohorts\n'
        'Results Anchor to Synthetic Phase Diagram Predictions',
        fontsize=13, fontweight='bold'
    )

    for row, (disease_key, disease_label, disease_color) in enumerate(diseases):
        df = real_data.get(disease_key)
        if df is None:
            for col in range(3):
                axes[row][col].text(0.5, 0.5, f'{disease_key}: data not found',
                                    ha='center', va='center', transform=axes[row][col].transAxes)
            continue

        for col, (metric, ylabel) in enumerate(metrics):
            ax = axes[row][col]

            for method in methods_to_show:
                sub = df[df['method'] == method].sort_values('rejection_rate')
                ls  = '--' if method == 'h_total' else '-'
                lw  = 2.5 if method == 'h_tau_ccr' else 1.8
                ax.plot(
                    sub['rejection_rate'], sub[metric],
                    color=PALETTE[method], label=LABELS[method],
                    linewidth=lw, linestyle=ls, zorder=3
                )

            ax.axvline(0.30, color='#7F8C8D', ls=':', lw=1.2, alpha=0.6)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, 0.80)

            if row == 0 and col == 0:
                ax.legend(fontsize=7.5, loc='lower right')

            if col == 0:
                ax.set_ylabel(f'{disease_label}\n\n{ylabel}', fontsize=9,
                              color=disease_color)
                ax.yaxis.label.set_color(disease_color)
            else:
                ax.set_ylabel(ylabel, fontsize=9)

            if row == 2:
                ax.set_xlabel('Rejection Rate', fontsize=10)

            if row == 0:
                ax.set_title(ylabel, fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig6_real_rejection_curves.pdf')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────
# FIG 7 — REAL-WORLD ASYMMETRY TEST
# ─────────────────────────────────────────────

def fig7_asymmetry_test(asymmetry_data: dict, out_dir: str):
    """
    3 panels (MS, PD, AD): ensemble variance vs distance from boundary.
    asymmetry_data: dict with keys 'MS', 'PD', 'AD', each a DataFrame with
                    columns [distance, variance_above_tau, variance_below_tau]
    """
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'Empirical Confirmation of Pathology 3: Asymmetric Predictive Capacity\n'
        'At Identical Distance from τ, Minority Patients Carry Systematically Higher Uncertainty',
        fontsize=12, fontweight='bold'
    )

    diseases = [
        ('MS', 'MS Cohort  (τ=0.11)', DISEASE_MARKERS['MS']['color']),
        ('PD', 'PD Cohort  (τ=0.29)', DISEASE_MARKERS['PD']['color']),
        ('AD', 'AD Cohort  (τ=0.54)', DISEASE_MARKERS['AD']['color']),
    ]

    for ax, (disease_key, label, color) in zip(axes, diseases):
        df = asymmetry_data.get(disease_key)
        if df is None:
            ax.text(0.5, 0.5, f'{disease_key}: data not found',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        ax.plot(df['distance'], df['variance_above_tau'],
                color='#C0392B', marker='o', lw=2.2, ms=6,
                label='Progressor class (above τ)')
        ax.plot(df['distance'], df['variance_below_tau'],
                color='#2980B9', marker='o', lw=2.2, ms=6,
                label='Stable class (below τ)')

        ax.fill_between(
            df['distance'],
            df['variance_above_tau'],
            df['variance_below_tau'],
            alpha=0.12, color='#C0392B',
            label='Uncertainty asymmetry gap'
        )

        ax.set_xlabel('Distance from Decision Boundary |p − τ|', fontsize=10)
        ax.set_ylabel('Mean Ensemble Variance', fontsize=10)
        ax.set_title(label, fontsize=11, color=color, fontweight='bold')
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig7_asymmetry_test.pdf')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def main(real_dir: str, out_dir: str):
  
    # ── Real-world figures (load from real_dir) ──
    print("\nLoading real-world data...")

    # Expected: one CSV per disease in real_dir
    # Each CSV: [rejection_rate, method, sensitivity, specificity, auprc]
    real_data = {}
    for disease in ['MS', 'PD', 'AD']:
        fpath = os.path.join(real_dir, f'{disease.lower()}_rejection_curves.csv')
        if os.path.exists(fpath):
            real_data[disease] = pd.read_csv(fpath)
            print(f"  Loaded {disease}: {fpath}")
        else:
            print(f"  [SKIP] {disease}: {fpath} not found")

    if real_data:
        print("Fig 6 — Real-World Rejection Curves...")
        fig6_real_rejection_curves(real_data, out_dir)

    # Asymmetry test data
    # Expected: one CSV per disease with [distance, variance_above_tau, variance_below_tau]
    asym_data = {}
    for disease in ['MS', 'PD', 'AD']:
        fpath = os.path.join(real_dir, f'{disease.lower()}_asymmetry.csv')
        if os.path.exists(fpath):
            asym_data[disease] = pd.read_csv(fpath)

    if asym_data:
        print("Fig 7 — Asymmetry Test...")
        fig7_asymmetry_test(asym_data, out_dir)



# main 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot real-world validation results.")
    parser.add_argument('--real_dir', type=str, required=True,
                        help='Directory containing real-world CSVs (e.g., rejection curves, asymmetry test)')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory to save the generated figures')
    args = parser.parse_args()

    real_dir = args.real_dir
    out_dir  = args.out_dir

    main(real_dir, out_dir)