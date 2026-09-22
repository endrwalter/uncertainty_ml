"""
Fig 6 : Real-world — Rejection Curves MS / PD / AD

python 5_plot_results_rejection_curves.py \
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
    Optimized for print-scale readability.
    """
    set_style()
    
    diseases = [
        ('MS', 'MS Cohort\n(π=0.13)'),
        ('PD', 'PD Cohort\n(π=0.39)'),
        ('AD', 'AD Cohort\n(π=0.54)'),
    ]
    
    metrics = [
        ('sensitivity', 'Sensitivity'),
        ('specificity', 'Specificity'),
        ('auprc',       'AUPRC'),
    ]
    
    methods_to_show = ['h_total', 'margin', 'h_tau', 'h_tau_ccr']

    style_map = {
        'h_tau_ccr': {'ls': '-',  'lw': 2.5, 'alpha': 1.0, 'zorder': 4, 'label': 'H-tau + CCR (Proposed)'},
        'h_tau':     {'ls': '--', 'lw': 1.5, 'alpha': 0.8, 'zorder': 3, 'label': 'H-tau (Global)'},
        'margin':    {'ls': '-.', 'lw': 1.5, 'alpha': 0.8, 'zorder': 2, 'label': 'Margin (Global)'},
        'h_total':   {'ls': ':',  'lw': 1.5, 'alpha': 0.8, 'zorder': 1, 'label': 'H-Total (Global)'}
    }

    # Reduced figsize to simulate actual print dimensions. 
    # This forces matplotlib to render fonts and lines proportionally larger.
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    letters = "ABCDEFGHI"

    for row, (disease_key, disease_label) in enumerate(diseases):
        df = real_data.get(disease_key)
        if df is None:
            for col in range(3):
                axes[row][col].text(0.5, 0.5, 'Data Missing', ha='center', va='center')
                axes[row][col].set_title(letters[row * 3 + col], loc='left', fontsize=12, fontweight='bold')
            continue

        for col, (metric, metric_name) in enumerate(metrics):
            ax = axes[row][col]
            plot_idx = row * 3 + col

            for method in methods_to_show:
                sub = df[df['method'] == method].sort_values('rejection_rate')
                style = style_map[method]
                
                ax.plot(
                    sub['rejection_rate'], sub[metric],
                    color=PALETTE[method], label=style['label'],
                    linewidth=style['lw'], linestyle=style['ls'], 
                    alpha=style['alpha'], zorder=style['zorder']
                )


            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, 0.80)
            
            # Adjust tick label size for print
            ax.tick_params(axis='both', which='major', labelsize=9)

            if col == 0:
                ax.set_ylabel(disease_label, fontsize=11, fontweight='bold', labelpad=10)
            else:
                ax.set_ylabel('')

            if row == 2:
                ax.set_xlabel('Rejection Rate', fontsize=10)
            else:
                ax.set_xlabel('')

            if row == 0:
                ax.set_title(metric_name, loc='center', fontsize=12, fontweight='bold', pad=8)

            ax.set_title(letters[plot_idx], loc='left', fontsize=12, fontweight='bold', pad=8)

    # Tightened padding so the subplots feel cohesive and don't waste canvas space
    plt.tight_layout(rect=[0, 0.08, 1, 1], h_pad=1.5, w_pad=1.5)

    handles, labels = axes[0][0].get_legend_handles_labels()
    
    if handles:
        by_label = dict(zip(labels, handles))
        ordered_handles = [by_label[style_map[m]['label']] for m in methods_to_show if style_map[m]['label'] in by_label]
        ordered_labels = [style_map[m]['label'] for m in methods_to_show if style_map[m]['label'] in by_label]
        
        # Switched to 2 columns if the figure is narrower, or 4 if it fits. 
        # Using 2 columns stacked ensures it doesn't spill off the edges on a 10-inch width.
        fig.legend(ordered_handles, ordered_labels,
                   loc='lower center', 
                   bbox_to_anchor=(0.5, 0.01),
                   ncol=2, 
                   fontsize=10, 
                   frameon=False)

    path = os.path.join(out_dir, 'fig6_real_rejection_curves.pdf')
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