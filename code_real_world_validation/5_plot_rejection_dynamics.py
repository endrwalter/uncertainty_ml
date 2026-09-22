"""
5_plot_rejection_dynamics.py
=============================================================
Generates a 2x3 grid plot illustrating the rejection queue dynamics 
(errors deferred vs. correct cases deferred in D0 and D1 queues) 
across MS, PD, and AD cohorts.

Inputs
------
--input: all_diseases_combined_dynamics.csv 
         (produced by 2_compute_uncertainty_metrics.py)

Outputs
-------
<out_dir>/figD1_reject_dynamics_clean.pdf

Usage
-----
python 5_plot_rejection_dynamics.py \
    --input ../results/real_world_results/all_diseases_combined_dynamics.csv \
    --out_dir ../figures/paper/
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
PALETTE = {
    'h_total':      '#C0392B',   # deep red
    'h_tau_ccr':    '#27AE60',   # forest green — proposed
    'h_total_ccr':  '#8E44AD',   # purple
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
# REBUTTAL PLOT (QUEUE DYNAMICS)
# ─────────────────────────────────────────────
def plot_rebuttal_queue_dynamics(csv_path, out_path):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find input file at '{csv_path}'")
        return

    df = pd.read_csv(csv_path)
    set_style()
    
    # 2x3 grid, scaled for print readability similar to Fig 6 (10 inches wide)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), sharey=True, sharex=True)
    letters = "ABCDEF"
    
    diseases = [
        ('MS', 'MS Cohort'),
        ('PD', 'PD Cohort'),
        ('AD', 'AD Cohort')
    ]
    
    # Plotting order: Baseline (Bottom), Ablation (Thick/Transparent Middle), Proposed (Thin Top)
    methods_to_plot = [
        ('h_total',     PALETTE['h_total'],     2.0, 1.0, 'Baseline (Global $H_{Total}$)'),
        ('h_total_ccr', PALETTE['h_total_ccr'], 5.0, 0.4, 'Ablation ($H_{Total}$ + CCR)'),
        ('h_tau_ccr',   PALETTE['h_tau_ccr'],   2.0, 1.0, 'Proposed ($H_\\tau$ + CCR)')
    ]
    
    for col, (disease_key, disease_label) in enumerate(diseases):
        dis_df = df[df['disease'] == disease_key]
        
        for method_key, color, lw, alpha, label in methods_to_plot:
            meth_df = dis_df[dis_df['method'] == method_key].sort_values('rejection_rate')
            if meth_df.empty: continue
            
            x_vals = meth_df['rejection_rate']
            
            # --- ROW 0: D0 Queue ---
            axes[0, col].plot(x_vals, meth_df['fn_rejection_pct'], 
                              color=color, linewidth=lw, linestyle='-', marker='o', markersize=3, alpha=alpha)
            axes[0, col].plot(x_vals, meth_df['tn_rejection_pct'], 
                              color=color, linewidth=lw*0.7, linestyle=':', alpha=alpha)

            # --- ROW 1: D1 Queue ---
            axes[1, col].plot(x_vals, meth_df['fp_rejection_pct'], 
                              color=color, linewidth=lw, linestyle='-', marker='o', markersize=3, alpha=alpha)
            axes[1, col].plot(x_vals, meth_df['tp_rejection_pct'], 
                              color=color, linewidth=lw*0.7, linestyle=':', alpha=alpha)

        # Formatting Subplots
        plot_idx_0 = col
        plot_idx_1 = 3 + col
        
        # Subplot Letters (Left) and Centered Titles (Row 0 only)
        axes[0, col].set_title(letters[plot_idx_0], loc='left', fontsize=12, fontweight='bold')
        axes[0, col].set_title(disease_label, loc='center', fontsize=12, fontweight='bold', pad=8)
        axes[1, col].set_title(letters[plot_idx_1], loc='left', fontsize=12, fontweight='bold')
        
        axes[1, col].set_xlabel('Rejection Rate', fontsize=10)
        axes[1, col].tick_params(axis='both', which='major', labelsize=9)
        axes[0, col].tick_params(axis='both', which='major', labelsize=9)
        
        axes[0, col].set_xlim(0, 0.80)
        axes[0, col].set_ylim(-2, 105) 
        
    axes[0, 0].set_ylabel('% Rejected from $D_0$', fontsize=11, fontweight='bold', labelpad=10)
    axes[1, 0].set_ylabel('% Rejected from $D_1$', fontsize=11, fontweight='bold', labelpad=10)
            
    # Unified Legend Construction
    custom_lines = [
        Line2D([0], [0], color=PALETTE['h_tau_ccr'], lw=2.0),
        Line2D([0], [0], color=PALETTE['h_total_ccr'], lw=5.0, alpha=0.4),
        Line2D([0], [0], color=PALETTE['h_total'], lw=2.0),
        Line2D([0], [0], color='none'), 
        Line2D([0], [0], color='black', lw=2.0, linestyle='-', marker='o', markersize=3),
        Line2D([0], [0], color='black', lw=1.5, linestyle=':')
    ]
    custom_labels = [
        'Proposed ($H_\\tau$ + CCR)',
        'Ablation ($H_{Total}$ + CCR)',
        'Baseline (Global $H_{Total}$)',
        '',
        'Errors Deferred (FNs in $D_0$, FPs in $D_1$)',
        'Correct Cases Deferred (TNs in $D_0$, TPs in $D_1$)'
    ]

    # Matching the compact 2-column legend style from Fig 6
    fig.legend(custom_lines, custom_labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), 
               ncol=2, frameon=False, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.12, 1, 1], h_pad=1.5, w_pad=1.5)
    
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path)
    print(f"\nPlot saved to: {os.path.abspath(out_path)}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to combined dynamics CSV")
    parser.add_argument('--out_dir', type=str, default='../figures/paper/')
    args = parser.parse_args()
    
    output_filepath = os.path.join(args.out_dir, 'figD1_reject_dynamics_clean.pdf')
    plot_rebuttal_queue_dynamics(args.input, output_filepath)