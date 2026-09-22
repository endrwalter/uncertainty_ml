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
from scipy.stats import gaussian_kde, wilcoxon # <-- Added wilcoxon

warnings.filterwarnings('ignore')

'''
python 5_plot_results_asymmetry_test.py \
    --real_dir  ../results/real_world_results/ \
    --out_dir   ../figures/paper/
'''
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
# FIG 7 — REAL-WORLD ASYMMETRY TEST
# ─────────────────────────────────────────────

def process_to_deciles(df):
    """
    Dynamically converts distance-binned data into aligned class percentiles.
    Processes classes independently to safely handle NaNs and missing tail data.
    """
    # --- Above tau (Minority / Progressor) ---
    df_above = df[['distance', 'n_above', 'variance_above_tau']].dropna()
    df_above = df_above[df_above['n_above'] > 0].sort_values('distance')
    
    pct_above = (df_above['n_above'].cumsum() / df_above['n_above'].sum()) * 100
    # Prepend 0 to ensure interpolation smoothly covers the first decile
    xp_above = np.concatenate([[0], pct_above.values])
    fp_above = np.concatenate([[df_above['variance_above_tau'].iloc[0]], df_above['variance_above_tau'].values])
    
    # --- Below tau (Majority / Stable) ---
    df_below = df[['distance', 'n_below', 'variance_below_tau']].dropna()
    df_below = df_below[df_below['n_below'] > 0].sort_values('distance')
    
    pct_below = (df_below['n_below'].cumsum() / df_below['n_below'].sum()) * 100
    xp_below = np.concatenate([[0], pct_below.values])
    fp_below = np.concatenate([[df_below['variance_below_tau'].iloc[0]], df_below['variance_below_tau'].values])
    
    # Standardize to deciles (10%, 20%, ... 100%)
    deciles = np.arange(10, 101, 10)
    
    # Interpolate variances at the exact decile marks
    var_above_interp = np.interp(deciles, xp_above, fp_above)
    var_below_interp = np.interp(deciles, xp_below, fp_below)
    
    # Calculate average sample size per decile bin
    n_above_per_decile = df_above['n_above'].sum() / 10
    n_below_per_decile = df_below['n_below'].sum() / 10
    
    return deciles, var_above_interp, var_below_interp, n_above_per_decile, n_below_per_decile


def fig7_asymmetry_test(asymmetry_data: dict, out_dir: str):
    """
    Print-optimized 3 panels (MS, PD, AD): ensemble variance vs distance percentiles.
    Uses significance stars instead of text boxes for a cleaner layout.
    """
    set_style()
    
    # Restored to the more compact 10x4 layout
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    diseases = [
        ('A', 'MS', 'MS Cohort  (π=0.13)', DISEASE_MARKERS['MS']['color']),
        ('B', 'PD', 'PD Cohort  (π=0.39)', DISEASE_MARKERS['PD']['color']),
        ('C', 'AD', 'AD Cohort  (π=0.54)', DISEASE_MARKERS['AD']['color']),
    ]

    for col, (ax, (letter, disease_key, label, color)) in enumerate(zip(axes, diseases)):
        full_title = f"{letter}   {label}"
        
        df = asymmetry_data.get(disease_key)
        if df is None:
            ax.text(0.5, 0.5, f'{disease_key}: data not found',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(full_title, loc='left', fontsize=12, fontweight='bold', pad=8)
            continue

        # 1. Transform data dynamically to deciles
        deciles, v_above, v_below, n_above, n_below = process_to_deciles(df)

        # 2. Plotting
        ax.plot(deciles, v_above,
                color='#C0392B', marker='o', lw=1.8, ms=4,
                label='Progressor class (above τ)')
        ax.plot(deciles, v_below,
                color='#2980B9', marker='o', lw=1.8, ms=4,
                label='Stable class (below τ)')

        ax.fill_between(
            deciles,
            v_above,
            v_below,
            alpha=0.12, color='#C0392B',
            label='Uncertainty asymmetry gap'
        )

        # 3. Statistical Testing (Stars only)
        try:
            stat, p_val = wilcoxon(v_above, v_below)
            
            if p_val < 0.001:
                sig_text = "***"
            elif p_val < 0.01:
                sig_text = "**"
            elif p_val < 0.05:
                sig_text = "*"
            else:
                sig_text = "ns"
            
            # Place the star(s) cleanly in the top right corner of the plot
            ax.text(0.95, 0.95, sig_text, transform=ax.transAxes, 
                    fontsize=12, fontweight='bold', ha='right', va='top')
        except ValueError:
            pass 

        # 4. Formatting
        ax.set_xlabel('Cumulative class proportion (%)', fontsize=10)
        ax.set_xlim(0, 105)
        ax.tick_params(labelsize=9)
        ax.set_title(full_title, loc='left', fontsize=12, fontweight='bold', pad=8)
        
        if col == 0:
            ax.set_ylabel('Mean ensemble variance (proxy)', fontsize=10)

    # Tight layout reserving bottom space for legend
    plt.tight_layout(rect=[0, 0.12, 1, 1], w_pad=1.5)

    # Global single-line legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(),
                   loc='lower center', 
                   bbox_to_anchor=(0.5, 0.01),
                   ncol=len(by_label), 
                   fontsize=10, 
                   frameon=False)

    path = os.path.join(out_dir, 'fig7_asymmetry_test_stat_val.pdf')
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


    # Asymmetry test data
    # Expected: one CSV per disease with [distance, variance_above_tau, variance_below_tau, n_above, n_below]
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