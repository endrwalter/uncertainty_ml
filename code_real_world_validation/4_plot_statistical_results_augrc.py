"""
plot_statistical_results.py
=============================================================
Loads all_statistical_tests.csv and produces:

1. A printed summary table for both Progressor and Stable ccAUGRC
2. A figure: grouped bar chart with 95% CI error bars
   and significance annotations for all three diseases

Usage
-----
    python 4_plot_statistical_results_augrc.py \
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
METHODS_ORDER = ['h_total', 'margin', 'h_tau', 'random_ccr', 'h_total_ccr', 'margin_ccr', 'h_tau_ccr']

METHOD_LABELS = {
    'h_total':    'H_Total\n(Global)',
    'margin':     'Margin\n(Global)',
    'h_tau':      'H_tau\n(Global)',
    'random_ccr': 'Random+ccr\n(Control)',
    'h_tau_ccr':  'H_tau+ccr\n(Proposed)',
    'h_total_ccr': 'H_Total + CCR (Control)',
    'margin_ccr': 'Margin + CCR (Control)',
}

METHOD_LABELS_SHORT = {
    'h_total':    'H_Total',
    'margin':     'Margin',
    'h_tau':      'H_tau',
    'random_ccr': 'Random+ccr',
    'h_tau_ccr':  'H_tau+ccr',
    'h_total_ccr': 'H_Total+ccr',
    'margin_ccr': 'Margin+ccr',
}

PALETTE = {
    'h_total':    '#C0392B', # Red
    'margin':     '#E67E22', # Orange
    'h_tau':      '#2980B9', # Blue
    'random_ccr': '#8E44AD', # Purple (New for Random+ccr)
    'h_tau_ccr':  '#27AE60', # Green
    'h_total_ccr': "#636161", # Gray
    'margin_ccr': "#D9B112", # Light Blue
}

DISEASES_ORDER = ['MS', 'PD', 'AD']
DISEASE_LABELS = {
    'MS': 'MS  (τ=0.11)',
    'PD': 'PD  (τ=0.39)',
    'AD': 'AD  (τ=0.54)',
}

REFERENCE = 'h_tau_ccr'



# ─────────────────────────────────────────────
# PRINTED TABLE
# ─────────────────────────────────────────────

def sig_marker(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def print_full_table(tests: pd.DataFrame, distributions: dict, metrics = [
        ('ccaugrc_progressor', 'Progressor ccAUGRC'),
        ('ccaugrc_stable',     'Stable ccAUGRC'),
        ('global_augrc',       'Global AUGRC'),
    ]):
    """
    Print a complete table with point estimates [95% CI] and
    significance markers for both Progressor and Stable ccAUGRC.
    """


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


def save_full_table_latex(tests: pd.DataFrame, distributions: dict, out_path: str):
    """
    Generates a publication-ready LaTeX table matching the strict multirow format.
    Combines point estimates, CIs, and significance superscripts into single cells.
    """
    # Mapping for  LaTeX method names
    latex_method_names = {
        'h_total':     r'$H_{\text{Total}}$ (Global)',
        'margin':      r'Margin (Global)',
        'h_tau':       r'$H_\tau$ (Global)',
        'random_ccr':  r'Random + CCR',
        'h_total_ccr': r'$H_{\text{Total}}$ + CCR',
        'margin_ccr':  r'Margin + CCR',
        'h_tau_ccr':   r'$H_\tau$ + CCR'
    }

    metrics = [
        ('ccaugrc_progressor', 'Progressor ccAUGRC'),
        ('ccaugrc_stable',     'Stable ccAUGRC'),
        ('global_augrc',       'Global AUGRC'),
    ]
        
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{l l l l l}")
    lines.append(r"\toprule")
    
    # Header Row
    lines.append(r"Dataset & Uncertainty Framework & Progressor ccAUGRC (95\% CI) & Stable ccAUGRC (95\% CI) & Global AUGRC (95\% CI) \\")
    
    num_methods = len(METHODS_ORDER)

    for i, disease in enumerate(DISEASES_ORDER):
        lines.append(r"\midrule")
        
        # Parse "MS  (τ=0.11)" into "MS" and "0.11"
        raw_label = DISEASE_LABELS[disease]
        dis_name = raw_label.split(' ')[0]
        tau_val = raw_label.split('=')[1].replace(')', '')
        
        multirow_def = f"\\multirow{{{num_methods}}}{{*}}{{\\textbf{{{dis_name}}} ($\\tau={tau_val}$)}}"
        
        dist_df = distributions.get(disease)
        
        for j, method in enumerate(METHODS_ORDER):
            # Dataset Column (only populated on the first row of the disease block)
            col_dataset = multirow_def if j == 0 else " "
            
            # Method Column (bold if it's the reference)
            meth_name = latex_method_names.get(method, method)
            col_method = f"\\textbf{{{meth_name}}}" if method == REFERENCE else meth_name
            
            row_cells = [col_dataset, col_method]
            
            for metric, _ in metrics:
                # 1. Get Point Estimate and CI
                if dist_df is not None and method in dist_df['method'].values:
                    vals = dist_df[dist_df['method'] == method][metric].dropna().values
                    if len(vals) > 0:
                        mean_val = np.nanmean(vals)
                        ci_low   = np.nanpercentile(vals, 2.5)
                        ci_high  = np.nanpercentile(vals, 97.5)
                        est_str  = f"{mean_val:.4f} [{ci_low:.4f}, {ci_high:.4f}]"
                    else:
                        est_str = "--"
                else:
                    est_str = "--"
                    
                # 2. Get Significance Superscript (Skip for Global AUGRC and Reference Method)
                sup_str = ""
                if metric != 'global_augrc' and method != REFERENCE and est_str != "--":
                    sub = tests[
                        (tests['disease'] == disease) &
                        (tests['metric'] == metric) &
                        (tests['comparison_method'] == method)
                    ]
                    if len(sub) > 0:
                        row = sub.iloc[0]
                        direction = r"\uparrow" if row['mean_difference'] > 0 else r"\downarrow"
                        p_marker = sig_marker(row['p_value'])
                        if p_marker == 'ns': 
                            p_marker = r"\text{ns}"
                        sup_str = f"$^{{{direction}{p_marker}}}$"
                
                # Combine Estimate, CI, and Superscript
                cell_str = f"{est_str}{sup_str}"
                
                # Bold the entire cell if it's the reference method
                if method == REFERENCE:
                    cell_str = f"\\textbf{{{cell_str}}}"
                    
                row_cells.append(cell_str)
                
            # Join row with & and end with \\
            lines.append(" & ".join(row_cells) + r" \\")
            
    # Close the table
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    
    # Save to file
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
        
    print(f"\nLaTeX table saved successfully to: {out_path}")

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

    available_metrics = [
            ('ccaugrc_progressor', 'Progressor ccAUGRC'),
            ('ccaugrc_stable',     'Stable ccAUGRC'),
            ('global_augrc',       'Global AUGRC')
    ]


    available_metrics_cs = [
            ('cs_ccaugrc_progressor', 'Progressor ccAUGRC (Common Support)'),
            ('cs_ccaugrc_stable',     'Stable ccAUGRC (Common Support)'),
            ('cs_global_augrc',       'Global AUGRC (Common Support)'),
        ]

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
    print_full_table(tests, distributions, metrics = available_metrics)
    save_full_table_latex(tests, distributions, out_path=os.path.join(args.out_dir, "statistical_summary.tex"))

