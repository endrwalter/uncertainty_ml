"""
4_print_statistical_results_augrc.py
=============================================================
Loads statistical tests and bootstrap distributions to produce 
printed summary tables for Progressor ccAUGRC, Stable ccAUGRC, 
and Global AUGRC. Applies Holm-Bonferroni correction and outputs 
publication-ready LaTeX tables.

Inputs
------
--tests:   all_statistical_tests.csv (produced by 3_compute_stats.py)
--distrib: Directory containing {disease}_distributions.csv files

Outputs
-------
<out_dir>/statistical_summary.tex
<out_dir>/statistical_summary_cs.tex (Common Support)

Usage
-----
    python 4_print_statistical_results_augrc.py \
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
METHODS_ORDER = ['h_total', 'margin', 'h_tau', 'random_ccr', 'h_total_ccr', 'h_tau_ccr']

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
    'MS': 'MS  (τ=0.13)',
    'PD': 'PD  (τ=0.39)',
    'AD': 'AD  (τ=0.54)',
}

REFERENCE = 'h_tau_ccr'

def apply_holm_bonferroni_column(tests_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Applies the Holm-Bonferroni step-down procedure per disease and metric family.
    Adds a 'passes_hb' boolean column to the dataframe.
    """
    tests = tests_df.copy()
    tests['passes_hb'] = False
    
    # We group by disease and metric. For each combination, we are comparing 
    # m = 5 methods against the reference. This constitutes our "family" of tests.
    for (disease, metric), group in tests.groupby(['disease', 'metric']):
        # Sort by raw p-value ascending
        sorted_group = group.sort_values('p_value')
        m = len(sorted_group)
        
        for k, (idx, row) in enumerate(sorted_group.iterrows()):
            # Holm-Bonferroni threshold: alpha / (m - k)
            # where k is 0-indexed here, so it effectively maps to (m + 1 - (k+1))
            threshold = alpha / (m - k)
            
            if row['p_value'] <= threshold:
                tests.loc[idx, 'passes_hb'] = True
            else:
                # Step-down procedure dictates that once one fails to reject, 
                # all subsequent hypotheses are also retained.
                break 
                
    return tests





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


                if 'global_augrc' in metric:
                    row_parts.append(f"{est_str:>30}")
                elif 'progressor' in metric:
                    row_parts.append(f"{est_str:>30} {sig_str:>12}")
                else:
                    row_parts.append(f"{est_str:>26} {sig_str:>12}")

            print(''.join(row_parts))

        print(f"\n  Reference method: {METHOD_LABELS_SHORT[REFERENCE]}")
        print(f"  *** p<0.001  ** p<0.01  * p<0.05  ns not significant")
        print(f"  ↑worse = comparison method has higher (worse) ccAUGRC than reference")
        print(f"  ↓better = comparison method has lower (better) ccAUGRC than reference")


def save_full_table_latex(tests: pd.DataFrame, distributions: dict, out_path: str, metrics: list):
    """
    Generates a publication-ready LaTeX table matching the strict multirow format.
    Combines point estimates, CIs, and significance superscripts into single cells.
    Colors the significance markers red if they pass Holm-Bonferroni correction.
    """
    # Mapping for LaTeX method names
    latex_method_names = {
        'h_total':     r'$H_{\text{Total}}$ (Global)',
        'margin':      r'Margin (Global)',
        'h_tau':       r'$H_\tau$ (Global)',
        'random_ccr':  r'Random + CCR',
        'h_total_ccr': r'$H_{\text{Total}}$ + CCR',
        'margin_ccr':  r'Margin + CCR',
        'h_tau_ccr':   r'$H_\tau$ + CCR'
    }
        
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
            # Dataset Column
            col_dataset = multirow_def if j == 0 else " "
            
            # Method Column
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
                    
                # 2. Get Significance Superscript (CORRETTO IL CONTROLLO 'global_augrc')
                sup_str = ""
                if 'global_augrc' not in metric and method != REFERENCE and est_str != "--":
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
                        else:
                            # Apply RED color if it passed Holm-Bonferroni
                            if row.get('passes_hb', False):
                                p_marker = f"\\textcolor{{red}}{{{p_marker}}}"
                                
                        sup_str = f"$^{{{direction}{p_marker}}}$"
                
                # Combine Estimate, CI, and Superscript
                cell_str = f"{est_str}{sup_str}"
                
                # Bold the entire cell if it's the reference method
                if method == REFERENCE:
                    cell_str = f"\\textbf{{{cell_str}}}"
                    
                row_cells.append(cell_str)
                
            lines.append(" & ".join(row_cells) + r" \\")
            
    # Close the table and add the Legend
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    
    # Legend detailing the Red text
    lines.append(r"\vspace{1ex}")
    lines.append(r"{\raggedright \footnotesize")
    lines.append(r"Significance vs. Proposed ($H_\tau$ + CCR): *** $p<0.001$, ** $p<0.01$, * $p<0.05$, ns: not significant. \\")
    lines.append(r"Arrows indicate if the baseline is structurally worse ($\uparrow$, higher risk) or better ($\downarrow$, lower risk). \\")
    lines.append(r"Red markers ($\textcolor{red}{*}$) indicate that statistical significance is maintained after Holm-Bonferroni correction ($\alpha=0.05$). \par}")
    
    lines.append(r"\end{table*}")
    
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
    tests_orig = tests[tests['comparison_method'].isin(METHODS_ORDER)]
    tests_cs = tests[tests['comparison_method'].isin(METHODS_ORDER) & tests['metric'].str.startswith('cs_')]
    tests = pd.concat([tests_orig, tests_cs], ignore_index=True)

    distributions = load_distributions(args.distrib)

    # Apply Holm-Bonferroni correction before generating tables
    corrected_tests_orig = apply_holm_bonferroni_column(tests_orig, alpha=0.05)
    corrected_tests_cs = apply_holm_bonferroni_column(tests_cs, alpha=0.05)

    print("\n" + "═"*90)
    print("STATISTICAL SUMMARY TABLE")
    print("═"*90)
    print("Note: Significance levels are adjusted for multiple comparisons using the Holm-Bonferroni method.")
    print_full_table(corrected_tests_orig, distributions, metrics=available_metrics)
    save_full_table_latex(corrected_tests_orig, distributions, out_path=os.path.join(args.out_dir, "statistical_summary.tex"), metrics=available_metrics)

    print("\n" + "═"*90)
    print("STATISTICAL SUMMARY TABLE (Common Support)")
    print("═"*90)
    print("Note: Significance levels are adjusted for multiple comparisons using the Holm-Bonferroni method.")
    print_full_table(corrected_tests_cs, distributions, metrics=available_metrics_cs)
    save_full_table_latex(corrected_tests_cs, distributions, out_path=os.path.join(args.out_dir, "statistical_summary_cs.tex"), metrics=available_metrics_cs)

