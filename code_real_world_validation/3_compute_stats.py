import argparse
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from rejection_utils import (
    DISEASES, METHODS, compute_common_support_areas, compute_signals, compute_ccaugrc, compute_global_augrc
)

warnings.filterwarnings('ignore')


"""
=============================================================
Computes bootstrap confidence intervals and statistical tests
for ccAUGRC differences between methods.

    Bootstraps the full aggregated ensemble (deployment reality).
    Gives 95% CI on the difference between methods.

Usage
-----
    # From aggregated summary 
    python 3_compute_stats.py \
        --ms_summary ../results/classification/ms_progression/progression_independent_from_relapses/aggregated/patient_mean_probs_progression_independent_from_relapses_ms_progression_model.csv \
        --pd_summary ../results/classification/pd_dyskinesia/FutureDyskinesia/aggregated/patient_mean_probs_FutureDyskinesia_pd_dyskinesia_model.csv \
        --ad_summary ../results/classification/mci_ad_conversion/label_bl_36m/aggregated/patient_mean_probs_label_bl_36m_mci_ad_conversion_model.csv \
        --out_dir ../results/statistical_tests/

"""




N_BOOTSTRAP = 1000

def compute_all_metrics(df: pd.DataFrame, tau: float) -> dict:
    df = compute_signals(df, tau)
    
    # 1. Temporarily store the full curve/support data for each method
    raw_progressor = {}
    raw_stable = {}
    raw_global = {}
    
    for method in METHODS:
        raw_progressor[method] = compute_ccaugrc(df, method, tau, 1)
        raw_stable[method]     = compute_ccaugrc(df, method, tau, 0)
        raw_global[method]     = compute_global_augrc(df, method, tau)
        
    # 2. Calculate the common-support areas across all methods for each metric
    cs_progressor = compute_common_support_areas(raw_progressor)
    cs_stable     = compute_common_support_areas(raw_stable)
    cs_global     = compute_common_support_areas(raw_global)
    
    # 3. Assemble the final reporting dictionary
    records = {}
    for method in METHODS:
        records[method] = {
            # Original Method-Dependent Areas
            'ccaugrc_progressor': raw_progressor[method]['area'],
            'ccaugrc_stable':     raw_stable[method]['area'],
            'global_augrc':       raw_global[method]['area'],
            
            # Report the support for each area
            'support_progressor': raw_progressor[method]['support'],
            'support_stable':     raw_stable[method]['support'],
            'support_global':     raw_global[method]['support'],
            
            # Common-Support Areas
            'cs_ccaugrc_progressor': cs_progressor.get(method, np.nan),
            'cs_ccaugrc_stable':     cs_stable.get(method, np.nan),
            'cs_global_augrc':       cs_global.get(method, np.nan),
        }
        
    return records

def approach2_bootstrap(summary_df: pd.DataFrame, tau: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    for b in range(N_BOOTSTRAP):
        idx = rng.choice(len(summary_df), size=len(summary_df), replace=True)
        sample = summary_df.iloc[idx].reset_index(drop=True)

        if sample['label'].nunique() < 2: continue

        metrics = compute_all_metrics(sample, tau)
        for method, vals in metrics.items():
            records.append({'bootstrap_iter': b, 'method': method, **vals})

        if (b + 1) % 200 == 0:
            print(f"      Bootstrap {b+1}/{N_BOOTSTRAP}")

    return pd.DataFrame(records)

def compute_ci_and_tests(iter_df: pd.DataFrame, disease: str, reference: str = 'h_tau_ccr', alpha: float = 0.05) -> pd.DataFrame:
    # Safely handle METHODS whether it's a list or a dict
    method_list = list(METHODS.keys()) if isinstance(METHODS, dict) else METHODS
    comparisons = [m for m in method_list if m != reference]
    
    records = []
    # Added the common-support metrics to the statistical testing loop
    metrics = [
        'ccaugrc_progressor', 'ccaugrc_stable', 'global_augrc',
        'cs_ccaugrc_progressor', 'cs_ccaugrc_stable', 'cs_global_augrc'
    ]

    for metric in metrics:
        if metric not in iter_df.columns:
            continue
            
        ref_series = iter_df[iter_df['method'] == reference].set_index('bootstrap_iter')[metric]
        ref_vals = ref_series.dropna().values
        
        if len(ref_vals) == 0: continue
        
        # FIX 2: Use nanpercentile to ignore any NaNs from the common-support calculations
        ref_ci_low, ref_ci_high = np.nanpercentile(ref_vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])

        for method in comparisons:
            comp_series = iter_df[iter_df['method'] == method].set_index('bootstrap_iter')[metric]
            common = ref_series.dropna().index.intersection(comp_series.dropna().index)

            if len(common) == 0: continue

            ref_al = ref_series.loc[common].values
            comp_al = comp_series.loc[common].values
            diff = comp_al - ref_al  # positive = comparison worse

            comp_ci_low, comp_ci_high = np.nanpercentile(comp_al, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            diff_ci_low, diff_ci_high = np.nanpercentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])

            prop_same_sign = (diff > 0).mean() if diff.mean() > 0 else (diff < 0).mean()
            pval = 2 * min(prop_same_sign, 1 - prop_same_sign)

            records.append({
                'disease': disease, 'metric': metric, 'reference_method': reference,
                'reference_mean': np.nanmean(ref_al), 'reference_ci_low': ref_ci_low, 'reference_ci_high': ref_ci_high,
                'comparison_method': method, 'comparison_mean': np.nanmean(comp_al),
                'comparison_ci_low': comp_ci_low, 'comparison_ci_high': comp_ci_high,
                'mean_difference': np.nanmean(diff), 'diff_ci_low': diff_ci_low, 'diff_ci_high': diff_ci_high,
                'ci_excludes_zero': not (diff_ci_low <= 0 <= diff_ci_high),
                'p_value': pval, 'test_name': 'Bootstrap sign test',
                'cohens_d': np.nanmean(diff) / (np.nanstd(diff) + 1e-10), 'n_samples': len(common),
            })

    return pd.DataFrame(records)


def run_disease(disease: str, summary_path: str, out_dir: str):
    if not summary_path or not os.path.exists(summary_path): return None
    tau = DISEASES[disease]['tau']
    print(f"\n{'═'*60}\n{disease}  (τ={tau})\n{'═'*60}")

    summary = pd.read_csv(summary_path)
    if 'mu' not in summary.columns: raise ValueError("Summary must have 'mu' column.")
    
    iter_df = approach2_bootstrap(summary, tau)
    iter_df.to_csv(os.path.join(out_dir, f'{disease.lower()}_distributions.csv'), index=False)

    tests = compute_ci_and_tests(iter_df, disease)
    tests.to_csv(os.path.join(out_dir, f'{disease.lower()}_statistical_tests.csv'), index=False)
    
    print("\n  Point estimates with 95% CI:")
    
    # Safe iteration over METHODS
    method_items = METHODS.items() if isinstance(METHODS, dict) else {m: m for m in METHODS}.items()
    
    for method, label in method_items:
        # Print Original
        vals = iter_df[iter_df['method'] == method]['ccaugrc_progressor'].dropna().values
        if len(vals) > 0:
            print(f"  {label:<32} (Original)       {np.mean(vals):.4f}  [{np.percentile(vals, 2.5):.4f}, {np.percentile(vals, 97.5):.4f}]")
            
        # Print Common Support
        cs_vals = iter_df[iter_df['method'] == method]['cs_ccaugrc_progressor'].dropna().values
        if len(cs_vals) > 0:
            print(f"  {label:<32} (Common Support) {np.mean(cs_vals):.4f}  [{np.percentile(cs_vals, 2.5):.4f}, {np.percentile(cs_vals, 97.5):.4f}]")

    return tests

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms_summary', default=None)
    parser.add_argument('--pd_summary', default=None)
    parser.add_argument('--ad_summary', default=None)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    all_tests = []
    for disease in ['MS', 'PD', 'AD']:
        path = getattr(args, f'{disease.lower()}_summary')
        res = run_disease(disease, path, args.out_dir)
        if res is not None: all_tests.append(res)

    if all_tests:
        pd.concat(all_tests, ignore_index=True).to_csv(os.path.join(args.out_dir, 'all_statistical_tests.csv'), index=False)