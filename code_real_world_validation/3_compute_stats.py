import argparse
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from rejection_utils import (
    DISEASES, METHODS, compute_signals, compute_ccaugrc, compute_global_augrc
)

warnings.filterwarnings('ignore')


"""
ccaugrc_statistical_test_v2.py
=============================================================
Computes bootstrap confidence intervals and statistical tests
for ccAUGRC differences between methods.

Automatically selects the appropriate approach based on
minority class size per test split.


Approach-> patient bootstrap: used when minority patients
    per split <= 30, or when only aggregated data is available.
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
    records = {}
    for method in METHODS:
        records[method] = {
            'ccaugrc_progressor': compute_ccaugrc(df, method, tau, 1),
            'ccaugrc_stable':     compute_ccaugrc(df, method, tau, 0),
            'global_augrc':       compute_global_augrc(df, method, tau),
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
    comparisons = [m for m in METHODS if m != reference]
    records = []
    metrics = ['ccaugrc_progressor', 'ccaugrc_stable', 'global_augrc']

    for metric in metrics:
        ref_series = iter_df[iter_df['method'] == reference].set_index('bootstrap_iter')[metric]
        ref_vals = iter_df[iter_df['method'] == reference][metric].values
        ref_ci_low, ref_ci_high = np.percentile(ref_vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])

        for method in comparisons:
            comp_series = iter_df[iter_df['method'] == method].set_index('bootstrap_iter')[metric]
            common = ref_series.index.intersection(comp_series.index)

            ref_al = ref_series.loc[common].values
            comp_al = comp_series.loc[common].values
            diff = comp_al - ref_al  # positive = comparison worse

            comp_ci_low, comp_ci_high = np.percentile(comp_al, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            diff_ci_low, diff_ci_high = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])

            prop_same_sign = (diff > 0).mean() if diff.mean() > 0 else (diff < 0).mean()
            pval = 2 * min(prop_same_sign, 1 - prop_same_sign)

            records.append({
                'disease': disease, 'metric': metric, 'reference_method': reference,
                'reference_mean': ref_al.mean(), 'reference_ci_low': ref_ci_low, 'reference_ci_high': ref_ci_high,
                'comparison_method': method, 'comparison_mean': comp_al.mean(),
                'comparison_ci_low': comp_ci_low, 'comparison_ci_high': comp_ci_high,
                'mean_difference': diff.mean(), 'diff_ci_low': diff_ci_low, 'diff_ci_high': diff_ci_high,
                'ci_excludes_zero': not (diff_ci_low <= 0 <= diff_ci_high),
                'p_value': pval, 'test_name': 'Bootstrap sign test',
                'cohens_d': diff.mean() / (diff.std() + 1e-10), 'n_samples': len(common),
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
    
    print("\n  Point estimates with 95% CI  (Progressor ccAUGRC):")
    for method, label in METHODS.items():
        vals = iter_df[iter_df['method'] == method]['ccaugrc_progressor'].values
        print(f"  {label:<32} {vals.mean():.4f}  [{np.percentile(vals, 2.5):.4f}, {np.percentile(vals, 97.5):.4f}]")

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