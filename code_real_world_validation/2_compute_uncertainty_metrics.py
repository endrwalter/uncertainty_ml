"""
2_compute_uncertainty_metrics.py
=============================================================
Computes all rejection metrics for the three real-world cohorts
(MS, PD, AD), following the exact same methodology as the
synthetic compute_rejection_metrics.py.

Input per disease
-----------------
patient_mean_probs_summary.csv produced by build_ensemble_csv:
    idx, mu, sigma, label, n_predictions, all_probs

Outputs
-----------------------------------------------
Per disease (<disease> is ms, pd, or ad):
    <out_dir>/<disease>_rejection_curves.csv
    <out_dir>/<disease>_scalar_summaries.csv
    <out_dir>/<disease>_asymmetry.csv
    <out_dir>/<disease>_combined_dynamics.csv

Aggregated across all diseases:
    <out_dir>/all_diseases_rejection_curves.csv
    <out_dir>/all_diseases_scalar_summaries.csv
    <out_dir>/all_diseases_asymmetry.csv
    <out_dir>/all_diseases_combined_dynamics.csv

Usage (here use the ".._public.." file for ADNI MCI->AD conversion):
-----
python 2_compute_uncertainty_metrics.py \
    --ms_path  ../results/classification/ms_progression/progression_independent_from_relapses/aggregated/patient_mean_probs_progression_independent_from_relapses_ms_progression_model.csv \
    --pd_path  ../results/classification/pd_dyskinesia/FutureDyskinesia/aggregated/patient_mean_probs_FutureDyskinesia_pd_dyskinesia_model.csv \
    --ad_path  ../results/classification/mci_ad_conversion/label_bl_36m/aggregated/patient_mean_probs_label_bl_36m_mci_ad_conversion_model.csv \
    --out_dir  ../results/real_world_results/
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import average_precision_score

from rejection_utils import (
    DISEASES, METHODS, REJECTION_RATES, 
    compute_signals, reject_patients, compute_ccaugrc, compute_global_augrc
)

warnings.filterwarnings('ignore')



FIXED_REJECTION = 0.30
DISTANCE_BINS = np.array([0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200])

def compute_metrics(retained: pd.DataFrame, tau: float) -> dict:
    if len(retained) == 0 or retained['label'].nunique() < 2:
        return dict(sensitivity=np.nan, specificity=np.nan, auprc=np.nan, n_retained=len(retained))

    y_true = retained['label'].values
    y_pred = (retained['mu'].values >= tau).astype(int)
    y_prob = retained['mu'].values

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()

    try: auprc = average_precision_score(y_true, y_prob)
    except Exception: auprc = np.nan

    return dict(
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        auprc = auprc,
        n_retained = len(retained),
    )

def compute_rejection_metrics(df: pd.DataFrame, tau: float, disease: str):
    curve_records, scalar_records = [], []
    n_min_total = (df['label'] == 1).sum()
    n_maj_total = (df['label'] == 0).sum()

    for method_col, method_label in METHODS.items():
        sensitivities, specificities, auprcs, min_coverages = [], [], [], []

        for rr in REJECTION_RATES:
            retained = reject_patients(df, method=method_col, rr=rr, tau=tau)
            m = compute_metrics(retained, tau)

            n_min_kept = (retained['label'] == 1).sum() if len(retained) > 0 else 0
            n_maj_kept = (retained['label'] == 0).sum() if len(retained) > 0 else 0

            curve_records.append({
                'disease': disease, 'tau': tau, 'method': method_col, 'method_label': method_label,
                'rejection_rate': rr, 'sensitivity': m['sensitivity'], 'specificity': m['specificity'],
                'auprc': m['auprc'], 'n_retained': m['n_retained'],
                'minority_coverage': n_min_kept / n_min_total if n_min_total > 0 else np.nan,
                'majority_coverage': n_maj_kept / n_maj_total if n_maj_total > 0 else np.nan,
            })

            sensitivities.append(m['sensitivity'])
            specificities.append(m['specificity'])
            min_coverages.append(n_min_kept / n_min_total if n_min_total > 0 else np.nan)

        # Scalars
        idx30 = np.argmin(np.abs(REJECTION_RATES - FIXED_REJECTION))
        sens_30 = sensitivities[idx30]
        spec_30 = specificities[idx30]
        mincov_30 = min_coverages[idx30]

        mask_50 = REJECTION_RATES <= 0.50
        valid = [(r, s) for r, s, m in zip(REJECTION_RATES, sensitivities, mask_50) if m and not np.isnan(s)]
        slope = linregress(*zip(*valid))[0] if len(valid) > 2 else np.nan

        # Extract the exact float value using the 'area' key
        ccaugrc_prog_dict = compute_ccaugrc(df, method_col, tau, 1)
        ccaugrc_prog = ccaugrc_prog_dict['area'] if isinstance(ccaugrc_prog_dict, dict) else ccaugrc_prog_dict
        
        ccaugrc_stable_dict = compute_ccaugrc(df, method_col, tau, 0)
        ccaugrc_stable = ccaugrc_stable_dict['area'] if isinstance(ccaugrc_stable_dict, dict) else ccaugrc_stable_dict
        
        global_augrc_dict = compute_global_augrc(df, method_col, tau)
        global_augrc = global_augrc_dict['area'] if isinstance(global_augrc_dict, dict) else global_augrc_dict

        scalar_records.append({
            'disease': disease, 'tau': tau, 'method': method_col, 'method_label': method_label,
            f'sensitivity_at_{int(FIXED_REJECTION*100)}pct': sens_30,
            f'specificity_at_{int(FIXED_REJECTION*100)}pct': spec_30,
            'sensitivity_stability': slope,
            f'minority_coverage_at_{int(FIXED_REJECTION*100)}pct': mincov_30,
            'class_separation': sens_30 / spec_30 if spec_30 and spec_30 > 0 else np.nan,
            'ccaugrc_progressor': ccaugrc_prog, 
            'ccaugrc_stable': ccaugrc_stable, 
            'global_augrc': global_augrc,
        })
        print(f"    {method_label:<30} sens@30%={sens_30:.3f} spec@30%={spec_30:.3f} stability={slope:.3f} prog_ccaugrc={ccaugrc_prog:.4f}")

    return pd.DataFrame(curve_records), pd.DataFrame(scalar_records)

def compute_asymmetry_test(df: pd.DataFrame, tau: float, disease: str) -> pd.DataFrame:
    df = df.copy()
    df['distance'] = np.abs(df['mu'] - tau)
    records = []
    
    for dist_upper in DISTANCE_BINS:
        dist_lower = max(0.0, dist_upper - (DISTANCE_BINS[1] - DISTANCE_BINS[0]))
        in_bin = (df['distance'] >= dist_lower) & (df['distance'] < dist_upper)

        above = df[in_bin & (df['predicted_class'] == 1)]
        below = df[in_bin & (df['predicted_class'] == 0)]

        records.append({
            'disease': disease, 'tau': tau, 'distance': round((dist_lower + dist_upper) / 2, 4),
            'variance_above_tau': above['ensemble_variance'].mean() if len(above) > 0 else np.nan,
            'variance_below_tau': below['ensemble_variance'].mean() if len(below) > 0 else np.nan,
            'n_above': len(above), 'n_below': len(below),
        })
    return pd.DataFrame(records)

def analyze_combined_rejection_dynamics(curves_df, df_raw, disease, method):
    """
    Computes rejection dynamics for BOTH the D0 (predicted stable) 
    and D1 (predicted progressor) queues across all rejection rates.
    """
    N1 = (df_raw['label'] == 1).sum()
    N0 = (df_raw['label'] == 0).sum()
    
    sub_df = curves_df[(curves_df['disease'] == disease) & (curves_df['method'] == method)]
    if sub_df.empty:
        return []
        
    base_row = sub_df[sub_df['rejection_rate'] == 0.0].iloc[0]
    
    # Initial D0 Queue Sizes (Predicted Stable)
    initial_fns = (1.0 - base_row['sensitivity']) * base_row['minority_coverage'] * N1
    initial_tns = base_row['specificity'] * base_row['majority_coverage'] * N0
    
    # Initial D1 Queue Sizes (Predicted Progressor)
    initial_tps = base_row['sensitivity'] * base_row['minority_coverage'] * N1
    initial_fps = (1.0 - base_row['specificity']) * base_row['majority_coverage'] * N0
    
    records = []
    for _, row in sub_df.iterrows():
        rr = row['rejection_rate']
        
        # Retained sizes
        retained_fns = (1.0 - row['sensitivity']) * row['minority_coverage'] * N1
        retained_tns = row['specificity'] * row['majority_coverage'] * N0
        
        retained_tps = row['sensitivity'] * row['minority_coverage'] * N1
        retained_fps = (1.0 - row['specificity']) * row['majority_coverage'] * N0
        
        # Calculate D0 Rejections (FN vs TN)
        rejected_fns = initial_fns - retained_fns
        rejected_tns = initial_tns - retained_tns
        fn_rej_pct = (rejected_fns / initial_fns) * 100 if initial_fns > 0 else 0
        tn_rej_pct = (rejected_tns / initial_tns) * 100 if initial_tns > 0 else 0
        
        # Calculate D1 Rejections (FP vs TP)
        rejected_tps = initial_tps - retained_tps
        rejected_fps = initial_fps - retained_fps
        tp_rej_pct = (rejected_tps / initial_tps) * 100 if initial_tps > 0 else 0
        fp_rej_pct = (rejected_fps / initial_fps) * 100 if initial_fps > 0 else 0
        
        records.append({
            'disease': disease,
            'method': method,
            'rejection_rate': rr,
            'fn_rejection_pct': fn_rej_pct,
            'tn_rejection_pct': tn_rej_pct,
            'fp_rejection_pct': fp_rej_pct,
            'tp_rejection_pct': tp_rej_pct
        })
        
    return records

def process_all_diseases(paths: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    
    all_curves, all_scalars, all_asymmetry, ccaugrc_records = [], [], [], []
    all_rej_dynamics = []
    disease_specific_rej_dynamics = []
    dyn_records = []  

    for disease, path in paths.items():
        disease_specific_rej_dynamics = []
        if not path or not os.path.exists(path): continue
        tau = DISEASES[disease]['tau']
        print(f"\n{'═'*60}\nProcessing {disease}  (τ={tau})\n{'═'*60}")

        df = pd.read_csv(path)
        df['tau'], df['disease'] = tau, disease
        df = compute_signals(df, tau)  # Uses shared lib

        curves, scalars = compute_rejection_metrics(df, tau, disease)
        asymmetry = compute_asymmetry_test(df, tau, disease)

        for m_col in METHODS.keys():    
            dyn_records = analyze_combined_rejection_dynamics(
                curves_df=curves, df_raw=df, disease=disease, method=m_col
            )
            disease_specific_rej_dynamics.extend(dyn_records)

        curves.to_csv(f"{out_dir}/{disease.lower()}_rejection_curves.csv", index=False)
        scalars.to_csv(f"{out_dir}/{disease.lower()}_scalar_summaries.csv", index=False)
        asymmetry.to_csv(f"{out_dir}/{disease.lower()}_asymmetry.csv", index=False)
        pd.DataFrame(disease_specific_rej_dynamics).to_csv(f"{out_dir}/{disease.lower()}_combined_dynamics.csv", index=False)

        all_curves.append(curves)
        all_scalars.append(scalars)
        all_asymmetry.append(asymmetry)
        all_rej_dynamics.extend(disease_specific_rej_dynamics)

    if all_curves:
        pd.concat(all_curves).to_csv(f"{out_dir}/all_diseases_rejection_curves.csv", index=False)
        pd.concat(all_scalars).to_csv(f"{out_dir}/all_diseases_scalar_summaries.csv", index=False)
        pd.concat(all_asymmetry).to_csv(f"{out_dir}/all_diseases_asymmetry.csv", index=False)
        # Convert the list of dicts to a DataFrame and save
        pd.DataFrame(all_rej_dynamics).to_csv(f"{out_dir}/all_diseases_combined_dynamics.csv", index=False)
        print(f"\n{'═'*60}\nAll outputs saved to: {out_dir}\n{'═'*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms_path')
    parser.add_argument('--pd_path')
    parser.add_argument('--ad_path')
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    process_all_diseases({'MS': args.ms_path, 'PD': args.pd_path, 'AD': args.ad_path}, args.out_dir)



