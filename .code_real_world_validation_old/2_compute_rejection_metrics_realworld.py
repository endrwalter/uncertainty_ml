"""
compute_rejection_metrics_realworld.py
=============================================================
Computes all rejection metrics for the three real-world cohorts
(MS, PD, AD), following the exact same methodology as the
synthetic compute_rejection_metrics.py.

Input per disease
-----------------
patient_mean_probs_summary.csv produced by build_ensemble_csv:
    idx, mu, sigma, label, n_predictions, all_probs

Output per disease (ready for paper_figures.py)
-----------------------------------------------
    <out_dir>/
        ms_rejection_curves.csv
        pd_rejection_curves.csv
        ad_rejection_curves.csv
        ms_scalar_summaries.csv
        pd_scalar_summaries.csv
        ad_scalar_summaries.csv
        ms_asymmetry.csv
        pd_asymmetry.csv
        ad_asymmetry.csv
        ccaugrc_real_world.csv          ← Fig 8 input

Usage
-----
python 2_compute_rejection_metrics_realworld.py \
    --ms_path  ../results/classification/ms_progression/progression_independent_from_relapses/aggregated/patient_mean_probs_progression_independent_from_relapses_ms_progression_model.csv \
    --pd_path  ../results/classification/pd_dyskinesia/FutureDyskinesia/aggregated/patient_mean_probs_FutureDyskinesia_pd_dyskinesia_model.csv \
    --ad_path  ../results/classification/mci_ad_conversion/label_bl_36m/aggregated/patient_mean_probs_label_bl_36m_mci_ad_conversion_model.csv \
    --out_dir  ../results/real_world_results/
"""

import argparse
import ast
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import average_precision_score

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# DISEASE CONFIGURATION
# ─────────────────────────────────────────────

DISEASES = {
    'MS': {'tau': 0.13, 'label': 'Multiple Sclerosis  (Severe, τ=0.13)'},
    'PD': {'tau': 0.39, 'label': 'Parkinson\'s Disease  (Moderate, τ=0.39)'},
    'AD': {'tau': 0.54, 'label': 'Alzheimer\'s Disease  (Mild, τ=0.54)'},
}

METHODS = {
    'h_total':    'H_Total (Global)',
    'margin':     'Margin (Global)',
    'h_tau':      'H_tau (Global)',
    'h_tau_ccr': 'H_tau + ccr (Proposed)',
    'random_ccr': 'Random + ccr (Control)',
}

INVERT_SIGNAL    = {'margin'}
REJECTION_RATES  = np.arange(0.0, 0.81, 0.05).round(2)
FIXED_REJECTION  = 0.30
DISTANCE_BINS    = np.array([0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200])


# ─────────────────────────────────────────────
# DATA LOADING AND PREPARATION
# ─────────────────────────────────────────────

def load_and_prepare(path: str, tau: float, disease: str) -> pd.DataFrame:
    """
    Load patient_mean_probs_summary.csv and compute all
    uncertainty signals needed for rejection analysis.

    Parameters
    ----------
    path  : path to patient_mean_probs_summary.csv
    tau   : disease-specific decision threshold
    disease: disease name for logging

    Returns
    -------
    df with columns:
        idx, mu, sigma, label, tau,
        h_total, margin, h_tau, h_tau_ccr,
        predicted_class, ensemble_variance
    """
    print(f"\n  Loading {disease}: {path}")
    df = pd.read_csv(path)

    # Validate required columns
    required = {'mu', 'sigma', 'label'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"{disease}: missing columns {missing}")

    df = df.copy()
    df['tau']     = tau
    df['disease'] = disease

    # ── Ensemble variance from all_probs if available ──
    # sigma from groupby std is already ensemble std; variance = sigma^2
    if 'sigma' in df.columns:
        df['ensemble_variance'] = df['sigma'] ** 2
    elif 'all_probs' in df.columns:
        def parse_probs(x):
            if isinstance(x, str):
                try:
                    return np.array(ast.literal_eval(x))
                except Exception:
                    return np.array([np.nan])
            return np.array(x) if hasattr(x, '__iter__') else np.array([np.nan])
        probs_arrays = df['all_probs'].apply(parse_probs)
        df['ensemble_variance'] = probs_arrays.apply(np.var)
    else:
        df['ensemble_variance'] = np.nan

    # ── Uncertainty signals ──
    p   = df['mu'].values.clip(1e-10, 1 - 1e-10)
    tau_val = tau

    # H_Total: standard Shannon entropy
    df['h_total'] = -p * np.log2(p) - (1-p) * np.log2(1-p)

    # Margin: absolute distance from boundary (low = uncertain)
    df['margin'] = np.abs(p - tau_val)

    # H_tau: piecewise-rescaled entropy
    p_tilde = np.where(
        p <= tau_val,
        p / (2 * tau_val),
        0.5 + (p - tau_val) / (2 * (1 - tau_val))
    ).clip(1e-10, 1 - 1e-10)
    df['h_tau'] = -p_tilde * np.log2(p_tilde) - (1-p_tilde) * np.log2(1-p_tilde)

    # Predicted class
    df['predicted_class'] = (df['mu'] >= tau_val).astype(int)

    # H_tau CCR: within-class percentile rank of h_tau
    df['h_tau_ccr'] = df.groupby('predicted_class')['h_tau'].rank(
        method='average', pct=True
    )


    n_min = (df['label'] == 1).sum()
    n_maj = (df['label'] == 0).sum()
    print(f"    N={len(df)}  minority={n_min}  majority={n_maj}  "
          f"actual_tau={n_min/len(df):.3f}  configured_tau={tau}")

    return df


# ─────────────────────────────────────────────
# REJECTION LOGIC (identical to synthetic script)
# ─────────────────────────────────────────────

def reject_patients(
    df: pd.DataFrame,
    uncertainty_col: str,
    rejection_rate: float,
    method: str,
    tau: float,
) -> pd.DataFrame:
    if rejection_rate == 0.0:
        return df.copy()
    if method == 'h_tau_ccr':
        return _ccr_reject(df, rejection_rate)
    elif method == 'random_ccr':
        return _ccr_random_reject(df, rejection_rate)
    else:
        return _global_reject(df, uncertainty_col, rejection_rate, method)


def _global_reject(df, uncertainty_col, rejection_rate, method):
    n_reject  = int(np.floor(len(df) * rejection_rate))
    if n_reject == 0:
        return df.copy()
    ascending = method in INVERT_SIGNAL
    sorted_df = df.sort_values(uncertainty_col, ascending=ascending)
    return df.loc[sorted_df.index[n_reject:]].copy()


def _ccr_reject(df, rejection_rate):
    retained = []
    for cls in [0, 1]:
        cls_df   = df[df['predicted_class'] == cls].copy()
        if cls_df.empty:
            continue
        n_reject = int(np.floor(len(cls_df) * rejection_rate))
        sorted_cls = cls_df.sort_values('h_tau_ccr', ascending=False)
        retained.append(sorted_cls.iloc[n_reject:])
    return pd.concat(retained) if retained else df.iloc[0:0].copy()

def _ccr_random_reject(df, rejection_rate):
    retained = []
    for cls in [0, 1]:
        cls_df   = df[df['predicted_class'] == cls].copy()
        if cls_df.empty:
            continue
        n_reject = int(np.floor(len(cls_df) * rejection_rate))
        sorted_cls = cls_df.sample(frac=1, random_state=42)  # Random shuffle
        retained.append(sorted_cls.iloc[n_reject:])
    return pd.concat(retained) if retained else df.iloc[0:0].copy()
# ─────────────────────────────────────────────
# METRIC COMPUTATION (identical to synthetic script)
# ─────────────────────────────────────────────

def compute_metrics(retained: pd.DataFrame, tau: float) -> dict:
    if len(retained) == 0 or retained['label'].nunique() < 2:
        return dict(sensitivity=np.nan, specificity=np.nan,
                    auprc=np.nan, n_retained=len(retained))

    y_true = retained['label'].values
    y_pred = (retained['mu'].values >= tau).astype(int)
    y_prob = retained['mu'].values

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()

    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = np.nan

    return dict(
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        auprc       = auprc,
        n_retained  = len(retained),
    )


def compute_ccaugrc(
    df: pd.DataFrame,
    uncertainty_col: str,
    method: str,
    tau: float,
    target_class: int,
) -> float:
    n_total = (df['label'] == target_class).sum()
    if n_total == 0:
        return np.nan

    risks, coverages = [], []
    for rr in REJECTION_RATES:
        retained     = reject_patients(df, uncertainty_col, rr, method, tau)
        retained_cls = retained[retained['label'] == target_class]

        if len(retained_cls) == 0:
            risks.append(0.0)
            coverages.append(0.0)
        else:
            y_true = retained_cls['label'].values
            y_pred = (retained_cls['mu'].values >= tau).astype(int)
            risks.append((y_pred != y_true).sum() / n_total)
            coverages.append(len(retained_cls) / n_total)

    pairs    = sorted(zip(coverages, risks))
    cov_arr  = np.array([p[0] for p in pairs])
    risk_arr = np.array([p[1] for p in pairs])
    return float(np.trapezoid(risk_arr, cov_arr))


# ─────────────────────────────────────────────
# REJECTION CURVES + SCALAR SUMMARIES
# ─────────────────────────────────────────────

def compute_rejection_metrics(
    df: pd.DataFrame,
    tau: float,
    disease: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute rejection curves and scalar summaries for one disease.

    Returns
    -------
    curves_df  : one row per (method, rejection_rate)
    scalars_df : one row per method
    """
    curve_records  = []
    scalar_records = []

    n_min_total = (df['label'] == 1).sum()
    n_maj_total = (df['label'] == 0).sum()

    for method_col, method_label in METHODS.items():
        sensitivities, specificities, auprcs, min_coverages = [], [], [], []

        for rr in REJECTION_RATES:
            retained = reject_patients(df, method_col, rr, method_col, tau)
            m        = compute_metrics(retained, tau)

            n_min_kept = (retained['label'] == 1).sum() if len(retained) > 0 else 0
            n_maj_kept = (retained['label'] == 0).sum() if len(retained) > 0 else 0

            curve_records.append({
                'disease':         disease,
                'tau':             tau,
                'method':          method_col,
                'method_label':    method_label,
                'rejection_rate':  rr,
                'sensitivity':     m['sensitivity'],
                'specificity':     m['specificity'],
                'auprc':           m['auprc'],
                'n_retained':      m['n_retained'],
                'minority_coverage': n_min_kept / n_min_total if n_min_total > 0 else np.nan,
                'majority_coverage': n_maj_kept / n_maj_total if n_maj_total > 0 else np.nan,
            })

            sensitivities.append(m['sensitivity'])
            specificities.append(m['specificity'])
            auprcs.append(m['auprc'])
            min_coverages.append(n_min_kept / n_min_total if n_min_total > 0 else np.nan)

        # ── Scalar summaries ──
        idx30     = np.argmin(np.abs(REJECTION_RATES - FIXED_REJECTION))
        sens_30   = sensitivities[idx30]
        spec_30   = specificities[idx30]
        mincov_30 = min_coverages[idx30]

        # Sensitivity stability: slope over 0-50%
        mask_50  = REJECTION_RATES <= 0.50
        valid    = [(r, s) for r, s, m in zip(REJECTION_RATES, sensitivities, mask_50)
                    if m and not np.isnan(s)]
        if len(valid) > 2:
            rr_v, s_v  = zip(*valid)
            slope, *_  = linregress(rr_v, s_v)
        else:
            slope = np.nan

        # ccAUGRC
        ccaugrc_prog   = compute_ccaugrc(df, method_col, method_col, tau, 1)
        ccaugrc_stable = compute_ccaugrc(df, method_col, method_col, tau, 0)

        # Global AUGRC
        g_risks = []
        for rr in REJECTION_RATES:
            ret = reject_patients(df, method_col, rr, method_col, tau)
            if len(ret) > 0 and ret['label'].nunique() > 1:
                y_t = ret['label'].values
                y_p = (ret['mu'].values >= tau).astype(int)
                g_risks.append((y_t != y_p).sum() / len(df))
            else:
                g_risks.append(0.0)
        global_augrc = float(np.trapezoid(g_risks, REJECTION_RATES))

        scalar_records.append({
            'disease':                disease,
            'tau':                    tau,
            'method':                 method_col,
            'method_label':           method_label,
            f'sensitivity_at_{int(FIXED_REJECTION*100)}pct': sens_30,
            f'specificity_at_{int(FIXED_REJECTION*100)}pct': spec_30,
            'sensitivity_stability':  slope,
            f'minority_coverage_at_{int(FIXED_REJECTION*100)}pct': mincov_30,
            'class_separation':       sens_30 / spec_30 if spec_30 and spec_30 > 0 else np.nan,
            'ccaugrc_progressor':     ccaugrc_prog,
            'ccaugrc_stable':         ccaugrc_stable,
            'global_augrc':           global_augrc,
        })

        print(f"    {method_label:<30}  "
              f"sens@30%={sens_30:.3f}  "
              f"spec@30%={spec_30:.3f}  "
              f"stability={slope:.3f}  "
              f"prog_ccaugrc={ccaugrc_prog:.4f}")

    return pd.DataFrame(curve_records), pd.DataFrame(scalar_records)


# ─────────────────────────────────────────────
# ASYMMETRY TEST  (Fig 7)
# ─────────────────────────────────────────────

def compute_asymmetry_test(
    df: pd.DataFrame,
    tau: float,
    disease: str,
) -> pd.DataFrame:
    """
    Empirical confirmation of Pathology 3.

    At each distance bin from tau, compute mean ensemble variance
    separately for patients above and below tau.

    This shows that at the same mathematical distance from the
    decision boundary, minority (above-tau) patients carry
    systematically higher ensemble uncertainty than majority patients.

    Returns
    -------
    DataFrame with columns:
        [distance, variance_above_tau, variance_below_tau,
         n_above, n_below, disease]
    """
    df = df.copy()
    df['distance'] = np.abs(df['mu'] - tau)

    records = []
    for dist_upper in DISTANCE_BINS:
        dist_lower = dist_upper - (DISTANCE_BINS[1] - DISTANCE_BINS[0])
        dist_lower = max(0.0, dist_lower)

        in_bin = (df['distance'] >= dist_lower) & (df['distance'] < dist_upper)

        above = df[in_bin & (df['predicted_class'] == 1)]  # minority / progressor
        below = df[in_bin & (df['predicted_class'] == 0)]  # majority / stable

        var_above = above['ensemble_variance'].mean() if len(above) > 0 else np.nan
        var_below = below['ensemble_variance'].mean() if len(below) > 0 else np.nan

        records.append({
            'disease':           disease,
            'tau':               tau,
            'distance':          round((dist_lower + dist_upper) / 2, 4),
            'variance_above_tau': var_above,
            'variance_below_tau': var_below,
            'n_above':            len(above),
            'n_below':            len(below),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def process_all_diseases(
    paths: dict,   # {'MS': path, 'PD': path, 'AD': path}
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    all_curves   = []
    all_scalars  = []
    all_asymmetry = []
    ccaugrc_records = []

    for disease, path in paths.items():
        if path is None or not os.path.exists(path):
            print(f"\n[SKIP] {disease}: path not provided or not found: {path}")
            continue

        tau = DISEASES[disease]['tau']
        print(f"\n{'═'*60}")
        print(f"Processing {disease}  (τ={tau})")
        print(f"{'═'*60}")

        # Load and prepare
        df = load_and_prepare(path, tau, disease)

        # Rejection curves + scalars
        curves, scalars = compute_rejection_metrics(df, tau, disease)

        # Asymmetry test
        asymmetry = compute_asymmetry_test(df, tau, disease)

        # Per-disease save
        curves.to_csv(f"{out_dir}/{disease.lower()}_rejection_curves.csv",  index=False)
        scalars.to_csv(f"{out_dir}/{disease.lower()}_scalar_summaries.csv", index=False)
        asymmetry.to_csv(f"{out_dir}/{disease.lower()}_asymmetry.csv",      index=False)

        # Accumulate for global files
        all_curves.append(curves)
        all_scalars.append(scalars)
        all_asymmetry.append(asymmetry)

        # ccAUGRC records for Fig 8
        for _, row in scalars.iterrows():
            ccaugrc_records.append({
                'disease':            disease,
                'tau':                tau,
                'method':             row['method'],
                'method_label':       row['method_label'],
                'global_augrc':       row['global_augrc'],
                'ccaugrc_progressor': row['ccaugrc_progressor'],
                'ccaugrc_stable':     row['ccaugrc_stable'],
            })

    # ── Global output files ──
    if all_curves:
        pd.concat(all_curves).to_csv(
            f"{out_dir}/all_diseases_rejection_curves.csv", index=False)
    if all_scalars:
        pd.concat(all_scalars).to_csv(
            f"{out_dir}/all_diseases_scalar_summaries.csv", index=False)
    if all_asymmetry:
        pd.concat(all_asymmetry).to_csv(
            f"{out_dir}/all_diseases_asymmetry.csv", index=False)
    if ccaugrc_records:
        ccaugrc_df = pd.DataFrame(ccaugrc_records)
        ccaugrc_df.to_csv(f"{out_dir}/ccaugrc_real_world.csv", index=False)

        # Print summary table for the paper
        _print_ccaugrc_summary(ccaugrc_df)

    print(f"\n{'═'*60}")
    print(f"All outputs saved to: {out_dir}")
    print(f"{'═'*60}")


def _print_ccaugrc_summary(ccaugrc_df: pd.DataFrame):
    """Print the real-world ccAUGRC table discussed in the paper."""
    print(f"\n{'─'*75}")
    print("ccAUGRC DECOMPOSITION — REAL WORLD  (lower = safer)")
    print(f"{'─'*75}")
    header = f"{'Disease':<6} {'τ':<6} {'Method':<28} {'Global':>8} {'Progressor':>12} {'Stable':>8}"
    print(header)
    print(f"{'─'*75}")

    for disease in ['MS', 'PD', 'AD']:
        sub = ccaugrc_df[ccaugrc_df['disease'] == disease]
        for method in ['h_total', 'margin', 'h_tau', 'h_tau_ccr', 'random_ccr']:
            row = sub[sub['method'] == method]
            if row.empty:
                continue
            row = row.iloc[0]
            print(
                f"{disease:<6} {row['tau']:<6.2f} "
                f"{row['method_label']:<28} "
                f"{row['global_augrc']:>8.4f} "
                f"{row['ccaugrc_progressor']:>12.4f} "
                f"{row['ccaugrc_stable']:>8.4f}"
            )
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms_path', default=None,
                        help='patient_mean_probs_summary.csv for MS cohort')
    parser.add_argument('--pd_path', default=None,
                        help='patient_mean_probs_summary.csv for PD cohort')
    parser.add_argument('--ad_path', default=None,
                        help='patient_mean_probs_summary.csv for AD cohort')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for all result CSVs')
    args = parser.parse_args()

    paths = {
        'MS': args.ms_path,
        'PD': args.pd_path,
        'AD': args.ad_path,
    }

    process_all_diseases(paths, args.out_dir)