"""
Stage 2: compute_rejection_metrics.py
=============================================================
For every condition × method × rejection rate, computes:
    - Sensitivity (minority recall)
    - Specificity (majority recall) 
    - AUPRC
    - Coverage (fraction of each class retained)
    - ccAUGRC (class-conditioned AUGRC) for both classes

Produces two output files:

1. rejection_curves.csv
   One row per (tau, d, n, method, rejection_rate)
   → used for per-condition rejection curve plots

2. scalar_summaries.csv  
   One row per (tau, d, n, method)
   → used for phase diagram heatmaps
   Scalar metrics:
     - sensitivity_at_30pct        : sensitivity at 30% rejection
     - specificity_at_30pct
     - sensitivity_stability       : slope of sensitivity vs rejection (0-50%)
     - class_separation_score      : sensitivity/specificity ratio at 30% rejection
     - minority_coverage_at_30pct  : fraction of minority class retained
     - ccaugrc_progressor          : class-conditioned AUGRC for class 1
     - ccaugrc_stable              : class-conditioned AUGRC for class 0
     - global_augrc                : standard AUGRC for comparison

Usage
-----
    python 5_compute_rejection_metrics.py \
        --input   ../data/synthetic_results/all_conditions_uncertainty.csv \
        --out_dir ../data/synthetic_results
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from scipy.stats import linregress


METHODS = {
    'h_total':   'H_Total (Global)',
    'margin':    'Margin (Global)',
    'h_tau':     'H_tau (Global)',
    'h_tau_ccrc': 'H_tau + CCRC (Proposed)',
}

# For margin: low value = high uncertainty (invert for ranking)
# For others: high value = high uncertainty
INVERT_SIGNAL = {'margin'}

REJECTION_RATES = np.arange(0.0, 0.81, 0.05).round(2)
FIXED_REJECTION = 0.30   # scalar summaries computed at this rate


# ─────────────────────────────────────────────
# REJECTION LOGIC
# ─────────────────────────────────────────────

def reject_patients(
    df: pd.DataFrame,
    uncertainty_col: str,
    rejection_rate: float,
    method: str,
    tau: float,
) -> pd.DataFrame:
    """
    Return the retained cohort after rejecting `rejection_rate` fraction
    of patients based on uncertainty signal.

    For H_tau + CCRC: rejection is class-conditioned.
        - Reject top rejection_rate/2 from each predicted class by H_tau rank
        - This preserves class proportions in the retained cohort

    For all other methods: global rejection by uncertainty score.
    """
    if rejection_rate == 0.0:
        return df.copy()

    if method == 'h_tau_ccrc':
        return _ccrc_reject(df, rejection_rate, tau)
    else:
        return _global_reject(df, uncertainty_col, rejection_rate, method)


def _global_reject(
    df: pd.DataFrame,
    uncertainty_col: str,
    rejection_rate: float,
    method: str,
) -> pd.DataFrame:
    """Reject the most uncertain patients globally."""
    n_reject = int(np.floor(len(df) * rejection_rate))
    if n_reject == 0:
        return df.copy()

    ascending = method in INVERT_SIGNAL  # margin: reject lowest values
    sorted_df = df.sort_values(uncertainty_col, ascending=ascending)
    retain_idx = sorted_df.index[n_reject:]
    return df.loc[retain_idx].copy()


def _ccrc_reject(
    df: pd.DataFrame,
    rejection_rate: float,
    tau: float,
) -> pd.DataFrame:
    """
    Class-conditioned rejection: reject rejection_rate/2 from each class.
    Uses h_tau_ccrc (within-class percentile rank) as the signal.
    """
    retained_parts = []
    for cls in [0, 1]:
        class_df = df[df['predicted_class'] == cls].copy()
        if class_df.empty:
            continue
        n_reject = int(np.floor(len(class_df) * rejection_rate))
        # Reject highest within-class uncertainty rank
        sorted_cls = class_df.sort_values('h_tau_ccrc', ascending=False)
        retained_parts.append(sorted_cls.iloc[n_reject:])

    if not retained_parts:
        return df.copy()
    return pd.concat(retained_parts)


# ─────────────────────────────────────────────
# METRIC COMPUTATION
# ─────────────────────────────────────────────

def compute_metrics_on_cohort(retained: pd.DataFrame) -> dict:
    """Compute classification metrics on a retained cohort."""
    if len(retained) == 0 or retained['label'].nunique() < 2:
        return {
            'sensitivity': np.nan, 'specificity': np.nan,
            'auprc': np.nan,
            'minority_coverage': 0.0, 'majority_coverage': 0.0,
            'n_retained': 0,
        }

    y_true = retained['label'].values
    y_pred = (retained['mu'].values >= retained['tau'].values[0]).astype(int)
    y_prob = retained['mu'].values

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = np.nan

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auprc': auprc,
        'n_retained': len(retained),
    }


def compute_augrc_class(
    df_full: pd.DataFrame,
    uncertainty_col: str,
    method: str,
    tau: float,
    target_class: int,
) -> float:
    """
    Class-conditioned AUGRC for a single class.
    
    Computes AUGRC by integrating generalized risk over rejection thresholds,
    evaluated only on patients of target_class.

    Generalized risk: errors_in_accepted / total_population (Traub et al. 2024)
    Uses trapezoidal integration over rejection rates 0-0.8.
    """
    class_df = df_full[df_full['label'] == target_class].copy()
    n_total  = len(class_df)

    if n_total == 0:
        return np.nan

    risks, coverages = [], []

    for rr in REJECTION_RATES:
        retained_full = reject_patients(df_full, uncertainty_col, rr, method, tau)
        retained_cls  = retained_full[retained_full['label'] == target_class]

        if len(retained_cls) == 0:
            risk     = 0.0
            coverage = 0.0
        else:
            y_true = retained_cls['label'].values
            y_pred = (retained_cls['mu'].values >= tau).astype(int)
            errors = (y_pred != y_true).sum()
            risk     = errors / n_total   # denominator is TOTAL class population
            coverage = len(retained_cls) / n_total

        risks.append(risk)
        coverages.append(coverage)

    # Integrate risk over coverage (trapezoidal)
    # Sort by coverage for proper integration
    pairs = sorted(zip(coverages, risks))
    cov_arr  = np.array([p[0] for p in pairs])
    risk_arr = np.array([p[1] for p in pairs])

    augrc = np.trapezoid(risk_arr, cov_arr)
    return float(augrc)


# ─────────────────────────────────────────────
# MAIN COMPUTATION
# ─────────────────────────────────────────────

def compute_all_rejection_metrics(
    input_path: str,
    out_dir: str,
):
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    conditions = df.groupby(['tau', 'd', 'n'])
    n_conditions = len(conditions)
    print(f"  {n_conditions} conditions  |  {len(METHODS)} methods  |  "
          f"{len(REJECTION_RATES)} rejection rates\n")

    curve_records   = []
    scalar_records  = []

    for i, ((tau, d, n), group) in enumerate(conditions):
        group = group.copy()
        group['tau'] = tau   # ensure tau is available as column

        for method_col, method_label in METHODS.items():
            # ── Rejection curve ──
            sensitivities, specificities, auprcs = [], [], []
            min_coverages, maj_coverages = [], []

            for rr in REJECTION_RATES:
                retained = reject_patients(group, method_col, rr, method_col, tau)
                metrics  = compute_metrics_on_cohort(retained)
                #if retained['label'].nunique() < 2:
                #    print(f"    Warning: Only one class retained for tau={tau:.2f}, d={d:.1f}, n={n}, "
                #          f"method={method_label}, rr={rr:.2f}. Metrics may be unreliable.")

                # Coverage per class
                n_min_total = (group['label'] == 1).sum()
                n_maj_total = (group['label'] == 0).sum()
                n_min_kept  = (retained['label'] == 1).sum() if len(retained) > 0 else 0
                n_maj_kept  = (retained['label'] == 0).sum() if len(retained) > 0 else 0

                curve_records.append({
                    'tau':             tau,
                    'd':               d,
                    'n':               n,
                    'method':          method_col,
                    'method_label':    method_label,
                    'rejection_rate':  rr,
                    'sensitivity':     metrics['sensitivity'],
                    'specificity':     metrics['specificity'],
                    'auprc':           metrics['auprc'],
                    'n_retained':      metrics['n_retained'],
                    'minority_coverage': n_min_kept / n_min_total if n_min_total > 0 else np.nan,
                    'majority_coverage': n_maj_kept / n_maj_total if n_maj_total > 0 else np.nan,
                })

                sensitivities.append(metrics['sensitivity'])
                specificities.append(metrics['specificity'])
                min_coverages.append(n_min_kept / n_min_total if n_min_total > 0 else np.nan)

            # ── Scalar summaries ──
            rr_arr   = REJECTION_RATES
            rr_idx30 = np.argmin(np.abs(rr_arr - FIXED_REJECTION))

            sens_30    = sensitivities[rr_idx30]
            spec_30    = specificities[rr_idx30]
            mincov_30  = min_coverages[rr_idx30]

            # Sensitivity stability: slope of sensitivity over 0-50% rejection
            rr_50_mask = rr_arr <= 0.50
            valid_sens  = [s for s, m in zip(sensitivities, rr_50_mask) if m and not np.isnan(s)]
            valid_rr    = [r for r, m in zip(rr_arr, rr_50_mask) if m][:len(valid_sens)]

            if len(valid_sens) > 2:
                slope, *_ = linregress(valid_rr, valid_sens)
                sens_stability = float(slope)
            else:
                sens_stability = np.nan

            # Class separation: sensitivity / specificity at 30% rejection
            class_separation = (
                sens_30 / spec_30
                if (spec_30 and not np.isnan(spec_30) and spec_30 > 0)
                else np.nan
            )

            # ccAUGRC for both classes
            ccaugrc_prog   = compute_augrc_class(group, method_col, method_col, tau, target_class=1)
            ccaugrc_stable = compute_augrc_class(group, method_col, method_col, tau, target_class=0)

            # Global AUGRC (all patients)
            risks_global = []
            for rr in REJECTION_RATES:
                retained = reject_patients(group, method_col, rr, method_col, tau)
                if len(retained) > 0 and retained['label'].nunique() > 1:
                    y_true = retained['label'].values
                    y_pred = (retained['mu'].values >= tau).astype(int)
                    errors = (y_pred != y_true).sum()
                    risk   = errors / len(group)
                else:
                    risk = 0.0
                risks_global.append(risk)
            global_augrc = float(np.trapezoid(risks_global, REJECTION_RATES))

            scalar_records.append({
                'tau':                    tau,
                'd':                      d,
                'n':                      n,
                'method':                 method_col,
                'method_label':           method_label,
                f'sensitivity_at_{int(FIXED_REJECTION*100)}pct': sens_30,
                f'specificity_at_{int(FIXED_REJECTION*100)}pct': spec_30,
                'sensitivity_stability':  sens_stability,
                'class_separation_score': class_separation,
                f'minority_coverage_at_{int(FIXED_REJECTION*100)}pct': mincov_30,
                'ccaugrc_progressor':     ccaugrc_prog,
                'ccaugrc_stable':         ccaugrc_stable,
                'global_augrc':           global_augrc,
            })

        if (i + 1) % 10 == 0 or (i + 1) == n_conditions:
            print(f"  [{i+1:>3}/{n_conditions}] tau={tau:.2f}  d={d:.1f}  n={n}")

    # ── Save ──
    curves_df  = pd.DataFrame(curve_records)
    scalars_df = pd.DataFrame(scalar_records)

    curves_path  = f"{out_dir}/rejection_curves.csv"
    scalars_path = f"{out_dir}/scalar_summaries.csv"

    curves_df.to_csv(curves_path,  index=False)
    scalars_df.to_csv(scalars_path, index=False)

    print(f"\nSaved rejection curves  : {curves_path}")
    print(f"Saved scalar summaries  : {scalars_path}")
    print(f"\nCurve rows  : {len(curves_df):,}")
    print(f"Scalar rows : {len(scalars_df):,}")

    return curves_df, scalars_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   required=True,
                        help='Path to all_conditions_uncertainty.csv')
    parser.add_argument('--out_dir', required=True,
                        help='Directory to write rejection_curves.csv and scalar_summaries.csv')
    args = parser.parse_args()

    compute_all_rejection_metrics(args.input, args.out_dir)
