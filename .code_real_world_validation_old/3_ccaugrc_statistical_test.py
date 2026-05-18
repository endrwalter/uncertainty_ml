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
    # From aggregated summary (forces Approach 2)
    python 10_ccaugrc_statistical_test_v2.py \
        --ms_summary ../results/classification/ms_progression/progression_independent_from_relapses/aggregated/patient_mean_probs_progression_independent_from_relapses_ms_progression_model.csv \
        --pd_summary ../results/classification/pd_dyskinesia/FutureDyskinesia/aggregated/patient_mean_probs_FutureDyskinesia_pd_dyskinesia_model.csv \
        --ad_summary ../results/classification/mci_ad_conversion/label_bl_36m/aggregated/patient_mean_probs_label_bl_36m_mci_ad_conversion_model.csv \
        --out_dir ../results/statistical_tests/

"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DISEASES = {'MS': 0.11, 'PD': 0.39, 'AD': 0.54}

METHODS = {
    'h_total':    'H_Total (Global)',
    'margin':     'Margin (Global)',
    'h_tau':      'H_tau (Global)',
    'h_tau_ccrc': 'H_tau + CCRC (Proposed)',
    'random_ccr': 'Random + ccr (Control)',

}

REJECTION_RATES  = np.arange(0.0, 0.81, 0.05).round(2)
N_BOOTSTRAP      = 1000
MIN_MINORITY_PER_SPLIT = 30   # threshold for approach selection


# ─────────────────────────────────────────────
# SIGNALS
# ─────────────────────────────────────────────

def compute_signals(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    df = df.copy()
    p = df['mu'].values.clip(1e-10, 1 - 1e-10)

    df['h_total'] = -p * np.log2(p) - (1-p) * np.log2(1-p)
    df['margin']  = np.abs(p - tau)

    p_tilde = np.where(
        p <= tau,
        p / (2 * tau),
        0.5 + (p - tau) / (2 * (1 - tau))
    ).clip(1e-10, 1 - 1e-10)
    df['h_tau'] = -p_tilde * np.log2(p_tilde) - (1-p_tilde) * np.log2(1-p_tilde)

    df['predicted_class'] = (df['mu'] >= tau).astype(int)
    df['h_tau_ccrc'] = df.groupby('predicted_class')['h_tau'].rank(
        method='average', pct=True
    )
    return df


# ─────────────────────────────────────────────
# REJECTION AND ccAUGRC
# ─────────────────────────────────────────────

def reject_patients(df, method, rr, tau):
    if rr == 0.0:
        return df.copy()
    if method == 'h_tau_ccrc':
        retained = []
        for cls in [0, 1]:
            cls_df   = df[df['predicted_class'] == cls].copy()
            n_reject = int(np.floor(len(cls_df) * rr))
            retained.append(
                cls_df.sort_values('h_tau_ccrc', ascending=False).iloc[n_reject:]
            )
        return pd.concat(retained) if retained else df.iloc[0:0]
    else:
        ascending = (method == 'margin')
        n_reject  = int(np.floor(len(df) * rr))
        return df.loc[
            df.sort_values(method, ascending=ascending).index[n_reject:]
        ].copy()


def ccaugrc(df, method, tau, target_class):
    n_total = (df['label'] == target_class).sum()
    if n_total == 0:
        return np.nan

    risks, coverages = [], []
    for rr in REJECTION_RATES:
        ret     = reject_patients(df, method, rr, tau)
        cls_ret = ret[ret['label'] == target_class]
        if len(cls_ret) == 0:
            risks.append(0.0); coverages.append(0.0)
        else:
            y_true = cls_ret['label'].values
            y_pred = (cls_ret['mu'].values >= tau).astype(int)
            risks.append((y_pred != y_true).sum() / n_total)
            coverages.append(len(cls_ret) / n_total)

    pairs = sorted(zip(coverages, risks))
    return float(np.trapezoid(
        [p[1] for p in pairs], [p[0] for p in pairs]
    ))


def global_augrc(df, method, tau):
    risks = []
    for rr in REJECTION_RATES:
        ret = reject_patients(df, method, rr, tau)
        if len(ret) > 0 and ret['label'].nunique() > 1:
            y_t = ret['label'].values
            y_p = (ret['mu'].values >= tau).astype(int)
            risks.append((y_t != y_p).sum() / len(df))
        else:
            risks.append(0.0)
    return float(np.trapezoid(risks, REJECTION_RATES))


def compute_all_metrics(df, tau):
    """Compute all ccAUGRC metrics for all methods on one cohort."""
    df = compute_signals(df, tau)
    records = {}
    for method in METHODS:
        records[method] = {
            'ccaugrc_progressor': ccaugrc(df, method, tau, 1),
            'ccaugrc_stable':     ccaugrc(df, method, tau, 0),
            'global_augrc':       global_augrc(df, method, tau),
        }
    return records







# ─────────────────────────────────────────────
# APPROACH 2: PATIENT BOOTSTRAP (PREFERRED)
# ─────────────────────────────────────────────

def approach2_bootstrap(
    summary_df: pd.DataFrame,
    tau: float,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap patients with replacement from the aggregated ensemble.

    This evaluates the deployment-ready product (the full aggregated
    ensemble) under patient sampling uncertainty. Each bootstrap sample
    represents a hypothetical different cohort drawn from the same
    clinical population.

    Note: CIs reflect patient-level sampling uncertainty, not model
    uncertainty. State this explicitly in the paper.
    """
    rng = np.random.default_rng(seed)
    records = []

    for b in range(n_bootstrap):
        idx    = rng.choice(len(summary_df), size=len(summary_df), replace=True)
        sample = summary_df.iloc[idx].reset_index(drop=True)

        if sample['label'].nunique() < 2:
            continue

        metrics = compute_all_metrics(sample, tau)
        for method, vals in metrics.items():
            records.append({'bootstrap_iter': b, 'method': method, **vals})

        if (b + 1) % 200 == 0:
            print(f"      Bootstrap {b+1}/{n_bootstrap}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# CONFIDENCE INTERVALS AND TESTS
# ─────────────────────────────────────────────

def compute_ci_and_tests(
    iter_df: pd.DataFrame,
    disease: str,
    iter_col: str,
    reference: str = 'h_tau_ccrc',
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    For each method vs reference:
    - 95% CI on the point estimate (percentile bootstrap)
    - 95% CI on the difference
    - Wilcoxon test if iter_col is 'iteration' (paired)
    - Sign test if iter_col is 'bootstrap_iter' (not paired)
    - Effect size (Cohen's d on the difference distribution)
    """
    comparisons = [m for m in METHODS if m != reference]
    records     = []
    metrics     = ['ccaugrc_progressor', 'ccaugrc_stable', 'global_augrc']

    for metric in metrics:
        ref_series = iter_df[iter_df['method'] == reference].set_index(iter_col)[metric]

        # CI on reference method
        ref_vals    = iter_df[iter_df['method'] == reference][metric].values
        ref_ci_low  = np.percentile(ref_vals, 100 * alpha / 2)
        ref_ci_high = np.percentile(ref_vals, 100 * (1 - alpha / 2))

        for method in comparisons:
            comp_series = iter_df[iter_df['method'] == method].set_index(iter_col)[metric]
            common      = ref_series.index.intersection(comp_series.index)

            ref_al   = ref_series.loc[common].values
            comp_al  = comp_series.loc[common].values
            diff     = comp_al - ref_al   # positive = comparison worse

            # CI on point estimate for comparison
            comp_ci_low  = np.percentile(comp_al, 100 * alpha / 2)
            comp_ci_high = np.percentile(comp_al, 100 * (1 - alpha / 2))

            # CI on difference
            diff_ci_low  = np.percentile(diff, 100 * alpha / 2)
            diff_ci_high = np.percentile(diff, 100 * (1 - alpha / 2))

            # Statistical test
            if iter_col == 'iteration':
                # Paired Wilcoxon (per-iteration approach)
                try:
                    stat, pval = wilcoxon(ref_al, comp_al, alternative='two-sided')
                    test_name  = 'Wilcoxon (paired)'
                except Exception:
                    stat, pval, test_name = np.nan, np.nan, 'Wilcoxon (failed)'
            else:
                # For bootstrap: CI-based inference
                # p-value approximated from proportion of differences
                # crossing zero (sign-flip probability)
                prop_same_sign = (diff > 0).mean() if diff.mean() > 0 \
                                 else (diff < 0).mean()
                pval      = 2 * min(prop_same_sign, 1 - prop_same_sign)
                stat      = prop_same_sign
                test_name = 'Bootstrap sign test'

            # Effect size: Cohen's d on differences
            d = diff.mean() / (diff.std() + 1e-10)

            records.append({
                'disease':              disease,
                'metric':               metric,
                'reference_method':     reference,
                'reference_mean':       ref_al.mean(),
                'reference_ci_low':     ref_ci_low,
                'reference_ci_high':    ref_ci_high,
                'comparison_method':    method,
                'comparison_mean':      comp_al.mean(),
                'comparison_ci_low':    comp_ci_low,
                'comparison_ci_high':   comp_ci_high,
                'mean_difference':      diff.mean(),
                'diff_ci_low':          diff_ci_low,
                'diff_ci_high':         diff_ci_high,
                'ci_excludes_zero':     not (diff_ci_low <= 0 <= diff_ci_high),
                'p_value':              pval,
                'test_name':            test_name,
                'cohens_d':             d,
                'n_samples':            len(common),
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────

def print_results(tests: pd.DataFrame, disease: str):
    print(f"\n  Results for {disease} — Progressor ccAUGRC:")
    print(f"  {'─'*80}")
    print(f"  {'Comparison':<35} {'Ref':>8} {'vs':>8} {'Δ':>8} "
          f"{'95% CI':>20} {'p':>8} {'sig':>5} {'d':>6}")
    print(f"  {'─'*80}")

    sub = tests[
        (tests['disease'] == disease) &
        (tests['metric'] == 'ccaugrc_progressor')
    ]

    for _, row in sub.iterrows():
        sig = '***' if row['p_value'] < 0.001 else \
              '**'  if row['p_value'] < 0.01  else \
              '*'   if row['p_value'] < 0.05  else 'ns'
        ci_str = f"[{row['diff_ci_low']:+.3f}, {row['diff_ci_high']:+.3f}]"
        excl   = '✓' if row['ci_excludes_zero'] else '✗'

        print(
            f"  vs {METHODS.get(row['comparison_method'], row['comparison_method']):<31} "
            f"{row['reference_mean']:>8.4f} "
            f"{row['comparison_mean']:>8.4f} "
            f"{row['mean_difference']:>+8.4f} "
            f"{ci_str:>20} "
            f"{row['p_value']:>8.4f} "
            f"{sig:>5} "
            f"{row['cohens_d']:>+6.2f}"
        )

    print(f"\n  ✓ = 95% CI excludes zero (difference is statistically meaningful)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_disease(
    disease: str,
    tau: float,
    raw_path: str = None,
    summary_path: str = None,
    out_dir: str = '.',
    idx_col: str = 'idx',
    prob_col: str = 'probs_cal',
    label_col: str = 'real y',
):
    print(f"\n{'═'*60}")
    print(f"{disease}  (τ={tau})")
    print(f"{'═'*60}")

    os.makedirs(out_dir, exist_ok=True)

    print(f"  → patient bootstrap — only summary data available")
    summary  = pd.read_csv(summary_path)
    if 'mu' not in summary.columns:
        raise ValueError(f"Summary must have 'mu' column. Found: {summary.columns.tolist()}")
    n_minority = (summary['label'] == 1).sum()
    print(f"  Total minority: {n_minority}")
    iter_df  = approach2_bootstrap(summary, tau)
    iter_col = 'bootstrap_iter'


    # ── Save bootstrap/iteration distributions ──
    iter_df.to_csv(
        os.path.join(out_dir, f'{disease.lower()}_distributions.csv'), index=False
    )

    # ── CI summary per method ──
    print(f"\n  Point estimates with 95% CI  (Progressor ccAUGRC):")
    print(f"  {'─'*65}")
    for method, label in METHODS.items():
        vals     = iter_df[iter_df['method'] == method]['ccaugrc_progressor'].values
        ci_low   = np.percentile(vals, 2.5)
        ci_high  = np.percentile(vals, 97.5)
        print(f"  {label:<32} {vals.mean():.4f}  [{ci_low:.4f}, {ci_high:.4f}]")


    # ── Statistical tests ──
    tests = compute_ci_and_tests(iter_df, disease, iter_col)
    print_results(tests, disease)

    tests.to_csv(
        os.path.join(out_dir, f'{disease.lower()}_statistical_tests.csv'), index=False
    )

    return tests


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Aggregated summary files (fallback)
    parser.add_argument('--ms_summary', default=None)
    parser.add_argument('--pd_summary', default=None)
    parser.add_argument('--ad_summary', default=None)

    parser.add_argument('--out_dir',    required=True)
    parser.add_argument('--idx_col',    default='idx')
    parser.add_argument('--prob_col',   default='probs_cal')
    parser.add_argument('--label_col',  default='real y')
    args = parser.parse_args()

    all_tests = []
    for disease in ['MS', 'PD', 'AD']:
        tau         = DISEASES[disease]
        raw_path    = getattr(args, f'{disease.lower()}_raw')
        summary_path = getattr(args, f'{disease.lower()}_summary')
        tests = run_disease(
            disease, tau, raw_path, summary_path, args.out_dir,
            args.idx_col, args.prob_col, args.label_col
        )
        if tests is not None:
            all_tests.append(tests)

    if all_tests:
        combined = pd.concat(all_tests, ignore_index=True)
        combined.to_csv(
            os.path.join(args.out_dir, 'all_statistical_tests.csv'), index=False
        )
        print(f"\nAll results saved to {args.out_dir}")
