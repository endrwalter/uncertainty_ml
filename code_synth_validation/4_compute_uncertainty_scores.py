"""
4_compute_uncertainty_scores.py
=============================================================
Loads aggregated synthetic results and computes uncertainty scores
for every patient × condition using base and class-conditioned metrics.

Inputs
------
--input: ../data/synthetic_results/all_conditions_summary.csv 
         (Produced by 3_build_h_ensembles.py)

Outputs
-------
--output: ../data/synthetic_results/all_conditions_uncertainty.csv
          Appends computed metrics (h_total, margin, h_tau) and their 
          class-conditioned ordinal rankings (h_tau_ccrc, margin_ccrc, 
          h_total_ccrc) to the input data.

Usage
-----
python 4_compute_uncertainty_scores.py \
    --input  ../data/synthetic_results/all_conditions_summary.csv \
    --output ../data/synthetic_results/all_conditions_uncertainty.csv
"""

import argparse
import numpy as np
import pandas as pd
from typing import Tuple


# ─────────────────────────────────────────────
# UNCERTAINTY METRICS
# ─────────────────────────────────────────────

def shannon_entropy(p: np.ndarray) -> np.ndarray:
    """Standard Shannon binary entropy. Maximum at p=0.5."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def decision_margin(p: np.ndarray, tau: float) -> np.ndarray:
    """Absolute distance from decision boundary. Lower = more uncertain."""
    return np.abs(p - tau)


def piecewise_rescale(p: np.ndarray, tau: float) -> np.ndarray:
    """
    Piecewise linear rescaling that maps tau -> 0.5.
    
    p_tilde = p / (2*tau)            if p <= tau
    p_tilde = 0.5 + (p-tau)/(2*(1-tau))  if p > tau
    
    This ensures H_tau achieves maximum at p=tau for any tau in (0,1).
    """
    p = np.clip(p, 1e-10, 1 - 1e-10)
    p_tilde = np.where(
        p <= tau,
        p / (2 * tau),
        0.5 + (p - tau) / (2 * (1 - tau))
    )
    return np.clip(p_tilde, 1e-10, 1 - 1e-10)


def h_tau(p: np.ndarray, tau: float) -> np.ndarray:
    """
    Piecewise-Rescaled Decision Entropy.
    Maximum at p=tau, reduces to Shannon entropy when tau=0.5.
    """
    p_tilde = piecewise_rescale(p, tau)
    return shannon_entropy(p_tilde)


def apply_ccrc(df_condition: pd.DataFrame, tau: float, metric='h_tau',) -> pd.DataFrame:
    """
    Class-Conditioned Rejection Curve ranking.
    
    Within each predicted class, rank patients by H_tau score (descending).
    The rank is expressed as a percentile within the class (0=least uncertain,
    1=most uncertain), so rejection thresholds are comparable across classes.
    
    Steps:
        1. Assign predicted class: 1 if mu >= tau, else 0
        2. Within each predicted class, rank by h_tau descending
        3. Convert to within-class percentile rank
    
    The CCRC score is the within-class percentile — used as the uncertainty
    signal for rejection decisions instead of the raw H_tau value.


    Parameters
    ----------
    df_condition : pd.DataFrame
        DataFrame containing patient-condition rows for a single condition.
    tau : float
        Decision threshold for predicted class assignment.
    metric : str, optional
        The uncertainty metric to rank by (default is 'h_tau').
    """
    df = df_condition.copy()
    df['predicted_class'] = (df['mu'] >= tau).astype(int)

    # Within-class percentile rank of H_tau (higher = more uncertain)
    df[f'{metric}_ccrc'] = df.groupby('predicted_class')[metric].rank(
        method='average', pct=True
    )

    return df


# ─────────────────────────────────────────────
# MAIN COMPUTATION
# ─────────────────────────────────────────────

def compute_all_scores(
    input_path: str,
    output_path: str,
) -> pd.DataFrame:

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    required = {'idx', 'mu', 'sigma', 'label', 'tau', 'd', 'n'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    print(f"  {len(df):,} patient-condition rows")
    print(f"  Conditions: {df[['tau','d','n']].drop_duplicates().shape[0]}")

    all_results = []
    conditions  = df.groupby(['tau', 'd', 'n'])

    for (tau, d, n), group in conditions:
        g = group.copy()

        # ── Four uncertainty signals ──
        g['h_total'] = shannon_entropy(g['mu'].values)
        g['margin']  = decision_margin(g['mu'].values, tau)
        g['h_tau']   = h_tau(g['mu'].values, tau)

        # ── CCRC: within-class percentile rank of H_tau ──
        g = apply_ccrc(g, tau, metric='h_tau')

        # ── CCRC: within-class percentile rank of Margin ──
        g = apply_ccrc(g, tau, metric='margin')

        # ── CCRC: within-class percentile rank of H_total ──
        g = apply_ccrc(g, tau, metric='h_total')

        all_results.append(g)

    result_df = pd.concat(all_results, ignore_index=True)


    result_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"Columns: {list(result_df.columns)}")


    return result_df



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True,
                        help='Path to all_conditions_summary.csv')
    parser.add_argument('--output', required=True,
                        help='Path to write all_conditions_uncertainty.csv')
    args = parser.parse_args()

    compute_all_scores(args.input, args.output)
