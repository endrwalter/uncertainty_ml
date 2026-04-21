""" Uncertainty Computation Functions """
from math import tau

import numpy as np
import pandas as pd

def tau_relative_entropy(p, tau, eps=1e-10):
    """
    Entropy that peaks at tau instead of 0.5
    """
    # Rescale probability space
    if p < tau:
        # Map [0, tau] → [0, 0.5]
        p_rescaled = 0.5 * (p / tau)
    else:
        # Map [tau, 1] → [0.5, 1]
        p_rescaled = 0.5 + 0.5 * ((p - tau) / (1 - tau))
    
    # Standard binary entropy on rescaled p
    p_rescaled = np.clip(p_rescaled, eps, 1-eps)
    H_tau = -p_rescaled * np.log2(p_rescaled) - (1-p_rescaled) * np.log2(1-p_rescaled)
    
    return H_tau

def compute_uncertainties(probas_df, tau=None):
    """
    Computes H (Total), C (Aleatoric), and I (Epistemic) uncertainty.
    
    - probas_df: DataFrame containing model probabilities and optionally 'label'.
    - tau: The optimal decision threshold. If None, defaults to the empirical prior.
    """
    eps = 1e-12

    if tau is None:
        if 'label' in probas_df.columns:
            tau = probas_df['label'].mean() # Empirical Prior
        else:
            tau = 0.5 # Fallback to standard balanced threshold

    # Ensure we only compute on probabilities
    prob_cols = [c for c in probas_df.columns if c not in ['idx', 'label', 'PatientID']]
    probs = probas_df[prob_cols].values
    
    # Clip to avoid exact 0s or 1s ruining the log2 calculation
    probs = np.clip(probs, eps, 1 - eps)
    
    # 1. Ensemble Mean
    mean_scaled_probs = np.nanmean(probs, axis=1)
    
    # 2. Total Uncertainty (H)
    H = -(mean_scaled_probs * np.log2(mean_scaled_probs) + 
          (1 - mean_scaled_probs) * np.log2(1 - mean_scaled_probs))
    
    # 3. Aleatoric Uncertainty (C)
    individual_entropies = -(probs * np.log2(probs) + 
                             (1 - probs) * np.log2(1 - probs))
    C = np.nanmean(individual_entropies, axis=1)
    
    # 4. Epistemic Uncertainty (I)
    I = H - C

    # 5. weighted mean Decision-Theoretic Uncertainty
    distance_to_tau = np.abs(mean_scaled_probs - tau) 

    # 6. Margin-weighted total uncertainty (# total uncertainty amplified by decision risk)
    mw_H = H / (distance_to_tau + 0.01)

    # 7. Margin-weighted ensemble std deviation (captures how much the ensemble disagrees, weighted by decision risk)
    mw_std = np.nanstd(probs, axis=1) / (distance_to_tau + 0.01)

    # 8. Tau-relative entropy
    H_tau = np.array([tau_relative_entropy(p, tau) for p in mean_scaled_probs])

    return pd.DataFrame({
        'H_Total': H, 
        'C_Aleatoric': C, 
        'I_Epistemic': I, 
        'Distance_to_Tau': distance_to_tau,
        'MW_H_Total': mw_H,
        'MW_Std': mw_std,
        'H_Tau': H_tau,
        'Final_Calibrated_Prob': mean_scaled_probs
    }, index=probas_df.index)



import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def extract_and_analyze_rejection(df, prior_threshold, rate, method='class_conditioned', 
                                  prob_col='Final_Calibrated_Prob', 
                                  sort_col='H_Total',
                                  uncertainty_cols=['H_Total', 'C_Aleatoric', 'I_Epistemic']):
    """
    Extracts KEPT and REMOVED patients at a specific rejection rate,
    and computes detailed statistical comparisons of their uncertainty profiles.
    """
    df = df.copy()
    
    # Standardize probability column names
    if prob_col not in df.columns and 'mu' in df.columns:
        prob_col = 'mu'
        
    # Generate predictions using the baseline prior
    if 'y_pred' not in df.columns:
        df['y_pred'] = (df[prob_col] >= prior_threshold).astype(int)
    
    if sort_col == "Distance_to_Tau":
        # invert the distance to tau for sorting (we want to drop those closest to tau first)
        df['Distance_to_Tau'] = -df['Distance_to_Tau']
        
    # ---------------------------------------------------------
    # 1. Split the Data based on Rejection Method
    # ---------------------------------------------------------
    if method == 'standard':
        n_drop = int(len(df) * (rate / 100.0))
        sorted_df = df.sort_values(by=sort_col, ascending=False)
        removed_df = sorted_df.iloc[:n_drop]
        kept_df = sorted_df.iloc[n_drop:]
        
    elif method == 'class_conditioned':
        pos_pool = df[df['y_pred'] == 1].copy()
        neg_pool = df[df['y_pred'] == 0].copy()
        
        n_drop_pos = int(len(pos_pool) * (rate / 100.0))
        n_drop_neg = int(len(neg_pool) * (rate / 100.0))
        
        # Process Positives
        sorted_pos = pos_pool.sort_values(by=sort_col, ascending=False)
        removed_pos = sorted_pos.iloc[:n_drop_pos] if n_drop_pos > 0 else pd.DataFrame()
        kept_pos = sorted_pos.iloc[n_drop_pos:] if n_drop_pos > 0 else pos_pool
        
        # Process Negatives
        sorted_neg = neg_pool.sort_values(by=sort_col, ascending=False)
        removed_neg = sorted_neg.iloc[:n_drop_neg] if n_drop_neg > 0 else pd.DataFrame()
        kept_neg = sorted_neg.iloc[n_drop_neg:] if n_drop_neg > 0 else neg_pool
        
        removed_df = pd.concat([removed_pos, removed_neg])
        kept_df = pd.concat([kept_pos, kept_neg])
    else:
        raise ValueError("Method must be 'standard' or 'class_conditioned'")

    # ---------------------------------------------------------
    # 2. Compute Statistical Uncertainty Summary
    # ---------------------------------------------------------
    stats = []
    
    # Only evaluate columns that actually exist in your dataframe
    valid_cols = [c for c in uncertainty_cols if c in df.columns]
    
    for col in valid_cols:
        kept_vals = kept_df[col].dropna()
        removed_vals = removed_df[col].dropna()
        
        # Mann-Whitney U test (non-parametric, perfect for uncertainty distributions)
        if len(kept_vals) > 0 and len(removed_vals) > 0:
            stat, p_val = mannwhitneyu(removed_vals, kept_vals, alternative='greater')
        else:
            p_val = np.nan
            
        stats.append({
            'Metric': col,
            'Kept_Mean': kept_vals.mean(),
            'Kept_Std': kept_vals.std(),
            'Removed_Mean': removed_vals.mean(),
            'Removed_Std': removed_vals.std(),
            'P_Value': p_val
        })
        
    stats_df = pd.DataFrame(stats)
    
    # ---------------------------------------------------------
    # --- Console Output ---
    # ---------------------------------------------------------
    print("="*80)
    print(f"REJECTION AUDIT: {method.upper()} @ {rate}%")
    print("="*80)
    print(f"Total Cohort:    {len(df)} patients")
    print(f"Total Kept:      {len(kept_df)} patients")
    print(f"Total Removed:   {len(removed_df)} patients")
    print("-" * 80)
    print("UNCERTAINTY PROFILE COMPARISON:")
    
    # Format for pretty printing
    display_df = stats_df.copy()
    display_df['Kept_Profile'] = display_df.apply(lambda x: f"{x['Kept_Mean']:.3f} ± {x['Kept_Std']:.3f}", axis=1)
    display_df['Removed_Profile'] = display_df.apply(lambda x: f"{x['Removed_Mean']:.3f} ± {x['Removed_Std']:.3f}", axis=1)
    display_df['P_Value'] = display_df['P_Value'].apply(lambda x: '<0.001' if x < 0.001 else f"{x:.3f}")
    
    print(display_df[['Metric', 'Kept_Profile', 'Removed_Profile', 'P_Value']].to_string(index=False))
    print("="*80)
    
    return kept_df, removed_df, stats_df