import ast
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────
# SHARED CONFIGURATION
# ─────────────────────────────────────────────
DISEASES = {
    'MS': {'tau': 0.13, 'label': 'Multiple Sclerosis  (Severe, τ=0.13)'},
    'PD': {'tau': 0.39, 'label': "Parkinson's Disease  (Moderate, τ=0.39)"},
    'AD': {'tau': 0.54, 'label': "Alzheimer's Disease  (Mild, τ=0.54)"},
}

METHODS = {
    'h_total':    'H_Total (Global)',
    'margin':     'Margin (Global)',
    'h_tau':      'H_tau (Global)',
    'h_tau_ccr':  'H_tau + CCR (Proposed)',
    'random_ccr': 'Random + CCR (Control)',
    'h_total_ccr': 'H_Total + CCR (Control)',
    'margin_ccr': 'Margin + CCR (Control)',
}

INVERT_SIGNAL = {'margin'}
REJECTION_RATES = np.arange(0.0, 0.81, 0.05).round(2)

# ─────────────────────────────────────────────
# SHARED SIGNAL COMPUTATION
# ─────────────────────────────────────────────
def compute_signals(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Computes all uncertainty signals and variance needed for analysis."""
    df = df.copy()
    
    # ── Ensemble variance (if available) ──
    if 'ensemble_variance' not in df.columns:
        if 'sigma' in df.columns:
            df['ensemble_variance'] = df['sigma'] ** 2
        elif 'all_probs' in df.columns:
            def parse_probs(x):
                if isinstance(x, str):
                    try: return np.array(ast.literal_eval(x))
                    except Exception: return np.array([np.nan])
                return np.array(x) if hasattr(x, '__iter__') else np.array([np.nan])
            df['ensemble_variance'] = df['all_probs'].apply(parse_probs).apply(np.var)
        else:
            df['ensemble_variance'] = np.nan

    # ── Uncertainty signals ──
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
    
    # H_tau CCR: within-class percentile rank
    df['h_tau_ccr'] = df.groupby('predicted_class')['h_tau'].rank(
        method='average', pct=True
    )
    # H_total CCR: within-class percentile rank
    df['h_total_ccr'] = df.groupby('predicted_class')['h_total'].rank(
        method='average', pct=True
    )
    # Margin CCR: within-class percentile rank
    df['margin_ccr'] = df.groupby('predicted_class')['margin'].rank(
        method='average', pct=True
    )

    
    return df

# ─────────────────────────────────────────────
# SHARED REJECTION LOGIC
# ─────────────────────────────────────────────
def reject_patients(df: pd.DataFrame, rr: float, tau: float, method: str) -> pd.DataFrame:
    if rr == 0.0:
        return df.copy()
        
    if method.endswith('ccr'):
        retained = []
        for cls in [0, 1]:
            cls_df = df[df['predicted_class'] == cls].copy()
            if cls_df.empty: 
                continue
                
            n_reject = int(np.floor(len(cls_df) * rr))
            
            if method == 'random_ccr':
                retained.append(cls_df.sample(frac=1, random_state=42).iloc[n_reject:])
            elif method == 'margin_ccr':
                retained.append(cls_df.sort_values(method, ascending=True).iloc[n_reject:])
            else:
                # Catches h_tau_ccr and any other CCR method
                retained.append(cls_df.sort_values(method, ascending=False).iloc[n_reject:])
                
        return pd.concat(retained) if retained else df.iloc[0:0].copy()
        
    else:
        n_reject = int(np.floor(len(df) * rr))
        if n_reject == 0: 
            return df.copy()
            
        ascending = method in INVERT_SIGNAL
        sorted_df = df.sort_values(method, ascending=ascending)
        return df.loc[sorted_df.index[n_reject:]].copy()


# ─────────────────────────────────────────────
# SHARED METRICS
# ─────────────────────────────────────────────
def compute_ccaugrc(df: pd.DataFrame, method: str, tau: float, target_class: int) -> dict:
    """
    Computes Class-Conditional AUGRC.
    Returns a dict with the area, support bounds, and raw curve data to address Reviewer Q5.
    """
    n_total = (df['label'] == target_class).sum()
    if n_total == 0: 
        return {'area': np.nan, 'support': (np.nan, np.nan)}

    risks, coverages = [], []
    for rr in REJECTION_RATES:
        ret = reject_patients(df, rr, tau, method)
        cls_ret = ret[ret['label'] == target_class]

        if len(cls_ret) == 0:
            risks.append(0.0)
            coverages.append(0.0)
        else:
            y_true = cls_ret['label'].values
            y_pred = (cls_ret['mu'].values >= tau).astype(int)
            risks.append((y_pred != y_true).sum() / n_total)
            coverages.append(len(cls_ret) / n_total)

    # Sort by coverage (x-axis) to correctly compute area
    pairs = sorted(zip(coverages, risks))
    cov_sorted = [p[0] for p in pairs]
    risk_sorted = [p[1] for p in pairs]
    
    # Calculate full empirical area
    area = float(np.trapezoid(risk_sorted, cov_sorted))
    
    return {
        'area': area,
        'support': (min(cov_sorted), max(cov_sorted)), # Reviewer Q5: "report support for each area"
        'coverages': cov_sorted,
        'risks': risk_sorted
    }


def compute_global_augrc(df: pd.DataFrame, method: str, tau: float) -> dict:
    """
    Computes Global AUGRC.
    Returns a dict with the area, support bounds, and raw curve data to address Reviewer Q5.
    """
    risks, coverages = [], []
    
    for rr in REJECTION_RATES:
        ret = reject_patients(df, rr, tau, method)
        
        if len(ret) > 0:
            y_t = ret['label'].values
            y_p = (ret['mu'].values >= tau).astype(int)
            risks.append((y_t != y_p).sum() / len(df))
            coverages.append(len(ret) / len(df))
        else:
            risks.append(0.0)
            coverages.append(0.0)
            
    pairs = sorted(zip(coverages, risks))
    cov_sorted = [p[0] for p in pairs]
    risk_sorted = [p[1] for p in pairs]
    
    area = float(np.trapezoid(risk_sorted, cov_sorted))
    
    return {
        'area': area,
        'support': (min(cov_sorted), max(cov_sorted)), # Reviewer Q5: "report support for each area"
        'coverages': cov_sorted,
        'risks': risk_sorted
    }


# ─────────────────────────────────────────────
# COMMON SUPPORT AREA
# ─────────────────────────────────────────────
def compute_common_support_areas(curves_dict: dict) -> dict:
    """
    Calculates the AUGRC over a common support range across multiple methods.
    
    Args:
        curves_dict: A dictionary mapping method names to their outputs from 
                     compute_global_augrc or compute_ccaugrc.
    Returns:
        A dictionary mapping method names to their common-support AUC.
    """
    # 1. Find the common support (maximum of the minimum coverages)
    # This is the lowest coverage reached by *all* methods.
    min_covs = [data['support'][0] for data in curves_dict.values() if not np.isnan(data['support'][0])]
    max_covs = [data['support'][1] for data in curves_dict.values() if not np.isnan(data['support'][1])]
    
    if not min_covs:
        return {}
        
    common_min = max(min_covs)
    common_max = min(max_covs) # Usually 1.0
    
    common_areas = {}
    for method, data in curves_dict.items():
        if np.isnan(data['area']):
            common_areas[method] = np.nan
            continue
            
        covs = np.array(data['coverages'])
        risks = np.array(data['risks'])
        
        # Interpolate the curve to find the exact risk at the common_min boundary
        interpolator = interp1d(covs, risks, kind='linear', fill_value="extrapolate")
        
        # Filter points that fall within the common support
        mask = (covs >= common_min) & (covs <= common_max)
        common_covs = covs[mask].tolist()
        common_risks = risks[mask].tolist()
        
        # Add the exact boundary points if they aren't already in the array
        if common_min not in common_covs:
            common_covs.insert(0, common_min)
            common_risks.insert(0, float(interpolator(common_min)))
            
        if common_max not in common_covs:
            common_covs.append(common_max)
            common_risks.append(float(interpolator(common_max)))
            
        # Calculate area over the new, standardized domain
        common_areas[method] = float(np.trapezoid(common_risks, common_covs))
        
    print(f"Calculated common-support area over coverage range: [{common_min:.3f}, {common_max:.3f}]")
    return common_areas