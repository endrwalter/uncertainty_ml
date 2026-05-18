import ast
import numpy as np
import pandas as pd

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
    return df

# ─────────────────────────────────────────────
# SHARED REJECTION LOGIC
# ─────────────────────────────────────────────
def reject_patients(df: pd.DataFrame, method: str, rr: float, tau: float) -> pd.DataFrame:
    if rr == 0.0:
        return df.copy()
        
    if method == 'h_tau_ccr':
        retained = []
        for cls in [0, 1]:
            cls_df = df[df['predicted_class'] == cls].copy()
            if cls_df.empty: continue
            n_reject = int(np.floor(len(cls_df) * rr))
            retained.append(cls_df.sort_values('h_tau_ccr', ascending=False).iloc[n_reject:])
        return pd.concat(retained) if retained else df.iloc[0:0].copy()
        
    elif method == 'random_ccr':
        retained = []
        for cls in [0, 1]:
            cls_df = df[df['predicted_class'] == cls].copy()
            if cls_df.empty: continue
            n_reject = int(np.floor(len(cls_df) * rr))
            retained.append(cls_df.sample(frac=1, random_state=42).iloc[n_reject:])
        return pd.concat(retained) if retained else df.iloc[0:0].copy()
        
    else:
        n_reject = int(np.floor(len(df) * rr))
        if n_reject == 0: return df.copy()
        ascending = method in INVERT_SIGNAL
        sorted_df = df.sort_values(method, ascending=ascending)
        return df.loc[sorted_df.index[n_reject:]].copy()

# ─────────────────────────────────────────────
# SHARED METRICS
# ─────────────────────────────────────────────
def compute_ccaugrc(df: pd.DataFrame, method: str, tau: float, target_class: int) -> float:
    n_total = (df['label'] == target_class).sum()
    if n_total == 0: return np.nan

    risks, coverages = [], []
    for rr in REJECTION_RATES:
        ret = reject_patients(df, method, rr, tau)
        cls_ret = ret[ret['label'] == target_class]

        if len(cls_ret) == 0:
            risks.append(0.0); coverages.append(0.0)
        else:
            y_true = cls_ret['label'].values
            y_pred = (cls_ret['mu'].values >= tau).astype(int)
            risks.append((y_pred != y_true).sum() / n_total)
            coverages.append(len(cls_ret) / n_total)

    pairs = sorted(zip(coverages, risks))
    return float(np.trapezoid([p[1] for p in pairs], [p[0] for p in pairs]))

def compute_global_augrc(df: pd.DataFrame, method: str, tau: float) -> float:
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