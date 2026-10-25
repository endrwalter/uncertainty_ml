"""
1r_sensitivity_analysis.py
=============================================================
Generates supplementary figures demonstrating the structural 
vulnerabilities of global uncertainty thresholds under varying 
cost matrices and distribution overlaps, strictly decoupling 
intrinsic class prevalence (pi) from decision thresholds (tau).

Inputs
------
../data/synthetic_results/all_conditions_uncertainty.csv 
(Produced by 4_compute_uncertainty_scores.py)

Outputs
-------
Supplementary_Elkan_Grid_<METRIC>.pdf (4x4 grid varying prevalence and tau)
Supplementary_Density_Ablation.pdf (1x3 grid proving Vulnerability 2)

Usage
-----
python 1r_sensitivity_analysis.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


# ─────────────────────────────────────────────────────────────
# CONFIGURAZIONE INIZIALE
# ─────────────────────────────────────────────────────────────
METRIC = 'sensitivity'  # Options: 'auprc', 'sensitivity', 'specificity'
INPUT_CSV = '../data/synthetic_results/all_conditions_uncertainty.csv'


def set_style():
    """Imposta lo stile grafico coerente con il paper."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8


PALETTE = {
    'h_tau_ccrc': '#1f77b4',
    'h_tau': '#2ca02c',
    'margin': '#ff7f0e',
    'h_total': '#7f7f7f',
}

LABELS = {
    'h_tau_ccrc': 'H_tau + CCRC (Proposed)',
    'h_tau': 'H_tau (Global)',
    'margin': 'Margin (Global)',
    'h_total': 'H_Total (Global)',
}

style_map = {
    'h_tau_ccrc': {'ls': '-', 'lw': 2.0, 'zorder': 4},
    'h_tau': {'ls': '--', 'lw': 1.5, 'zorder': 3},
    'margin': {'ls': '-.', 'lw': 1.5, 'zorder': 2},
    'h_total': {'ls': ':', 'lw': 1.5, 'zorder': 1},
}


# ─────────────────────────────────────────────────────────────
# FUNZIONI DI RICALCOLO ESATTO (da 4_compute_uncertainty_scores.py)
# ─────────────────────────────────────────────────────────────

def shannon_entropy(p: np.ndarray) -> np.ndarray:
    """Standard Shannon binary entropy. Maximum at p=0.5."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def decision_margin(p: np.ndarray, tau: float) -> np.ndarray:
    """Absolute distance from decision boundary. Lower = more uncertain."""
    return np.abs(p - tau)


def piecewise_rescale(p: np.ndarray, tau: float) -> np.ndarray:
    """Piecewise linear rescaling that maps tau -> 0.5."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    p_tilde = np.where(
        p <= tau,
        p / (2 * tau),
        0.5 + (p - tau) / (2 * (1 - tau))
    )
    return np.clip(p_tilde, 1e-10, 1 - 1e-10)


def compute_h_tau(p: np.ndarray, tau: float) -> np.ndarray:
    """Piecewise-Rescaled Decision Entropy around threshold tau."""
    return shannon_entropy(piecewise_rescale(p, tau))


def recompute_for_threshold(base_df: pd.DataFrame, tau_val: float) -> pd.DataFrame:
    """
    Preserves the row's generating prevalence dataset (base_df) and recomputes
    all threshold-dependent uncertainty scores and rankings at tau_val.
    """
    df = base_df.copy()
    mu = df['mu'].values

    df['predicted_class'] = (mu >= tau_val).astype(int)
    df['h_total'] = shannon_entropy(mu)
    df['margin'] = decision_margin(mu, tau_val)
    df['h_tau'] = compute_h_tau(mu, tau_val)

    # Within-class percentile rank of H_tau (higher = more uncertain)
    df['h_tau_ccrc'] = df.groupby('predicted_class')['h_tau'].rank(
        method='average', pct=True
    )
    return df


# ─────────────────────────────────────────────────────────────
# PLOTTING FUNCTIONS
# ─────────────────────────────────────────────────────────────

def plot_elkan_grid(df_all: pd.DataFrame):
    """
    4x4 Grid independently varying intrinsic prevalence (pi, rows) 
    and cost-derived decision thresholds (tau, columns).
    """
    selected_pis = [0.10, 0.20, 0.40, 0.50]
    target_d = 2.0
    target_n = 300
    rejection_rates = np.arange(0.0, 0.81, 0.05)

    set_style()
    fig, axes = plt.subplots(len(selected_pis), 4, figsize=(10, 9.0), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.94, wspace=0.15, hspace=0.15)

    print(f'\nGenerazione della griglia 4x4 corretta (Metrica: {METRIC.upper()}) in corso...')

    for i, pi_val in enumerate(selected_pis):
        # 1. Seleziona il dataset generato con prevalenza intrinseca pi_val
        # (nella pipeline sintetica, la prevalenza generativa è indicizzata dalla colonna 'tau')
        base_group = df_all[
            np.isclose(df_all['tau'], pi_val) &
            np.isclose(df_all['d'], target_d) &
            (df_all['n'] == target_n)
        ].copy()

        if base_group.empty:
            print(f'Attenzione: Nessun dato trovato per pi={pi_val}, d={target_d}, n={target_n}')
            continue

        cost_scenarios = {
            'Baseline': (None, None),
            'Elkan 1': (4.0, 1.0),  # tau = 0.20
            'Elkan 2': (1.0, 1.0),  # tau = 0.50
            'Elkan 3': (1.0, 2.0),  # tau = 0.67
        }

        col_taus = []
        col_titles = []

        for name, (c_fn, c_fp) in cost_scenarios.items():
            if name == 'Baseline':
                tau_val = pi_val
                title = '$\\tau = \\pi$ (Baseline)'
            else:
                tau_val = c_fp / (c_fn + c_fp)
                if c_fn > c_fp:
                    title = f'$\\tau = {tau_val:.2f}$ ($C_{{FN}} = {int(c_fn)}C_{{FP}}$)'
                elif c_fn == c_fp:
                    title = f'$\\tau = {tau_val:.2f}$ ($C_{{FN}} = C_{{FP}}$)'
                else:
                    title = f'$\\tau = {tau_val:.2f}$ ($C_{{FP}} = {int(c_fp)}C_{{FN}}$)'

            col_taus.append(tau_val)
            col_titles.append(title)

        # 2. Per ogni colonna, MANTIENI base_group (pi fisso) e ricalcola gli score attorno a tau_val
        for j, tau_val in enumerate(col_taus):
            ax = axes[i, j]
            sub_cond = recompute_for_threshold(base_group, tau_val)

            for method in ['h_total', 'margin', 'h_tau', 'h_tau_ccrc']:
                metric_values = []
                for rr in rejection_rates:
                    if rr == 0.0:
                        retained = sub_cond.copy()
                    else:
                        if method == 'h_tau_ccrc':
                            retained_parts = []
                            for cls in [0, 1]:
                                class_df = sub_cond[sub_cond['predicted_class'] == cls]
                                if class_df.empty:
                                    continue
                                n_reject = int(np.floor(len(class_df) * rr))
                                sorted_cls = class_df.sort_values('h_tau_ccrc', ascending=False)
                                retained_parts.append(sorted_cls.iloc[n_reject:])
                            retained = pd.concat(retained_parts) if retained_parts else sub_cond.copy()
                        else:
                            n_reject = int(np.floor(len(sub_cond) * rr))
                            if n_reject == 0:
                                retained = sub_cond.copy()
                            else:
                                ascending = (method == 'margin')
                                sorted_df = sub_cond.sort_values(method, ascending=ascending)
                                retained = sorted_df.iloc[n_reject:].copy()

                    if len(retained) == 0:
                        metric_values.append(np.nan)
                        continue

                    y_true = retained['label'].values
                    y_pred = (retained['mu'].values >= tau_val).astype(int)
                    y_prob = retained['mu'].values

                    if METRIC.lower() == 'sensitivity':
                        tp = ((y_pred == 1) & (y_true == 1)).sum()
                        fn = ((y_pred == 0) & (y_true == 1)).sum()
                        val = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                    elif METRIC.lower() == 'specificity':
                        tn = ((y_pred == 0) & (y_true == 0)).sum()
                        fp = ((y_pred == 1) & (y_true == 0)).sum()
                        val = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                    elif METRIC.lower() == 'auprc':
                        if retained['label'].nunique() < 2:
                            val = np.nan
                        else:
                            try:
                                val = average_precision_score(y_true, y_prob)
                            except Exception:
                                val = np.nan
                    else:
                        raise ValueError(f'Metrica non supportata: {METRIC}')

                    metric_values.append(val)

                ls = style_map.get(method, {}).get('ls', '-')
                lw = style_map.get(method, {}).get('lw', 1.5)
                zo = style_map.get(method, {}).get('zorder', 2)

                ax.plot(
                    rejection_rates, metric_values,
                    label=LABELS.get(method, method), color=PALETTE.get(method, '#333333'),
                    linewidth=lw, linestyle=ls, zorder=zo
                )

            ax.axvline(0.30, color='#bdc3c7', ls='--', lw=1.0, alpha=0.6, zorder=0)

            if i == 0:
                ax.set_title(col_titles[j], fontsize=10, pad=6)

            ax.tick_params(labelsize=9)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, 0.8)
            ax.grid(True, linestyle='--', alpha=0.4)

            if i == len(selected_pis) - 1:
                ax.set_xlabel('Rejection Rate', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'$\\pi = {pi_val:.2f}$\n{METRIC.upper()}', fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(), by_label.keys(),
            loc='lower center', bbox_to_anchor=(0.5, 0.02),
            ncol=min(4, len(by_label)), fontsize=11, frameon=False
        )

    out_path = f'Supplementary_Elkan_Grid_{METRIC.upper()}.pdf'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'PDF salvato: {out_path}')


def plot_distribution_density_ablation(df_all: pd.DataFrame):
    """
    1x3 Grid varying the distribution distance (d) at fixed prevalence (pi=0.20)
    and fixed threshold (tau=0.50), with all scores recomputed around tau=0.50.
    """
    distances = [1.0, 2.0, 3.0]
    pi_val = 0.20
    tau_val = 0.50
    target_n = 300
    rejection_rates = np.arange(0.0, 0.81, 0.05)

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.25, top=0.85, wspace=0.1)

    print('\nGenerazione ablation su densità corretta (Vulnerabilità 2)...')

    for j, target_d in enumerate(distances):
        ax = axes[j]

        base_group = df_all[
            np.isclose(df_all['tau'], pi_val) &
            np.isclose(df_all['d'], target_d) &
            (df_all['n'] == target_n)
        ].copy()

        if base_group.empty:
            print(f'Nessun dato base per d={target_d}')
            continue

        # Ricalcola tutti gli score attorno a tau_val = 0.50 mantenendo pi_val = 0.20
        sub_cond = recompute_for_threshold(base_group, tau_val)
        original_minority_count = (sub_cond['label'] == 1).sum()

        for method in ['h_total', 'margin', 'h_tau', 'h_tau_ccrc']:
            retention_values = []

            for rr in rejection_rates:
                if rr == 0.0:
                    retained = sub_cond.copy()
                else:
                    if method == 'h_tau_ccrc':
                        retained_parts = []
                        for cls in [0, 1]:
                            class_df = sub_cond[sub_cond['predicted_class'] == cls]
                            if class_df.empty:
                                continue
                            n_reject = int(np.floor(len(class_df) * rr))
                            sorted_cls = class_df.sort_values('h_tau_ccrc', ascending=False)
                            retained_parts.append(sorted_cls.iloc[n_reject:])
                        retained = pd.concat(retained_parts) if retained_parts else sub_cond.copy()
                    else:
                        n_reject = int(np.floor(len(sub_cond) * rr))
                        if n_reject == 0:
                            retained = sub_cond.copy()
                        else:
                            ascending = (method == 'margin')
                            sorted_df = sub_cond.sort_values(method, ascending=ascending)
                            retained = sorted_df.iloc[n_reject:].copy()

                if len(retained) == 0 or original_minority_count == 0:
                    retention_values.append(np.nan)
                else:
                    current_minority = (retained['label'] == 1).sum()
                    retention_values.append(current_minority / original_minority_count)

            ls = style_map.get(method, {}).get('ls', '-')
            lw = style_map.get(method, {}).get('lw', 1.5)
            ax.plot(
                rejection_rates, retention_values,
                label=LABELS.get(method, method), color=PALETTE.get(method, '#333333'),
                linewidth=lw, linestyle=ls
            )

        ax.set_title(f'Distance $d = {target_d:.1f}$', fontsize=11, pad=8)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 0.8)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel('Global Rejection Rate', fontsize=10)

        if j == 0:
            ax.set_ylabel('Minority Class Retention', fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower center',
        bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=10, frameon=False
    )

    out_path = 'Supplementary_Density_Ablation.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'PDF salvato: {out_path}')


# ─────────────────────────────────────────────────────────────
# ESECUZIONE MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'Loading {INPUT_CSV}...')
    df_all_data = pd.read_csv(INPUT_CSV)

    plot_elkan_grid(df_all_data)
    plot_distribution_density_ablation(df_all_data)