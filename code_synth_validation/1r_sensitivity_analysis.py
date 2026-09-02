import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# ─────────────────────────────────────────────────────────────
# CONFIGURAZIONE INIZIALE
# ─────────────────────────────────────────────────────────────
METRIC = 'specificity'  # Options: 'auprc' or 'sensitivity' or 'specificity'
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


def plot_elkan_grid(df_all):
    """
    4x4 Grid varying prevalence (pi) and cost-derived thresholds (tau).
    """
    selected_pis = [0.10, 0.20, 0.40, 0.50]
    target_d = 2.0  
    target_n = 300  
    rejection_rates = np.arange(0.0, 0.81, 0.05)

    set_style()
    fig, axes = plt.subplots(len(selected_pis), 4, figsize=(10, 9.0), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.94, wspace=0.15, hspace=0.15)

    print(f'\nGenerazione della griglia 4x4 (Metrica: {METRIC.upper()}) in corso...')

    for i, pi_val in enumerate(selected_pis):
        base_group = df_all[
            (df_all['tau'] == pi_val) & 
            (df_all['d'] == target_d) & 
            (df_all['n'] == target_n)
        ].copy()

        if base_group.empty:
            continue

        cost_scenarios = {
            'Baseline': (None, None),
            'Elkan 1': (4.0, 1.0),  
            'Elkan 2': (1.0, 1.0),  
            'Elkan 3': (1.0, 2.0),  
        }

        col_taus = []
        col_titles = []

        for name, (c_fn, c_fp) in cost_scenarios.items():
            if name == 'Baseline':
                tau_val = pi_val
                title = f'$\\tau = \\pi = {pi_val:.2f}$ (Baseline)'
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

        for j, tau_val in enumerate(col_taus):
            ax = axes[i, j]
            sub_cond = df_all[
                (df_all['tau'] == tau_val) & 
                (df_all['d'] == target_d) & 
                (df_all['n'] == target_n)
            ]
            
            if sub_cond.empty:
                sub_cond = base_group.copy()
                sub_cond['predicted_class'] = (sub_cond['mu'] >= tau_val).astype(int)
            else:
                sub_cond = sub_cond.copy()

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
                                if class_df.empty: continue
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
                ax.set_ylabel(f'$\\pi = {pi_val}$\n{METRIC.upper()}', fontsize=10)

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


def plot_distribution_density_ablation(df_all):
    """
    1x3 Grid varying the distribution distance (d) to prove Vulnerability 2.
    Plots Minority Class Retention to explicitly show how class-conditional 
    overlap triggers the disproportionate rejection artifact.
    """
    distances = [1.0, 2.0, 3.0]
    pi_val = 0.20
    tau_val = 0.50  # Costringe un'enorme disparità di margine 
    target_n = 300
    rejection_rates = np.arange(0.0, 0.81, 0.05)
    
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.25, top=0.85, wspace=0.1)
    
    print('\nGenerazione ablation su densità (Vulnerabilità 2)...')
    
    for j, target_d in enumerate(distances):
        ax = axes[j]
        
        # 1. Recupera il gruppo base usando pi_val (nel CSV salvato sotto 'tau')
        base_group = df_all[
            (df_all['tau'] == pi_val) & 
            (df_all['d'] == target_d) & 
            (df_all['n'] == target_n)
        ].copy()
        
        if base_group.empty:
            print(f'Nessun dato base per d={target_d}')
            continue
            
        # 2. Applica la soglia decisionale sfalsata (tau_val = 0.50)
        sub_cond = base_group.copy()
        sub_cond['predicted_class'] = (sub_cond['mu'] >= tau_val).astype(int)
        
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
                            if class_df.empty: continue
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
                
                # Calcolo: Quanti positivi (minority) sono sopravvissuti rispetto all'inizio?
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
    
    # Esegue il primo plot e chiude il canvas in sicurezza
    plot_elkan_grid(df_all_data)
    
    # Esegue il plot dell'ablazione sulla densità
    plot_distribution_density_ablation(df_all_data)