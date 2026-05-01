import os
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_uncertainty_distributions(uncertainty_results, H_bl, save_path=None):
    """
    Plots the distributions of Total, Aleatoric, and Epistemic uncertainty 
    in a single 1x3 figure.
    
    Parameters:
    - uncertainty_results: DataFrame containing H_Total, C_Aleatoric, I_Epistemic
    - H_bl: Float, the computed baseline entropy
    - save_path: Optional string path to save the figure (e.g., 'results/dists.png')
    """
    # Create a 1x3 grid of subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ---------------------------------------------------------
    # 1. Total Uncertainty (H_Total)
    # ---------------------------------------------------------
    sns.histplot(uncertainty_results['H_Total'], bins=50, kde=True, 
                 color='skyblue', edgecolor='black', alpha=0.7, ax=axes[0])
    axes[0].axvline(H_bl, color='red', linestyle='--', linewidth=2, 
                    label=f'Baseline Entropy (H_bl={H_bl:.3f})')
    axes[0].set_title('Total Uncertainty (H_Total)', fontsize=14)
    axes[0].set_xlabel('H_Total', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)

    # ---------------------------------------------------------
    # 2. Aleatoric Uncertainty (Data Noise / C_Aleatoric)
    # ---------------------------------------------------------
    sns.histplot(uncertainty_results['C_Aleatoric'], bins=50, kde=True, 
                 color='orange', edgecolor='black', alpha=0.7, ax=axes[1])
    axes[1].axvline(H_bl, color='red', linestyle='--', linewidth=2, 
                    label=f'Noise Ceiling (H_bl={H_bl:.3f})')
    axes[1].axvline(0, color='green', linestyle='--', linewidth=2, 
                    label='Ideal Feature Clarity (C=0)')
    axes[1].set_title('Aleatoric Uncertainty (Data Noise)', fontsize=14)
    axes[1].set_xlabel('Aleatoric Uncertainty (C)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)

    # ---------------------------------------------------------
    # 3. Epistemic Uncertainty (Model Disagreement / I_Epistemic)
    # ---------------------------------------------------------
    sns.histplot(uncertainty_results['I_Epistemic'], bins=50, kde=True, 
                 color='purple', edgecolor='black', alpha=0.7, ax=axes[2])
    axes[2].axvline(0, color='green', linestyle='--', linewidth=2, 
                    label='Perfect Agreement (I=0)')
    axes[2].set_title('Epistemic Uncertainty (Model Disagreement)', fontsize=14)
    axes[2].set_xlabel('Epistemic Uncertainty (I)', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].legend()
    axes[2].grid(axis='y', linestyle='--', alpha=0.3)

    # Adjust layout so titles and labels don't overlap
    plt.tight_layout()

    # ----------------------------------------------------------
    # 4. Tau-relative entropy (H_Tau)
    if 'H_Tau' in uncertainty_results.columns:
        plt.figure(figsize=(6, 5))
        sns.histplot(uncertainty_results['H_Tau'], bins=50, kde=True, 
                     color='teal', edgecolor='black', alpha=0.7)
        axes[2].axvline(0, color='green', linestyle='--', linewidth=2, 
                        label='Perfect Agreement (H_Tau=0)')
        axes[2].axvline(H_bl, color='red', linestyle='--', linewidth=2, 
                        label=f'Baseline Entropy (H_bl={H_bl:.3f})')
        plt.title('Tau-relative Entropy (H_Tau)', fontsize=14)
        plt.xlabel('Tau-relative Entropy (H_Tau)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()


    if 'imbalance_aware_margin' in uncertainty_results.columns:
        # plot the imbalance-aware uncertainty distribution
        plt.figure(figsize=(6, 5))
        sns.histplot(uncertainty_results['imbalance_aware_margin'], bins=50, kde=True, 
                        color='coral', edgecolor='black', alpha=0.7)
        axes[2].axvline(0, color='green', linestyle='--', linewidth=2,
                        label='Ideal Balanced Certainty (H=0)')
        axes[2].axvline(H_bl, color='red', linestyle='--', linewidth=2,
                        label=f'Baseline Entropy (H_bl={H_bl:.3f})')
        plt.title('Imbalance-Aware margin', fontsize=14)
        plt.xlabel('Imbalance-Aware margin', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()

    if 'imbalance_aware_uncertainty' in uncertainty_results.columns:
        # plot the imbalance-aware uncertainty distribution
        plt.figure(figsize=(6, 5))
        sns.histplot(uncertainty_results['imbalance_aware_uncertainty'], bins=50, kde=True, 
                        color='coral', edgecolor='black', alpha=0.7)
        axes[2].axvline(0, color='green', linestyle='--', linewidth=2,
                        label='Ideal Balanced Certainty (H=0)')
        axes[2].axvline(H_bl, color='red', linestyle='--', linewidth=2,
                        label=f'Baseline Entropy (H_bl={H_bl:.3f})')
        plt.title('Imbalance-Aware Uncertainty', fontsize=14)
        plt.xlabel('Imbalance-Aware Uncertainty', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()


    # Save the figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure successfully saved to: {save_path}")

    # Display the plot
    plt.show()




def plot_combined_uncertainty_analysis_v2(uncertainty_df, y_true, uncertainty_type='H_Total', save_path=None):
    # --- Configuration (Synchronized with Phenotype Template) ---
    border_width = 1.5
    base_fontsize = 10
    rc_params = {
        "axes.linewidth": border_width,
        "xtick.major.width": border_width,
        "ytick.major.width": border_width,
        "font.size": base_fontsize,
        "axes.edgecolor": "black"
    }
    sns.set_theme(style="ticks", rc=rc_params)
    
    fig = plt.figure(figsize=(14, 7))
    fig.set_facecolor('white')
    
    gs_master = gridspec.GridSpec(1, 2, wspace=0.3)

    # ---------------------------------------------------------
    # --- Subplot A: Phase-Plane ---
    # ---------------------------------------------------------
    gs_left = gridspec.GridSpecFromSubplotSpec(
        5, 2, subplot_spec=gs_master[0],
        height_ratios=[0.2, 1, 0.15, 0.10, 0.03], 
        width_ratios=[1, 0.2], 
        hspace=0.05, wspace=0.05
    )
    
    ax_main = fig.add_subplot(gs_left[1, 0])
    ax_hist_x = fig.add_subplot(gs_left[0, 0], sharex=ax_main)
    ax_hist_y = fig.add_subplot(gs_left[1, 1], sharey=ax_main)
    
    # New Axis for the Marginal Legend in the empty top-right corner
    ax_legend_marginal = fig.add_subplot(gs_left[0, 1])
    ax_legend_marginal.axis('off')
    
    # Bottom Legend Axis
    ax_legend_a = fig.add_subplot(gs_left[3, 0])
    ax_legend_a.axis('off')

    gs_cbar_container = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_left[4, 0], width_ratios=[0.25, 0.5, 0.25]
    )
    ax_cbar = fig.add_subplot(gs_cbar_container[0, 1])

    # --- THE FIX: Calculate Global Prior & Shift Colormap ---
    optimal_threshold = y_true.mean()
    
    # Keep the physical scale linear (0 to 1), but force the white center to sit at the threshold
    color_nodes = [
        (0.0, plt.cm.coolwarm(0.0)),
        (optimal_threshold, plt.cm.coolwarm(0.5)),
        (1.0, plt.cm.coolwarm(1.0))
    ]
    shifted_coolwarm = mcolors.LinearSegmentedColormap.from_list('shifted_coolwarm', color_nodes)

    # Plotting Data
    markers_map = {0: 'o', 1: '^'}
    color_tn, color_tp = '#b0b0b0', '#4d4d4d' 
    scatter_plot = None
    
    for true_class in [0, 1]:
        subset = uncertainty_df[uncertainty_df['label'] == true_class]
        if subset.empty: continue
        scatter = ax_main.scatter(
            subset['C_Aleatoric'], subset['I_Epistemic'], 
            c=subset['mu'], 
            cmap=shifted_coolwarm, 
            vmin=0.0, vmax=1.0, # Scale remains purely linear
            alpha=0.8, edgecolors='k', linewidths=0.5,
            s=58, marker=markers_map[true_class]
        )
        scatter_plot = scatter

    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, uncertainty_df['I_Epistemic'].max() * 1.1)

    # Marginal Histograms (Stacked)
    sns.histplot(data=uncertainty_df, x='C_Aleatoric', hue='label', ax=ax_hist_x, 
                 palette={0: color_tn, 1: color_tp}, multiple="stack", alpha=0.6, legend=False)
    sns.histplot(data=uncertainty_df, y='I_Epistemic', hue='label', ax=ax_hist_y, 
                 palette={0: color_tn, 1: color_tp}, multiple="stack", alpha=0.6, legend=False)

    # Upper Right Legend for Marginal Histograms
    marginal_legend_elements = [
        Patch(facecolor=color_tn, alpha=0.6, label='Actual Negative'),
        Patch(facecolor=color_tp, alpha=0.6, label='Actual Positive')
    ]
    ax_legend_marginal.legend(handles=marginal_legend_elements, loc='center left', 
                              bbox_to_anchor=(-0.2, 0.5), frameon=False, fontsize=base_fontsize, title='Marginals')

    # Legend A (Bottom) - Hollow Shapes Only
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Actual Negative', 
               markerfacecolor='none', markersize=8, markeredgecolor='k', markeredgewidth=1.2),
        Line2D([0], [0], marker='^', color='w', label='Actual Positive',
               markerfacecolor='none', markersize=8, markeredgecolor='k', markeredgewidth=1.2)
    ]
    ax_legend_a.legend(handles=legend_elements, loc='center', ncol=2, frameon=False, fontsize=base_fontsize)

    # ---------------------------------------------------------
    # --- Subplot B: Rejection Curves ---
    # ---------------------------------------------------------
    gs_right = gridspec.GridSpecFromSubplotSpec(
        5, 1, subplot_spec=gs_master[1],
        height_ratios=[0.2, 1, 0.15, 0.10, 0.03], hspace=0.05
    )
    ax_b = fig.add_subplot(gs_right[1, 0])
    
    ax_legend_b = fig.add_subplot(gs_right[3, 0])
    ax_legend_b.axis('off')
    
    # Generate binary predictions using the custom clinical threshold
    y_pred_class = (uncertainty_df['mu'] > optimal_threshold).astype(int)
    
    # Extract the sorting metric safely without altering the original dataframe
    sort_series = uncertainty_df[uncertainty_type].copy()
    
    if uncertainty_type in ["Distance_to_Tau", "imbalance_aware_margin"]:
        # Invert only the temporary series so we drop those closest to tau first
        sort_series = -sort_series

    # Sort based on the temporary series
    sorted_indices = sort_series.sort_values(ascending=False).index
    rejection_rates = np.linspace(0, 0.80, 20)
    
    mcc_scores, sens_scores, spec_scores = [], [], []
    for rate in rejection_rates:
        n_rejected = int(len(sorted_indices) * rate)
        idx = sorted_indices[n_rejected:]
        if len(np.unique(y_true.loc[idx])) < 2:
            mcc_scores.append(np.nan); sens_scores.append(np.nan); spec_scores.append(np.nan)
        else:
            y_t, y_p = y_true.loc[idx], y_pred_class.loc[idx]
            mcc_scores.append(matthews_corrcoef(y_t, y_p))
            tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
            sens_scores.append(tp/(tp+fn)); spec_scores.append(tn/(tn+fp))

    ax_b.plot(rejection_rates*100, mcc_scores, marker='s', color='navy', 
              linewidth=2, markersize=5, label='MCC')
    ax_b.plot(rejection_rates*100, sens_scores, marker='o', color='forestgreen', 
              linewidth=2, markersize=5, label='Sensitivity')
    ax_b.plot(rejection_rates*100, spec_scores, marker='^', color='darkorange', 
              linewidth=2, markersize=5, label='Specificity')
    
    ax_b.set_ylim(-0.25, 1.05) 
    ax_b.axhline(0, color='black', lw=1, ls='--', alpha=0.3)
    
    handles, labels = ax_b.get_legend_handles_labels()
    ax_legend_b.legend(handles=handles, labels=labels, loc='center', ncol=3, frameon=False, fontsize=base_fontsize)

    # ---------------------------------------------------------
    # --- Final Formatting & Colorbar Integration ---
    # ---------------------------------------------------------
    for ax in [ax_main, ax_b]:
        for spine in ax.spines.values():
            spine.set_linewidth(border_width)
        ax.tick_params(width=border_width, length=5)
        ax.grid(False)

    for ax_h in [ax_hist_x, ax_hist_y]:
        ax_h.axis('off')

    # Colorbar Generation
    cbar = fig.colorbar(scatter_plot, cax=ax_cbar, orientation='horizontal')
    cbar.outline.set_linewidth(border_width)
    cbar.ax.tick_params(width=border_width, labelsize=base_fontsize)
    # Increased labelpad slightly to make room for the floating text
    cbar.set_label('Consensus Predicted Risk', fontsize=base_fontsize, labelpad=15)

    # 1. Use evenly spaced standard ticks so the numbers aren't cramped
    cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
    cbar.set_ticklabels(['0.0', '0.25', '0.50', '0.75', '1.0'])
    
    # 2. Draw a hard vertical line exactly at the clinical threshold
    cbar.ax.axvline(optimal_threshold, color='black', linewidth=2.5, linestyle='-')
    
    # 3. Add the floating annotation right above the line to label it cleanly
    cbar.ax.text(
        optimal_threshold, 1.05, 
        f'Cutoff ({optimal_threshold:.2f})', 
        ha='center', va='bottom', 
        fontsize=base_fontsize - 1, 
        fontweight='bold',
        transform=cbar.ax.get_xaxis_transform()
    )

    ax_main.set_xlabel('Aleatoric Uncertainty')
    ax_main.set_ylabel('Epistemic Uncertainty')
    ax_b.set_xlabel('Rejection Rate (%)')
    ax_b.set_ylabel('Score')

    ax_hist_x.text(-0.02, 1.0, 'A', transform=ax_hist_x.transAxes, 
                   fontsize=14, fontweight='bold', color='black')
    
    ax_b_header = fig.add_subplot(gs_right[0, 0])
    ax_b_header.axis('off')
    ax_b_header.text(-0.02, 1.0, 'B', transform=ax_b_header.transAxes, 
                     fontsize=14, fontweight='bold', color='black')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_class_conditioned_rejection_curve(df, threshold, uncertainty_col='H_Total', rates=np.linspace(0, 80, 17)):
    """
    Calculates TP, TN, FP, FN, MCC, Sensitivity, and Specificity 
    using a CLASS-CONDITIONED REJECTION rule and plots the resulting curves.
    """
    results = []
    df = df.copy()
    if uncertainty_col in ["Distance_to_Tau", "imbalance_aware_margin"]:
    # invert the distance to tau for sorting (we want to drop those closest to tau first)
        df[uncertainty_col] = -df[uncertainty_col]
    
    # Generate predictions using your calculated clinical baseline prior
    if 'y_pred' not in df.columns:
        df['y_pred'] = (df['mu'] >= threshold).astype(int)
        
    for rate in rates:
        if rate > 0:
            # Split the global dataframe by PREDICTED class
            pos_pool = df[df['y_pred'] == 1].copy()
            neg_pool = df[df['y_pred'] == 0].copy()
            
            # Calculate the drop quotas proportionally for EACH class
            n_drop_pos = int(len(pos_pool) * (rate / 100.0))
            n_drop_neg = int(len(neg_pool) * (rate / 100.0))
            
            # Sort by uncertainty and drop the most uncertain inside each class
            if n_drop_pos > 0:
                kept_pos = pos_pool.sort_values(by=uncertainty_col, ascending=False).iloc[n_drop_pos:]
            else:
                kept_pos = pos_pool
                
            if n_drop_neg > 0:
                kept_neg = neg_pool.sort_values(by=uncertainty_col, ascending=False).iloc[n_drop_neg:]
            else:
                kept_neg = neg_pool
            
            # Recombine the surviving patients back into a single global cohort
            global_kept = pd.concat([kept_pos, kept_neg])
        else:
            global_kept = df.copy()
            
        # Evaluate the metrics
        y_true = global_kept['label']
        y_pred = global_kept['y_pred']
        
        labels_present = np.unique(np.concatenate((y_true, y_pred))) if len(global_kept) > 0 else []
        
        if len(labels_present) > 1:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            mcc = matthews_corrcoef(y_true, y_pred)
        elif len(labels_present) == 1:
            tp = sum((y_true == 1) & (y_pred == 1))
            tn = sum((y_true == 0) & (y_pred == 0))
            fp = sum((y_true == 0) & (y_pred == 1))
            fn = sum((y_true == 1) & (y_pred == 0))
            mcc = 0.0
        else:
            tp = tn = fp = fn = 0
            mcc = 0.0
            
        # Safely calculate Sensitivity and Specificity (handle division by zero)
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            
        results.append({
            'Rejection (%)': rate,
            'Total Kept': len(global_kept),
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'MCC': mcc if not pd.isna(mcc) else np.nan,
            'Sensitivity': sens,
            'Specificity': spec
        })
            
    # Format as a clean DataFrame
    results_df = pd.DataFrame(results)
    
    # ---------------------------------------------------------
    # --- Plotting the Rejection Curve ---
    # ---------------------------------------------------------
    sns.set_theme(style="ticks")
    plt.figure(figsize=(8, 5))
    
    plt.plot(results_df['Rejection (%)'], results_df['MCC'], 
             marker='s', color='navy', linewidth=2, markersize=6, label='MCC')
    plt.plot(results_df['Rejection (%)'], results_df['Sensitivity'], 
             marker='o', color='forestgreen', linewidth=2, markersize=6, label='Sensitivity')
    plt.plot(results_df['Rejection (%)'], results_df['Specificity'], 
             marker='^', color='darkorange', linewidth=2, markersize=6, label='Specificity')
    
    plt.ylim(-0.25, 1.05)
    plt.axhline(0, color='black', lw=1, ls='--', alpha=0.3)
    
    plt.title('Class-Conditioned Rejection Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Rejection Rate (%)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    
    # Place legend neatly below the plot
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False, fontsize=11)
    
    # Clean up formatting
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # --- Console Output ---
    # ---------------------------------------------------------
    print("="*95)
    print("GLOBAL CLASS-CONDITIONED REJECTION: METRICS EVOLUTION")
    print("="*95)
    
    # Create a copy just for printing so the rounding doesn't destroy the actual returned data
    display_df = results_df.copy()
    display_df[['MCC', 'Sensitivity', 'Specificity']] = display_df[['MCC', 'Sensitivity', 'Specificity']].round(3)
    display_df['Rejection (%)'] = display_df['Rejection (%)'].astype(int)
    
    print(display_df.to_string(index=False))
    print("="*95)

    # reverting the distance to tau back to its original form in case it was modified for sorting
    if uncertainty_col in ["Distance_to_Tau", "imbalance_aware_margin"]:
    # invert the distance to tau for sorting (we want to drop those closest to tau first)
        df[uncertainty_col] = -df[uncertainty_col]
    
    return results_df


def plot_comprehensive_rejection_dashboard(df, threshold, prob_col='Final_Calibrated_Prob', 
                                           uncertainty_col='H_Total', rates=np.linspace(0, 80, 17)):
    """
    Plots a 2x2 grid showing the survival trajectories of TP, TN, FP, and FN,
    comparing Standard Rejection vs. Class-Conditioned Rejection.
    """
    results = []
    df = df.copy()

    if uncertainty_col in ["Distance_to_Tau", "imbalance_aware_margin"]:
    # invert the distance to tau for sorting (we want to drop those closest to tau first)
        df[uncertainty_col] = -df[uncertainty_col]

    # Standardize probability column names
    if prob_col not in df.columns and 'mu' in df.columns:
        prob_col = 'mu'
        
    # Generate predictions using the baseline prior
    if 'y_pred' not in df.columns:
        df['y_pred'] = (df[prob_col] >= threshold).astype(int)
        
    for rate in rates:
        if rate == 0:
            kept_std = df.copy()
            kept_cc = df.copy()
        else:
            # 1. Standard Rejection (Global Drop)
            n_drop_std = int(len(df) * (rate / 100.0))
            kept_std = df.sort_values(by=uncertainty_col, ascending=False).iloc[n_drop_std:]
            
            # 2. Class-Conditioned Rejection (Proportional Drop)
            pos_pool = df[df['y_pred'] == 1].copy()
            neg_pool = df[df['y_pred'] == 0].copy()
            
            n_drop_pos = int(len(pos_pool) * (rate / 100.0))
            n_drop_neg = int(len(neg_pool) * (rate / 100.0))
            
            kept_pos = pos_pool.sort_values(by=uncertainty_col, ascending=False).iloc[n_drop_pos:] if n_drop_pos > 0 else pos_pool
            kept_neg = neg_pool.sort_values(by=uncertainty_col, ascending=False).iloc[n_drop_neg:] if n_drop_neg > 0 else neg_pool
            
            kept_cc = pd.concat([kept_pos, kept_neg])

        # Helper function to extract all 4 matrix components
        def get_matrix_components(df_kept):
            if len(df_kept) == 0: return 0, 0, 0, 0
            y_true, y_pred = df_kept['label'], df_kept['y_pred']
            labels_present = np.unique(np.concatenate((y_true, y_pred)))
            if len(labels_present) > 1:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            elif len(labels_present) == 1:
                tp = sum((y_true == 1) & (y_pred == 1))
                tn = sum((y_true == 0) & (y_pred == 0))
                fp = sum((y_true == 0) & (y_pred == 1))
                fn = sum((y_true == 1) & (y_pred == 0))
            else:
                tp = tn = fp = fn = 0
            return tp, tn, fp, fn

        tp_std, tn_std, fp_std, fn_std = get_matrix_components(kept_std)
        tp_cc, tn_cc, fp_cc, fn_cc = get_matrix_components(kept_cc)
        
        results.append({
            'Rejection (%)': rate,
            'TP_Standard': tp_std, 'TN_Standard': tn_std, 'FP_Standard': fp_std, 'FN_Standard': fn_std,
            'TP_ClassCond': tp_cc, 'TN_ClassCond': tn_cc, 'FP_ClassCond': fp_cc, 'FN_ClassCond': fn_cc
        })
        
    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # --- Plotting the 2x2 Dashboard ---
    # ---------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Evolution of Confusion Matrix Components During Rejection', fontsize=16, fontweight='bold', y=0.98)
    
    # Common styling kwargs to keep code clean
    kws_std = dict(marker='X', color='crimson', linestyle='--', linewidth=2, markersize=6, label='Standard Rejection')
    kws_cc = dict(marker='o', color='forestgreen', linestyle='-', linewidth=2.5, markersize=6, label='Class-Conditioned')

    # [0, 0] True Positives (TP)
    axes[0, 0].plot(results_df['Rejection (%)'], results_df['TP_Standard'], **kws_std)
    axes[0, 0].plot(results_df['Rejection (%)'], results_df['TP_ClassCond'], **kws_cc)
    axes[0, 0].set_title('True Positives (Correctly Identified Progressors)', fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel('Number of Patients', fontsize=11)
    
    # [0, 1] True Negatives (TN)
    axes[0, 1].plot(results_df['Rejection (%)'], results_df['TN_Standard'], **kws_std)
    axes[0, 1].plot(results_df['Rejection (%)'], results_df['TN_ClassCond'], **kws_cc)
    axes[0, 1].set_title('True Negatives (Correctly Identified Healthy)', fontsize=13, fontweight='bold')
    
    # [1, 0] False Positives (FP)
    axes[1, 0].plot(results_df['Rejection (%)'], results_df['FP_Standard'], **kws_std)
    axes[1, 0].plot(results_df['Rejection (%)'], results_df['FP_ClassCond'], **kws_cc)
    axes[1, 0].set_title('False Positives (Healthy misclassified as Progressors)', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Rejection Rate (%)', fontsize=11)
    axes[1, 0].set_ylabel('Number of Patients', fontsize=11)
    
    # [1, 1] False Negatives (FN)
    axes[1, 1].plot(results_df['Rejection (%)'], results_df['FN_Standard'], **kws_std)
    axes[1, 1].plot(results_df['Rejection (%)'], results_df['FN_ClassCond'], **kws_cc)
    axes[1, 1].set_title('False Negatives (Progressors misclassified as Healthy)', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Rejection Rate (%)', fontsize=11)

    # Formatting clean-up
    for ax in axes.flat:
        ax.legend(frameon=True, fontsize=10)
        # Ensure y-axis always starts at 0 for honest visual scaling
        bottom, top = ax.get_ylim()
        ax.set_ylim(0, top * 1.05) 

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    plt.show()

        # reverting the distance to tau back to its original form in case it was modified for sorting
    if uncertainty_col in ["Distance_to_Tau", "imbalance_aware_margin"]:
    # invert the distance to tau for sorting (we want to drop those closest to tau first)
        df[uncertainty_col] = -df[uncertainty_col]

    return results_df


def plot_side_by_side_distributions(df, threshold, rate=25, 
                                    prob_col='Final_Calibrated_Prob', 
                                    uncertainty_col='H_Total'):
    """
    Plots a 1x2 figure comparing the Kernel Density Estimates (KDE) of uncertainty 
    for Standard vs. Class-Conditioned rejection methods.
    """
    df = df.copy()
    
    # Standardize probability column names
    if prob_col not in df.columns and 'mu' in df.columns:
        prob_col = 'mu'
        
    # Generate predictions using the baseline prior
    if 'y_pred' not in df.columns:
        df['y_pred'] = (df[prob_col] >= threshold).astype(int)
    
    if uncertainty_col == "Distance_to_Tau":
        df['Distance_to_Tau'] = -df['Distance_to_Tau']


    # ---------------------------------------------------------
    # 1. Data Splitting Logic (Both Methods)
    # ---------------------------------------------------------
    # --- Standard Method ---
    n_drop_std = int(len(df) * (rate / 100.0))
    sorted_df_std = df.sort_values(by=uncertainty_col, ascending=False)
    discarded_std = sorted_df_std.iloc[:n_drop_std]
    kept_std = sorted_df_std.iloc[n_drop_std:]
    
    # --- Class-Conditioned Method ---
    pos_pool = df[df['y_pred'] == 1].copy()
    neg_pool = df[df['y_pred'] == 0].copy()
    
    n_drop_pos = int(len(pos_pool) * (rate / 100.0))
    n_drop_neg = int(len(neg_pool) * (rate / 100.0))
    
    sorted_pos = pos_pool.sort_values(by=uncertainty_col, ascending=False)
    discarded_pos = sorted_pos.iloc[:n_drop_pos] if n_drop_pos > 0 else pd.DataFrame()
    kept_pos = sorted_pos.iloc[n_drop_pos:] if n_drop_pos > 0 else pos_pool
    
    sorted_neg = neg_pool.sort_values(by=uncertainty_col, ascending=False)
    discarded_neg = sorted_neg.iloc[:n_drop_neg] if n_drop_neg > 0 else pd.DataFrame()
    kept_neg = sorted_neg.iloc[n_drop_neg:] if n_drop_neg > 0 else neg_pool
    
    discarded_cc = pd.concat([discarded_pos, discarded_neg])
    kept_cc = pd.concat([kept_pos, kept_neg])

    # ---------------------------------------------------------
    # 2. Plotting the 1x2 Figure
    # ---------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define colors
    color_kept = '#2b8cbe'
    color_discarded = '#f03b20'
    
    # Helper function to plot a single axis
    def plot_kde(ax, df_kept, df_discarded, title):
        sns.kdeplot(data=df_kept, x=uncertainty_col, fill=True, color=color_kept, 
                    alpha=0.5, linewidth=2, label=f'Kept Cohort (N={len(df_kept)})', ax=ax)
        sns.kdeplot(data=df_discarded, x=uncertainty_col, fill=True, color=color_discarded, 
                    alpha=0.6, linewidth=2, label=f'Discarded Cohort (N={len(df_discarded)})', ax=ax)
        
        mean_kept = df_kept[uncertainty_col].mean()
        mean_discarded = df_discarded[uncertainty_col].mean()
        
        ax.axvline(mean_kept, color=color_kept, linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(mean_discarded, color=color_discarded, linestyle='--', linewidth=2, alpha=0.8)
        
        # Dynamic text positioning
        y_max = ax.get_ylim()[1]
        ax.text(mean_kept, y_max * 0.9, f' Mean:\n {mean_kept:.2f}', 
                 color=color_kept, fontweight='bold', ha='right', fontsize=11)
        ax.text(mean_discarded, y_max * 0.85, f' Mean:\n {mean_discarded:.2f}', 
                 color=color_discarded, fontweight='bold', ha='left', fontsize=11)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel(f'{uncertainty_col} Score', fontsize=12)
        ax.set_ylabel('Density of Patients', fontsize=12)
        ax.legend(frameon=True, fontsize=11, loc='upper right')
    
    # Plot Standard on Left
    plot_kde(axes[0], kept_std, discarded_std, f'Standard Method ({rate}% Rejection)')
    
    # Plot Class-Conditioned on Right
    plot_kde(axes[1], kept_cc, discarded_cc, f'Class-Conditioned Method ({rate}% Rejection)')

    sns.despine()
    plt.tight_layout()
    plt.show()

    if uncertainty_col == "Distance_to_Tau":
        df['Distance_to_Tau'] = -df['Distance_to_Tau']
