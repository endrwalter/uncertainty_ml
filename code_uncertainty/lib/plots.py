from matplotlib import pyplot as plt
import seaborn as sns
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
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

    # Save the figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure successfully saved to: {save_path}")

    # Display the plot
    plt.show()





def plot_combined_uncertainty_analysis(uncertainty_df, y_true, save_path=None):
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

    # Plotting Data
    markers_map = {0: 'o', 1: '^'}
    color_tn, color_tp = '#b0b0b0', '#4d4d4d' 
    scatter_plot = None

    optimal_threshold = y_true.mean()
    div_norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=optimal_threshold, vmax=1.0)

    for true_class in [0, 1]:
        subset = uncertainty_df[uncertainty_df['label'] == true_class]
        if subset.empty: continue
        scatter = ax_main.scatter(
            subset['C_Aleatoric'], subset['I_Epistemic'], 
            c=subset['mu'], cmap='coolwarm', 
            vmin=0.0, vmax=1.0, alpha=0.8, edgecolors='k', linewidths=0.5,
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

    # Upper Right Legend for Marginal Histograms (Outside Main Plot)
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
    
    # threshold to classify as positive or negative
    threshold = y_true.mean()  # or set to 0.5 if you want a fixed threshold

    y_pred_class = (uncertainty_df['mu'] > threshold).astype(int)
    sorted_indices = uncertainty_df['H_Total'].sort_values(ascending=False).index
    rejection_rates = np.linspace(0, 0.80, 20)
    
    mcc_scores, sens_scores, spec_scores = [], [] ,[]
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
    # --- Final Formatting & Thickness Enforcement ---
    # ---------------------------------------------------------
    for ax in [ax_main, ax_b]:
        for spine in ax.spines.values():
            spine.set_linewidth(border_width)
        ax.tick_params(width=border_width, length=5)
        ax.grid(False)

    for ax_h in [ax_hist_x, ax_hist_y]:
        ax_h.axis('off')

    cbar = fig.colorbar(scatter_plot, cax=ax_cbar, orientation='horizontal')
    cbar.outline.set_linewidth(border_width)
    cbar.ax.tick_params(width=border_width, labelsize=base_fontsize)
    cbar.set_label('Consensus Predicted Risk', fontsize=base_fontsize, labelpad=15)

    # 1. Set evenly spaced, linear ticks so the distances make sense
    cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
    cbar.set_ticklabels(['0.0', '0.25', '0.50', '0.75', '1.0'])
    
    # 2. Draw a hard vertical line exactly at your clinical threshold
    cbar.ax.axvline(optimal_threshold, color='black', linewidth=2.5, linestyle='-')
    
    # 3. Add an annotation right above the line to label it
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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import matplotlib.gridspec as gridspec

def plot_combined_uncertainty_analysis_v2(uncertainty_df, y_true, save_path=None):
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
    
    sorted_indices = uncertainty_df['H_Total'].sort_values(ascending=False).index
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