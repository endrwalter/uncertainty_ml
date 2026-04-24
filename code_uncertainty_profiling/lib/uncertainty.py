""" Uncertainty Computation Functions """
from math import tau

import numpy as np
import pandas as pd

def H_tau_imbalance_aware(p, tau, alpha=None, eps=1e-10):
    """
    Imbalance-aware entropy with bounded [0,1] output
    
    Parameters:
    - p: predicted probability  
    - tau: decision threshold (typically = prior)
    - alpha: asymmetry parameter (default: imbalance ratio)
    """
    if alpha is None:
        # Default: use imbalance ratio
        alpha = (1 - tau) / tau
    
    # Standard symmetric H_tau
    if p < tau:
        p_rescaled = 0.5 * (p / tau)
    else:
        p_rescaled = 0.5 + 0.5 * ((p - tau) / (1 - tau))
    
    p_rescaled = np.clip(p_rescaled, eps, 1-eps)
    H_base = -p_rescaled * np.log2(p_rescaled + eps) - (1-p_rescaled) * np.log2(1-p_rescaled + eps)
    
    # Asymmetric weighting
    distance = np.abs(p - tau)
    
    if p >= tau:
        # Minority side: weight increases with distance
        # Linear model: weight = 1 + alpha * (distance / (1-tau))
        # This gives weight=1 at tau, weight=1+alpha at p=1
        weight = 1 + alpha * (distance / (1 - tau))
    else:
        # Majority side: constant weight
        weight = 1.0
    
    # Weighted entropy
    H_weighted = H_base * weight
    
    # Normalize to [0, 1]
    # Max possible value is H_base=1.0 * weight_max=(1+alpha)
    H_normalized = H_weighted / (1 + alpha)
    
    return H_normalized


def imbalance_aware_margin(p, tau, prior, eps=0.01):
    """
    Asymmetric uncertainty that accounts for differential sampling
    
    Parameters:
    - p: predicted probability
    - tau: decision threshold (typically equals prior)
    - prior: baseline disease prevalence
    """
    margin = np.abs(p - tau)
    
    # Compute imbalance ratio
    imbalance_ratio = (1 - prior) / prior
    
    # Weight minority-class predictions by imbalance
    if p >= tau:
        # Positive side (minority): amplify uncertainty
        weight = imbalance_ratio
    else:
        # Negative side (majority): standard weight
        weight = 1.0
    
    return weight / (margin + eps)


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

def compute_uncertainties(probas_df, global_stats, tau=None):
    """
    Computes H (Total), C (Aleatoric), and I (Epistemic) uncertainty.
    
    - probas_df: DataFrame containing model probabilities and optionally 'label'.
    - global_stats: DataFrame containing global statistics for each patient.
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
    mean_probs = np.nanmean(probs, axis=1)
    uncertainty_metrics = {}

    # 1. Standard Shannon entropy (baseline)
    H_total = -mean_probs * np.log2(mean_probs + eps) - (1-mean_probs) * np.log2(1-mean_probs + eps)
    uncertainty_metrics['H_total'] = H_total

    # 2. Pure margin (distance from boundary)
    margin = np.abs(mean_probs - tau)
    uncertainty_metrics['Margin'] = margin  # Note: LOW margin = HIGH uncertainty

    # 3. Inverse margin
    inverse_margin = 1 / (margin + 0.01)
    uncertainty_metrics['Inverse_Margin'] = inverse_margin

    # 4. Ensemble variance (raw epistemic proxy)
    ensemble_var = np.nanvar(probs, axis=1)
    uncertainty_metrics['Ensemble_Variance'] = ensemble_var

    # 5. Standard deviation (sqrt of variance)
    ensemble_std = np.nanstd(probs, axis=1)
    uncertainty_metrics['Ensemble_Std'] = ensemble_std

    # 6. Tau-relative entropy (symmetric)
    H_tau_symmetric = np.array([tau_relative_entropy(p, tau) for p in mean_probs])
    uncertainty_metrics['H_tau_symmetric'] = H_tau_symmetric

    # 7. Imbalance-aware H_tau
    H_tau_imbalanced = np.array([H_tau_imbalance_aware(p, tau) for p in mean_probs])
    uncertainty_metrics['H_tau_imbalanced'] = H_tau_imbalanced

    # 9. Ensemble variance weighted by inverse margin
    var_margin_weighted = ensemble_var / (margin + 0.01)
    uncertainty_metrics['Var_over_margin'] = var_margin_weighted

    # 10. Distance to tau (absolute)
    distance_tau = np.abs(mean_probs - tau)
    uncertainty_metrics['Distance_to_tau'] = distance_tau  # LOW = uncertain

    return uncertainty_metrics, tau

import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_uncertainty_distributions(uncertainty_results, metrics_to_plot=None, H_bl=None, save_path=None):
    """
    Plots overlapping distributions of selected uncertainty metrics on a single figure.
    
    Parameters:
    - uncertainty_results: DataFrame containing the uncertainty metrics.
    - metrics_to_plot: List of strings (column names) to plot. 
                       Defaults to ['H_Total', 'C_Aleatoric', 'I_Epistemic'].
    - H_bl: Float, the computed baseline entropy (optional).
    - save_path: Optional string path to save the figure (e.g., 'results/dists.png').
    """
    # Default metrics if none are provided
    if metrics_to_plot is None:
        metrics_to_plot = ['H_Total', 'C_Aleatoric', 'I_Epistemic']
        
    # Filter out metrics that don't actually exist in the DataFrame to prevent errors
    valid_metrics = [m for m in metrics_to_plot if m in uncertainty_results.columns]
    
    if not valid_metrics:
        print("None of the requested metrics were found in the DataFrame.")
        return

    # Set up the figure
    plt.figure(figsize=(10, 6))
    
    # Define a custom color palette so distributions are easily distinguishable
    colors = sns.color_palette("Set1", n_colors=len(valid_metrics))
    
    # Plot each valid metric
    for i, metric in enumerate(valid_metrics):
        sns.histplot(
            data=uncertainty_results, 
            x=metric,
            bins=50, 
            kde=True, 
            element="step", # 'step' is better for overlaps than standard filled bars
            fill=True,
            alpha=0.3,      # High transparency so we can see overlapping areas
            color=colors[i],
            label=metric
        )

    # ---------------------------------------------------------
    # Add important vertical lines (if applicable)
    # ---------------------------------------------------------
    if H_bl is not None:
        plt.axvline(H_bl, color='red', linestyle='--', linewidth=2, 
                    label=f'Baseline Entropy (H_bl={H_bl:.3f})')
        
    plt.axvline(0, color='green', linestyle='--', linewidth=2, 
                label='Ideal / Perfect Agreement (0)')

    # Formatting the plot
    plt.title('Overlapping Uncertainty Distributions', fontsize=16)
    plt.xlabel('Uncertainty Value (Bits / Nats)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    
    # Move the legend outside the plot so it doesn't cover the data
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Adjust layout to accommodate the external legend
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure successfully saved to: {save_path}")

    # Display the plot
    plt.show()