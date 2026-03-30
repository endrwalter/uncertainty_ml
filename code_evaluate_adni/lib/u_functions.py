import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_curve, roc_auc_score, auc

def binary_entropy(p):
    """Calculates Shannon entropy for binary probabilities."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return - (p * np.log2(p) + (1 - p) * np.log2(1 - p))

def calc_aleatoric(probs_list):
    """Calculates aleatoric uncertainty (mean of individual entropies)."""
    return np.mean([binary_entropy(p) for p in probs_list])

# --- Main Evaluation Logic ---
def run_tte_evaluation(ensemble_path, dataset_path, target_sensitivity=0.85):
    """
    Loads data, merges, calculates uncertainty, and computes bucketed metrics.
    Returns: (eval_df, df_results)
    """
    # Load Data
    ensemble_df = pd.read_csv(ensemble_path)
    dataset_df = pd.read_csv(dataset_path) 

    # Merge Data
    eval_df = ensemble_df.merge(
        dataset_df[['subject_id', 'month', 'months_to_conversion']], 
        on=['subject_id', 'month'], 
        how='inner'
    )

    # Define Buckets
    bins = [0, 6, 12, 18, 24, 30, 36]
    labels = ['0-6m', '6-12m', '12-18m', '18-24m', '24-30m', '30-36m']
    eval_df['tte_bucket'] = pd.cut(
        eval_df['months_to_conversion'], 
        bins=bins, labels=labels, include_lowest=True
    )

    # Calculate Row-Level Uncertainties (The eval_df part)
    eval_df['all_probs'] = eval_df['all_probs'].apply(ast.literal_eval)
    eval_df['total_uncertainty'] = eval_df['mu'].apply(binary_entropy)
    eval_df['aleatoric_uncertainty'] = eval_df['all_probs'].apply(calc_aleatoric)
    eval_df['epistemic_uncertainty'] = eval_df['total_uncertainty'] - eval_df['aleatoric_uncertainty']

    # Compute Summary Results (The df_results part)
    stable_df = eval_df[eval_df['label'] == 0]
    results = []

    for bucket in labels:
        bucket_mask = (eval_df['label'] == 1) & (eval_df['tte_bucket'] == bucket)
        bucket_df = eval_df[bucket_mask]
        
        if len(bucket_df) == 0: continue
            
        combined_df = pd.concat([stable_df, bucket_df])
        y_true, y_probs = combined_df['label'], combined_df['mu']
        
        # Metrics
        auc_score = roc_auc_score(y_true, y_probs)
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        prc_score = auc(recall, precision)
        
        # Dynamic Thresholding for Sensitivity
        threshold = np.percentile(bucket_df['mu'], 100 * (1 - target_sensitivity))
        y_pred = (y_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        results.append({
            'Time to Event': bucket,
            'Pos Cases': len(bucket_df),
            'AUC': round(auc_score, 3),
            'PRC': round(prc_score, 3),
            'MCC': round(matthews_corrcoef(y_true, y_pred), 3),
            'Req. Threshold': round(threshold, 3),
            'TP': tp, 'FN': fn, 'TN': tn, 'FP': fp,
            'Sensitivity': round(tp/(tp+fn), 3),
            'Specificity': round(tn/(tn+fp), 3),
            'Tot Unc': round(bucket_df['total_uncertainty'].mean(), 3),
        })

    return eval_df, pd.DataFrame(results)




# --- Dashboard Visualization ---
def plot_comprehensive_dashboard(df_results, eval_df):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('MCI-AD Conversion Performance Dashboard', fontsize=20, fontweight='bold')
    
    x = df_results['Time to Event']
    
    # Plot 1: Performance Decay
    axes[0,0].plot(x, df_results['AUC'], marker='o', label='AUC', linewidth=3)
    axes[0,0].plot(x, df_results['PRC'], marker='s', label='PRC', linewidth=3)
    axes[0,0].set_title('Predictive Power Decay', fontsize=14)
    axes[0,0].legend()

    # Plot 2: Clinical Metrics
    axes[0,1].plot(x, df_results['Sensitivity'], marker='o', color='green', label='Sensitivity')
    axes[0,1].plot(x, df_results['Specificity'], marker='s', color='red', label='Specificity')
    axes[0,1].set_title('Clinical Utility (Fixed Sens=0.85)', fontsize=14)
    axes[0,1].legend()

    # Plot 3: Threshold & Density
    ax3_twin = axes[1,0].twinx()
    ax3_twin.bar(x, df_results['Pos Cases'], color='gray', alpha=0.15, label='N Pos Cases')
    axes[1,0].plot(x, df_results['Req. Threshold'], marker='D', color='navy', label='Threshold')
    axes[1,0].set_title('Threshold Requirements & Sample Density', fontsize=14)
    axes[1,0].set_ylabel('Probability Threshold')
    ax3_twin.set_ylabel('Number of Cases')

    # Plot 4: Uncertainty Decomposition
    pos_unc = eval_df[eval_df['label'] == 1].groupby('tte_bucket', observed=True).mean(numeric_only=True)
    axes[1,1].plot(x, pos_unc['total_uncertainty'], color='black', lw=3, label='Total Uncertainty')
    axes[1,1].fill_between(x, 0, pos_unc['aleatoric_uncertainty'], color='orange', alpha=0.3, label='Aleatoric')
    axes[1,1].plot(x, pos_unc['epistemic_uncertainty'], '--', color='blue', label='Epistemic')
    axes[1,1].set_title('Uncertainty Decomposition (Positives)', fontsize=14)
    axes[1,1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def plot_training_strategy_comparison(df_aug, df_base):
    """
    Combines and plots the results from the Augmented and Baseline-only models.
    """
    # 1. Tag and Combine the DataFrames
    df_aug = df_aug.copy()
    df_base = df_base.copy()
    
    df_aug['Training Strategy'] = 'Augmented (All Visits)'
    df_base['Training Strategy'] = 'Baseline-Only'
    
    # Combine into a single dataframe for easy seaborn plotting
    combined_df = pd.concat([df_base, df_aug], ignore_index=True)
    
    # 2. Set up the plotting environment
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Comparison on Baseline Test Set: Augmented vs. Baseline-Only', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Define standard colors so the legend is consistent across plots
    palette = {'Baseline-Only': 'teal', 'Augmented (All Visits)': 'darkorange'}
    
    # --- Plot 1: AUC (Overall Discriminative Power) ---
    sns.lineplot(data=combined_df, x='Time to Event', y='AUC', hue='Training Strategy', 
                 marker='o', linewidth=3, markersize=8, ax=axes[0, 0], palette=palette)
    axes[0, 0].set_title('AUC Decay Over Time', fontsize=14)
    axes[0, 0].set_ylabel('AUC')
    
    # --- Plot 2: PRC (Precision-Recall Area) ---
    sns.lineplot(data=combined_df, x='Time to Event', y='PRC', hue='Training Strategy', 
                 marker='s', linewidth=3, markersize=8, ax=axes[0, 1], palette=palette)
    axes[0, 1].set_title('PRC Decay Over Time', fontsize=14)
    axes[0, 1].set_ylabel('PRC')
    
    # --- Plot 3: Specificity (Holding Sensitivity at ~0.85) ---
    sns.lineplot(data=combined_df, x='Time to Event', y='Specificity', hue='Training Strategy', 
                 marker='^', linewidth=3, markersize=8, ax=axes[1, 0], palette=palette)
    axes[1, 0].set_title('Specificity (At Fixed 85% Sensitivity)', fontsize=14)
    axes[1, 0].set_ylabel('Specificity')
    
    # --- Plot 4: False Positives (Clinical Cost) ---
    sns.barplot(data=combined_df, x='Time to Event', y='FP', hue='Training Strategy', 
                ax=axes[1, 1], palette=palette, alpha=0.85)
    axes[1, 1].set_title('Number of False Positives Triggered', fontsize=14)
    axes[1, 1].set_ylabel('False Positives Count')

    # 3. Clean up the formatting
    for ax in axes.flat:
        ax.set_xlabel('Months to Conversion', fontsize=12)
        ax.tick_params(axis='x', rotation=0)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()