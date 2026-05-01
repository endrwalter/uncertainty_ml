""" Functions """



from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, matthews_corrcoef, silhouette_score
from sklearn.mixture import GaussianMixture
from scipy.stats import mannwhitneyu
import pandas as pd
from matplotlib.lines import Line2D

def optimize_pca_and_gmm_kaiser(X_clinical, max_clusters=6):
    """
    Finds the optimal number of PCs using the Kaiser Rule (Eigenvalue > 1) 
    and the optimal number of GMM clusters based on those PCs.
    """
    # ==========================================
    # STEP 0: Preprocessing
    # ==========================================
    imputer = SimpleImputer(strategy='median')
    # Scaling is mandatory for the Kaiser rule so total variance = number of features
    X_scaled = StandardScaler().fit_transform(imputer.fit_transform(X_clinical))
    
    # ==========================================
    # STEP 1: Optimize PCA (Kaiser Rule)
    # ==========================================
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Calculate Eigenvalues (Explained variance for standard-scaled data)
    eigenvalues = pca_full.explained_variance_

    # The Kaiser Rule: Keep PCs with eigenvalue > 1
    n_components_opt = np.sum(eigenvalues > 1.0)

    print(f"Number of original clinical features: {X_scaled.shape[1]}")
    print(f"Optimal PCs according to the Kaiser Rule: {n_components_opt}")
    print("\nTop 5 Eigenvalues:")
    print(np.round(eigenvalues[:5], 2))
    
    # ==========================================
    # STEP 2: Optimize GMM (using Kaiser-optimal PCs)
    # ==========================================
    pca_opt = PCA(n_components=n_components_opt, random_state=42)
    X_pca = pca_opt.fit_transform(X_scaled)
    
    bics = []
    silhouettes = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        labels = gmm.fit_predict(X_pca)
        
        bics.append(gmm.bic(X_pca))
        silhouettes.append(silhouette_score(X_pca, labels))
        
    # ==========================================
    # PLOTTING THE JUSTIFICATION DASHBOARD
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.set_facecolor('white')
    
    # Plot 1: True Scree Plot (Eigenvalues)
    axes[0].plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', lw=2, color='#3498DB')
    axes[0].axhline(y=1.0, color='gray', linestyle='--', label='Kaiser Threshold (Eigenvalue = 1.0)')
    axes[0].axvline(x=n_components_opt, color='#E74C3C', linestyle='--', label=f'Optimal PCs: {n_components_opt}')
    axes[0].set_title('Step 1: PCA Scree Plot (Eigenvalues)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Eigenvalue (Variance)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: GMM BIC Score
    axes[1].plot(k_range, bics, marker='s', lw=2, color='#E67E22')
    axes[1].set_title(f'Step 2: GMM BIC Score\n(Using {n_components_opt} PCs)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('BIC Score (LOWER is better)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: GMM Silhouette Score
    axes[2].plot(k_range, silhouettes, marker='^', lw=2, color='#2ECC71')
    axes[2].set_title(f'Step 3: GMM Silhouette Score\n(Using {n_components_opt} PCs)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Number of Clusters (k)')
    axes[2].set_ylabel('Silhouette Score (HIGHER is better)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n--- Optimization Results ---")
    print(f"1. Optimal PCs retained (Kaiser Rule, Eigenvalue > 1.0): {n_components_opt}")
    print(f"2. Best GMM model according to BIC (lowest complexity penalty): k = {k_range[np.argmin(bics)]}")
    print(f"3. Best GMM model according to Silhouette (highest separation): k = {k_range[np.argmax(silhouettes)]}")


def profile_clinical_clusters(X_clinical, analysis_df):
    """
    Compares the clinical features between the two discovered phenotypes.
    """
    # 1. Ensure X_clinical and the cluster labels are aligned
    df_compare = X_clinical.copy()
    df_compare['Phenotype'] = analysis_df['Phenotype'].values
    
    # 2. Initialize a list to store our results
    results = []
    
    # 3. Iterate through each clinical feature to compare the two groups
    features = [col for col in df_compare.columns if col != 'Phenotype']
    
    for feature in features:
        # Get data for each phenotype
        group_A = df_compare[df_compare['Phenotype'] == 'Phenotype A (Predictable - advanced)'][feature].dropna()
        group_B = df_compare[df_compare['Phenotype'] == 'Phenotype B (Invisible - mild)'][feature].dropna()
        
        # Skip if a group is empty
        if len(group_A) == 0 or len(group_B) == 0:
            continue
            
        # Calculate Means
        mean_A = group_A.mean()
        mean_B = group_B.mean()
        
        # Statistical Test (Mann-Whitney U is robust for non-normal clinical data)
        stat, p_val = mannwhitneyu(group_A, group_B, alternative='two-sided')
        
        results.append({
            'Clinical_Feature': feature,
            'Mean_Phenotype_A': round(mean_A, 2),
            'Mean_Phenotype_B': round(mean_B, 2),
            'P_Value': p_val
        })
        
    # 4. Create a DataFrame and sort by the most significant differences
    profile_df = pd.DataFrame(results)
    profile_df.sort_values('P_Value', inplace=True)
    
    # Format P-value for readability
    profile_df['P_Value'] = profile_df['P_Value'].apply(lambda x: '<0.001' if x < 0.001 else round(x, 4))
    
    print("--- Clinical Signature of the Phenotypes ---")
    print(profile_df.head(10).to_string(index=False)) # Show top 10 most different features
    
    return profile_df

# --- Configuration ---
PALETTE = {0.0: '#3498DB', 1.0: '#E74C3C'} # Bright Blue / Red
RATES_TO_SHOW = [0, 25, 50] # The rejection "snapshots" for scatter plots
MCC_RANGE = np.arange(0, 61, 5) # The full range for the metrics curve at the bottom
BASE_FONT_SIZE = 11
plt.rcParams.update({'font.size': BASE_FONT_SIZE, 'axes.grid': False})

def calculate_metrics_series(df, uncertainty_col, rates):
    """Helper to calculate MCC, Sensitivity, and Specificity curve data."""
    mcc_scores = []
    sensitivities = []
    specificities = []
    
    threshold_prior = df['label'].mean() # Prior probability of positive class (for sanity check)
    # Create the binary prediction column if it doesn't exist
    if 'y_pred' not in df.columns:
        df['y_pred'] = (df['mu'] >= threshold_prior).astype(int)
        
    for rate in rates:
        n_reject = int(len(df) * (rate / 100.0))
        if n_reject > 0:
            kept = df.sort_values(by=uncertainty_col, ascending=False).iloc[n_reject:]
        else:
            kept = df
        
        y_true = kept['label']
        y_pred = kept['y_pred']
        
        # Calculate MCC
        if len(y_true.unique()) > 1:
            mcc = matthews_corrcoef(y_true, y_pred)
        else:
            mcc = 0.0
            
        # Calculate Sensitivity & Specificity
        if len(y_true) > 0:
            # Safely calculate confusion matrix even if one class is entirely rejected
            labels_present = np.unique(np.concatenate((y_true, y_pred)))
            if len(labels_present) > 1:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            else:
                tp = sum((y_true == 1) & (y_pred == 1))
                tn = sum((y_true == 0) & (y_pred == 0))
                fp = sum((y_true == 0) & (y_pred == 1))
                fn = sum((y_true == 1) & (y_pred == 0))
                
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            sens, spec = 0.0, 0.0
            
        mcc_scores.append(mcc)
        sensitivities.append(sens)
        specificities.append(spec)
        
    return rates, mcc_scores, sensitivities, specificities

def plot_enhanced_scatter(ax, df, rate, max_uncertainty, uncertainty_col, threshold_prior=0.5):
    """Plots scatter with crucial fix: RED DOTS ON TOP. Dynamically maps Y-axis."""
    # Split data by class
    df_healthy = df[df['label'] == 0.0]
    df_incident = df[df['label'] == 1.0]
    
    # 1. Plot Healthy (Background, Transparent)
    ax.scatter(
        df_healthy['mu'], df_healthy[uncertainty_col], c=PALETTE[0.0], 
        alpha=0.3, s=40, edgecolor='none', zorder=1
    )
    
    # 2. Plot Incident (Foreground, Opaque, Larger edges)
    ax.scatter(
        df_incident['mu'], df_incident[uncertainty_col], c=PALETTE[1.0], 
        alpha=1.0, s=55, edgecolor='white', linewidth=0.8, zorder=2
    )
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.01, max_uncertainty)
    ax.axvline(x=threshold_prior, color='grey', linestyle='--', linewidth=1, zorder=0)
    ax.text(0.05, max_uncertainty*0.85, f'Rate: {rate}%\nN={len(df)}', fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_metrics_context(ax, rates, mcc_scores, sensitivities, specificities):
    """Plots the MCC, Sensitivity, and Specificity curves with markers."""
    # Plot the three lines with distinct styles
    ax.plot(rates, mcc_scores, color='#555555', linewidth=3, marker='o', markersize=6, label='MCC')
    ax.plot(rates, sensitivities, color='#E74C3C', linewidth=2, linestyle='--', marker='s', markersize=5, label='Sensitivity (TP Rate)')
    ax.plot(rates, specificities, color='#3498DB', linewidth=2, linestyle='-.', marker='^', markersize=5, label='Specificity (TN Rate)')
    
    ax.set_ylim(-0.05, 1.05) # Standardize scale to fit Sensitivity/Specificity
    ax.set_xlim(0, 60)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_ylabel('Metric Score')
    ax.set_xlabel('Rejection Rate (%)')
    
    # Add vertical lines linking to the scatter plots above
    for r in RATES_TO_SHOW:
        ax.axvline(x=r, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=1)

def generate_forensic_dashboard(df, uncertainty_col='sigma'):
    phenotypes = ['Phenotype A (Predictable - advanced)', 'Phenotype B (Invisible - mild)']
    
    # Dictionary to format Y-axis labels dynamically
    axis_labels = {
        'sigma': 'Std Dev (σ)',
        'C_Aleatoric': 'Aleatoric Uncertainty (C)',
        'H_Total': 'Total Uncertainty (H)',
        'I_Epistemic': 'Epistemic Uncertainty (I)'
    }
    y_axis_label = axis_labels.get(uncertainty_col, uncertainty_col)
    
    fig = plt.figure(figsize=(14, 12))
    fig.set_facecolor('white')
    outer_grid = GridSpec(1, 2, figure=fig, wspace=0.25, width_ratios=[1, 1])
    
    max_uncertainty = df[uncertainty_col].max() * 1.1
    
    # --- Main Loop through Phenotypes (Columns) ---
    for i, pheno in enumerate(phenotypes):
        inner_grid = GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_grid[i], 
                                             height_ratios=[1, 1, 1, 1.5], hspace=0.4)
        pheno_df = df[df['Phenotype'] == pheno].copy()
        
        # Title for the whole column
        col_title = f"{pheno}\n({'moderate-to-advanced patients' if i==0 else 'early-stage patients'})"
        fig.text(0.28 + i*0.44, 0.92, col_title, ha='center', fontsize=16, fontweight='bold')

        # --- Rows 1-3: Scatter Plot Evolution ---
        for j, rate in enumerate(RATES_TO_SHOW):
            ax = plt.Subplot(fig, inner_grid[j])
            n_drop = int(len(pheno_df) * (rate / 100))
            if n_drop > 0:
                subset = pheno_df.sort_values(by=uncertainty_col, ascending=False).iloc[n_drop:]
            else:
                subset = pheno_df
                
            # Pass the uncertainty_col to the plotting function
            plot_enhanced_scatter(ax, subset, rate, max_uncertainty, uncertainty_col, threshold_prior=df['label'].mean())
            
            # Dynamically set the Y-axis label
            if i == 0: ax.set_ylabel(y_axis_label)
            if j == 2: ax.set_xlabel('Mean Probability (μ)')
            fig.add_subplot(ax)
            
        # --- Row 4: Metrics Curve Context ---
        ax_metrics = plt.Subplot(fig, inner_grid[3])
        rates, mcc_scores, sens_scores, spec_scores = calculate_metrics_series(pheno_df, uncertainty_col, MCC_RANGE)
        
        plot_metrics_context(ax_metrics, rates, mcc_scores, sens_scores, spec_scores) 
        if i == 1: ax_metrics.set_ylabel('')
        fig.add_subplot(ax_metrics)

    # --- Custom Legend ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Healthy (y=0, Background)', 
               markerfacecolor=PALETTE[0.0], markersize=12, alpha=0.6),
        Line2D([0], [0], marker='o', color='w', label='Progressor (y=1, Foreground)', 
               markerfacecolor=PALETTE[1.0], markersize=12, markeredgecolor='k'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label='Rejection Snapshots shown above')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
               frameon=False, ncol=3, fontsize=12)
    
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.show()

def explore_confusion_matrix_evolution(df, uncertainty_col='H_Total', rates=[0, 25, 50]):
    """
    Calculates the exact TP, TN, FP, FN, and MCC at specific rejection rates 
    to prove why the model collapses on Phenotype B but survives on Phenotype A.
    """
    results = []
    # Make sure we have a binary prediction column
    if 'y_pred' not in df.columns:
        df['y_pred'] = (df['mu'] >= 0.5).astype(int)
        
    phenotypes = sorted(df['Phenotype'].unique())
    
    for pheno in phenotypes:
        pheno_df = df[df['Phenotype'] == pheno].copy()
        
        for rate in rates:
            # Calculate rejection threshold
            n_drop = int(len(pheno_df) * (rate / 100.0))
            if n_drop > 0:
                kept = pheno_df.sort_values(by=uncertainty_col, ascending=False).iloc[n_drop:]
            else:
                kept = pheno_df
                
            y_true = kept['label']
            y_pred = kept['y_pred']
            
            # Safely calculate confusion matrix even if classes are missing
            labels_present = np.unique(np.concatenate((y_true, y_pred)))
            if len(labels_present) > 1:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                mcc = matthews_corrcoef(y_true, y_pred)
            else:
                # If only one class is predicted/present, MCC is 0 and we manually count
                tp = sum((y_true == 1) & (y_pred == 1))
                tn = sum((y_true == 0) & (y_pred == 0))
                fp = sum((y_true == 0) & (y_pred == 1))
                fn = sum((y_true == 1) & (y_pred == 0))
                mcc = 0.0
                
            results.append({
                'Phenotype': pheno[:11] + '...', # Shorten for clean display
                'Rejection (%)': rate,
                'Total Kept': len(kept),
                'True Positives (TP)': tp,
                'False Positives (FP)': fp,
                'True Negatives (TN)': tn,
                'False Negatives (FN)': fn,
                'MCC': round(mcc, 3)
            })
            
    # Format as a clean DataFrame
    results_df = pd.DataFrame(results)
    
    # Print formatted output for the presentation
    print("="*80)
    print("CONFUSION MATRIX EVOLUTION BY PHENOTYPE")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    return results_df