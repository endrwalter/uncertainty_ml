""" Uncertainty Computation Functions """
import numpy as np
import pandas as pd


def compute_uncertainties(probas_df):
    """
    Computes H (Total), C (Aleatoric), and I (Epistemic) uncertainty.
    """
    eps = 1e-12
    
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
    
    return pd.DataFrame({
        'H_Total': H, 
        'C_Aleatoric': C, 
        'I_Epistemic': I, 
        'Final_Calibrated_Prob': mean_scaled_probs
    }, index=probas_df.index)