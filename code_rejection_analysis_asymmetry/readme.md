# Clinical Uncertainty & Rejection Analysis Pipeline

This repository contains the pipeline for uncertainty estimation and selective prediction (rejection analysis) in clinical machine learning. The framework is designed to evaluate heterogeneous ensembles predicting future disease onset or progression from baseline clinical data. 

Standard global rejection frameworks often fail in clinical datasets because predicting the onset of a future disease is inherently noisier than diagnosing baseline stability. This pipeline introduces **Class-Conditioned Rejection Curves (CCRC)** to decouple rejection thresholds from asymmetric aleatoric uncertainty, allowing models to safely say "I don't know" without crashing sensitivity for rare disease classes.

## Supported Datasets & Tasks
This pipeline has been validated across three distinct neurological prognostic tasks:
1. **Parkinson's Disease (PD):** Future dyskinesia prediction within 3 years.
2. **Multiple Sclerosis (MS):** Progression Independent of Relapse Activity (PIRA).
3. **Alzheimer's Disease (ADNI):** Mild Cognitive Impairment (MCI) to Alzheimer's conversion (24m/36m horizon).

## Key Features
* **Uncertainty Decomposition:** Breaks down ensemble variance into Aleatoric ($C_{Aleatoric}$) and Epistemic ($I_{Epistemic}$) uncertainty, alongside Total Uncertainty ($H_{Total}$).
* **Phase-Plane Analysis:** Visualizes the relationship between data noise (Aleatoric) and model ignorance (Epistemic).
* **Global vs. Class-Conditioned Rejection:** Compares standard rejection (which disproportionately deletes positive cases) against CCRC, which preserves minority class sensitivity.
* **Comprehensive Dashboards:** Generates confusion matrix evolution plots, clinical distribution comparisons for kept vs. rejected cohorts, and dynamic metric tracking.

## Pipeline Overview

1. **Ensemble Aggregation (`build_ensemble_csv`)**: Loads multi-fold predictions from a heterogeneous ensemble.
2. **Uncertainty Computation (`compute_uncertainties`)**: Expands raw probability arrays and calculates entropy-based metrics.
3. **Baseline Risk Calculation**: Computes the dataset-specific baseline entropy ($H_{bl}$) to contextualize model confidence.
4. **Clinical Clustering (Optional)**: Attempts GMM/PCA clustering to identify latent phenotypes (can be bypassed if clinical separation is poor).
5. **Class-Conditioned Rejection (CCRC)**: Sweeps rejection rates (0% to 80%) conditionally based on the optimal prior threshold. (comparison with global rejection, note that rejection analysis can be performed with any uncertainty metric)
6. **Rejection Extraction**: Isolates the "kept" vs. "removed" cohorts at a specific clinical operating point (e.g., 25% rejection) for side-by-side biomarker analysis.

## Uncertainty Metrics
- **$C_{Aleatoric}$**: Captures inherent data noise, reflecting the irreducible uncertainty in the clinical features. (do not trust this metric alone for rejection)
- **$I_{Epistemic}$**: Quantifies model ignorance, indicating how much the ensemble disagrees on a given prediction. (do not trust this metric alone for rejection)
- **$H_{Total}$**: The combined uncertainty, representing the overall confidence of the model's prediction.
- **$H_{bl}$**: The baseline entropy of the dataset, serving as a reference point for interpreting uncertainty values.
- **$MW_{H_{Total}}$**: Margin-Weighted Total Uncertainty, which adjusts $H_{Total}$ by the distance to the decision threshold, providing a more clinically relevant uncertainty measure for rejection decisions. (decision threshold could be the prior probability of the positive class, suboptimal)

## Usage
Modify the `MAIN SETTINGS` block in the script to target your specific dataset and outcome:

```python
dataset = 'ms_neuro'  # 'pd', 'ms_neuro', or 'adni'
analysis_name = 'ms_progression'
outcome_name = 'progression_independent_from_relapses'
```







## CHANGES TO CIBB 2026
modified plot considerning the new threshold for MCC computation