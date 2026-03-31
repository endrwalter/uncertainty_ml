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
5. **Class-Conditioned Rejection (CCRC)**: Sweeps rejection rates (0% to 80%) conditionally based on the optimal prior threshold.
6. **Rejection Extraction**: Isolates the "kept" vs. "removed" cohorts at a specific clinical operating point (e.g., 25% rejection) for side-by-side biomarker analysis.

## Usage
Modify the `MAIN SETTINGS` block in the script to target your specific dataset and outcome:

```python
dataset = 'ms_neuro'  # 'pd', 'ms_neuro', or 'adni'
analysis_name = 'ms_progression'
outcome_name = 'progression_independent_from_relapses'
```




# Results: Parkinson's Disease (3-Year Dyskinesia Prediction)

## 1. Context and Baseline Uncertainty Profile
The task of predicting future dyskinesia in Parkinson's Disease (PD) within a 3-year horizon presents a challenging prognostic scenario characterized by substantial Aleatoric Uncertainty (inherent data noise). Phase-plane analysis reveals that high predictive uncertainty is structurally concentrated within the progressor class (Class 1). This asymmetry arises because predicting the onset of future clinical complications is biologically and observationally noisier than predicting continued stability.

## 2. The Catastrophic Failure of Global Rejection
When a standard, global uncertainty rejection threshold is applied to the PD cohort, the model undergoes a systemic clinical failure. 
* **Metric Collapse:** Global rejection acts as a naive minority-class deletion filter. As the rejection rate increases, Sensitivity rapidly decays while Specificity artificially inflates.
* **Confusion Matrix Dynamics:** Tracking the raw confusion matrix components reveals the exact mechanism of failure. At a 25% global rejection rate, the standard method discards more than half of its True Positives (dropping from 50 to ~22) while discarding zero True Negatives. It achieves higher confidence not by improving discrimination, but by systematically refusing to predict the "progressor" trajectory for ambiguous patients.

## 3. The Class-Conditioned Rejection Curve (CCRC) Solution
To decouple the rejection threshold from the asymmetric noise profile, Class-Conditioned Rejection Curves (CCRC) were implemented. By evaluating and rejecting the most uncertain patients proportionally from *both* predicted classes, the model's clinical utility is rescued.
* **Metric Stabilization:** Under CCRC, both Sensitivity and Specificity are maintained. At a 25% rejection rate, the model retains a highly balanced performance profile (Sensitivity ~0.80, Specificity ~0.62).
* **Proportional Trimming:** Confusion matrix tracking proves that CCRC forces a balanced deferral. False Positives are successfully filtered (dropping from 58 to 42), while the vast majority of True Positives are safely preserved (39 out of 50 retained).

## 4. Distributional Proof of Mechanism (The 25% Threshold)
Analyzing the Total Uncertainty ($H_{Total}$) distributions of the kept versus discarded cohorts at the 25% rejection threshold provides clear visual proof of the CCRC mechanism:
* **Standard Method:** The discarded cohort forms a massive, single-peaked density at the extreme upper boundary of uncertainty ($H_{Total} > 0.95$). Because progressors are inherently more uncertain, this rigid truncation effectively massacres the disease class.
* **Class-Conditioned Method:** The discarded cohort forms a distinctly **bimodal** distribution. This mathematically proves that the algorithm successfully identified and rejected two separate populations: the most uncertain *predicted stable* patients (the lower-uncertainty peak) and the most uncertain *predicted progressors* (the higher-uncertainty peak). 

## 5. Conclusion for PD
For medium-term prognostic tasks in Parkinson's disease, predictive uncertainty is highly asymmetric. Standard global rejection frameworks are clinically unsafe, as they quietly erase the high-risk patient population. Implementing CCRC is strictly necessary to safely defer ambiguous clinical profiles while protecting the model's diagnostic sensitivity over the disease class.



# Results: Multiple Sclerosis (Progression Independent of Relapse Activity)

## 1. Context and Baseline Uncertainty Profile
Predicting Progression Independent of Relapse Activity (PIRA) in Multiple Sclerosis represents the most difficult task evaluated in this study. The dataset is characterized by extreme phenotypic heterogeneity and a severe class imbalance (disease prior ~0.11). Phase-plane analysis reveals that the inherent biological noise (Aleatoric Uncertainty) is overwhelmingly concentrated in the rare progressor class, as silent progression is notoriously difficult to distinguish from stable disease or normal aging.

## 2. The Catastrophic Failure of Global Rejection
Applying a standard global uncertainty rejection threshold to the MS cohort triggers a complete systemic collapse. 
* **Metric Collapse:** Because the model associates high uncertainty almost exclusively with the positive class, global rejection acts as a ruthless minority-class deletion filter. By a 35% rejection rate, Sensitivity plummets entirely to 0.0, rendering the model clinically useless.
* **Confusion Matrix Dynamics:** The evolution of the confusion matrix explicitly shows this failure mode. Under standard rejection, True Positives are aggressively purged from step one (dropping from 58 to 0 by 35% rejection), while True Negatives are largely ignored. The model artificially achieves "certainty" by defaulting to a blanket prediction of "Stable" for all patients.

## 3. The Class-Conditioned Rejection Curve (CCRC) Solution
In this high-noise, highly imbalanced environment, Class-Conditioned Rejection Curves (CCRC) transition from an optimization tool to a mandatory safety mechanism. 
* **The Minority Class Rescue:** By enforcing proportional rejection based on the baseline prior, CCRC prevents the artificial erasure of the disease class. While overall ratios like MCC remain mathematically flat (~0.29), this flatlining is not a failure—it is the signature of a stabilized model. 
* **Metric Improvement:** At a 25% rejection rate, CCRC actually *improves* Sensitivity (rising from 0.707 to 0.771) while maintaining Specificity (~0.71). 
* **Proportional Trimming:** The confusion matrix confirms that CCRC successfully isolates noisy predictions from both classes. At 25% rejection, it safely defers 38 False Positives while preserving the core of the rare disease class (retaining 37 out of 58 True Positives).

## 4. Distributional Proof of Mechanism (The 25% Threshold)
The Total Uncertainty ($H_{Total}$) distributions of the kept versus discarded cohorts at the 25% threshold provide stark visual evidence of how CCRC rescues the model:
* **Standard Method:** The discarded cohort is represented by a single, massive distribution peak heavily skewed to the extreme right (Mean: 0.74). This confirms that the standard algorithm is simply slicing off the highest-uncertainty tail, which corresponds almost entirely to the minority progressor class.
* **Class-Conditioned Method:** The discarded cohort forms a highly complex, **multimodal** distribution (Mean: 0.59). This mathematically proves that CCRC is selectively extracting distinct sub-populations of noisy predictions: the most ambiguous *predicted stable* patients (the lower-uncertainty peaks) and the most ambiguous *predicted progressors* (the rightmost peak), ensuring no single clinical trajectory is unfairly penalized.

## 5. Conclusion for MS
In highly imbalanced and noisy clinical tasks like MS progression, global uncertainty rejection fails entirely, equating "uncertainty" with the "minority class." CCRC successfully breaks this false equivalence. It allows the model to safely abstain from predicting highly elusive clinical endpoints while actively protecting its diagnostic power over the rare disease class.


# Results: Alzheimer's Disease (36-Month MCI to AD Conversion)

## 1. Context and Baseline Uncertainty Profile
Predicting MCI-to-AD conversion at a 36-month horizon using only baseline clinical data presents a scenario with a relatively balanced prior (Baseline Entropy $H_{bl} \approx 0.995$) but massive Aleatoric Uncertainty. Phase-plane analysis reveals that because projecting a 3-year biological trajectory from a single Day 0 visit is inherently ambiguous, a massive volume of patients from both classes cluster near the maximum uncertainty boundary ($H_{Total} > 0.90$). 

## 2. The Skew of Global Rejection
Because the AD dataset does not suffer from the extreme class imbalance seen in Multiple Sclerosis, standard global rejection does not trigger a catastrophic collapse. The overall Matthews Correlation Coefficient (MCC) climbs steadily under global truncation. However, it introduces a distinct clinical skew:
* **Metric Divergence:** As the global rejection rate increases, Sensitivity aggressively outpaces Specificity.
* **Confusion Matrix Dynamics:** The evolution of the confusion matrix reveals the exact nature of this bias. Under standard rejection, False Negatives (progressors missed by the model) remain entirely ignored and flatlined until a 35% rejection rate. Meanwhile, True Negatives are discarded much faster than True Positives. The standard model achieves higher confidence by disproportionately penalizing the stable class, artificially inflating Sensitivity at the expense of balanced diagnostic power.

## 3. The Class-Conditioned Rejection Curve (CCRC) Solution
In this scenario, Class-Conditioned Rejection Curves (CCRC) function not as a rescue mechanism, but as a strict calibration enforcer. By evaluating uncertainty independently within the predicted progressor and predicted stable classes, the model's diagnostic symmetry is restored.
* **Perfect Calibration:** Under CCRC, Sensitivity and Specificity converge and rise in near-perfect tandem. At a 25% rejection rate, the model achieves a highly balanced 0.808 Sensitivity and 0.804 Specificity. By 50% rejection, both metrics cross 0.90.
* **Proportional Error Reduction:** Confusion matrix tracking shows that CCRC immediately and linearly reduces False Negatives from step one, ensuring that high-risk ambiguous cases are safely deferred rather than falsely reassured.

## 4. Distributional Proof of Mechanism (The 25% Threshold)
The Total Uncertainty ($H_{Total}$) distributions of the kept versus discarded cohorts at the 25% threshold highlight the nuance of CCRC in a balanced dataset:
* **Standard Method:** The discarded cohort is represented by a single, narrow peak isolated at the extreme right boundary (Mean: 0.98). The algorithm rigidly lops off the highest absolute uncertainty, ignoring the relative difficulty of the predictions.
* **Class-Conditioned Method:** The discarded cohort shares the major peak at 0.98 but reveals a distinct, secondary density bump between 0.60 and 0.80. This mathematically proves that CCRC dips further into the distribution to extract the most uncertain patients *relative to their specific class*, ensuring that moderately ambiguous cases in the "easier" trajectory are appropriately flagged for review, rather than just blindly chopping the tail.

## 5. Conclusion for AD
For prognostic tasks with strong biological signals but high temporal decay (e.g., 3-year predictions from baseline), global rejection frameworks create unbalanced clinical tools that favor one trajectory over the other. Implementing CCRC enforces strict calibration, ensuring the AI remains equally trustworthy and symmetric for diagnosing both future progression and continued stability.






## CHANGES TO CIBB 2026
modified plot considerning the new threshold for MCC computation