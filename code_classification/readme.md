## MAIN CHANGES FROM CIBB SHORTPAPER:

- SHAP Explanation Fix: Added a dummy base_values array and enforced numpy array formatting in shap.Explanation to stop the beeswarm and summary plots from crashing.XGBoost/SHAP 

- Bug Bypass: Moved 'xgbclassifier' to your KernelExplainer list to gracefully bypass the known TreeExplainer parsing bug caused by the latest XGBoost update.

- Slurm Job Array: Built an sbatch array script to seamlessly launch multiple Python configurations in parallel using your production .sqsh container.

- Target Data Sanitization: Added automated type-checking to load_data to instantly convert any True/False booleans or strings into strict 0 and 1 integers so XGBoost doesn't crash.

- AUPRC for Optimization: Switched the refit metric in your Grid Search and the scoring in Permutation Importance to average_precision (AUPRC). This forces the algorithms to optimize for ranking the true biological signal, avoiding the 0.5 threshold trap.

- Post-Hoc Decision Thresholding: Instead of using the default 0.5 cutoff, the pipeline now calibrates the probabilities  and automatically uses the training set's prior prevalence as the medically accurate decision threshold for the test set.

- Threshold Storage: Added code to append and save the prior-based threshold for every fold into optimal_thresholds.csv. You will need this file to correctly calculate MCC when you evaluate the Classification-Rejection Curves (CRCs) in the next phase of your research.



------------------------------------------------------
## Methodological Update: Optimization and Threshold Selection
1. Decoupling Signal Ranking from Decision Making
In highly imbalanced clinical datasets, standard hyperparameter tuning using hard-classification metrics (such as the Matthews Correlation Coefficient) often fails due to the default 0.5 probability threshold. To ensure the models learned the true biological signal of Levodopa-Induced Dyskinesia (LID) onset, the hyperparameter optimization (refit metric) and Permutation Importance scoring were strictly evaluated using the Area Under the Precision-Recall Curve (AUPRC). This threshold-independent metric forces the ensemble to optimize for ranking true positive patients over majority-class negative patients, isolating the discriminative capability of the baseline clinical features.

2. Probability Calibration and the 0.5 Threshold Trap
To accurately decompose predictive uncertainty into Aleatoric and Epistemic components, the models must output mathematically pure, real-world probabilities. Consequently, Platt Scaling was applied to the optimal base models. However, calibrating probabilities to a heavily imbalanced dataset naturally shifts the average predictive output toward the target's baseline prevalence (e.g., ~11.4%). Under these calibrated conditions, applying a standard 0.5 decision threshold results in an artificial collapse of sensitivity, as true progressors rarely reach a 50% absolute probability despite having a risk profile exponentially higher than the population average.

3. Clinically Anchored Post-Hoc Thresholding
To bridge the gap between calibrated probabilities and binary clinical decisions, a dynamic, post-hoc thresholding strategy was implemented. Rather than executing a potentially overfitted empirical search for the maximum MCC, the decision boundary was dynamically set to the positive class prior (prevalence) of the training fold. This approach provides a robust, clinically interpretable anchor: any patient whose calibrated risk exceeds the baseline population prevalence is classified as a positive trajectory.

4. Integration with Uncertainty-Informed Rejection
To conduct the uncertainty-informed rejection analysis using Classification-Rejection Curves (CRCs) , it is necessary to re-evaluate the MCC at increasing rejection thresholds. Because discarding the most uncertain patients fundamentally alters the distribution of the remaining cohort, the fold-specific prior thresholds were saved (optimal_thresholds.csv) during training. This ensures that the exact, data-driven decision boundaries are consistently applied when computing the confusion matrices and performance metrics across all stages of the rejection analysis.