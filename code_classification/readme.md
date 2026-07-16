## `main_calibrate.py`: Calibrated Ensemble & Classification Pipeline

This script serves as the core execution engine for the machine learning classification pipeline. It orchestrates data loading, hyperparameter tuning, model training, probability calibration, and interpretability (SHAP/Permutation Importance) across multiple nested cross-validation splits.

### Overview of Features

*   **Robust Splitting:** Supports both standard stratified train-test splits and `GroupShuffleSplit` (e.g., ensuring multiple visits from the same patient stay in the same fold).
*   **Automated Tuning:** Implements grid search for hyperparameter tuning using predefined parameter distributions, with a fallback for tuning-free models like TabPFN.
*   **Probability Calibration:** Optionally calibrates model probabilities post-training, recalculating the optimal classification threshold (based on MCC) for improved reliability in clinical decision-making.
*   **Interpretability:** Computes Permutation Feature Importance, standard SHAP values, and Calibrated SHAP values to explain both raw and calibrated model predictions.
*   **Comprehensive Logging:** Automatically aggregates and saves raw prediction outputs, calibration metrics, performance distributions, and visualization plots.

### Usage

Run the script from the command line by passing a valid configuration file (YAML, depending on your `load_config` implementation):

```bash
python main_calibrate.py --config path/to/config.json
```

### Execution Workflow

1.  **Initialization:** Loads settings from the provided `--config` file and sets fixed random seeds for reproducibility.
2.  **Data Ingestion:** Loads the dataset, feature types, and stratification targets. Applies optional grouping and drops random patients if specified.
3.  **Classifier Iteration:** Loops through each algorithm specified in `config['classifiers_list']`.
4.  **Nested Cross-Validation:** For each predefined random seed (`rs_list`):
    *   Splits data into training and test sets.
    *   Constructs a modeling pipeline (imputation, feature selection, scaling, classification).
    *   Tunes hyperparameters via Grid Search on the training set.
    *   Evaluates the raw, uncalibrated model on the test set.
    *   *(Optional)* Fits a calibration layer on the training data, determines an optimal threshold, and re-evaluates the test set using calibrated probabilities.
    *   Calculates SHAP values and permutation importance.
5.  **Aggregation & Export:** Averages metrics across all folds and writes results to disk.

### Generated Outputs

For every classifier run, the script automatically generates a dedicated subdirectory containing the following artifacts:

**Metrics & Data:**
*   `calibration_metrics_raw.csv` / `calibration_metrics_calibrated.csv`: Brier scores and expected calibration errors (ECE) before and after calibration.
*   `optimal_thresholds.csv`: The threshold selected to maximize the MCC per fold.
*   **Raw Results:** CSVs containing ground truth vs. predicted probabilities for every patient across all folds (useful for computing AUGRC/rejection curves downstream).
*   **Performance Metrics:** Aggregated test metrics (MCC, F1, AUROC, AUPRC, Sensitivity, Specificity) and training metrics from the Grid Search.

**Visualizations & Interpretability:**
*   **ROC Curves:** Mean ROC curve plots across all folds.
*   **Calibration Plots:** Reliability diagrams for both raw and calibrated probabilities.
*   **Feature Importance:** Output matrices and plots for SHAP values (raw and calibrated) and Permutation Feature Importance.