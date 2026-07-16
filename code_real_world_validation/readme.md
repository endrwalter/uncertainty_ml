**Important Note:** Before executing these steps, please ensure you check and update all file and directory paths in the commands below to match your local repository structure.

### Step 1 - Generate Heterogeneous Ensembles
Creates the aggregated ensemble predictions for Parkinson's Disease (PD), Multiple Sclerosis (MS), and Alzheimer's Disease (AD).
```bash
python3 1_h_ensemble.py ../results/classification/pd_dyskinesia/FutureDyskinesia
python3 1_h_ensemble.py ../results/classification/ms_progression/progression_independent_from_relapses
python3 1_h_ensemble.py ../results/classification/mci_ad_conversion/label_bl_36m
```

### Step 2 - Compute Uncertainty Metrics
Computes total predictive uncertainty and class-conditioned rejection curves.
```bash
python3 2_compute_uncertainty_metrics.py \
    --ms_path  ../results/classification/ms_progression/progression_independent_from_relapses/aggregated/patient_mean_probs_progression_independent_from_relapses_ms_progression_model.csv \
    --pd_path  ../results/classification/pd_dyskinesia/FutureDyskinesia/aggregated/patient_mean_probs_FutureDyskinesia_pd_dyskinesia_model.csv \
    --ad_path  ../results/classification/mci_ad_conversion/label_bl_36m/aggregated/patient_mean_probs_label_bl_36m_mci_ad_conversion_model.csv \
    --out_dir  ../results/real_world_results/
```

### Step 3 - Compute Statistics
Computes bootstrap confidence intervals and statistical tests for ccAUGRC differences between methods. Bootstraps the full aggregated ensemble (deployment reality) and provides a 95% CI on the difference between methods.
```bash
python3 3_compute_stats.py \
    --ms_summary ../results/classification/ms_progression/progression_independent_from_relapses/aggregated/patient_mean_probs_progression_independent_from_relapses_ms_progression_model.csv \
    --pd_summary ../results/classification/pd_dyskinesia/FutureDyskinesia/aggregated/patient_mean_probs_FutureDyskinesia_pd_dyskinesia_model.csv \
    --ad_summary ../results/classification/mci_ad_conversion/label_bl_36m/aggregated/patient_mean_probs_label_bl_36m_mci_ad_conversion_model.csv \
    --out_dir ../results/statistical_tests/
```

### Step 4 - Plot Statistical Tests (AUGRC)
Generates a printed summary table for both Progressor and Stable ccAUGRC, alongside a grouped bar chart figure with 95% CI error bars and significance annotations for all three diseases.
```bash
python3 4_plot_statistical_results_augrc.py \
    --tests   ../results/statistical_tests/all_statistical_tests.csv \
    --distrib ../results/statistical_tests/ \
    --out_dir ../figures/paper/
```

### Step 5 - Plot Results
Generates the final manuscript figures:
* **Fig 6**: Real-world — Rejection Curves MS / PD / AD
* **Fig 7**: Real-world — Asymmetry Test
* **Fig 8**: Real-world — ccAUGRC Decomposition Table

```bash
python3 5_plot_results.py \
    --real_dir  ../results/real_world_results/ \
    --out_dir   ../figures/paper/
```