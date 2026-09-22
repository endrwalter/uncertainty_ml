**Important Note:** Before executing these steps, please ensure you check and update all file and directory paths in the commands below to match your local repository structure.

### Step 1 - Generate the data 
```bash
python3 1_synthetic_generator.py
```
*Creates `synthetic_data/` with condition folders and a `manifest.csv`.*

### Step 2 - Generate configs and commands file (once)
```bash
python3 2_generate_synthetic_configs.py \
    --manifest  ../data/synthetic_data/manifest.csv \
    --template  config_synthetic_template.ini \
    --code_path ../code_classification \
    --script    main_calibrate.py \
    --output    ../data/synthetic_data/synthetic_commands.txt
```

### Step 3 - Submit the array
```bash
sbatch sbatch_synthetic_.sh
```

### Step 4 - Create the ensemble predictions
```bash
python3 3_build_h_ensembles.py \
    --manifest     ../data/synthetic_data/manifest.csv \
    --results_root ../data/synthetic_results
```

### Step 5 - Compute uncertainty scores
```bash
python3 4_compute_uncertainty_scores.py \
    --input  ../data/synthetic_results/all_conditions_summary.csv \
    --output ../data/synthetic_results/all_conditions_uncertainty.csv
```

### Step 6 - Compute rejection metrics
```bash
python3 5_compute_rejection_metrics.py \
    --input   ../data/synthetic_results/all_conditions_uncertainty.csv \
    --out_dir ../data/synthetic_results
```

### Step 7 - Generate Main Paper Figures
Generates the core synthetic manuscript figures (Sensitivity Stability, Failure Map, Separability Interaction, and Appendix Rejection Curves):
```bash
python3 7_paper_figures.py \
    --scalars   ../data/synthetic_results/scalar_summaries.csv \
    --curves    ../data/synthetic_results/rejection_curves.csv \
    --out_dir   ../figures/paper/
```

### Step 8 - Structural Divergence Validation (Asymmetry Test)
Generates the synthetic asymmetry test validation figure evaluating uncertainty gaps across prevalence thresholds:
```bash
python3 8_structural_divergence_validation.py \
    --in_dir  ../data/synthetic_results/ \
    --out_dir ../figures/paper/
```

### Step 9 - Sensitivity Analysis & Density Ablation
Generates supplementary figures demonstrating structural vulnerabilities under varying cost matrices and distribution overlaps:
```bash
python3 1r_sensitivity_analysis.py
```