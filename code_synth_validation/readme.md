Step 1 - Generate the data 

    python3 1_synthetic_generator.py
    # creates synthetic_data/ with 56 condition folders + manifest.csv

Step 2 - Generate configs and commands file (once)

    python3 2_generate_synthetic_configs.py \
        --manifest  ../data/synthetic_data/manifest.csv \
        --template  config_synthetic_template.ini \ 
        --code_path ../code_classification \ 
        --script    main_calibrate.py \ 
        --output    ../data/synthetic_data/synthetic_commands.txt

Step 3 - Submit the array

    sbatch sbatch_synthetic_array.sh
    
Step 4 - Create the ensembles predictions
    python3 3_build_h_ensembles.py \
        --manifest     ../data/synthetic_data/manifest.csv \
        --results_root ../data/synthetic_results

Step 5  — uncertainty scores (~1 min)
python 4_compute_uncertainty_scores.py \
    --input  ../data/synthetic_results/all_conditions_summary.csv \
    --output ../data/synthetic_results/all_conditions_uncertainty.csv


Step 6 — rejection metrics (~10-20 min for 56 conditions)
python 5_compute_rejection_metrics.py \
    --input   ../data/synthetic_results/all_conditions_uncertainty.csv \
    --out_dir ../data/synthetic_results


Step 7 — all figures
python 6_visualize_phase_diagram.py \
    --scalars ../data/synthetic_results/scalar_summaries.csv \
    --curves  ../data/synthetic_results/rejection_curves.csv \
    --out_dir ../figures/synthetic