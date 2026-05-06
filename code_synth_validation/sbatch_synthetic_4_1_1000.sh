#!/bin/bash
#SBATCH --job-name=synth_4
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --time=4-00:00:00     # IMPORTANT: Give it plenty of time (e.g., 4 days)
#SBATCH --mem=60G            # High memory to prevent OOM kills

#SBATCH --output=/storage/DSH/projects/neuroart/uncertainty_ml/.sbatch_logs/%x_%j_stdOut.txt
#SBATCH --error=/storage/DSH/projects/neuroart/uncertainty_ml/.sbatch_logs/%x_%j_stdErr.txt

# --- Container Configuration ---
#SBATCH --container-image="/storage/DSH/projects/neuroart/uncertainty_ml/.container_files/ml_environment.sqsh"
#SBATCH --container-mounts="/storage/DSH/projects/neuroart/"

# --------------------------------------------
cd /storage/DSH/projects/neuroart/uncertainty_ml/code_classification/
python3 main_calibrate.py --config ../data/synthetic_data/tau_0.40_d_1.0_n_1000/config.ini &
python3 main_calibrate.py --config ../data/synthetic_data/tau_0.40_d_2.0_n_300/config.ini &
wait
