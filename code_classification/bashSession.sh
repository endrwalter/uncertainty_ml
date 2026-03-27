#!/bin/bash
#SBATCH --job-name=ml_array_train
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --array=0-2
#SBATCH --output=/storage/DSH/projects/neuroart/uncertainty_ml/.sbatch_logs/%x_%A_%a_stdOut.txt
#SBATCH --error=/storage/DSH/projects/neuroart/uncertainty_ml/.sbatch_logs/%x_%A_%a_stdErr.txt

# --- Container Configuration ---
#SBATCH --container-image="/storage/DSH/projects/neuroart/uncertainty_ml/.container_files/ml_environment.sqsh"
#SBATCH --container-mounts="/storage/DSH/projects/neuroart/"

# 1. Navigate to the working directory
cd /storage/DSH/projects/neuroart/uncertainty_ml/code_classification

# 2. Ensure Python can find your 'lib' folder
export PYTHONPATH="/storage/DSH/projects/neuroart/uncertainty_ml:$PYTHONPATH"

# 3. Define the list of commands you want to run
# You can put completely different python scripts here, or the same script with different configs!
COMMANDS=(
    "python3 main_calibrate.py --config ../data/adni/config.ini"
    "python3 main_calibrate.py --config ../data/ms_neuro/config.ini"
    "python3 main_calibrate.py --config ../data/pd_neuro/config.ini"
)

# 4. Extract the specific command for this exact array task
CURRENT_COMMAND=${COMMANDS[$SLURM_ARRAY_TASK_ID]}

echo "Starting Array Task: $SLURM_ARRAY_TASK_ID"
echo "Running Command: $CURRENT_COMMAND"

# 5. Execute it
eval $CURRENT_COMMAND