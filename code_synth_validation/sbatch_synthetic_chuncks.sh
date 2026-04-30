#!/bin/bash
#SBATCH --job-name=synthetic_chunked
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4         # ONLY 5 jobs submitted!
#SBATCH --time=3-00:00:00     # Time needed for ~12 tasks running sequentially
#SBATCH --mem=15G       
#SBATCH --output=/storage/DSH/projects/neuroart/uncertainty_ml/.sbatch_logs/%x_%A_%a_stdOut.txt
#SBATCH --error=/storage/DSH/projects/neuroart/uncertainty_ml/.sbatch_logs/%x_%A_%a_stdErr.txt

# --- Container Configuration ---
#SBATCH --container-image="/storage/DSH/projects/neuroart/uncertainty_ml/.container_files/ml_environment.sqsh"
#SBATCH --container-mounts="/storage/DSH/projects/neuroart/"

COMMANDS_FILE="/storage/DSH/projects/neuroart/uncertainty_ml/data/synthetic_data/synthetic_commands.txt"

cd /storage/DSH/projects/neuroart/uncertainty_ml/code_classification
export PYTHONPATH="/storage/DSH/projects/neuroart/uncertainty_ml:$PYTHONPATH"

# 1. Calculate how many commands each of the 5 jobs should handle
TOTAL_COMMANDS=$(wc -l < "$COMMANDS_FILE")
JOBS_IN_ARRAY=5
# Math to round up (ceiling) so no commands are left behind
CHUNK_SIZE=$(( (TOTAL_COMMANDS + JOBS_IN_ARRAY - 1) / JOBS_IN_ARRAY )) 

# 2. Determine the start and end lines for this specific array task
START_LINE=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE + 1 ))
END_LINE=$(( START_LINE + CHUNK_SIZE - 1 ))

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Array Task ID : $SLURM_ARRAY_TASK_ID"
echo "Processing lines $START_LINE to $END_LINE from commands file"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 3. Extract just those lines and run them sequentially
sed -n "${START_LINE},${END_LINE}p" "$COMMANDS_FILE" | while IFS= read -r CURRENT_COMMAND; do
    # Skip empty lines (in case the last chunk overshoots the file length)
    if [ -n "$CURRENT_COMMAND" ]; then
        echo "[Task $SLURM_ARRAY_TASK_ID] Running: $CURRENT_COMMAND"
        eval "$CURRENT_COMMAND"
    fi
done