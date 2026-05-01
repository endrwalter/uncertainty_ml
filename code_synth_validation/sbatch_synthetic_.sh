#!/bin/bash
#SBATCH --job-name=synth_sequential
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4-00:00:00     # IMPORTANT: Give it plenty of time (e.g., 4 days)
#SBATCH --mem=60G            # High memory to prevent OOM kills

#SBATCH --output=/storage/DSH/projects/neuroart/uncertainty_ml/.sbatch_logs/%x_%j_stdOut.txt
#SBATCH --error=/storage/DSH/projects/neuroart/uncertainty_ml/.sbatch_logs/%x_%j_stdErr.txt

# --- Container Configuration ---
#SBATCH --container-image="/storage/DSH/projects/neuroart/uncertainty_ml/.container_files/ml_environment.sqsh"
#SBATCH --container-mounts="/storage/DSH/projects/neuroart/"

COMMANDS_FILE="/storage/DSH/projects/neuroart/uncertainty_ml/data/synthetic_data/synthetic_commands.txt"

cd /storage/DSH/projects/neuroart/uncertainty_ml/code_classification
export PYTHONPATH="/storage/DSH/projects/neuroart/uncertainty_ml:$PYTHONPATH"

# --- THE FIX for host environment leakage ---
export PATH="/opt/conda/bin:/usr/bin:/bin:$PATH"
hash -r
# --------------------------------------------

echo "Starting sequential execution of all commands..."

# Read the file line by line and execute
while IFS= read -r CURRENT_COMMAND; do
    # Skip empty lines
    if [ -n "$CURRENT_COMMAND" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Running: $CURRENT_COMMAND"
        eval "$CURRENT_COMMAND"
    fi
done < "$COMMANDS_FILE"

echo "All commands processed!"