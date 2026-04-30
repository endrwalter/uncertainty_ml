"""
generate_synthetic_configs.py
=============================================================
Reads synthetic_data/manifest.csv and generates:
  - One config.ini per condition under synthetic_data/<condition>/config.ini
  - A commands file (synthetic_commands.txt) listing one command per line,
    ready to be indexed by SLURM_ARRAY_TASK_ID in the sbatch script

Usage
-----
    python generate_synthetic_configs.py \
        --manifest  ../data/synthetic_data/manifest.csv \
        --template  config_synthetic_template.ini \
        --code_path /storage/DSH/projects/neuroart/uncertainty_ml/code_classification \
        --script    main_calibrate.py \
        --output    ../data/synthetic_data/synthetic_commands.txt

All paths should be relative to the sbatch working directory, or absolute.
"""

import argparse
import os
import pandas as pd


def generate_configs(
    manifest_path: str,
    template_path: str,
    code_path: str,
    script_name: str,
    commands_output_path: str,
):
    manifest = pd.read_csv(manifest_path)

    with open(template_path, "r") as f:
        template = f.read()

    commands = []

    for _, row in manifest.iterrows():
        # X.csv lives here; pipeline finds y.csv automatically
        input_path  = row["X_path"]

        # Results go into a parallel results tree mirroring the data tree
        # e.g.  synthetic_data/tau_0.10_d_0.5_n_300/  →
        #        synthetic_results/tau_0.10_d_0.5_n_300/
        data_dir    = os.path.dirname(row["X_path"])
        output_path = data_dir.replace("synthetic_data", "synthetic_results") + "/"

        os.makedirs(output_path, exist_ok=True)

        # Fill template
        config_content = template.format(
            input_path=input_path,
            output_path=output_path,
        )

        # Write config.ini next to X.csv
        config_path = os.path.join(data_dir, "config.ini")
        with open(config_path, "w") as f:
            f.write(config_content)

        # Build the command that sbatch will run for this condition
        cmd = f"python3 {script_name} --config {config_path}"
        commands.append(cmd)

    # Write commands file — one line per condition
    os.makedirs(os.path.dirname(commands_output_path) or ".", exist_ok=True)
    with open(commands_output_path, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")

    print(f"Generated {len(commands)} configs.")
    print(f"Commands written to: {commands_output_path}")
    print(f"\nSet your sbatch --array=0-{len(commands)-1}")
    print(f"Total conditions: {len(commands)}")

    # Print a summary table of conditions for quick inspection
    print("\n── Condition summary ──")
    print(
        manifest[["tau", "d", "n", "n_minority", "n_majority"]]
        .to_string(index=False)
    )

    return commands


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",  required=True, help="Path to manifest.csv")
    parser.add_argument("--template",  required=True, help="Path to config_synthetic_template.ini")
    parser.add_argument("--code_path", required=True, help="Path to classification code directory")
    parser.add_argument("--script",    default="main_calibrate.py", help="Python script name")
    parser.add_argument("--output",    required=True, help="Path to write synthetic_commands.txt")
    args = parser.parse_args()

    generate_configs(
        manifest_path=args.manifest,
        template_path=args.template,
        code_path=args.code_path,
        script_name=args.script,
        commands_output_path=args.output,
    )