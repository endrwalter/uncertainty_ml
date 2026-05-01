"""
build_ensemble_synthetic.py
=============================================================
Adapts build_ensemble_csv() to the synthetic experiment structure.

Key differences from the real-world version:
  - Iterates over manifest.csv conditions instead of analysis_name/outcome_name
  - Results live in synthetic_results/<condition>/ mirroring synthetic_data/<condition>/
  - Patient index is a simple integer (row position), not a clinical ID
  - Adds condition metadata (tau, d, n) to both aggregated and summary CSVs
  - Produces a cross-condition summary table for downstream phase diagram analysis

Output per condition:
  synthetic_results/tau_0.10_d_0.5_n_300/aggregated/
      all_predictions_aggregated.csv      (all models × all iterations)
      patient_mean_probs_summary.csv      (mu, sigma, label per patient)

Output global:
  synthetic_results/
      all_conditions_summary.csv          (one row per patient × condition,
                                           with tau/d/n columns attached)
"""

import os
import pandas as pd


MODELS_TO_INCLUDE = [
    'randomforestclassifier',
    'extratreesclassifier',
    'xgbclassifier',
    'logisticregression',
    'svc',
]


# ─────────────────────────────────────────────
# SINGLE CONDITION
# ─────────────────────────────────────────────

def build_ensemble_csv_synthetic(
    condition_results_path: str,
    tau: float,
    d: float,
    n: int,
    models_to_include: list = MODELS_TO_INCLUDE,
    index_col: str = 'idx',
) -> pd.DataFrame | None:
    """
    Build aggregated ensemble CSV for a single synthetic condition.

    Parameters
    ----------
    condition_results_path : str
        Path to the results folder for this condition, e.g.
        '../data/synthetic_results/tau_0.10_d_0.5_n_300'
    tau, d, n : condition parameters, attached as columns for downstream use
    models_to_include : list of model names to aggregate
    index_col : patient index column name in raw_results_calibration.csv

    Returns
    -------
    patient_stats : pd.DataFrame with condition metadata attached, or None if
                    no results found (condition may not have finished yet)
    """
    if not os.path.exists(condition_results_path):
        print(f"  [SKIP] Results folder not found: {condition_results_path}")
        return None

    all_data_list = []

    for root, dirs, files in os.walk(condition_results_path):
        if 'raw_results_calibration.csv' not in files:
            continue

        path_parts = root.split(os.sep)
        model_name = path_parts[-1]

        if model_name not in models_to_include:
            continue

        df = pd.read_csv(os.path.join(root, 'raw_results_calibration.csv'))
        df['model'] = model_name

        # Attach condition metadata at row level for traceability
        df['tau'] = tau
        df['d']   = d
        df['n']   = n

        all_data_list.append(df)

    if not all_data_list:
        print(f"  [WARN] No raw_results_calibration.csv found in {condition_results_path}")
        return None

    out_dir = os.path.join(condition_results_path, 'aggregated')
    os.makedirs(out_dir, exist_ok=True)

    # ── All predictions (all models × all iterations) ──
    full_df = pd.concat(all_data_list, ignore_index=True)
    full_df.to_csv(
        os.path.join(out_dir, 'all_predictions_aggregated.csv'),
        index=False
    )

    # ── Per-patient summary (mu, sigma, label) ──
    patient_stats = (
        full_df
        .groupby(index_col)
        .agg(
            mu            = ('probs_cal', 'mean'),
            sigma         = ('probs_cal', 'std'),
            label         = ('real y',    'first'),
            n_predictions = ('probs_cal', 'count'),
            all_probs     = ('probs_cal', list),
        )
        .reset_index()
    )

    # Attach condition metadata so rows are self-describing
    patient_stats['tau'] = tau
    patient_stats['d']   = d
    patient_stats['n']   = n

    patient_stats.to_csv(
        os.path.join(out_dir, 'patient_mean_probs_summary.csv'),
        index=False
    )

    print(
        f"  [OK]  tau={tau:.2f}  d={d:.1f}  n={n:>4}  |  "
        f"{len(full_df):>6} predictions  |  "
        f"{len(patient_stats):>4} patients"
    )

    return patient_stats


# ─────────────────────────────────────────────
# ALL CONDITIONS (manifest-driven)
# ─────────────────────────────────────────────

def build_all_synthetic_ensembles(
    manifest_path: str,
    synthetic_results_root: str,
    models_to_include: list = MODELS_TO_INCLUDE,
    index_col: str = 'idx',
) -> pd.DataFrame:
    """
    Iterate over all conditions in the manifest and aggregate results.

    Parameters
    ----------
    manifest_path : str
        Path to synthetic_data/manifest.csv
    synthetic_results_root : str
        Root folder where condition results live, e.g.
        '../data/synthetic_results'
        Expected subfolder structure:
        <synthetic_results_root>/tau_0.10_d_0.5_n_300/<model>/raw_results_calibration.csv
    models_to_include : list
    index_col : str

    Returns
    -------
    all_conditions_df : pd.DataFrame
        One row per patient × condition with tau/d/n attached.
        Saved to <synthetic_results_root>/all_conditions_summary.csv
    """
    manifest = pd.read_csv(manifest_path)
    all_stats = []

    print(f"Processing {len(manifest)} conditions from manifest...\n")

    for _, row in manifest.iterrows():
        tau = row['tau']
        d   = row['d']
        n   = int(row['n'])

        # Mirror the data folder naming convention
        condition_name    = f"tau_{tau:.2f}_d_{d:.1f}_n_{n}"
        condition_results = os.path.join(synthetic_results_root, condition_name)

        stats = build_ensemble_csv_synthetic(
            condition_results_path=condition_results,
            tau=tau,
            d=d,
            n=n,
            models_to_include=models_to_include,
            index_col=index_col,
        )

        if stats is not None:
            all_stats.append(stats)

    if not all_stats:
        print("\nNo conditions processed. Check that jobs have completed.")
        return pd.DataFrame()

    # ── Cross-condition summary ──
    all_conditions_df = pd.concat(all_stats, ignore_index=True)

    out_path = os.path.join(synthetic_results_root, 'all_conditions_summary.csv')
    all_conditions_df.to_csv(out_path, index=False)

    completed   = len(all_stats)
    total       = len(manifest)
    missing     = total - completed

    print(f"\n{'─'*55}")
    print(f"Completed : {completed}/{total} conditions")
    if missing:
        print(f"Missing   : {missing} (jobs may still be running)")
    print(f"Summary   : {out_path}")
    print(f"Rows      : {len(all_conditions_df):,} patient-condition pairs")
    print(f"{'─'*55}")

    return all_conditions_df


# ─────────────────────────────────────────────
# PARTIAL RUN (conditions completed so far)
# ─────────────────────────────────────────────

def check_completion_status(
    manifest_path: str,
    synthetic_results_root: str,
) -> pd.DataFrame:
    """
    Quick status check — shows which conditions have finished
    without loading any data. Useful while the array job is running.

    Returns a DataFrame with columns [tau, d, n, status, n_models_found]
    """
    manifest = pd.read_csv(manifest_path)
    records  = []

    for _, row in manifest.iterrows():
        tau  = row['tau']
        d    = row['d']
        n    = int(row['n'])
        name = f"tau_{tau:.2f}_d_{d:.1f}_n_{n}"
        path = os.path.join(synthetic_results_root, name)

        # Count how many model subfolders contain raw_results_calibration.csv
        n_found = 0
        if os.path.exists(path):
            for root, _, files in os.walk(path):
                if 'raw_results_calibration.csv' in files:
                    n_found += 1

        status = (
            'complete' if n_found == len(MODELS_TO_INCLUDE)
            else 'partial' if n_found > 0
            else 'missing'
        )

        records.append({
            'tau':            tau,
            'd':              d,
            'n':              n,
            'status':         status,
            'n_models_found': n_found,
        })

    status_df = pd.DataFrame(records)

    # Summary counts
    counts = status_df['status'].value_counts()
    print("\n── Completion status ──")
    print(status_df.to_string(index=False))
    print(f"\nComplete : {counts.get('complete', 0)}")
    print(f"Partial  : {counts.get('partial',  0)}")
    print(f"Missing  : {counts.get('missing',  0)}")

    return status_df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--manifest',
        default='../data/synthetic_data/manifest.csv',
        help='Path to manifest.csv'
    )
    parser.add_argument(
        '--results_root',
        default='../data/synthetic_results',
        help='Root folder for synthetic results'
    )

    args = parser.parse_args()

    status = check_completion_status(args.manifest, args.results_root)
    n_complete = (status['status'] == 'complete').sum()

    if n_complete == 0:
        print("\nNo complete conditions found. Exiting.")
    else:
        print(f"\nAggregating {n_complete} complete conditions...\n")
        build_all_synthetic_ensembles(
            manifest_path=args.manifest,
            synthetic_results_root=args.results_root,
        )