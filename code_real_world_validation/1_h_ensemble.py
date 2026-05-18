"""
Usage:
    python 1_h_ensemble.py ../results/classification/pd_dyskinesia/FutureDyskinesia
    python 1_h_ensemble.py ../results/classification/ms_progression/progression_independent_from_relapses
    python 1_h_ensemble.py ../results/classification/mci_ad_conversion/label_bl_36m
To see all available options:
    python 1_h_ensemble.py -h
"""

import os
import pandas as pd
import argparse

def build_ensemble_csv(base_path, index_col='idx', models_to_include=None):
    if models_to_include is None:
        models_to_include = ['randomforestclassifier', 'extratreesclassifier', 'xgbclassifier', 
                             'logisticregression', 'svc']
                             
    # Clean up the path and extract analysis/outcome names to name the output files
    base_path = os.path.normpath(base_path)
    outcome_name = os.path.basename(base_path)
    analysis_name = os.path.basename(os.path.dirname(base_path))

    all_data_list = []

    if not os.path.exists(base_path):
        print(f"Directory {base_path} not found. Please check the path.")
        return

    for root, dirs, files in os.walk(base_path):
        # Searching for 'raw_results_calibration.csv' as specified
        if 'raw_results_calibration.csv' in files:
            path_parts = root.split(os.sep)
            model_name = path_parts[-1]
            folder_outcome = path_parts[-2] 

            if model_name not in models_to_include:
                print(f"Skipping {model_name} as it's not in the list of models to include.")
                continue

            file_path = os.path.join(root, 'raw_results_calibration.csv')
            df = pd.read_csv(file_path)
            
            # Attach metadata
            df['model'] = model_name
            df['outcome'] = folder_outcome
            
            all_data_list.append(df)

    if not all_data_list:
        print("No result files found. Verify the file name (e.g., 'raw_results_calibration.csv').")
        return

    # Ensure output directory exists before saving
    out_dir = f'{base_path}/aggregated'
    os.makedirs(out_dir, exist_ok=True)

    # Combine all models and iterations
    full_df = pd.concat(all_data_list, ignore_index=True)
    full_df.to_csv(f'{out_dir}/all_predictions_{analysis_name}_{outcome_name}_aggregated.csv', index=False)
    
    patient_stats = full_df.groupby(index_col).agg(
        mu=('probs_cal', 'mean'),
        sigma=('probs_cal', 'std'),
        label=('real y', 'first'),
        n_predictions=('probs_cal', 'count'), 
        all_probs=('probs_cal', list)
    ).reset_index()
    
    # Save summary for the next step
    patient_stats.to_csv(f'{out_dir}/patient_mean_probs_{outcome_name}_{analysis_name}_model.csv', index=False)
    print(f"Successfully aggregated {len(full_df)} predictions across {len(patient_stats)} unique patients.")   
    print(f"Saved patient-level summary to {out_dir}/patient_mean_probs_{outcome_name}_{analysis_name}_model.csv")


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Aggregate model predictions into an ensemble CSV.")
    
    # Required positional argument for the path
    parser.add_argument("path", help="Path to the directory containing model result folders")
    
    # Optional arguments
    parser.add_argument("-i", "--index_col", default="idx", help="Column name to use as the index (default: idx)")
    parser.add_argument("-m", "--models", nargs="+", 
                        default=['randomforestclassifier', 'extratreesclassifier', 'xgbclassifier', 'logisticregression', 'svc'],
                        help="List of models to include, separated by spaces")

    # Parse the arguments from the terminal
    args = parser.parse_args()

    # Run the function with the parsed arguments
    build_ensemble_csv(
        base_path=args.path,
        index_col=args.index_col,
        models_to_include=args.models
    )