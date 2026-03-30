import os
import pandas as pd

import os
import pandas as pd

def build_ensemble_csv(analysis_name, outcome_name, index_col='idx'):
    base_path = f'../results/classification/{analysis_name}/{outcome_name}'
    all_data_list = []

    if not os.path.exists(base_path):
        print(f"Directory {base_path} not found. Please check the path.")
        return

    for root, dirs, files in os.walk(base_path):
        # Searching for 'raw_results_calibration.csv' as specified
        if 'raw_results_calibration.csv' in files:
            path_parts = root.split(os.sep)
            model_name = path_parts[-1]
            folder_outcome = path_parts[-2] # Renamed to avoid overwriting the function argument
            
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
    
    # Calculate mean (mu) and std (sigma) per patient (idx)
    # FIX: Column updated to 'probs_cal' to match your actual data
    # check if index col is an array of two elements, if so, we need to handle the multi index case
    
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



#TODO def build ensemble (with filtering for specific models)