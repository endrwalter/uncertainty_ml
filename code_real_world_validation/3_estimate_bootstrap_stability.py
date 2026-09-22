import argparse
import os
import pandas as pd
import numpy as np

'''
python 3_estimate_bootstrap_stability.py \
    --ms_dist ../results/statistical_tests/ms_distributions.csv \
    --pd_dist ../results/statistical_tests/pd_distributions.csv \
    --ad_dist ../results/statistical_tests/ad_distributions.csv \
    --out_csv ../results/statistical_tests/bootstrap_stability_table.csv
'''


def check_bootstrap_stability(iter_df: pd.DataFrame, disease: str, method: str = 'h_tau_ccr', metric: str = 'ccaugrc_progressor', step: int = 100):
    """Computes cumulative CIs to evaluate stability and returns the records."""
    
    # Extract the ordered bootstrap results for the chosen method
    if method not in iter_df['method'].values:
        # Fallback to the first available method if 'h_tau_ccr' isn't in the file
        method = iter_df['method'].iloc[0]
        
    method_data = iter_df[iter_df['method'] == method].sort_values('bootstrap_iter')
    vals = method_data[metric].dropna().values
    
    records = []
    prev_low, prev_high = None, None
    
    for n in range(step, len(vals) + 1, step):
        subset = vals[:n]
        if len(subset) < 2: continue
        
        ci_low, ci_high = np.percentile(subset, [2.5, 97.5])
        
        d_low = f"{abs(ci_low - prev_low):.5f}" if prev_low is not None else "-"
        d_high = f"{abs(ci_high - prev_high):.5f}" if prev_high is not None else "-"
        
        records.append({
            'Cohort': disease,
            'Metric': metric,
            'Method': method,
            'Iterations': n,
            'Lower CI': f"{ci_low:.4f}",
            'Upper CI': f"{ci_high:.4f}",
            'Δ Lower': d_low,
            'Δ Upper': d_high
        })
        
        prev_low, prev_high = ci_low, ci_high
        
    return records

def run_disease_from_file(disease: str, dist_path: str):
    if not dist_path or not os.path.exists(dist_path): 
        print(f"Skipping {disease}: File not found ({dist_path})")
        return []
        
    print(f"Processing {disease} from: {dist_path}")
    iter_df = pd.read_csv(dist_path)
    
    # You can change the metric here if needed
    return check_bootstrap_stability(iter_df, disease, metric='ccaugrc_progressor', step=100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Point these to the existing *_distributions.csv files
    parser.add_argument('--ms_dist', default=None)
    parser.add_argument('--pd_dist', default=None)
    parser.add_argument('--ad_dist', default=None)
    # Output file
    parser.add_argument('--out_csv', required=True, help="Path to save the summary CSV")
    args = parser.parse_args()

    all_records = []
    for disease in ['MS', 'PD', 'AD']:
        path = getattr(args, f'{disease.lower()}_dist')
        records = run_disease_from_file(disease, path)
        all_records.extend(records)
        
    if all_records:
        df_out = pd.DataFrame(all_records)
        
        # Make sure output directory exists
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        
        df_out.to_csv(args.out_csv, index=False)
        print(f"\nStability results saved to: {args.out_csv}")
        print("You can now open this file in Excel and copy the table directly into Word.")
    else:
        print("No data processed. Please check your file paths.")