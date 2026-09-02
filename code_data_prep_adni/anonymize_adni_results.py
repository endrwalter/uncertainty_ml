import pandas as pd
import numpy as np
import random
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

INPUT_FILE = "../results/classification/mci_ad_conversion/label_36m/aggregated/all_predictions_mci_ad_conversion_label_36m_aggregated.csv"        
OUTPUT_PUBLIC = "../results/classification/mci_ad_conversion/label_36m/aggregated/all_predictions_mci_ad_conversion_label_36m_aggregated_public.csv"  
SECRET_MAPPING = "../results/classification/mci_ad_conversion/label_36m/aggregated/adni_mapping_key.csv"      

def anonymize_data():
    print(f"Caricamento dati da {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    
    unique_subjects = df['subject_id'].unique().tolist()
    
    # 2. random IDs
    random.seed(42) # reproducible mapping on this machine
    random.shuffle(unique_subjects)
    
    # 3. dict subject_id -> dummy_id
    mapping_dict = {
        subj: f"patient_{i+1:04d}" for i, subj in enumerate(unique_subjects)
    }
    
    # 4. store key mapping
    mapping_df = pd.DataFrame(list(mapping_dict.items()), columns=['original_subject_id', 'dummy_id'])
    mapping_df.to_csv(SECRET_MAPPING, index=False)
    print(f"{SECRET_MAPPING}")
    
    # 5. apply map
    df['dummy_id'] = df['subject_id'].map(mapping_dict)
    
    cols = ['dummy_id'] + [c for c in df.columns if c not in ['subject_id', 'dummy_id']]
    df_public = df[cols]
    
    df_public.to_csv(OUTPUT_PUBLIC, index=False, float_format='%.8f')
    print(f"File anonimizzato PRONTO per GitHub salvato in: {OUTPUT_PUBLIC}")

if __name__ == "__main__":
    anonymize_data()