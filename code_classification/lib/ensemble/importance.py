

import pathlib
from pydoc import pathdirs
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_values(classifier_name, model, X_test_transformed, colnames, rs, res_dir):
    """
    Computes SHAP values using the 'Raw Score' strategy.
    Updated to handle 3D array outputs from VotingClassifier/KernelExplainer.
    """
    
    # --- 1. PREPARE DATA ---
    if isinstance(X_test_transformed, pd.DataFrame):
        X_bg = X_test_transformed
        X_test_numpy = X_test_transformed.values
    else:
        X_bg = pd.DataFrame(X_test_transformed, columns=colnames)
        X_test_numpy = X_test_transformed

    # --- 2. SELECT EXPLAINER & TARGET ---
    
    # A. LINEAR MODELS
    if classifier_name == 'logisticregression':
        explainer = shap.LinearExplainer(model, X_bg)
        shap_values_obj = explainer(X_test_numpy)
        
        if len(shap_values_obj.values.shape) == 2:
            shap_values_class_1_array = shap_values_obj.values
        else:
            shap_values_class_1_array = shap_values_obj.values[:, :, 1]

    # B. KERNEL MODELS (SVC, Stacking, Voting)
    elif classifier_name in ['svc', 'voting', 'stacking', 'tabpfn', 'xgbclassifier']:
        
        # Target Function
        if hasattr(model, 'decision_function'):
            target_func = model.decision_function
        else:
            target_func = model.predict_proba

        # Optimization
        if classifier_name == 'tabpfn':
            background_summary = shap.kmeans(X_bg, 5)
        else:
            background_summary = shap.kmeans(X_bg, 30)
            
        explainer = shap.KernelExplainer(target_func, background_summary)
        shap_vals_result = explainer.shap_values(X_test_numpy)
        
        # --- ROBUST OUTPUT HANDLING (THE FIX) ---
        # 1. If it's a list (e.g., [Class0, Class1]), take Class 1
        if isinstance(shap_vals_result, list):
            if len(shap_vals_result) == 2:
                shap_values_class_1_array = shap_vals_result[1]
            else:
                shap_values_class_1_array = shap_vals_result[0] # Fallback
        
        # 2. If it's a Numpy Array, check dimensions
        else:
            # Case: (Samples, Features, Classes) -> (50, 26, 2)
            # This is exactly what caused your error.
            if len(shap_vals_result.shape) == 3:
                shap_values_class_1_array = shap_vals_result[:, :, 1]
            
            # Case: (Samples, Features) -> (50, 26)
            else:
                shap_values_class_1_array = shap_vals_result

    # C. TREE MODELS
    else:
             
        explainer = shap.TreeExplainer(model)
        shap_values_obj = explainer(X_test_numpy)
        
        if len(shap_values_obj.values.shape) == 3:
            shap_values_class_1_array = shap_values_obj.values[:, :, 1]
        else:
            shap_values_class_1_array = shap_values_obj.values

    # --- 3. FORMATTING OUTPUT ---
    # Now shap_values_class_1_array is guaranteed to be 2D
    df_vals = pd.DataFrame(shap_values_class_1_array, columns=colnames)
    df_data = pd.DataFrame(X_test_numpy, columns=colnames)

    # --- 4. PLOTTING ---
    plt.figure()
    shap.summary_plot(shap_values_class_1_array, features=df_data, feature_names=colnames, show=False)
    plt.tight_layout()
    
    single_shap = pathlib.Path(res_dir / 'single_shaps')
    single_shap.mkdir(parents=True, exist_ok=True)

    plt.savefig(single_shap / f'summary_{rs}.png', dpi=400)
    plt.close()

    return df_vals, df_data



import shap
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import numpy as np
import traceback # Import this to see real errors

def compute_calibrated_shap_values(model, X_test_raw, colnames, rs, res_dir):
    """
    Computes SHAP values specifically for a CalibratedClassifierCV object.
    Robustly handles Pipeline input requirements by reconstructing DataFrames.
    """
    print(f"Computing SHAP for Calibrated Model (KernelExplainer)... this may be slow.")

    # --- 1. PREPARE DATA ---
    # Ensure X_bg is a DataFrame with correct columns
    if isinstance(X_test_raw, pd.DataFrame):
        X_bg = X_test_raw
    else:
        X_bg = pd.DataFrame(X_test_raw, columns=colnames)
        
    # 2. Optimize Background Data
    # Use shap.utils.sample (safer for mixed types/strings than kmeans)
    background_summary = shap.utils.sample(X_bg, 10)

    # 3. Define the Target Function (Wrapper)
    # This wrapper reconstructs the DataFrame so the Pipeline doesn't crash
    def probability_class_1(X):
        # DEBUG: If X comes in as a Numpy array (which SHAP often does), 
        # convert it back to DataFrame using the known column names.
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=colnames)
            
            # Explicitly fix data types if necessary (Pandas usually infers, but safety first)
            # This helps if SHAP converted everything to object/string
            X = X.infer_objects() 

        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            # If it still crashes, print the REAL error so we can see it
            traceback.print_exc()
            raise

    # 4. Initialize KernelExplainer
    # We pass the function and the background data
    explainer = shap.KernelExplainer(probability_class_1, background_summary)

    # 5. Compute SHAP Values
    # shap.utils.sample might return a dataframe, so we pass X_bg directly
    shap_values = explainer.shap_values(X_bg, nsamples='auto', silent=True)

    # Handle Output Format (List vs Array)
    if isinstance(shap_values, list):
        shap_values_class_1 = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_values_class_1 = shap_values

    # 6. Create DataFrame & Plot
    df_vals = pd.DataFrame(shap_values_class_1, columns=colnames)
    
    plt.figure()
    shap.summary_plot(shap_values_class_1, features=X_bg, feature_names=colnames, show=False)
    plt.tight_layout()
    
    calibrated_shap_dir = pathlib.Path(res_dir / 'calibrated_single_shaps')
    calibrated_shap_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(calibrated_shap_dir / f'calibrated_summary_{rs}.png', dpi=400)
    plt.close()

    return df_vals





def store_importances( res_dir_feat, perm_imp_list, feature_names_orig):

	# df of permutation importance values for each iteration
	df_perm_imp = pd.DataFrame(perm_imp_list, columns=feature_names_orig)
	df_perm_imp.to_csv(res_dir_feat / 'all_permutation_importances.csv', index = False)


def shap_analysis(shap_dir, shap_val_list, shap_data_list):

	df_shap_values = pd.concat([df for df in shap_val_list], axis=0)
	df_shap_values.to_csv(shap_dir / 'df_shap_vals_filtered.csv', index=False)

	df_shap_data = pd.concat([df for df in shap_data_list], axis=0)
	df_shap_data.to_csv(shap_dir / 'df_shap_data_filtered.csv', index=False)
	

	num_rows = 500 if len(df_shap_values) > 700 else len(df_shap_values)  # Adjust as needed
	random_indices = np.random.choice(len(df_shap_values), size=num_rows, replace=False)

	# Sample the same rows from both DataFrames
	df_shap_vals = df_shap_values.iloc[random_indices]
	df_shap_data = df_shap_data.iloc[random_indices]

	shap_values_obj = shap.Explanation(values=df_shap_vals.values, data=df_shap_data, 
									feature_names=df_shap_data.columns,
        base_values=np.zeros(len(df_shap_vals))) # BASE VALUES kept as default..


	# swarmplot
	plt.figure()
	shap.plots.beeswarm(shap_values_obj, max_display=10, show=False)
	plt.tight_layout()
	plt.savefig(shap_dir / 'beeswarmplot.png', dpi=400)
	plt.close()

	plt.figure()
	shap.summary_plot(shap_values_obj, show=False)
	plt.tight_layout()
	plt.savefig(shap_dir / 'summary_plot.png', dpi=400)
	plt.close()


def shap_analysis_calibrated(shap_dir, shap_val_list, shap_data_list):

    df_shap_values = pd.concat([df for df in shap_val_list], axis=0)
    df_shap_values.to_csv(shap_dir / 'df_shap_vals_filtered_calibrated.csv', index=False) 
    
    df_shap_data = pd.concat([df for df in shap_data_list], axis=0)
    df_shap_data.to_csv(shap_dir / 'df_shap_data_filtered_calibrated.csv', index=False)

    num_rows = 500 if len(df_shap_values) > 700 else len(df_shap_values)  # Adjust as needed
    random_indices = np.random.choice(len(df_shap_values), size=num_rows, replace=False)

	# Sample the same rows from both DataFrames
    df_shap_vals = df_shap_values.iloc[random_indices]
    df_shap_data = df_shap_data.iloc[random_indices]
    shap_values_obj = shap.Explanation(values=df_shap_vals.values, data=df_shap_data, 
									feature_names=df_shap_data.columns,
                                    base_values=np.zeros(len(df_shap_vals))) # BASE VALUES kept as default..


	# swarmplot
    plt.figure()
    shap.plots.beeswarm(shap_values_obj, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(shap_dir / 'beeswarmplot_calibrated.png', dpi=400)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values_obj, show=False)
    plt.tight_layout()
    plt.savefig(shap_dir / 'summary_plot_calibrated.png', dpi=400)
    plt.close()