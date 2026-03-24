import argparse
import pathlib
import random
import sys

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# Custom Imports
from lib.ensemble.store import get_patient_prob_results, mean_roc_curve_plot, save_raw_results, save_raw_results_w_cal, store_classification_metrics
from lib.ensemble.importance import compute_shap_values, shap_analysis, shap_analysis_calibrated, store_importances, compute_calibrated_shap_values
from lib.ensemble.utils import create_result_dirs, generate_paths, load_config, load_data, load_param_distributions, save_config
from lib.ensemble.pipeline import define_pipeline, evaluate_model, get_final_transformed_test_data, get_score, my_grid_search
# Added calibrate_best_model to imports
from lib.ensemble.calibration import get_calibration_metrics, plot_aggregated_calibration_curve, calibrate_best_model

def main(config_file) -> int:
    
    # reproducibility settings
    random_state = 42
    random.seed(0)

    # load config dictionary from config path
    config = load_config(config_file)
    
    # load parameter distributions for gridsearch
    param_distributions = load_param_distributions('params.yaml')

    # get input dir and create res dir (Pathlib dir)
    input_dir, res_dir = generate_paths(config['y_label'], config['input_path'], config['output_path'])

    # store config options for this run
    save_config(config, res_dir)

    # rs list for train-test splits
    rs_list = random.sample(range(1, 100), config['nested_iterations'])

    # load data
    X_orig, y_orig, feature_types, strat_col, n_drop = load_data(input_dir, config['path_to_feature_types'],
                                   y_label=config['y_label'], col_to_drop=config['col_to_drop'], 
                                   stratify_on_symptom=True, 
                                   drop_bl_info=config['drop_bl_info'])
    


    # metrics definition
    test_metrics = ['mcc', 'f1', 'roc_auc', 'confusion_matrix', 'auprc', 'sensitivity', 'specificity']
    train_metrics = ['mean_test_mcc', 'std_test_mcc', 'mean_test_roc_auc', 'std_test_roc_auc', 'mean_test_f1', 'std_test_f1']

    # classifiers loop
    for classifier in config['classifiers_list']:

        res_dir_cl, feat_dir, shap_dir = create_result_dirs(res_dir, classifier, config['shap_analysis'], config['perm_importance'])
        print('-----------------------------------')
        print(f'Running {classifier}')  
        
        # pipeline definition (Calibration removed from here)
        use_calibration = config.get('use_calibration', False)

        pipeline, param_dist = define_pipeline(classifier, param_distribution=param_distributions, 
                                                handle_imb_data = config['handle_imb_data'],
                                                include_feature_selector = config['include_feature_selector'], 
                                                n_of_features = config['n_of_features'], 
                                                feature_types = feature_types, 
                                                random_state=random_state)



        # Init Storage
        pred_prob_list_final, pred_prob_list_raw = [], []
        pred_y_list_final = []
        test_metrics_list_final= []
        
        best_train_metrics_list, best_params_list = [], []

        # SHAP and Permutation importance storage
        shap_val_list, shap_data_list, shap_cal_val_list, shap_cal_data_list, perm_imp_list = [], [], [], [], []

        real_y_list, X_test_idx_list = [], []

        # Lists for calibration metrics
        calib_metrics_cal_list = []
        calib_metrics_raw_list = []

        # Common FPR for ROC curves
        tpr_list, fpr_list, roc_auc_list = [], [], []

        fpr_common = np.linspace(0, 1, 100)
        
        print('Pipeline : ', pipeline)
        print('--')
        print('Param dist: ', param_dist)
        
        # train-test combinations loop
        for rs in rs_list:
            calibrated_model = None  # Safety reset
            
            if config['drop_random_patients']:
                _, drop_idx = train_test_split(
                    X_orig.index, 
                    test_size=n_drop,
                    stratify=strat_col,
                    random_state=rs 
                    )
                X_temp = X_orig.drop(index=drop_idx, errors='ignore')
                y_temp = y_orig.drop(index=drop_idx, errors='ignore')
            else:
                X_temp = X_orig.copy()
                y_temp = y_orig.copy()

            if strat_col is not None:
                if config['drop_random_patients']:
                    strat_col_temp = strat_col.drop(index=drop_idx, errors='ignore')
                else:
                    strat_col_temp = strat_col.copy()
            
            X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, stratify=strat_col_temp, test_size=config['train_test_split_size'], random_state=rs)
            
            
            # --- 1. Train Base Model (Grid Search) ---
            if param_dist:
                grid_model = my_grid_search(X_train, y_train, pipeline, param_dist, 
                                        config['n_jobs'], verbose=True, 
                                        refit_metric=config['refit'], 
                                        scoring=get_score(),
                                        cv_repeats=config['cv_repeats'], 
                                        cv_splits=config['cv_splits'], 
                                        n_iter=config['n_iter'], random_state=random_state)
                # Extract best estimator for potential calibration
                best_model_raw = grid_model.best_estimator_
            else: 
                # TabPFN / No Search
                grid_model = pipeline
                grid_model.fit(X_train, y_train)
                best_model_raw = grid_model

            X_test_idx_list.append(X_test.index)
            
            # --- 2. Evaluate RAW (Uncalibrated) Model ---
            # Only transform if SHAP is enabled to save cluster time
            # Important: pass best_model_raw to get_final_transformed to extract features correctly
            if config['shap_analysis']:
                X_test_transformed, final_colnames = get_final_transformed_test_data(best_model_raw, X_test)
            # Used for "Before" plot and fallback if calibration is off
            y_pred_raw, y_pred_proba_raw, metrics_raw, tpr_raw, roc_auc_raw = evaluate_model(best_model_raw, X_test, y_test, fpr_common)
            


            # --- 3. Calibration Logic ---
            if use_calibration:
                # Calculate & Store Raw Calibration Metrics
                calib_metrics_raw = get_calibration_metrics(y_test.values, y_pred_proba_raw)
                calib_metrics_raw_list.append(calib_metrics_raw)
                
                # Calibrate the Model
                calibrated_model = calibrate_best_model(best_model_raw, X_train, y_train, config)
                
                # Evaluate Calibrated Model
                # Note: We don't need to re-extract X_test_transformed as features shouldn't change, 
                # but passing calibrated_model to evaluate is key.
                y_pred_cal, y_prob_cal_2d, metrics_cal, tpr_cal, roc_auc_cal = evaluate_model(calibrated_model, X_test, y_test, fpr_common)
                
                # Ensure 1D extraction for calibrated probs
                y_pred_proba_cal = y_prob_cal_2d[:, 1] if y_prob_cal_2d.ndim == 2 else y_prob_cal_2d
                
                # Store Calibrated Metrics
                calib_metrics_cal = get_calibration_metrics(y_test.values, y_pred_proba_cal)
                calib_metrics_cal_list.append(calib_metrics_cal)

                # Set Final variables with Calibrated results
                y_pred_final = y_pred_cal
                y_pred_proba_final = y_pred_proba_cal
                metrics_final = metrics_cal # We want to report the IMPROVED metrics
                tpr_final = tpr_cal
                roc_auc_final = roc_auc_cal
            else:
                # Set Final variables with NON Calibrated results (since no calibration is done)
                y_pred_final = y_pred_raw
                y_pred_proba_final = y_pred_proba_raw
                metrics_final = metrics_raw
                tpr_final = tpr_raw
                roc_auc_final = roc_auc_raw

            # Append Results to Storage Lists ---
            real_y_list.append(y_test)
            
            # Append to FINAL lists (Used for main reports/plots)
            pred_y_list_final.append(y_pred_final)
            pred_prob_list_final.append(y_pred_proba_final)
            test_metrics_list_final.append(metrics_final)
            tpr_list.append(tpr_final)
            fpr_list.append(fpr_common)
            roc_auc_list.append(roc_auc_final)
            
            # Append RAW predictions to a separate list  
            pred_prob_list_raw.append(y_pred_proba_raw)


            # Store Training Metrics (from the Grid Search - non calibrated, since calibration is a post-processing step)
            if param_dist:
                best_params_list.append(grid_model.best_params_)
                best_train_metrics_list.append([
                    grid_model.cv_results_.get(key)[grid_model.best_index_]
                    for key in train_metrics
                ])
            else:
                best_params_list.append({'params': 'N/A'})
                best_train_metrics_list.append(['N/A'] * len(train_metrics))

            # --- SHAP ---
            if config['shap_analysis']:
                # DIRECTLY use best_model_raw -> This is the pipeline/model fitted on the full X_train
                base_estimator = best_model_raw
                
                # Logic to find the final step if it is a Pipeline
                if isinstance(base_estimator, Pipeline):
                    classifier_model = base_estimator.named_steps['classifier']
                else:
                    classifier_model = base_estimator

                # Compute SHAP for the RAW model (since SHAP is typically done on the base model before calibration)
                df_shap, df_data = compute_shap_values(
                    classifier, 
                    classifier_model, 
                    X_test_transformed, 
                    final_colnames, 
                    rs, 
                    res_dir=shap_dir
                )
                shap_val_list.append(df_shap)
                shap_data_list.append(df_data)

                if use_calibration and calibrated_model is not None:

                    # Compute Calibrated SHAP 
                    df_shap_cal = compute_calibrated_shap_values(
                        calibrated_model,
                        X_test,
                        X_test.columns,
                        rs,
                        res_dir=shap_dir
                    )
                    shap_cal_val_list.append(df_shap_cal)

                    shap_cal_data_list.append(X_test.reset_index(drop=True))

                # Permutation importance (uses X_train/y_train and raw grid_model)
                if config['perm_importance']:
                    perm = permutation_importance(grid_model, X_train, y_train, n_repeats=10, random_state=random_state,
                                            scoring=make_scorer(matthews_corrcoef, greater_is_better=True))
                    perm_imp_list.append(np.mean(perm.importances, axis=1))

            # --- END OF LOOP ---
        
        print(f'Saving metrics in {res_dir_cl}')

        # 1. Feature Importances
        if config['perm_importance']:
            store_importances(feat_dir, perm_imp_list, feature_names_orig=X_orig.columns)
     
        if config['shap_analysis']:
            shap_analysis(shap_dir, shap_val_list, shap_data_list)
            if use_calibration and len(shap_cal_val_list) > 0:
                shap_analysis_calibrated(shap_dir, shap_cal_val_list, shap_cal_data_list)
            
        
        # 2. Save Classification Performance (Metrics)
        # We use test_metrics_list_final which contains the best version (Calibrated or Raw)
        if classifier == 'tabpfn': 
            store_classification_metrics(test_metrics_list_final, test_metrics, None, None, None, res_dir_cl)
        else:
            store_classification_metrics(test_metrics_list_final, test_metrics, best_train_metrics_list, train_metrics, best_params_list, res_dir_cl)

        # 3. Save Calibration Data & Plots
        if use_calibration:
            # Save CSVs
            calib_metrics_raw_df = pd.DataFrame(calib_metrics_raw_list)
            calib_metrics_raw_df.to_csv(pathlib.Path(res_dir_cl) / 'calibration_metrics_raw.csv', index=False)
            
            calib_metrics_cal_df = pd.DataFrame(calib_metrics_cal_list)
            calib_metrics_cal_df.to_csv(pathlib.Path(res_dir_cl) / 'calibration_metrics_calibrated.csv', index=False)
            
            # Plot "After" (Calibrated) -> Use FINAL list
            plot_aggregated_calibration_curve(real_y_list, pred_prob_list_final, calib_metrics_cal_df, res_dir_cl, classifier + '_calibrated')
            
            # Plot "Before" (Raw) -> Use RAW list
            plot_aggregated_calibration_curve(real_y_list, pred_prob_list_raw, calib_metrics_raw_df, res_dir_cl, classifier + '_raw')

        # 4. Store Raw Results (Predictions)
        # Use pred_prob_list_final to ensure we save the probs that match the metrics
        save_raw_results(X_test_idx_list, real_y_list, pred_y_list_final, pred_prob_list_final, res_dir_cl)

        if use_calibration:
            save_raw_results_w_cal(X_test_idx_list, real_y_list, pred_prob_list_raw, pred_prob_list_final, res_dir_cl)

        get_patient_prob_results(X_orig.index, X_test_idx_list, pred_prob_list_final, rs_list, res_dir_cl)
        
        # 5. Plot ROC Curve (Mean)
        mean_roc_curve_plot(tpr_list, roc_auc_list, fpr_common, res_dir_cl)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis with optional configuration.")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    sys.exit(main(config_file=args.config))