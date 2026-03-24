import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from scipy.special import logit
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt


def get_calibration_metrics(y_true, y_prob):
    """
    Computes 'Weak Calibration' metrics recommended for small datasets.
    """
    # 1. Brier Score (lower is better)
    bs = brier_score_loss(y_true, y_prob)
    
    # 2. Calibration-in-the-large (Mean Calibration)
    # If mean_pred > mean_obs -> Overestimation
    mean_pred = np.mean(y_prob)
    mean_obs = np.mean(y_true)
    
    # 3. Calibration Slope and Intercept
    # Avoid log(0) or log(1) by clipping slightly
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    
    # Calculate log-odds (logit) of predictions
    # This transforms probabilities (0 to 1) to linear space (-inf to +inf)
    logit_preds = logit(y_prob_clipped).reshape(-1, 1)
    
    # Fit Logistic Regression: y_true ~ intercept + slope * logit_preds
    # We use a standard LogisticRegression but we must ensure no regularization (C=huge)
    # to get the raw statistical relationship.
    calib_model = LogisticRegression(C=1e9, solver='lbfgs') 
    calib_model.fit(logit_preds, y_true)
    
    slope = calib_model.coef_[0][0]
    intercept = calib_model.intercept_[0]
    
    return {
        'brier_score': bs,
        'mean_prediction': mean_pred,
        'mean_observation': mean_obs,
        'calib_slope': slope,
        'calib_intercept': intercept
    }


    import matplotlib.pyplot as plt
import numpy as np
import pathlib
from sklearn.calibration import calibration_curve

def plot_aggregated_calibration_curve(y_true_list, y_prob_list, metrics_df, output_dir, classifier_name):

    """
    Plots a calibration curve by aggregating (concatenating) data from all cross-validation folds.
    This creates a single 'master' curve effectively using N_total samples.
    
    Args:
        y_true_list: List of array-like true labels (0/1) for each fold.
        y_prob_list: List of array-like predicted probabilities (class 1) for each fold.
        metrics_df: DataFrame containing 'calib_slope', 'calib_intercept', 'brier_score' to show stats.
        output_dir: Path object or string for saving.
        classifier_name: String name of the model.
    """
    plt.figure(figsize=(7, 7))
    
    # 1. Aggregate Data (Concatenation)
    # This pools all 100 test sets into one large set of ~5000 predictions
    y_true_all = np.concatenate(y_true_list)
    y_prob_all = np.concatenate(y_prob_list)
    
    # 2. Calculate the Curve on this massive pooled data
    # n_bins=10 is now safe because we have thousands of points
    prob_true, prob_pred = calibration_curve(y_true_all, y_prob_all, n_bins=10, strategy='uniform')
    
    # 3. Plotting
    # Diagonal line (Perfect Calibration)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    # The Aggregated Curve
    plt.plot(prob_pred, prob_true, "s-", color='blue', label=f"{classifier_name} (Aggregated)")
    
    # Optional: Histogram to show where the probabilities actually fall
    plt.hist(y_prob_all, range=(0, 1), bins=10, color='gray', alpha=0.1, 
             weights=np.ones_like(y_prob_all) / len(y_prob_all), label="Distribution")

    # 4. Add Text Stats (Mean +/- STD from your metrics DF)
    # We display the *variance* of the metrics here, even though the plot is aggregated
    mean_slope = metrics_df['calib_slope'].mean()
    std_slope = metrics_df['calib_slope'].std()
    mean_int = metrics_df['calib_intercept'].mean()
    mean_bs = metrics_df['brier_score'].mean()
    
    stats_text = (
        f"Mean Slope: {mean_slope:.2f} ± {std_slope:.2f}\n"
        f"Mean Intercept: {mean_int:.2f}\n"
        f"Mean Brier: {mean_bs:.3f}"
    )
    
    # Position the text box in bottom right or relevant empty space
    plt.text(0.55, 0.05, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray"))

    # 5. Styling and Saving
    plt.ylabel("Observed Fraction")
    plt.xlabel("Predicted Probability")
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="upper left")
    plt.title(f"Calibration Curve: {classifier_name}\n(Aggregated over {len(y_true_list)} folds)")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    save_path = pathlib.Path(output_dir) / f'aggregated_calibration_curve_{classifier_name}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()



from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

def calibrate_best_model(best_estimator, X_train, y_train, config):
    """
    Wraps the best estimator from GridSearch in a CalibratedClassifierCV.
    
    CRITICAL: This does NOT use the pre-fitted weights. It retrains the model
    on internal cross-validation folds to learn a clean calibration mapping
    without data leakage.
    
    Args:
        best_estimator: The winning estimator from RandomizedSearchCV.
        X_train, y_train: The training data for the current external fold.
        config: Dictionary containing 'n_jobs' and other settings.
        
    Returns:
        A fitted CalibratedClassifierCV
    """
    
    # 1. Identify if we should skip calibration (e.g., TabPFN)
    # Check if the step is named 'classifier' and it is a TabPFN instance
    # Depending on your pipeline structure, we might need to look deeper.
    # A robust check is looking at the class name of the final step.
    
    if isinstance(best_estimator, Pipeline):
        final_step = best_estimator.steps[-1][1]
        model_type = final_step.__class__.__name__.lower()
    else:
        model_type = best_estimator.__class__.__name__.lower()

    if 'tabpfn' in model_type:
        print("Skipping calibration for TabPFN (natively calibrated).")
        return best_estimator

    # 2. Apply "Weak Calibration" Strategy (Sigmoid + CV)
    # We clone the estimator to ensure we start fresh (ignoring previous fit)
    unfitted_estimator = clone(best_estimator)
    
    calibrated_model = CalibratedClassifierCV(
        estimator=unfitted_estimator,
        method='sigmoid',   # Safe for small N (approx 250 samples)
        cv=5,               # 5-fold internal CV for calibration
        n_jobs=config['n_jobs']
    )
    
    # 3. Fit on X_train
    # This triggers the internal splitting: Train on 80%, Calibrate on 20%
    calibrated_model.fit(X_train, y_train)
    
    return calibrated_model