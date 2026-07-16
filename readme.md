## Research Concept
# Uncertainty and Rejection under Clinical Class Imbalance

*(Add DOI/Publication badge here once published)*

## Research Concept
Standard machine learning evaluation assumes predictive uncertainty can be thresholded independently of a dataset's class prior. In clinical practice, this assumption fails catastrophically. To safely translate predictive models into actionable Clinical Decision Support Systems (CDSS) without abandoning high-risk minority patients, we implemented the following workflow:

**0. Dynamic Calibrated Ensemble Generation**
The mathematical integrity of any uncertainty-based rejection system depends entirely on the reliability of the underlying probability distributions. The pipeline enforces strict probability calibration (Platt scaling) across 30 bootstrapped iterations of heterogeneous base classifiers. This asynchronous ensemble captures both algorithmic and data variance, providing a highly robust, patient-specific distribution of probabilities ($p$) and epistemic variance ($\sigma^2$).

**1. Resolving the Threshold-Entropy Mismatch ($H_\tau$)**
Standard Shannon binary entropy assumes maximum ambiguity always occurs at $p = 0.5$. In imbalanced clinical datasets, the decision boundary ($\tau$) naturally shifts (e.g., $\tau = 0.13$). Instead of applying uncorrected entropy, we introduce **Piecewise-rescaled decision entropy ($H_\tau$)**. This mathematically warps the asymmetric probability space so that the clinical boundary ($\tau$) sits exactly at the physical center of the entropy curve, realigning maximum mathematical uncertainty with maximum clinical ambiguity.

**2. Class-Conditioned Rejection (CCR)**
To prevent global uncertainty thresholds from acting as a naive minority-class deletion filter, the workflow utilizes **Class-Conditioned Rejection (CCR)**. By evaluating the ordinal ranking of $H_\tau$ strictly within localized predicted trajectories, CCR neutralizes prior-driven bias, bypasses epistemic sparsity, and safely defers ambiguous patients without mathematically erasing the disease class.

**3. Class-Conditioned Evaluation (ccAUGRC)**
Standard multi-threshold metrics (like AUGRC) suffer from "risk-masking" under imbalance: models can artificially improve their global score by systematically substituting high-risk minority patients with low-risk majority patients. We resolve this by computing generalized risk-coverage independently for each clinical trajectory (**ccAUGRC**), proving whether safety is maintained across all patient groups.

**4. Cross-Disease Validation**
To ensure the pipeline is robust across varying degrees of biological noise and class imbalance, we mapped the phase diagram of algorithmic failure across 56 synthetic conditions and evaluated the framework on three neuro-predictive cohorts:
* **Parkinson’s Disease (3-Year Dyskinesia, $\tau=0.39$):** A moderately imbalanced cohort testing the algorithm's ability to stabilize asymmetric noise.
* **Multiple Sclerosis (PIRA, $\tau=0.13$):** A severely imbalanced cohort testing CCR as a mandatory rescue mechanism against systemic minority-class collapse.
* **Alzheimer’s Disease (36-Month MCI to AD, $\tau=0.54$):** A near-balanced cohort testing the framework's ability to safely converge to baseline behavior.

**5. synthetic validation**
---

### Main Findings 

**1. Structural Uncertainty Divergence**
Across multiple clinical forecasting domains, we observed that predictive uncertainty is structurally concentrated on the minority class. Due to inherent epistemic sparsity, minority-class predictions carry systematically higher ensemble variance than majority predictions—even when evaluated at the exact same distance from the decision boundary.

**2. Global Rejection Acts as a Naive Deletion Filter**
Our analysis proves that applying standard global rejection thresholds (like $H_{Total}$ or Margin) weaponizes this uncertainty divergence. Global rejection acts as a naive deletion filter, systematically discarding the most difficult-to-predict minority patients and causing sensitivity to artificially crash.

**3. Breaking the Sensitivity-Specificity Trade-off**
We demonstrate that CCR + $H_\tau$ is the only framework capable of safely discarding ambiguous patients without erasing the minority class. By realigning the entropy signal and isolating triage queues, it mathematically prevents the systemic deletion of high-risk patients while stabilizing overall performance.

**4. Unmasking Algorithmic Bias in Evaluation**
We establish that standard global evaluation metrics are dangerously misleading under imbalance. Frameworks that appeared "safe" globally were actively abandoning rapid progressors. ccAUGRC exposes this bias and proves our proposed framework is the only method that bounds risk for all patient trajectories simultaneously.

**5. The Scaling Law of Class Imbalance**
Our cross-disease application revealed that the clinical necessity and functional role of CCR scale directly with the baseline balance of the dataset:
* In severely imbalanced datasets (MS), CCR acts as a **Mandatory Rescue Mechanism** to prevent the total collapse of Sensitivity.
* In moderately imbalanced datasets (PD), it acts as a **Stabilization Tool** to halt progressive minority erosion.
* In near-balanced datasets (AD), the risk of erasure dissolves, and the framework acts as a **Safe Convergence Engine**, matching standard baseline performance without introducing artificial penalties.

**6. Operationalizing the Clinical "Abstain" Option**
CCR transitions predictive models from rigid binary classifiers into safe CDSS. By accurately bounding uncertainty without destroying class integrity, the framework provides a mathematically rigorous mechanism for models to "abstain" from prediction. Deferring the top 20-30% of class-conditioned ambiguous cases to secondary screening represents the necessary operational blueprint for deploying AI in high-stakes biological environments.

---

## Repository Structure

The codebase is organized into modular directories handling the core classification pipeline, validation on both real-world and synthetic datasets, and dataset-specific preprocessing.

### Core Pipeline
* **`code_classification/`**: Contains the main classification pipeline, including data preprocessing, model training, and evaluation scripts.
  * `main_calibrate.py`: Implements dynamic calibrated ensemble generation, nested cross-validation, and Sigmoid calibration for uncertainty quantification.

### Real-World Validation
* **`code_real_world_validation/`**: Implements the uncertainty quantification (UQ) and rejection protocol for cross-disease validation on real-world clinical datasets.
  * `1_h_ensemble.py`: Handles the asynchronous aggregation of all machine learning models.
  * `2_compute_uncertainy_metrics.py`: Computes total predictive uncertainty and class-conditioned rejection curves.
  * `3_compute_stats.py`: Calculates bootstrap confidence intervals and performs statistical tests for ccAUGRC differences between methods.
  * `4_plots`, `5_plots`, `6_plots`: Scripts to generate figures and visualizations for the real-world validation results.

### Synthetic Validation
* **`code_synth_validation/`**: Contains scripts for generating synthetic data and conducting controlled experiments to test method robustness.
  * `1_synthetic_generator.py`: Generates synthetic datasets with configurable levels of class imbalance and noise.
  * `2_generate_synthetic_configs.py`: Reads `synthetic_data/manifest.csv` to generate condition-specific `config.ini` files and a SLURM-compatible commands file (`synthetic_commands.txt`) indexed by `SLURM_ARRAY_TASK_ID`.
  * `3_compute_stats.py`: Calculates bootstrap confidence intervals and statistical tests for ccAUGRC differences.
  * `4_compute_uncertainty_scores.py`: Computes total predictive uncertainty and class-conditioned rejection curves for the synthetic datasets.
  * `5_compute_rejection_metrics.py`: Calculates specialized rejection metrics.
  * `7_paper_figures.py`: Generates figures for the manuscript based on synthetic validation results.

### ADNI-Specific Pipelines
* **`code_data_prep_adni/`**: Handles ADNI dataset preprocessing, including data cleaning, feature extraction, and formatting for model training.
* **`code_evaluate_adni/`**: Evaluates model performance specifically on the ADNI dataset, including metric calculation and results visualization.
