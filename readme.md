## Research Concept
Standard machine learning evaluation assumes that all predictive errors carry equal weight and that predictive uncertainty can be evaluated independently of the dataset's class prior. In clinical practice, this assumption fails catastrophically. To safely translate predictive models into actionable Clinical Decision Support Systems (CDSS), we implemented the following workflow:

**0. Dynamic Calibrated Ensemble Generation**

The mathematical integrity of any uncertainty-based rejection system depends entirely on the reliability and depth of the underlying probability distributions. Because standard machine learning algorithms frequently produce uncalibrated outputs, the pipeline first enforces strict probability calibration via nested cross-validation and internal Sigmoid calibration (Platt scaling). Furthermore, because the test set composition shifts dynamically across 30 resampling iterations, each patient receives a variable-length ensemble of up to 150 calibrated predictions (spanning 5 distinct ML algorithms). This asynchronous ensemble approach captures both algorithmic variance and training-data variance, providing a highly robust, patient-specific distribution of probabilities that serves as the foundation for accurate entropy evaluation.

**1. Uncertainty Quantification**

Rather than relying on the absolute purity of mathematically entangled Information-Theoretic metrics (Aleatoric vs. Epistemic uncertainty), the workflow calculates Total Predictive Uncertainty ($H_{total}$). This captures both inherent biological noise and out-of-distribution anomalies while remaining robust in finite-data clinical regimes.
 - NEW : Standard Shannon binary entropy is mathematically perfect, but its physical assumption is that maximum ambiguity always occurs at $p = 0.5$. In clinical datasets with extreme class imbalance, the baseline prior can be as low as 10-20%, meaning maximum ambiguity occurs at $p = 0.1$ or $p = 0.2$. Instead of inventing a completely new entropy formula, $H_\tau$ works by taking the asymmetric clinical probability space and warping it so that the clinical boundary ($\tau$) sits exactly at the physical center ($0.5$). Once the space is mathematically centered, we simply apply standard Shannon entropy.

**2. Class-Conditioned Rejection (CCRC)**
- NEW :
To prevent global uncertainty thresholds from acting as a naive minority-class deletion filter, the workflow utilizes Class-Conditioned Rejection Curves (CCRC). By evaluating the ordinal ranking of $H_{\tau}$ strictly within localized predicted classes, CCRC applies proportional rejection thresholds. This neutralizes prior-driven bias and safely defers ambiguous patients without mathematically erasing the disease class.

 - we compare CCRC + $H_\{tau}$ against a global rejection protocol and against a margin-based rejection protocol (which is the current SOTA method for operationalizing Epistemic uncertainty). We demonstrate that CCRC + $H_\{tau}$ is the only method that can safely discard ambiguous patients without erasing the minority class, while also achieving stable overall performance across all three clinical scenarios. 

**3. Clinical Evaluation via Confusion Matrix Dynamics**

The workflow shifts the evaluation paradigm away from global statistical ratios (e.g., MCC or AUC) toward raw confusion matrix dynamics. Because proportional rejection simultaneously removes False Positives (preventing toxic overtreatment) and False Negatives (preventing false reassurance), global ratios often mathematically plateau. The workflow reframes these "flatlining" metrics not as algorithmic failures, but as the expected signature of a safely calibrated clinical model.

**4. Cross-Disease Validation**

To ensure the pipeline is robust across varying degrees of biological noise and class imbalance, the workflow evaluates the CCRC algorithm across three distinct neuro-predictive scenarios:

* Parkinson’s Disease (3-Year Dyskinesia): A moderately imbalanced cohort testing the algorithm's ability to stabilize asymmetric noise.

* Multiple Sclerosis (Progression Independent of Relapse Activity): A severely imbalanced cohort testing CCRC as a mandatory rescue mechanism against systemic minority-class collapse.

* Alzheimer’s Disease (36-Month MCI to AD Conversion): A balanced cohort testing CCRC's ability to act as a strict calibration enforcer.

### Main findings 
**1. The Asymmetric Penalty of Clinical Uncertainty**

Across multiple clinical forecasting domains, we observed that predictive uncertainty is structurally concentrated on the minority class. Because Bayesian algorithms default toward the dataset's baseline prior in the presence of ambiguous or noisy biological features, minority-class predictions require an exponentially higher standard of evidence to achieve confidence. Consequently, the minority class inherently carries higher baseline Shannon Entropy, even when data noise is equally distributed across both trajectories.

**2. Global Rejection Acts as a Naive Deletion Filter**

Our analysis proves that applying a standard global rejection threshold to clinical data does not inherently isolate "bad" predictions. Because of the uncertainty asymmetry, global rejection acts as a naive minority-class deletion filter, systematically discarding the most difficult-to-predict patients and causing Sensitivity to artificially crash.

**3. Bypassing SOTA Entanglement to Operationalize Total Uncertainty**

While current Information-Theoretic (IT) methods emphasize disentangling Aleatoric and Epistemic uncertainty, recent literature demonstrates these metrics are deeply entangled in finite-data regimes. We demonstrate that CCRC provides a pragmatic, mathematically safe mechanism to operationalize Total Predictive Uncertainty ($H_{total}$). By evaluating ordinal ranking strictly within localized predicted classes, CCRC neutralizes prior-driven bias, allowing models to safely discard ambiguous patients without requiring unreliable IT disentanglement and without erasing the rare disease class.

**4. Redefining Algorithmic Failure: The "Flatlining" Paradox**

We establish that standard global metrics (such as MCC) are dangerously misleading when evaluating clinical rejection protocols. In severely imbalanced cohorts, overall performance ratios mathematically plateaued under CCRC. While standard literature interprets flatlining metrics as algorithmic failure, our analysis of raw confusion matrix dynamics proves this plateau occurs because CCRC proportionally removes False Positives (preventing toxic overtreatment) and False Negatives (preventing false reassurance) at equal rates. A flatlining metric under CCRC is the mathematical signature of a rigorously calibrated model prioritizing clinical safety over inflated statistical ratios.

**5. The Scaling Law of Class Imbalance**

Our cross-disease application revealed that the clinical necessity and functional role of CCRC scale directly with the baseline balance of the dataset.

- In severely imbalanced datasets (e.g., MS at ~11% prior), CCRC acts as a mandatory Rescue Mechanism to prevent the total collapse of Sensitivity.
- In moderately imbalanced datasets (e.g., PD at ~29%), it acts as a Stabilization Tool.
- In highly balanced datasets (e.g., Alzheimer's at ~54%), the risk of minority-class erasure disappears, and CCRC transitions into an Optimization Engine, perfectly synchronizing Sensitivity and Specificity by targeting Epistemic disagreement.

**6. Operationalizing the Clinical "Abstain" Option**

Ultimately, CCRC transitions predictive models from rigid binary classifiers into safe Clinical Decision Support Systems (CDSS). By accurately bounding uncertainty without destroying class integrity, the framework provides a mathematically rigorous mechanism for models to "abstain" from prediction. Deferring the top 20-30% of class-conditioned ambiguous cases to secondary screening or "watchful waiting" represents the necessary operational blueprint for deploying AI in high-stakes, noisy biological environments.

## Repository Structure

code_classification/ 
    folder contains the code for the classification pipeline, including data preprocessing, model training, and evaluation scripts. files:

    - main_calibrate.py : Implements the dynamic calibrated ensemble generation and uncertainty quantification steps, including nested cross-validation and Sigmoid calibration.



code_real_world_validation/ f
    older contains uncertainty quantification and rejection protocol implementation, along with scripts for cross-disease validation for real-world clinical datasets. files: 

    - 1_h_ensemble.py : implements the asynchronous aggregation of all ml models
    - 2_compute_uncertainy_metrics.py : implements the computation of total predictive uncertainty and class-conditioned rejection curves
    - 3_compute_stats.py : implements the computation of bootstrap confidence intervals and statistical tests for ccAUGRC differences between methods.
    - 4,5,6 plots


code_synth_validation/ 
    folder contains scripts for synthetic data generation and validation, allowing for controlled experiments to test the robustness of the proposed methods. files:
    1_synthetic_generator.py : generates synthetic datasets with varying levels of class imbalance and noise.
    2_generate_synthetic_configs.py : Reads synthetic_data/manifest.csv and generates:
        - One config.ini per condition under synthetic_data/<condition>/config.ini
        - A commands file (synthetic_commands.txt) listing one command per line, ready to be indexed by SLURM_ARRAY_TASK_ID in the sbatch script
    3_compute_stats.py :
    Computes bootstrap confidence intervals and statistical tests for ccAUGRC differences between methods.

    4_compute_uncertainty_scores.py : implements the computation of total predictive uncertainty and class-conditioned rejection curves for synthetic datasets.

    5_compute_rejection_metrics.py : implements the computation of rejection metrics for synthetic datasets.

    7_paper_figures.py : figures.

    
adni-specific folders:
 - code_data_prep_adni/ folder contains scripts for preprocessing the ADNI dataset, including data cleaning, feature extraction, and formatting for model training.
 -code_evaluate_adni/ folder contains scripts for evaluating model performance on the ADNI dataset, including metrics calculation and visualization of results.



