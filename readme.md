## Research Concept
Standard machine learning evaluation assumes that all predictive errors carry equal weight and that predictive uncertainty can be evaluated independently of the dataset's class prior. In clinical practice, this assumption fails catastrophically. To safely translate predictive models into actionable Clinical Decision Support Systems (CDSS), we implemented a four-step uncertainty-aware workflow:

**1. Uncertainty Quantification (Bypassing SOTA Entanglement)**

Rather than relying on the absolute purity of mathematically entangled Information-Theoretic metrics (Aleatoric vs. Epistemic uncertainty), the workflow calculates Total Predictive Uncertainty ($H_{total}$). This captures both inherent biological noise and out-of-distribution anomalies while remaining robust in finite-data clinical regimes.

**2. Class-Conditioned Rejection (CCRC)**

To prevent global uncertainty thresholds from acting as a naive minority-class deletion filter, the workflow utilizes Class-Conditioned Rejection Curves (CCRC). By evaluating the ordinal ranking of $H_{total}$ strictly within localized predicted classes, CCRC applies proportional rejection thresholds. This neutralizes prior-driven bias and safely defers ambiguous patients without mathematically erasing the disease class.

**3. Clinical Evaluation via Confusion Matrix Dynamics**

The workflow shifts the evaluation paradigm away from global statistical ratios (e.g., MCC or AUC) toward raw confusion matrix dynamics. Because proportional rejection simultaneously removes False Positives (preventing toxic overtreatment) and False Negatives (preventing false reassurance), global ratios often mathematically plateau. The workflow reframes these "flatlining" metrics not as algorithmic failures, but as the expected signature of a safely calibrated clinical model.

**4. Cross-Disease Validation**

To ensure the pipeline is robust across varying degrees of biological noise and class imbalance, the workflow evaluates the CCRC algorithm across three distinct neuro-predictive scenarios:

* Parkinson’s Disease (3-Year Dyskinesia): A moderately imbalanced cohort testing the algorithm's ability to stabilize asymmetric noise.

* Multiple Sclerosis (Progression Independent of Relapse Activity): A severely imbalanced cohort testing CCRC as a mandatory rescue mechanism against systemic minority-class collapse.

* Alzheimer’s Disease (36-Month MCI to AD Conversion): A balanced cohort testing CCRC's ability to act as a strict calibration enforcer.
* 

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

## Repository Structure & Usage

*(specific repository structure, installation instructions, and script usage examples will be added here)*
