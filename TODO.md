# PREPROCESSING
- check GENDER variable in adni dataset. currently coded as 1 and 2. But also -4 is present.

# ENSEMBLE METHODS
- How to handle calibration? Do we need to use specific calibration settings for each ml model? 
- check for adni training if the multi index effectively prevents data leakage (visits of the same patient should not be in train and test sets)

- should we compare the results of the rolling window approach to a more traditional approach where we only use the baseline visit of each patient? This is to check if the rolling window approach is actually improving the performance of the models and the quality of the uncertainty measures.

- Include month as a feature: If the model explicitly knows "this is a Month 0 visit" versus "this is a Month 18 visit," it can conditionally adjust its internal thresholds. It learns that a mild ADAS13 score at Month 0 is alarming, but the same score at Month 18 is expected.

# UNCERTAINTY ANALYSIS
- After uncertainty decomposition, what are the next steps?
    - Classification Rejection Curves using the different uncertainty measures (aleatoric, epistemic, total) and compare them to the baseline (random rejection curve).
    - CRCs but discarding proportionally samples classified as 0 and classified as 1 (instead of discarding the most uncertain samples regardless of their predicted class). This is to check if the uncertainty measures are more effective at identifying misclassified samples in one class vs the other.

    - Do we want to perform GMM clustering using clinical data (as we did for pd) to check if the uncertainty measures are more effective at identifying misclassified samples in one cluster vs the other? This is to check if the uncertainty measures are more effective at identifying misclassified samples in one cluster vs the other.

    - Class condition rejection curves used on the entire dataset as an agnostic way.


**Absolute uncertainty is not a fair metric for rejection in imbalanced clinical data. You have to judge a prediction's uncertainty relative to the baseline difficulty of its class. -> Thats why conditional rejection curves are more appropriate for clinical data.**

# ADNI UNCERTANTY ANALYSIS
- should we consider all augmented samples or only real baseline samples for the uncertainty analysis? Atm seems like only baseline samples provide better results in general. 
However if we consider class-condition rejection on the augmented model considering only real bl patients we have an increase in performance.
- Both analyses make sense to me. The first one is more general and the second one is more specific to the task of "baseline visit screening".
    - If we consider all augmented samples, we are essentially evaluating the model's uncertainty across the entire disease progression spectrum. Here we have to take into account weird distributions (?) 
    - if we consider only real baseline samples, we are evaluating the model's uncertainty specifically for the task of "baseline visit screening". This is a more focused analysis that directly assesses the model's performance on the most critical time point for early detection.



# WHAT'S MISSING? 
- The **Clinical Translation** (How does a doctor actually use this?)
You have proven the math works, but clinical informatics papers require a "Translational Workflow." If a hospital buys your algorithm tomorrow, how does the UI/UX actually work for the physician? 
    - What to add: A short paragraph defining the "Wait-and-See" Protocol. If a patient's uncertainty score exceeds the CCRC threshold, the AI does not output a diagnosis. Instead, it outputs: "Inconclusive: High Aleatoric Noise. Defer to clinical judgment or schedule a 6-month follow-up."



# other things (gemini suggestions):

# Project TODOs: Pipeline Fixes & Manuscript Defenses

## Part 1: Codebase & Pipeline Adjustments
These are the critical structural fixes required in your Python scripts to ensure mathematical validity before running the final results.

- [x] **Prevent Ensemble "Double Counting":**
  - *Action:* Modify the `build_ensemble_csv` aggregation logic. Ensure that Total Entropy ($H_{Total}$), Aleatoric ($C_{Aleatoric}$), and Epistemic ($I_{Epistemic}$) uncertainties are calculated **only** using the independent base models (Random Forest, Extra Trees, XGBoost, Logistic Regression, SVC). 
  - *Reason:* Including Meta-models (Voting, Stacking) in the variance calculation forces a false mathematical consensus, which artificially depresses the true Model Disagreement ($I_{Epistemic}$).
- [ ] **Document AUPRC Refitting:**
  - *Action:* Add comments in the code and explicitly state in the manuscript methodology that GridSearch refitting was optimized using Area Under the Precision-Recall Curve (AUPRC).
  - *Reason:* AUPRC is the mathematically optimal metric for highly imbalanced datasets (like the MS cohort) as it avoids the distortion caused by massive True Negative counts.

## Part 2: Manuscript & Methodological Defenses
These are the critical narrative points that must be explicitly addressed in the Discussion or Limitations section of your paper to preempt reviewer critiques.

- [ ] **Address the "Predicted Class" Dependency:**
  - *Action:* Acknowledge that CCRC conditions rejection on the *Consensus Predicted Risk* (since the true label is unknown). 
  - *Defense:* Defend this by highlighting that the heterogeneous ensemble actively mitigates the risk of a model being "confidently wrong." Because diverse architectures disagree on difficult cases, Epistemic Uncertainty rises, successfully flagging the patient for rejection even if one individual model is overly confident.
- [ ] **Defend "Flatlining" Metrics via Clinical Harm Reduction:**
  - *Action:* When discussing the MS dataset, explicitly address the fact that the MCC flatlines around 0.28-0.30 during rejection.
  - *Defense:* Pivot the argument from statistical ratios to raw clinical utility using the Confusion Matrix tracking. Prove that a flatlining MCC under CCRC is a success because it prevents dozens of healthy patients from receiving toxic treatments (avoiding False Positives) without artificially erasing the minority disease class.
- [ ] **Explain the Inadequacy of Standard Calibration for Rejection:**
  - *Action:* Preempt the question: *"Why not just calibrate the probabilities and use standard global rejection?"*
  - *Defense:* Explain that standard calibration (Platt Scaling, Isotonic Regression) maps predictions to true empirical likelihoods, but it *does not* fix Aleatoric asymmetry. A perfectly calibrated model will still output maximum uncertainty (~0.5) for the hardest cases. If those cases belong predominantly to the disease class, standard global rejection will still cause minority-class collapse. CCRC is required regardless of base calibration.
- [ ] **Define the Rejection Threshold Selection Process:**
  - *Action:* Clarify how a real-world clinic would choose the "25%" or "30%" threshold highlighted in the results.
  - *Defense:* Explain that the specific rejection threshold is selected *retrospectively* on a validation set based on a specific clinic's risk tolerance and resource constraints (e.g., MRI bandwidth). Emphasize that CCRC ensures the diagnostic balance remains mathematically intact regardless of which operational threshold the hospital ultimately chooses.