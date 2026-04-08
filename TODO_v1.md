# Project TODOs: Pipeline Fixes, Analysis, & Manuscript Defenses

## PART 1: PREPROCESSING & PIPELINE ADJUSTMENTS
These are the critical structural fixes required in the Python scripts to ensure mathematical validity before running the final results.

- [x] **Check GENDER variable in ADNI dataset:** Currently coded as 1 and 2. But -4 is also present (missing/unknown). Handle missing values appropriately.
- [x] **Refit ADNI models:** Rerun the grid search/training using the newly preprocessed ADNI data (accounting for the fixed gender/education variables).
- [x] **Verify ADNI Data Leakage Prevention:** Ensure the GroupShuffleSplit (or MultiIndex logic) effectively prevents visits from the same patient appearing in both train and test sets.
- [x] **Rolling Window vs. Baseline-Only Comparison:** Test if the rolling window approach actually improves model performance and uncertainty quality compared to a traditional "baseline visit only" approach.
- [?] **Include 'Month' as a Feature:** Test explicitly feeding "Month X" to the model. *Hypothesis:* The model can conditionally adjust its internal thresholds (e.g., learning that a mild ADAS13 score at Month 0 is alarming, but expected at Month 18).
- [?] **Ensemble Aggregation Rule - Strict Calibration:** Ensure all base models (RF, SVC, XGBoost, etc.) are strictly calibrated (Platt/Isotonic) *before* their probabilities are averaged. Uncalibrated scores from different hypothesis spaces cannot be directly averaged.
- [x] **Prevent Ensemble "Double Counting":** Modify the `build_ensemble_csv` logic. Ensure $H_{Total}$, $C_{Aleatoric}$, and $I_{Epistemic}$ are calculated using ONLY the independent base models. Exclude Meta-models (Voting, Stacking) to prevent a forced mathematical consensus that artificially depresses Epistemic Uncertainty.
- [?] **Refit Metric Evaluation (AUPRC vs. ROC-AUC):** - Document that AUPRC was used for MS/NeuroART (highly imbalanced).
    - *Decision needed:* ADNI is more balanced. Should we use ROC-AUC or MCC for ADNI grid-search refitting instead of AUPRC?

## PART 2: UNCERTAINTY ANALYSIS & PLOTTING
Tasks related to extracting metrics, running rejection curves, and generating the necessary plots for the manuscript.

- [x] **Calculate the IT Decomposition:** Extract Total ($H$), Aleatoric ($C$), and Epistemic ($I$) uncertainty via standard Information-Theoretic formulas, recognizing their mathematical entanglement. (Note: We extract $C$ and $I$ not to use them for clinical thresholds, but to expose their mathematical entanglement and justify abandoning them for rejection.)

- [x] **Generate Aleatoric vs. Epistemic Phase Planes:** Plot $C$ (x-axis) vs $I$ (y-axis) for the cohorts. *Goal:* Visually prove that $I$ is universally low (< 0.1) and $H$ is dominated by $C$. Critically, use this plot to demonstrate the "Suppression Effect" (de Jong, 2024): show how massive $C$ (biological noise) mathematically crushes $I$ for Class 1, proving disentangled $I$ is a dangerous metric for detecting model failure.

- [x] **Implement the "Two-Tiered" Rejection Strategy:**
    - **The Mechanism:** Use ONLY $H_{Total}$ for the actual CCRC sorting and rejection. (It is computationally pure, bypasses entanglement critiques, and acts as a clinical safety net).
    - **The Explanation:** Use $C_{Aleatoric}$ and the Phase Plane in the text to explain why $H_{Total}$ is behaving the way it is (proving the Total Uncertainty is driven by asymmetric biological noise, while Epistemic is mathematically hidden).

- [ ] **Class-Conditioned Rejection Curves (CCRC):** Run CCRC tracking confusion matrix dynamics. Compare to Standard Global Rejection and Random Rejection baselines.

- [x] **ADNI Sample Scope Analysis:** Run uncertainty analysis on *all* augmented samples vs. *only* real baseline samples.
    - *Note:* All augmented evaluates the full disease progression spectrum. Baseline-only evaluates the specific clinical task of "baseline screening." Compare both.

- [ ] **Statistical Rejection Audit (Mann-Whitney U):** Run non-parametric tests comparing the uncertainty profiles ($H, C, I$) of the "Kept" vs. "Removed" cohorts to prove CCRC targets the statistical tails of *both* classes.

## PART 3: MANUSCRIPT NARRATIVE & DEFENSES
Narrative points that must be explicitly addressed in the Discussion or Limitations sections to preempt SOTA critiques.

- [ ] **The "SOTA Entanglement" Defense (Wimmer, de Jong):**  Explicitly acknowledge that IT metrics ($H, C, I$) are mathematically flawed. Specifically detail the "Suppression Effect": because $H = C + I$, high Aleatoric noise ($C$) mathematically suppresses Epistemic uncertainty ($I$). Argue that this creates a "clinical blindspot": if doctors rely on $I$ to detect model failure, they will miss highly ambiguous Class 1 patients because their $I$ score is artificially crushed by biological noise.

- [ ] **The " $H_{Total}$ Clinical Safety Net" Defense:** Defend the use of $H_{Total}$ by arguing that because $I$ is suppressed in the hardest cases, relying on disentangled metrics is clinically dangerous. $H_{Total}$ is the only mathematically pure metric that guarantees ambiguous patients are caught.

- [ ] **The "Ordinal vs. Absolute" Defense (Why CCRC works):** Defend CCRC by explaining it shifts reliance from the *absolute mathematical purity* of the uncertainty score to its *ordinal ranking* within a localized clinical trajectory (grading on a curve).

- [ ] **The "0.5 Illusion" Defense:** Preempt the critique "Why not just reject based on distance to 0.5?" Explain that $|p - 0.5|$ collapses aleatoric (inherent ambiguity) and epistemic (model failure) variance into the same bucket, and ignores calibrated disease prevalence priors.

- [ ] **Defend "Flatlining" Metrics via Clinical Harm Reduction:** Explicitly address the flatlining MCC (e.g., ~0.28-0.30 in MS). Pivot from statistical ratios to raw Confusion Matrix counts. Prove that flatlining is the mathematical signature of safely deferring equal proportions of False Positives and False Negatives.

- [ ] **The "Predicted Class" Dependency:** Acknowledge that CCRC relies on the *predicted* risk class. Defend this by stating the heterogeneous ensemble mitigates "confident errors" (Epistemic $I$ spikes when diverse architectures disagree, ensuring the patient is still rejected).

- [ ] **Inadequacy of Standard Calibration for Rejection:** Explain that even perfectly calibrated models output maximum uncertainty for the hardest cases. If biological noise is asymmetric, standard calibration + global rejection still causes minority-class collapse. CCRC is required *on top* of base calibration.

- [ ] **The Clinical Translation (Wait-and-See Protocol):** Add a UI/UX paragraph for the physician. Explain that if a patient exceeds the CCRC threshold, the AI does not force a diagnosis. It outputs: *"Inconclusive: High Aleatoric Noise. Defer to clinical judgment or schedule advanced testing/follow-up."*

- [ ] **Define Threshold Selection for Hospitals:** Clarify that a real-world clinic chooses the specific rejection cutoff (e.g., 25%) retrospectively on a validation set based on their specific risk tolerance and resource bandwidth (e.g., MRI availability).

- [ ] **The "Unobserved Covariates" Defense:** Clarify the definition of Aleatoric uncertainty in a medical context. Explicitly state that what the model perceives as irreducible noise ($C$) is often just the limitation of its finite feature space. Defend the rejection mechanism as a necessary "clinical escalation" to a human physician who possesses a wider observational bandwidth and can interpret context beyond the recorded data. --> **idea of Model-Bounded Aleatoric Noise**

- **A Note on Uncertainty Terminology** In this work, we extract standard Information-Theoretic (IT) metrics commonly referred to in the machine learning literature as 'Aleatoric' ($C$) and 'Epistemic' ($I$) uncertainty. However, aligning with recent critiques by Bickford-Smith & van der Wilk (2025), we recognize that this traditional dichotomy is often insufficiently expressive and conflates distinct mathematical quantities.
Therefore, rather than treating these metrics as pure, absolute representations of 'chance' and 'knowledge', we operationalize them strictly according to their tangible clinical and statistical equivalents. We treat the Aleatoric estimator ($C$) as a proxy for biological statistical dispersion (the inherent signal-to-noise ratio of a specific clinical trajectory). Conversely, we treat the Epistemic estimator ($I$) strictly as a measure of procedural parameter uncertainty (disagreement across our heterogeneous ensemble due to finite training data). By mapping these ambiguous terms to precise statistical behaviors, we evaluate their downstream impact on clinical triage more rigorously. **

- Absolute uncertainty is not a fair metric for rejection in imbalanced clinical data. You have to judge a prediction's uncertainty relative to the baseline difficulty of its class. -> Thats why conditional rejection curves are more appropriate for clinical data.
