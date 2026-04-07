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

- [ ] **Calculate the IT Decomposition:** Extract Total ($H$), Aleatoric ($C$), and Epistemic ($I$) uncertainty via standard Information-Theoretic formulas, recognizing their mathematical entanglement.
- [ ] **Generate Aleatoric vs. Epistemic Phase Planes:** Plot $C$ (x-axis) vs $I$ (y-axis) for the cohorts. *Goal:* Visually prove that $I$ is universally low (< 0.1) and $H$ is dominated by $C$ (biological noise).
- [ ] **Implement the "Two-Tiered" Rejection Strategy:**
    - **The Mechanism:** Use $H_{Total}$ for the actual CCRC sorting and rejection. (It is computationally pure, bypasses entanglement critiques, and acts as a clinical safety net for unseen Out-of-Distribution/Epistemic spikes).
    - **The Explanation:** Use $C_{Aleatoric}$ and the Phase Plane to *explain* the rejection in the paper (proving rejection is driven by asymmetric biological noise).
- [ ] **Class-Conditioned Rejection Curves (CCRC):** Run CCRC tracking confusion matrix dynamics. Compare to Standard Global Rejection and Random Rejection baselines.
- [x] **ADNI Sample Scope Analysis:** Run uncertainty analysis on *all* augmented samples vs. *only* real baseline samples.
    - *Note:* All augmented evaluates the full disease progression spectrum. Baseline-only evaluates the specific clinical task of "baseline screening." Compare both.
- [ ] **Statistical Rejection Audit (Mann-Whitney U):** Run non-parametric tests comparing the uncertainty profiles ($H, C, I$) of the "Kept" vs. "Removed" cohorts to prove CCRC targets the statistical tails of *both* classes.

## PART 3: MANUSCRIPT NARRATIVE & DEFENSES
Narrative points that must be explicitly addressed in the Discussion or Limitations sections to preempt SOTA critiques.

- [ ] **The "SOTA Entanglement" Defense (Wimmer, de Jong):** Explicitly acknowledge that IT metrics ($H, C, I$) are mathematically entangled and flawed as absolute estimators. Argue that this entanglement is *exactly* why standard Global Rejection acts as a catastrophic minority-class deletion filter. 
- [ ] **The "Ordinal vs. Absolute" Defense (Why CCRC works):** Defend CCRC by explaining it shifts reliance from the *absolute mathematical purity* of the uncertainty score to its *ordinal ranking* within a localized clinical trajectory (grading on a curve).
- [ ] **The "0.5 Illusion" Defense:** Preempt the critique "Why not just reject based on distance to 0.5?" Explain that $|p - 0.5|$ collapses aleatoric (inherent ambiguity) and epistemic (model failure) variance into the same bucket, and ignores calibrated disease prevalence priors.
- [ ] **Defend "Flatlining" Metrics via Clinical Harm Reduction:** Explicitly address the flatlining MCC (e.g., ~0.28-0.30 in MS). Pivot from statistical ratios to raw Confusion Matrix counts. Prove that flatlining is the mathematical signature of safely deferring equal proportions of False Positives and False Negatives.
- [ ] **The "Predicted Class" Dependency:** Acknowledge that CCRC relies on the *predicted* risk class. Defend this by stating the heterogeneous ensemble mitigates "confident errors" (Epistemic $I$ spikes when diverse architectures disagree, ensuring the patient is still rejected).
- [ ] **Inadequacy of Standard Calibration for Rejection:** Explain that even perfectly calibrated models output maximum uncertainty for the hardest cases. If biological noise is asymmetric, standard calibration + global rejection still causes minority-class collapse. CCRC is required *on top* of base calibration.
- [ ] **The Clinical Translation (Wait-and-See Protocol):** Add a UI/UX paragraph for the physician. Explain that if a patient exceeds the CCRC threshold, the AI does not force a diagnosis. It outputs: *"Inconclusive: High Aleatoric Noise. Defer to clinical judgment or schedule advanced testing/follow-up."*
- [ ] **Define Threshold Selection for Hospitals:** Clarify that a real-world clinic chooses the specific rejection cutoff (e.g., 25%) retrospectively on a validation set based on their specific risk tolerance and resource bandwidth (e.g., MRI availability).
