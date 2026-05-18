## Data Preprocessing Pipeline

This notebook contains the preprocessing pipeline to transform raw, fragmented ADNI longitudinal tables into tabular datasets. 

The pipeline operates by first generating a universal **Extended Longitudinal Cohort** that maps the entire temporal history and rolling-window outcomes for all patients. Once this master dataset is built, specific analytic sub-cohorts (such as the static baseline cohort used in our manuscript) are generated simply by filtering the extended cohort.

---

### Phase 1: Building the Extended Longitudinal Cohort
*These steps process the raw ADNI tables (DXSUM, PTDEMOG, ADAS, etc.) into a continuous temporal format with localized outcomes.*

#### 1. Data Standardization & Fusion
* **Unified Keys:** Standardizes `PTID` to `subject_id` and `VISCODE` to `visit` across all ADNI tables.
* **Temporal Mapping:** Converts visit codes (bl, m06, m12, etc.) into a continuous `month` integer to enable chronological sorting.
* **Feature Fusion:** Fuses longitudinal clinical scales including ADAS-Cog 13, MMSE, CDR-SB, and FAQ. Static demographics (Age, Gender, Education, and APOE4 status) are extracted from the baseline visit and broadcasted across all subsequent visits.

#### 2. Intra-Visit Consolidation (SC/BL)
To maximize feature completeness at the patient's entry point, the script performs an intra-visit merge of **Screening (sc)** and **Baseline (bl)** data:
* Data points from these two visits (typically occurring within weeks of each other) are treated as a single $t=0$ timepoint.
* **Stitching:** Missing values in the baseline visit are back-filled or forward-filled using the screening visit (e.g., if MMSE was captured at SC but ADAS was captured at BL).
* Redundant screening rows are purged post-merge to prevent data duplication.

#### 3. Diagnosis Gatekeeper (MCI Filter)
To ensure the cohort represents a specific clinical triage scenario for Mild Cognitive Impairment, the script applies a strict "MCI-only" filter:
* Every valid row must correspond to a visit where the patient has a confirmed clinical diagnosis of **MCI** (Stable MCI, Early MCI, or Late MCI).
* CN (Cognitively Normal) or AD (Alzheimer's Disease) diagnoses are excluded as starting points.

#### 4. Rolling-Window Labeling (Trajectory Outcomes)
For every valid MCI visit, the script scans the subsequent longitudinal history to assign local prognostic labels.
* **36-Month Horizon:** The script looks ahead up to 36 months from the current visit.
* **Class 1 (Progressor):** Assigned if a diagnosis of AD (Conversion) is recorded within the window.
* **Class 0 (Stable):** Assigned if the patient is confirmed to remain MCI for the full window duration. (Right-censored patients who drop out before the window closes without converting are flagged/excluded).

---

### Phase 2: Extracting the Manuscript-Specific Baseline Cohort
*The following steps filter the Extended Longitudinal Cohort down to the specific static $N=314$ feature matrix utilized in the published manuscript to evaluate the uncertainty framework.*

#### 5. Baseline Filtering & Cross-Sectional Restriction
Rather than using the continuous trajectories, the dataset is strictly filtered to simulate a real-world predictive scenario at the time of initial triage:
* **Filter to Baseline:** Only rows corresponding to the initial $t=0$ (Baseline) visit are retained. All subsequent longitudinal visits are discarded from the feature space.
* This yields the strict 36-month prognostic cohort: 170 MCI-to-AD converters and 144 Stable MCI patients.

#### 6. Missing Data & Scaling (Leak-Proof)
To prepare the final tabular datasets for machine learning optimization without introducing data leakage:
* **Filtration:** Features with a missing data proportion exceeding 50% across the analytic cohort are excluded.
* **Imputation:** Missing baseline values are imputed using the median (for continuous variables) or mode (for categorical variables). *Note: In cross-validation settings, these statistics are computed strictly on the training folds.*
* **Scaling:** All continuous numerical features undergo $z$-score normalization (standardized to a mean of 0 and standard deviation of 1) to ensure uniform weight distribution across the machine learning classifiers.