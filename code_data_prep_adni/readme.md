## Data Preprocessing Pipeline

The preprocessing pipeline transforms raw, fragmented ADNI longitudinal tables into a high-quality, clinical-grade training dataset for MCI-to-AD progression prediction.

### 1. Data Standardization & Fusion
* **Unified Keys:** Standardizes `PTID` to `subject_id` and `VISCODE` to `visit` across all ADNI tables (DXSUM, PTDEMOG, ADAS, and My_Table).
* **Temporal Mapping:** Converts visit codes (bl, m06, m12, etc.) into a continuous `month` integer to enable chronological sorting.
* **Feature Fusion:** Fuses longitudinal clinical scales including ADAS-Cog 13, MMSE, CDR-SB, and FAQ. Static demographics (Age, Gender, Education, and APOE4 status) are extracted from the baseline visit and broadcasted across all subsequent visits.

### 2. Intra-Visit Consolidation (SC/BL)
To maximize feature completeness at the study "starting line," the script performs an intra-visit merge of **Screening (sc)** and **Baseline (bl)** data:
* Data points from these two visits (typically occurring within weeks of each other) are treated as a single $t=0$ timepoint.
* **Stitching:** Missing values in the baseline visit are back-filled or forward-filled using the screening visit (e.g., if MMSE was captured at SC but ADAS was captured at BL).
* Redundant screening rows are purged post-merge to prevent data duplication.

### 3. Diagnosis Gatekeeper (MCI Filter)
To ensure the model is a specific triage tool for Mild Cognitive Impairment, the script applies a strict "MCI-only" filter:
* Every training row (starting line) must correspond to a visit where the patient has a confirmed clinical diagnosis of **MCI** (Stable MCI, Early MCI, or Late MCI).
* Visits where the patient is Cognitively Normal (CN) or has already progressed to Alzheimer’s Disease (AD) are excluded as prediction starting points.

### 4. Rolling-Window Labeling (24-Month Horizon)
The dataset is expanded using a rolling-window approach to capture the clinical evolution of the disease:
* **Window:** For every valid MCI visit, the script scans the subsequent **24 months** of the patient's history.
* **Class 1 (Progressor):** Labeled if any diagnosis of AD (Conversion) is recorded within the 24-month window.
* **Class 0 (Stable MCI):** Labeled only if the patient is confirmed to remain MCI for the full 24-month duration (Right-Censoring handled).

### 5. Leak-Proof Imputation Strategy
To handle missing clinical values without introducing data leakage:
* **Step A (LOCF):** Last Observation Carried Forward is applied within each patient’s timeline. This is temporally safe as it only uses past data to fill the present.
* **Step B (Median Fallback):** Any remaining missing values (primarily at the true baseline) are filled using the **median of the Training Set only**. This prevents information from the Test Set from leaking into the training process.

### 6. Group-Aware Validation
To prevent "Patient Leakage," the data is split using a **GroupShuffleSplit**:
* All longitudinal rows belonging to a single patient are kept together.
* A patient’s entire history is assigned to either the Training set or the Test set, ensuring the model cannot "memorize" specific individuals to inflate accuracy.