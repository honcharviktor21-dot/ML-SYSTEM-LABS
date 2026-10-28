# Laboratory 4: Reproducible Data Pipeline & Quality Validation Report

**Student:** Viktor Honchar  
**Student ID:** 72014  
**Course:** Machine Learning Laboratory  

---

## 1. Problem Definition & System Objectives
This engineering report documents the construction of a reproducible, leakage-free data pipeline utilizing the UCI Adult dataset. The system accepts 14 demographic and employment features to perform binary classification predicting whether an individual's annual income exceeds $50K. 

The primary engineering constraints met in this pipeline include:
* Prevention of data leakage via train/test splitting prior to imputation or scaling.
* One-command reproducible execution (`python main.py`).
* Comprehensive schema validation and data quality auditing.
* Cryptographic SHA256 hashing of processed dataset artifacts.

---

## 2. Ingestion & Schema Validation Report
* **Raw Data Shape:** 48,842 rows and 15 columns loaded.
* **Range Validation:** Strict boundaries verified with zero violations:
  * `age`: [17, 90]
  * `hours-per-week`: [1, 99]
  * `capital-gain`: [0, 99999]
  * `capital-loss`: [0, 4356]
* **Validation Verdict:** **All range checks passed.**

---

## 3. Label-Quality & Integrity Audit
* **Audit Result:** Found **5 instances** of duplicate features with conflicting labels.

---

## 4. Leakage-Free Cleaning & Transformation Strategy
* **Stratified Splitting:** 70/30 Train/Test split (`random_state=72014`) before cleaning.
* **Duplicate Handling:** Dropped exact duplicates from training split only.
* **Numeric Transformations:** Clipped to 1st/99th percentiles, median-imputed, standard-scaled.
* **Categorical Transformations:** Mode-imputed, one-hot encoded.
* **Artifact Storage:** Preprocessor saved to `results/preprocessor.joblib`. Processed CSV files saved to `data/processed/`.

---

## 5. Feature Engineering & Model Evaluation
Engineered derived features (`capital_net`, `hours_per_age`) improved logistic regression baseline metrics:
* **Accuracy:** 0.8535
* **F1 Score:** 0.6615

---

## 6. Pipeline Reproducibility & Cryptographic Hashing
* **Execution Command:** `python main.py`
* **Configuration File:** `config.yaml`