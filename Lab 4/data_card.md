# Data Card: Processed UCI Adult Dataset

## 1. Dataset Overview
* **Source:** UCI Adult Dataset via scikit-learn `fetch_openml(name='adult', version=2)`
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Size:** 
  * Raw dataset: 48,842 rows and 15 columns
  * Processed dataset split 70/30 stratified train/test
* **Pipeline Engineer:** Viktor Honchar (ID: 72014)

## 2. Dataset Pipeline & Transformation
* **Splitting Strategy:** 70/30 stratified split before cleaning to eliminate data leakage.
* **Missing Values:** Categorical features use mode imputation; numeric features use median imputation (fitted on train set only).
* **Outlier Handling:** Numeric features clipped to 1st and 99th percentiles derived from the training set.
* **Feature Engineering:** 
  * `capital_net` = `capital-gain` - `capital-loss`
  * `hours_per_age` = `hours-per-week` / (`age` - 16)

## 3. Validation & Audits
* **Schema Validation Checks:** Range enforcement on age (17-90), hours (1-99), capital-gain (0-99999), capital-loss (0-4356).
* **Label Quality Audit:** Conducted duplicate-conflict audit to flag duplicate feature rows with conflicting income labels.

## 4. Known Biases & Privacy Concerns
* **Known Biases:** Historical 1994 US Census data contains outdated demographic structures and socio-economic biases.
* **Privacy Concerns:** High-granularity demographic feature combinations could allow re-identification attacks in sparse demographic subsets.