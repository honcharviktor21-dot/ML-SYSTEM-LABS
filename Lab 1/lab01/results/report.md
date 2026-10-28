# Lab 01 Final Report: Baseline Training & Profiling

## 1. Environment & Package Versions
- **Python**: 3.11.9
- **Core ML**: `numpy==1.26.4`, `pandas==2.2.2`, `scikit-learn==1.5.0`, `torch==2.3.1+cpu`
- **Tracking & Web**: `mlflow==2.14.1`, `fastapi==0.111.0`, `codecarbon==2.4.2`

---

## 2. Baseline Model Performance
Evaluation metrics on the Breast Cancer Wisconsin dataset (20% test split):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9825** | **0.9861** | **0.9861** | **0.9861** | **0.9954** |
| **Decision Tree** | 0.9123 | 0.9559 | 0.9028 | 0.9286 | 0.9157 |
| **Random Forest** | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9937 |

---

## 3. Resource & Performance Profiling

| Model | Execution Time (s) | RAM Used (MB) | Emissions (kg CO2) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | **0.0128** | ~0.332 | 2.40e-07 |
| **Random Forest** | 0.1226 | ~0.285 | 1.18e-06 |

---

## 4. Key Takeaways
1. **Logistic Regression** outperformed tree-based models on this dataset, achieving the highest Accuracy (**98.25%**) and the top **ROC-AUC (0.9954)**.
2. The linear model executed nearly **10x faster** than Random Forest with lower total emissions, making it the most efficient choice for deployment.