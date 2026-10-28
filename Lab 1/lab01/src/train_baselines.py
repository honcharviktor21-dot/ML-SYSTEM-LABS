import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_evaluate():
    # 1. Завантаження датасету
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # 2. Розділення даних (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Масштабування ознак
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Моделі для тренування
    models = {
        "LogisticRegression": LogisticRegression(random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42)
    }

    results = {}

    # 5. Навчання та оцінка моделей
    for name, model in models.items():
        if name == "LogisticRegression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds)), 4),
            "recall": round(float(recall_score(y_test, preds)), 4),
            "f1_score": round(float(f1_score(y_test, preds)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4)
        }

    # 6. Збереження результатів у json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(current_dir, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, "baseline_metrics.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print("Baseline metrics saved successfully:")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    train_and_evaluate()