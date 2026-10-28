import os
import time
import json
import psutil
from codecarbon import EmissionsTracker
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def profile_execution():
    # Завантаження даних
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LogisticRegression": (LogisticRegression(random_state=42), X_train_scaled, X_test_scaled),
        "RandomForest": (RandomForestClassifier(random_state=42), X_train, X_test)
    }

    profile_results = {}
    process = psutil.Process(os.getpid())

    # Налаштування директорії для результатів
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(current_dir, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    for name, (model, X_tr, X_te) in models.items():
        # Трекер CodeCarbon для вимірювання енергоспоживання
        tracker = EmissionsTracker(project_name=f"lab01_{name}", output_dir=results_dir, save_to_file=False)
        tracker.start()

        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        start_time = time.perf_counter()

        # Навчання та інференс
        model.fit(X_tr, y_train)
        _ = model.predict(X_te)

        end_time = time.perf_counter()
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        emissions = tracker.stop()

        profile_results[name] = {
            "execution_time_sec": round(end_time - start_time, 6),
            "memory_used_mb": round(max(0, mem_after - mem_before), 4),
            "emissions_kg_co2": round(float(emissions) if emissions else 0.0, 8)
        }

    output_file = os.path.join(results_dir, "profiling_results.json")
    with open(output_file, "w") as f:
        json.dump(profile_results, f, indent=4)

    print("Profiling completed successfully:")
    print(json.dumps(profile_results, indent=4))

if __name__ == "__main__":
    profile_execution()