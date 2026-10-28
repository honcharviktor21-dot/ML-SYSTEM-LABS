import os
import json
import hashlib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(config):
    print("--- 4. Evaluation & Hashing Step ---")
    train_data = pd.read_csv(config['pipeline']['paths']['train_data'])
    test_data = pd.read_csv(config['pipeline']['paths']['test_data'])

    X_train, y_train = train_data.drop('income', axis=1), train_data['income']
    X_test, y_test = test_data.drop('income', axis=1), test_data['income']

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = {
        "engineered_model": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred)), 4)
        }
    }
    
    print(f"Model Performance: Accuracy={metrics['engineered_model']['accuracy']}, F1={metrics['engineered_model']['f1_score']}")

    os.makedirs(os.path.dirname(config['pipeline']['paths']['metrics']), exist_ok=True)
    with open(config['pipeline']['paths']['metrics'], 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {config['pipeline']['paths']['metrics']}")

    dataset_string = X_train.to_csv(index=False).encode()
    dataset_hash = hashlib.sha256(dataset_string).hexdigest()
    print(f"SHA256 Hash of processed training data: {dataset_hash}\n")