import yaml
from src.ingestion import ingest_data
from src.validation import validate_data
from src.preprocessing import preprocess_data
from src.evaluation import evaluate_model

if __name__ == "__main__":
    print("Initializing Data Pipeline...\n")
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    df = ingest_data(config)
    validate_data(df, config)
    preprocess_data(df, config)
    evaluate_model(config)

    print("Pipeline Execution Complete. All artifacts saved.")