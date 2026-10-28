import os
import pandas as pd
from sklearn.datasets import fetch_openml

def ingest_data(config):
    print("--- 1. Ingestion Step ---")
    adult = fetch_openml(name=config['pipeline']['dataset']['name'], 
                         version=config['pipeline']['dataset']['version'], 
                         as_frame=True)
    df = adult.frame
    df = df.rename(columns={'class': 'income'})
    
    os.makedirs(os.path.dirname(config['pipeline']['paths']['raw_data']), exist_ok=True)
    df.to_csv(config['pipeline']['paths']['raw_data'], index=False)
    
    print(f"Raw Data saved to {config['pipeline']['paths']['raw_data']}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
    return df