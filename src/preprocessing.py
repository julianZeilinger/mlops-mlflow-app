import pandas as pd
import yaml
import mlflow
import os

def preprocess_data(params):
    df = pd.read_csv('data/raw/iris.csv')
    # No preprocessing needed for this dataset
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/iris_processed.csv', index=False)
    mlflow.log_artifact('data/processed/iris_processed.csv')

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)['preprocessing']
    with mlflow.start_run(run_name="Preprocessing"):
        preprocess_data(params)