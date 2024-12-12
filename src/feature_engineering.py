import pandas as pd
import yaml
import mlflow
import os

def engineer_features(params):
    df = pd.read_csv('data/processed/iris_processed.csv')
    # Example feature engineering: add a new feature
    df['sepal_area'] = df['sepal_length'] * df['sepal_width']
    df['petal_area'] = df['petal_length'] * df['petal_width']

    os.makedirs('data/features', exist_ok=True)
    df.to_csv('data/features/iris_features.csv', index=False)
    mlflow.log_artifact('data/features/iris_features.csv')

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)['feature_engineering']
    with mlflow.start_run(run_name="Feature Engineering"):
        engineer_features(params)