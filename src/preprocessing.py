import pandas as pd
import mlflow
import click
import os


def preprocess_data(input_data):
    # Load and preprocess data
    df = pd.read_csv(input_data)

    # Save processed data
    os.makedirs("processed_data", exist_ok=True)
    processed_data_path = "processed_data/iris_processed.csv"
    df.to_csv(processed_data_path, index=False)
    mlflow.log_artifact(processed_data_path, artifact_path="processed_data")
