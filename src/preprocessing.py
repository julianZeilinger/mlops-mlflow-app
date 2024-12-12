import pandas as pd
import mlflow
import click
import os


@click.command()
@click.option("--input-data", type=str, required=True, help="Path to validated data")
def preprocess_data(input_data):
    with mlflow.start_run(run_name="Preprocessing"):
        # Load and preprocess data
        df = pd.read_csv(input_data)

        # Save processed data
        os.makedirs("processed_data", exist_ok=True)
        processed_data_path = "processed_data/iris_processed.csv"
        df.to_csv(processed_data_path, index=False)
        mlflow.log_artifact(processed_data_path, artifact_path="processed_data")


if __name__ == "__main__":
    preprocess_data()