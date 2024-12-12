import pandas as pd
import mlflow
import click
import os

def engineer_features(input_data):
    with mlflow.start_run(run_name="Feature Engineering"):
        # Load data and engineer features
        df = pd.read_csv(input_data)
        df["sepal_area"] = df["sepal_length"] * df["sepal_width"]
        df["petal_area"] = df["petal_length"] * df["petal_width"]

        # Save feature data
        os.makedirs("features", exist_ok=True)
        feature_data_path = "features/iris_features.csv"
        df.to_csv(feature_data_path, index=False)
        mlflow.log_artifact(feature_data_path, artifact_path="features")
