import pandas as pd
import mlflow
import mlflow.sklearn
import click
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

def train_model(input_data, best_params):
    with mlflow.start_run(run_name="Training"):
        # Load data and parameters
        df = pd.read_csv(input_data)
        with open(best_params, "r") as f:
            params = yaml.safe_load(f)

        X = df.drop("species", axis=1)
        y = df["species"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Save and log model
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")
        mlflow.sklearn.log_model(model, artifact_path="model")