import pandas as pd
import yaml
import mlflow
import click
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import os

def hyperparameter_tuning(input_data, param_grid):
    with mlflow.start_run(run_name="Hyperparameter Tuning"):
        # Load data
        df = pd.read_csv(input_data)
        X = df.drop("species", axis=1)
        y = df["species"]

        # Load parameter grid
        with open(param_grid, "r") as f:
            param_grid = yaml.safe_load(f)

        # Perform grid search
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3)
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        mlflow.log_params(best_params)

        # Save best parameters
        os.makedirs("params", exist_ok=True)
        best_params_path = "params/best_params.yaml"
        with open(best_params_path, "w") as f:
            yaml.dump(best_params, f)
        mlflow.log_artifact(best_params_path, artifact_path="params")
