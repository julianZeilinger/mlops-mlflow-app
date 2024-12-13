import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

def train_model(input_data, best_params_path):
    # Load data and parameters
    df = pd.read_csv(input_data)
    with open(best_params_path, "r") as f:
        params = yaml.safe_load(f)
    
    X = df.drop("variety", axis=1)
    y = df["variety"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Save and log model
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.joblib"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, artifact_path="models")
    
    # Log the model with MLflow
    mlflow.sklearn.log_model(model, artifact_path="model")
    
    # Save test data for evaluation
    test_data = X_test.copy()
    test_data["variety"] = y_test
    test_data_path = "test_data.csv"
    test_data.to_csv(test_data_path, index=False)
    mlflow.log_artifact(test_data_path, artifact_path="test_data")