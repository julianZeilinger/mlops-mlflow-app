import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model(params):
    df = pd.read_csv('data/features/iris_features.csv')
    X = df.drop('species', axis=1)
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/model.joblib'
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    # Log the model with MLflow
    mlflow.sklearn.log_model(model, artifact_path="model")

if __name__ == "__main__":
    with open("params/best_params.yaml") as f:
        params = yaml.safe_load(f)['training']
    with mlflow.start_run(run_name="Training"):
        train_model(params)