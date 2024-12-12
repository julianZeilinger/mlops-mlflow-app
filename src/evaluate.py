import pandas as pd
import mlflow
import click
from sklearn.metrics import accuracy_score
import joblib

def evaluate_model(model_path, test_data):
    # Load model and data
    model = joblib.load(model_path)
    df = pd.read_csv(test_data)
    X = df.drop("species", axis=1)
    y = df["species"]

    # Evaluate model
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    mlflow.log_metric("accuracy", accuracy)

