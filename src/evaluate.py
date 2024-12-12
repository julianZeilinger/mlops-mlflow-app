import pandas as pd
import mlflow
import click
from sklearn.metrics import accuracy_score
import joblib


@click.command()
@click.option("--model-path", type=str, required=True, help="Path to trained model")
@click.option("--test-data", type=str, required=True, help="Path to feature data")
def evaluate_model(model_path, test_data):
    with mlflow.start_run(run_name="Evaluation"):
        # Load model and data
        model = joblib.load(model_path)
        df = pd.read_csv(test_data)
        X = df.drop("species", axis=1)
        y = df["species"]

        # Evaluate model
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        mlflow.log_metric("accuracy", accuracy)


if __name__ == "__main__":
    evaluate_model()