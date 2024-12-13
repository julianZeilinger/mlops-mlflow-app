import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, classification_report
import joblib

def evaluate_model(model_path, test_data_path):
    # Load model
    model = joblib.load(model_path)
    
    # Load test data
    df = pd.read_csv(test_data_path)
    X_test = df.drop("variety", axis=1)
    y_test = df["variety"]
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    for class_label, metrics in report.items():
        if isinstance(metrics, dict):
            mlflow.log_metric(f"{class_label}_precision", metrics["precision"])
            mlflow.log_metric(f"{class_label}_recall", metrics["recall"])
            mlflow.log_metric(f"{class_label}_f1_score", metrics["f1-score"])
    
    # Optionally, log the classification report as a JSON artifact
    import json
    report_path = "evaluation/classification_report.json"
    os.makedirs("evaluation", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact(report_path, artifact_path="evaluation")