import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score
import joblib

def evaluate_model():
    df = pd.read_csv('data/features/iris_features.csv')
    X = df.drop('species', axis=1)
    y = df['species']

    model = joblib.load('models/model.joblib')
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    mlflow.log_metric('accuracy', accuracy)

if __name__ == "__main__":
    with mlflow.start_run(run_name="Evaluation"):
        evaluate_model()