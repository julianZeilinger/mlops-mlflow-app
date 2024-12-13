import pandas as pd
import mlflow
import os
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_data):
    # Load data
    df = pd.read_csv(input_data)
    
    # Handle missing values if any (for demo, we'll drop them)
    df.dropna(inplace=True)
    
    # Feature scaling
    features = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Save scaler for future use
    scaler_path = "preprocessing/scaler.joblib"
    os.makedirs("preprocessing", exist_ok=True)
    import joblib
    joblib.dump(scaler, scaler_path)
    mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
    
    # Save processed data
    processed_data_path = "processed_data/iris_processed.csv"
    os.makedirs("processed_data", exist_ok=True)
    df.to_csv(processed_data_path, index=False)
    mlflow.log_artifact(processed_data_path, artifact_path="processed_data")