import pandas as pd
import mlflow
import os

def engineer_features(input_data):
    # Load data
    df = pd.read_csv(input_data)
    
    # Create new features
    df["sepal.area"] = df["sepal.length"] * df["sepal.width"]
    df["petal.area"] = df["petal.length"] * df["petal.width"]
    
    # Save feature data
    feature_data_path = "features/iris_features.csv"
    os.makedirs("features", exist_ok=True)
    df.to_csv(feature_data_path, index=False)
    mlflow.log_artifact(feature_data_path, artifact_path="features")