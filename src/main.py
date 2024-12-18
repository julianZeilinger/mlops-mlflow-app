import mlflow
from data_validation import validate_data
from preprocessing import preprocess_data
from feature_engineering import engineer_features
from hyperparameter_tuning import hyperparameter_tuning
from train import train_model
from evaluate import evaluate_model

def main():
    print("MLflow Version:", mlflow.version.VERSION)
    print("Current Tracking URI:", mlflow.get_tracking_uri())
    
    # Set the MLflow tracking URI (ensure this URI is accessible)
    mlflow.set_tracking_uri("http://admin:admin@mlflow-tracking.mlflow.svc.cluster.local:80")
    print("Updated Tracking URI:", mlflow.get_tracking_uri())
    
    # End any active runs to prevent conflicts
    if mlflow.active_run():
        print(f"Ending stale run: {mlflow.active_run().info.run_id}")
        mlflow.end_run()
    
    with mlflow.start_run(run_name="Full Pipeline") as parent_run:
        print(f"Parent run ID: {parent_run.info.run_id}")
        
        # Step 1: Data Validation
        with mlflow.start_run(nested=True, run_name="Data Validation"):
            validate_data("data/raw/iris.csv")
        
        # Step 2: Preprocessing
        with mlflow.start_run(nested=True, run_name="Preprocessing"):
            preprocess_data("validated_data.csv")
        
        # Step 3: Feature Engineering
        with mlflow.start_run(nested=True, run_name="Feature Engineering"):
            engineer_features("processed_data/iris_processed.csv")
        
        # Step 4: Hyperparameter Tuning
        with mlflow.start_run(nested=True, run_name="Hyperparameter Tuning"):
            hyperparameter_tuning("features/iris_features.csv", "param_grid.yaml")
        
        # Step 5: Training
        with mlflow.start_run(nested=True, run_name="Training"):
            train_model("features/iris_features.csv", "params/best_params.yaml")
        
        # Step 6: Evaluation
        with mlflow.start_run(nested=True, run_name="Evaluation"):
            evaluate_model("models/model.joblib", "test_data.csv")

if __name__ == "__main__":
    main()