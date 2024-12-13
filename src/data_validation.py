import pandas as pd
import mlflow

def validate_data(input_data="data/raw/iris.csv"):
    # Read and validate data
    print(f"Validating data from: {input_data}")
    df = pd.read_csv(input_data)
    
    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head())
    
    # Basic validation checks
    validation_passed = True
    required_columns = ["sepal.length", "sepal.width", "petal.length", "petal.width", "variety"]
    
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing column: {col}")
            validation_passed = False
    
    # Log validation result
    mlflow.log_metric("data_validation_passed", int(validation_passed))
    
    if not validation_passed:
        raise ValueError("Data validation failed. Required columns are missing.")
    
    # Save validated data
    validated_data_path = "validated_data.csv"
    df.to_csv(validated_data_path, index=False)
    mlflow.log_artifact(validated_data_path, artifact_path="validated_data")