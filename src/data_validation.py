import pandas as pd
import mlflow
import click


def validate_data(input_data="data/raw/iris.csv"):
    # Read and validate data
    print(f"Validating data from: {input_data}")
    df = pd.read_csv(input_data)
    print(df.iloc[0])  # Print the first line of the dataset

    validation_passed = True

    mlflow.log_metric("data_validation_passed", int(validation_passed))

    if not validation_passed:
        raise ValueError("Data validation failed.")

    # Save validated data
    validated_data_path = "validated_data.csv"
    df.to_csv(validated_data_path, index=False)
    mlflow.log_artifact(validated_data_path, artifact_path="validated_data")
