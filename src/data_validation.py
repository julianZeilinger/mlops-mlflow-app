import pandas as pd
import mlflow
import click


@click.command()
@click.option("--input-data", type=str, default="data/raw/iris.csv", help="Input dataset path")
def validate_data(input_data):
    with mlflow.start_run(run_name="Data Validation"):
        # Read and validate data
        df = pd.read_csv(input_data)
        print(df.iloc[0])  # Print the first line of the dataset

        validation_passed = True
        if df.isnull().values.any():
            validation_passed = False
        if df.duplicated().any():
            validation_passed = False

        mlflow.log_metric("data_validation_passed", int(validation_passed))

        if not validation_passed:
            raise ValueError("Data validation failed.")

        # Save validated data
        validated_data_path = "validated_data.csv"
        df.to_csv(validated_data_path, index=False)
        mlflow.log_artifact(validated_data_path, artifact_path="validated_data")


if __name__ == "__main__":
    validate_data()