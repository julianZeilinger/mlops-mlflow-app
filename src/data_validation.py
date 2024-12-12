import pandas as pd
import mlflow

def validate_data():
    df = pd.read_csv('data/raw/iris.csv')
    validation_passed = True

    # Check for missing values
    if df.isnull().values.any():
        validation_passed = False

    # Check for duplicates
    if df.duplicated().any():
        validation_passed = False

    mlflow.log_metric("data_validation_passed", int(validation_passed))

    if not validation_passed:
        raise ValueError("Data validation failed.")

if __name__ == "__main__":
    with mlflow.start_run(run_name="Data Validation"):
        validate_data()