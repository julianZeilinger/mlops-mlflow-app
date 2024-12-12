import os
import mlflow
from click import command, option
from mlflow.tracking import MlflowClient


def _get_or_run(entry_point, parameters):
    """
    Launches an MLflow run or reuses a cached run if one exists with the same parameters.
    """
    client = MlflowClient()
    existing_run = None
    for run in client.search_runs(experiment_ids=["0"], order_by=["start_time desc"]):
        if run.data.tags.get("mlflow.project.entryPoint") == entry_point:
            if all(run.data.params.get(k) == str(v) for k, v in parameters.items()):
                existing_run = run
                break
    if existing_run:
        print(f"Using cached run for {entry_point} with parameters {parameters}")
        return existing_run

    print(f"Starting new run for {entry_point} with parameters {parameters}")
    return mlflow.run(".", entry_point, parameters=parameters, env_manager="local")


@command()
def workflow():
    """
    Orchestrates the ML pipeline.
    """
    # Step 1: Data validation
    validation_run = _get_or_run("data_validation", {"input_data": "data/raw/iris.csv"})
    validated_data_uri = os.path.join(validation_run.info.artifact_uri, "validated_data.csv")

    # Step 2: Preprocessing
    preprocessing_run = _get_or_run("preprocessing", {"input_data": validated_data_uri})
    processed_data_uri = os.path.join(preprocessing_run.info.artifact_uri, "processed_data/iris_processed.csv")

    # Step 3: Feature Engineering
    feature_engineering_run = _get_or_run("feature_engineering", {"input_data": processed_data_uri})
    feature_data_uri = os.path.join(feature_engineering_run.info.artifact_uri, "features/iris_features.csv")

    # Step 4: Hyperparameter Tuning
    hyperparameter_tuning_run = _get_or_run(
        "hyperparameter_tuning", {"input_data": feature_data_uri, "param_grid": "param_grid.yaml"}
    )
    best_params_uri = os.path.join(hyperparameter_tuning_run.info.artifact_uri, "params/best_params.yaml")

    # Step 5: Training
    training_run = _get_or_run(
        "training", {"input_data": feature_data_uri, "best_params": best_params_uri}
    )
    model_uri = os.path.join(training_run.info.artifact_uri, "models/model.joblib")

    # Step 6: Evaluation
    _get_or_run("evaluation", {"model_path": model_uri, "test_data": feature_data_uri})


if __name__ == "__main__":
    workflow()