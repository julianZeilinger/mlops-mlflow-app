#!/usr/bin/env bash
set -e

dvc pull

mlflow run . -e data_validation --env-manager=local -P run_name="Data Validation"
mlflow run . -e preprocessing --env-manager=local -P run_name="Preprocessing"
mlflow run . -e feature_engineering --env-manager=local -P run_name="Feature Engineering"
mlflow run . -e hyperparameter_tuning --env-manager=local -P run_name="Hyperparameter Tuning"
mlflow run . -e training --env-manager=local -P run_name="Training"
mlflow run . -e evaluation --env-manager=local -P run_name="Evaluation"