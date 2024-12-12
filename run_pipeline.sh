#!/usr/bin/env bash
set -e
dvc pull
mlflow run . -e data_validation
mlflow run . -e preprocessing
mlflow run . -e feature_engineering
mlflow run . -e hyperparameter_tuning
mlflow run . -e training
mlflow run . -e evaluation