#!/usr/bin/env bash
set -e

dvc pull

steps=("data_validation" "preprocessing" "feature_engineering" "hyperparameter_tuning" "training" "evaluation")
failed_steps=()

for step in "${steps[@]}"; do
    echo "Running step: $step"
    if ! mlflow run . -e "$step" --env-manager=local; then
        echo "Step $step failed!"
        failed_steps+=("$step")
    fi
done

if [ ${#failed_steps[@]} -ne 0 ]; then
    echo "The following steps failed: ${failed_steps[@]}"
    exit 1
else
    echo "All steps completed successfully!"
fi