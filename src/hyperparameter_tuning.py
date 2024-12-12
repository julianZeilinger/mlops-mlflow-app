import yaml
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

def hyperparameter_tuning(params):
    df = pd.read_csv('data/features/iris_features.csv')
    X = df.drop('species', axis=1)
    y = df['species']

    param_grid = params['param_grid']

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    mlflow.log_params(best_params)

    # Save best parameters
    os.makedirs('params', exist_ok=True)
    with open('params/best_params.yaml', 'w') as f:
        yaml.dump({'training': best_params}, f)

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)['hyperparameter_tuning']
    with mlflow.start_run(run_name="Hyperparameter Tuning"):
        hyperparameter_tuning(params)