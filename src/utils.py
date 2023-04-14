import os

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression


def train_and_store_model(model_name: str, STORAGE: str = "locally"):
    """Train a logistic regression model on the training set"""

    training_data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feature1": [0.8, 0.3, 0.9],
            "feature2": [0.5, 0.1, 0.7],
            "target": [1, 0, 1],
        }
    )
    X = training_data.drop(["id", "target"], axis=1)
    y = training_data["target"]
    model = LogisticRegression()
    model.fit(X, y)
    if STORAGE == "locally":
        joblib.dump(model, f"{model_name}.joblib")
    elif STORAGE == "mlflow":
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, model_name)
            run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{model_name}"
        mlflow.register_model(model_uri, model_name)


def store_config(config: dict, config_path: str):
    """Store a config file at a given path"""
    if not os.path.exists("config"):
        os.makedirs("config")
    with open(config_path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
