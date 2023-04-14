import importlib

import mlflow
import pandas as pd
import yaml
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from src.utils import store_config, train_and_store_model
from src.validators import InputData, OutputData


class AIService:
    """A class for performing inference with a pre-trained machine
    learning model."""

    def __init__(self, config_path):
        """Initialize the MyModel object with the configuration file path."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Load the model
        model_version = config["model"]["version"]
        model_version = None if model_version == "latest" else int(model_version)
        model_uri = f"models:/{config['model']['name']}/{model_version}"
        self.model = mlflow.sklearn.load_model(model_uri)

        # Load preprocessing logic
        preprocess_module = importlib.import_module(
            config["preprocessing"]["module_path"], "src"
        )
        self.preprocess = getattr(
            preprocess_module, config["preprocessing"]["function_name"]
        )

    def validate_dataframe(
        self, schema: BaseModel, dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """Validates a dataframe for a given pydantic basemodel"""
        list_validated_data = [schema(**row) for _, row in dataframe.iterrows()]
        return pd.DataFrame(jsonable_encoder(list_validated_data))

    def inference(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        """Perform inference with the preprocessed data."""
        return self.model.predict(preprocessed_data)

    def postprocess(self, input_data, predictions: pd.DataFrame) -> pd.DataFrame:
        """Postprocess the model's predictions."""
        return pd.DataFrame(
            {
                "id": input_data["id"],
                "prediction": predictions,
            }
        )

    def orchestrate(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Run the entire inference pipeline on the input data."""
        validated_input = self.validate_dataframe(InputData, input_data)
        preprocessed_data = self.preprocess(validated_input)
        predictions = self.inference(preprocessed_data)
        postprocessed_data = self.postprocess(input_data, predictions)
        return self.validate_dataframe(OutputData, postprocessed_data)


def main():
    """Main function to prep abd run the example"""

    config = {
        "preprocessing": {
            "module_path": "src.sklearn_linear_regression.preprocessing",
            "function_name": "preprocess",
        },
        "model": {"name": "sklearn-linear-regression", "version": "latest"},
    }
    config_path = "config/config.yaml"
    store_config(config, config_path)

    model_name = "sklearn-linear-regression"
    train_and_store_model(model_name, "mlflow")

    inference_data = pd.DataFrame(
        {"id": [1, 2, 3], "feature1": [1, 0.2, 0.3], "feature2": [0.0, 0.5, 0.6]}
    )
    my_model_service = AIService(config_path)
    result = my_model_service.orchestrate(inference_data)
    print(result)


if __name__ == "__main__":
    main()
