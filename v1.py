import joblib
import pandas as pd

from src.utils import train_and_store_model


class AIService:
    """A class for performing inference with a pre-trained machine
    learning model."""

    def __init__(self, model_path: str) -> None:
        """Initialize the AIService object with a pre-trained machine learning model."""
        self.model = joblib.load(model_path)

    def preprocess(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input data."""
        return input_data.drop(["id"], axis=1)

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
        preprocessed_data = self.preprocess(input_data)
        predictions = self.inference(preprocessed_data)
        return self.postprocess(input_data, predictions)


def main():
    """Main function to prep abd run the example"""

    model_name = "sklearn-linear-regression"
    train_and_store_model(model_name, "locally")

    inference_data = pd.DataFrame(
        {"id": [1, 2, 3], "feature1": [1, 0.2, 0.3], "feature2": [0.0, 0.5, 0.6]}
    )
    my_model_service = AIService(f"{model_name}.joblib")
    result = my_model_service.orchestrate(inference_data)
    print(result)


if __name__ == "__main__":
    main()
