import os
import sys
import pandas as pd

from src.utils import load_object
from src.logger import logging
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Starting prediction pipeline...")

            # Load saved model and preprocessor
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            logging.info("Model and preprocessor loaded.")

            # Preprocess input features
            transformed_data = preprocessor.transform(features)

            # Predict using the trained model
            preds = model.predict(transformed_data)

            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    Collects all 30 input features for the Breast Cancer model and prepares them as a DataFrame.
    """
    def __init__(self, **kwargs):
        self.feature_dict = kwargs

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            df = pd.DataFrame([self.feature_dict])
            return df
        except Exception as e:
            raise CustomException(e, sys)
