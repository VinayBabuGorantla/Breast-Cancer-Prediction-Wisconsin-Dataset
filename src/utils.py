import os
import sys
import dill
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path: str, obj):
    """
    Saves a Python object to a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str):
    """
    Loads a pickled Python object.
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict) -> dict:
    """
    Evaluates multiple models using GridSearchCV and returns their test accuracies.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging_message = f"Evaluating model: {model_name}"
            print(logging_message)

            grid = GridSearchCV(model, param[model_name], cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            best_model.fit(X_train, y_train)

            y_pred_test = best_model.predict(X_test)
            test_score = accuracy_score(y_test, y_pred_test)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
