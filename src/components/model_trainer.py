import os
import sys
from dataclasses import dataclass
from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: any, test_array: any) -> float:
        try:
            logging.info("Starting model training component.")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models: Dict[str, any] = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
            }

            params: Dict[str, dict] = {
                "Logistic Regression": {
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear'],
                    'C': [0.01, 0.1, 1.0, 10.0]
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
            }

            # Evaluate all models using GridSearchCV
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            best_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with score: {best_score}")

            # Reject model if accuracy is below threshold
            if best_score < 0.85:
                raise CustomException(f"No good model found. Best score: {best_score}")

            # Retrain best model on full train set with best params
            best_model.fit(X_train, y_train)
            save_object(self.config.trained_model_path, best_model)

            logging.info(f"Trained model saved to: {self.config.trained_model_path}")

            y_pred = best_model.predict(X_test)
            final_accuracy = accuracy_score(y_test, y_pred)

            logging.info(f"Final Accuracy on Test Set: {final_accuracy}")
            return final_accuracy

        except Exception as e:
            raise CustomException(e, sys)
