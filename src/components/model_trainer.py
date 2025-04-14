import os
import sys
from dataclasses import dataclass
from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

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

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            best_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with score: {best_score}")

            if best_score < 0.85:
                raise CustomException(f"No good model found. Best score: {best_score}", sys)

            # Start MLflow experiment logging
            mlflow.set_experiment("BreastCancer_Classification")

            with mlflow.start_run():
                mlflow.log_param("model_name", best_model_name)
                for param_name, param_value in params[best_model_name].items():
                    mlflow.log_param(param_name, param_value)

                # Train and evaluate model
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                final_accuracy = accuracy_score(y_test, y_pred)

                mlflow.log_metric("accuracy", final_accuracy)
                mlflow.sklearn.log_model(best_model, "model")

                logging.info(f"MLflow logging completed with accuracy: {final_accuracy:.4f}")

            save_object(self.config.trained_model_path, best_model)
            logging.info(f"Trained model saved to: {self.config.trained_model_path}")

            return final_accuracy

        except Exception as e:
            raise CustomException(e, sys)
