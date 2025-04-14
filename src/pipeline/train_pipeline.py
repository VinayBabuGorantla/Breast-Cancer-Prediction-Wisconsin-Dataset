import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

if __name__ == "__main__":
    try:
        logging.info(">>> Training pipeline started.")

        # 1. Ingest data
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # 2. Transform data
        transformer = DataTransformation()
        train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)

        # 3. Train and evaluate model
        trainer = ModelTrainer()
        accuracy = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Training pipeline completed successfully with test accuracy: {accuracy:.4f}")

    except Exception as e:
        raise CustomException(e, sys)
