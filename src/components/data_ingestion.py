import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_train_test
import pandas as pd

from src.components.data_transformation import DataTransformer
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entering the data ingestion method or component")
        try:
            df = pd.read_csv("src\data\stud.csv")
            logging.info("Read the dataset as dataframe")
            save_train_test(
                df,
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
        
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_set,test_set = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformer()
    train_df,test_df,_ = data_transformation.initiate_data_transformation(train_set,test_set)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_df,test_df))
    
    