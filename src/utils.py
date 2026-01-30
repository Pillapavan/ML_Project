import numpy as np
import pandas as pd
import os
import sys
import dill
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.logger import logging

def save_train_test(df, raw_path: str,train_path: str, test_path: str):
    try:
        dir_path = os.path.dirname(raw_path)
        os.makedirs(dir_path, exist_ok=True)
        df.to_csv(raw_path, index=False, header=True)
        logging.info("Train test split initiated")
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        train_set.to_csv(train_path, index=False, header=True)
        test_set.to_csv(test_path, index=False, header=True)
        logging.info("Ingestion of the data is completed")
    except Exception as e:
        logging.info(f"Error occurred while saving train and test data: {CustomException(e, sys)}")
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        logging.info(f"Error occurred while saving object: {CustomException(e, sys)}")
        raise CustomException(e, sys)
    
    
    