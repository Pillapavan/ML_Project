import os
import sys
import dill
import pickle
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from src.logger import logging
import warnings
warnings.filterwarnings("ignore")

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
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            logging.info(f"Training model started: {list(models.keys())[i]}")
            rs = RandomizedSearchCV(model, param, cv=5, n_iter=30, n_jobs=-1, verbose=2, scoring='r2', random_state=42)
            rs.fit(X_train, y_train)
            model.set_params(**rs.best_params_)
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
            logging.info(f"Training model completed: {list(models.keys())[i]} with R2 score: {test_model_score}")
        return report
    except Exception as e:  
        logging.info(f"Error occurred during model evaluation: {CustomException(e, sys)}")
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info(f"Error occurred while loading object: {CustomException(e, sys)}")
        raise CustomException(e, sys)
            