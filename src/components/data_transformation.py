import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformerConfig:
     preprocessor_obj_path:str = os.path.join('artifacts','preprocessor.pkl')
     
class DataTransformer:
    def __init__(self):
         self.preprocessor_config = DataTransformerConfig()
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]  
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )
            
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", numerical_pipeline, numerical_columns),
                    ("cat_pipeline", categorical_pipeline, categorical_columns)
                ]
            ) 
            return preprocessor   
        except Exception as e:
            logging.warning(CustomException(e, sys))
            raise CustomException(e, sys)
            
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessor object")
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "math_score"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_df_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_df_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Saved preprocessing object")
            
            save_object(
                self.preprocessor_config.preprocessor_obj_path,
                preprocessing_obj
            )
            
            return (
                train_df_arr,
                test_df_arr,
                self.preprocessor_config.preprocessor_obj_path
            )
            
        except Exception as e:
            logging.warning(CustomException(e, sys))
            raise CustomException(e, sys)