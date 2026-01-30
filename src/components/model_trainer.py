import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "KNeighbors Regressor": KNeighborsRegressor()
            }
            param_grids = {
                "Random Forest": {
                    "n_estimators": [100, 300, 500],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                    "bootstrap": [True, False]
                },
                "Decision Tree": {
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": [None, "sqrt", "log2"],
                "criterion": ["squared_error", "friedman_mse"]
            },
            "Gradient Boosting": {
                "n_estimators": [100, 200, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [2, 3, 5],
                "subsample": [0.6, 0.8, 1.0],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 3, 5]
            },
            "Linear Regression": {
                "fit_intercept": [True, False],
                "positive": [True, False]
            },
            "XGBRegressor": {
                "n_estimators": [200, 500, 800],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7, 10],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "gamma": [0, 0.1, 0.3],
                "reg_alpha": [0, 0.1, 1],
                "reg_lambda": [1, 1.5, 2]
            },
            "CatBoosting Regressor": {
                "iterations": [300, 600, 1000],
                "learning_rate": [0.01, 0.05, 0.1],
                "depth": [4, 6, 8, 10],
                "l2_leaf_reg": [1, 3, 5, 7],
                "bagging_temperature": [0, 0.5, 1],
                "border_count": [32, 64, 128]
            },
            "AdaBoost Regressor": {
                "n_estimators": [50, 100, 300],
                "learning_rate": [0.01, 0.05, 0.1, 1.0],
                "loss": ["linear", "square", "exponential"]
            },
            "KNeighbors Regressor": {
                "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
                "p": [1, 2],              # 1 = Manhattan, 2 = Euclidean (used only if metric=minkowski)
                "algorithm": ["auto", "ball_tree", "kd_tree"]
            }
        }
            
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models,param_grids)
            logging.info(f"Model report: {model_report}")
            best_model_value = max(sorted(model_report.values()))
            best_model = list(model_report.keys())[
                list(model_report.values()).index(best_model_value)
            ]
            logging.info(f"Best model: {best_model}")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            return {best_model: best_model_value}
        except Exception as e:
            logging.warning(CustomException(e,sys))
            raise CustomException(e,sys)