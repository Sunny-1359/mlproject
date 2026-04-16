import sys
import os
from dataclasses import dataclass


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import  XGBRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            
            logging.info("spliting training and test input data")
            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models ={
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()

            }

            params = {
                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                },

                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree"],
                    "p": [1, 2]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },

                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "criterion": ["squared_error", "absolute_error"],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["sqrt", "log2"]
                },

                "XGBRegressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                },

                "CatBoosting Regressor": {
                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "depth": [4, 6, 8],
                    "l2_leaf_reg": [1, 3, 5],
                    "loss_function": ["RMSE", "MAE"]
                },
                
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1],
                    "loss": ["linear", "square", "exponential"]
                },
                
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5],
                    "subsample": [0.8, 1.0],
                    "loss": ["squared_error", "absolute_error", "huber"]
                }
            }


            model_report:dict=evaluate_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models,params=params)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best found model on both trainnig and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_scores=r2_score(Y_test,predicted)
            return r2_scores,best_model_name

        except Exception as e:
            raise CustomException(e,sys)