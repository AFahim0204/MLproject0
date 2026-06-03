import os 
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline   
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor   
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score 
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object , evaluate_models



@dataclass
class ModelTrainerConfig :
    trained_model_path = os.path.join('artifacts' , 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainerConfig()

    def initiate_model_train(self,train_array,test_array):
        try:
            logging.info('Split training and test data')
            X_train,y_train , X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                    'Linear Regression': LinearRegression(),
                    'lasso Regression': Lasso(),
                    'Ridge Regression': Ridge(),
                    'KNN Regression': KNeighborsRegressor(),
                    'Random Forest Regression': RandomForestRegressor(),
                    'Gradient Boosting Regression': GradientBoostingRegressor()
            }

            params = {

                'Linear Regression': {} ,
                'Lasso' : {
                    'alpha' : [0.001 , 0.01 , 0.1 , 1 , 10]} ,
                'Ridge' : {'alpha' : [0.001 , 0.01 , 0.1 , 1 , 10]} , 

                "KNN Regression": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"]
                },
                'Random Forest Regression': { 
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                    },

                "Gradient Boosting Regression": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0]
                }
            }

            model_report:dict=evaluate_models(X_train=X_train , y_train=y_train , X_test=X_test, y_test=y_test ,models=models , param=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found')
            
            logging.info('Best model found')

            save_object(
                file_path=self.model_train_config.trained_model_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test , predicted)
            return r2_square
        

        except Exception as e:
            raise CustomException(e,sys)
        
