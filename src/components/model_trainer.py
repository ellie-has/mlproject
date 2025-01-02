import os
import sys
from dataclasses import dataclass
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifact', 'model.pkl')

class ModelTrainer:
    """Handles model training and evaluation"""
    def __init__(self):
            self.model_trainer_config = ModelTrainerConfig()
    
    def train_model(self,train_arry, test_arry):
        """
        Initialize model trainer with configuration        
        Train the model using the provided configuration
        Args:
            train_arry : arry to train the model
            test_arry: arry to test the model
        """
        try:
            logging.info('Training model')
            # Load the training data
            X_train, X_test, y_train, y_test = (
                train_arry[:, :-1], test_arry[:, :-1], train_arry[:, -1], test_arry[:, -1]
            )
            # Define a dictionary of models to train
            models={
                "Linear Regression":LinearRegression(),
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "K-Neighbour Regressor":KNeighborsRegressor(),
                "XGB Regressor":XGBRegressor(),
                "CatBoost Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor()
            }

            # Define hyperparameters for the models
            params = {
                "Linear Regression":{},
                'Decision Tree': {'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                'Random Forest': {'n_estimators': [8,16,32,64,128,256]},
                'Gradient Boosting': {'n_estimators': [8,16,32,64,128,256],'learning_rate':[.1,.01,.05,.001]},
                'AdaBoost Regressor': {'learning_rate':[.1,.01,0.5,.001],'n_estimators': [8,16,32,64,128,256]},
                'K-Neighbour Regressor': {'n_neighbors': [5,7,9,11]},
                'XGB Regressor': {'n_estimators': [8,16,32,64,128,256],'learning_rate':[.1,.01,.05,.001]},
                'CatBoost Regressor': {'learning_rate': [0.01, 0.05, 0.1], 'depth': [6,8,10], 'iterations': [30,50,100]}
            }

            # Create evaluated models report
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,params=params)
            logging.info(f'Model report: {model_report}')

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                 raise CustomException("No best model found")
            logging.info(f'Best model: {best_model_name} with score: {best_model_score}')

            #Save the best model
            save_object(
                obj=best_model,
                file_path=self.model_trainer_config.trained_model_file_path
            )
            
            logging.info('Model trained and saved successfully')
            
            predicted_values = best_model.predict(X_test)
            score = r2_score(y_test, predicted_values)
            logging.info(f'R2 score of the model: {score}')
            return score
        
        except Exception as e:
            logging.error(f'Error training model: {str(e)}')
            raise CustomException(str(e), sys)
       

