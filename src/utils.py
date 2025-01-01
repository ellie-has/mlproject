import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(obj, file_path):
    """
    Save an object to a file using dill
    Args:
        obj: Object to be saved
        file_path: Path to save the object
    """
    try:
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(f"Save object failed: {str(e)}", sys)
    
    def evaluate_models(X_train, y_train, X_test, y_test, models, params):
        """
        Evaluate a list of models on the given data
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            models: List of models to evaluate
            param: Dictionary of hyperparameters for the models
        """
        try:
            model_report = {}
            for model in range(len(list(models))):
                model = list(models.values())[model]
                param = list(params.keys())[model]
                grid = GridSearchCV(model, param, refit=True, verbose=3)
                grid.fit(X_train, y_train)
                model_report[model] = grid.score(X_test, y_test)
            return model_report
        
        except Exception as e:
            raise CustomException(e, sys)