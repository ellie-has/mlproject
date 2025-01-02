import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
    



# def evaluate_models(X_train,y_train,X_test,y_test,models,params,cv=3,n_jobs=3,verbose=1,refit=False):
#     try:
#         report={}
#         for i in range(len(models)):
            
#             model=list(models.values())[i]
#             para=params[list(models.keys())[i]]
            
#             gs = GridSearchCV(model,para,cv=cv,n_jobs=n_jobs,verbose=verbose,refit=refit)
#             gs.fit(X_train,y_train)
            
#             model.set_params(**gs.best_params_)
#             model.fit(X_train,y_train)
#             y_test_pred=model.predict(X_test)

#             test_model_score=r2_score(y_test, y_test_pred)
#             report[list(models.keys())[i]]=test_model_score
     

#         return report
#     except Exception as e:
#         raise CustomException(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params, 
                   n_iter=10, cv=3, n_jobs=3, verbose=1):
    """
    Evaluate multiple models using RandomizedSearchCV for parameter tuning
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        models: Dictionary of models to evaluate
        params: Dictionary of parameters for each model
        n_iter: Number of parameter settings sampled
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
    
    Returns:
        Dictionary with model names as keys and best scores as values
    """
    try:
        report = {}
        
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param_dist = params[model_name]
            
            # Skip parameter search if no parameters to tune
            if not param_dist:
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)
                report[model_name] = test_model_score
                continue
                
            # Use RandomizedSearchCV for parameter tuning
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=42
            )
            
            # Fit and evaluate model
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Store results
            report[model_name] = test_model_score
            
            # Optional: Print best parameters for each model
            if verbose:
                print(f"\nBest parameters for {model_name}:")
                print(random_search.best_params_)
                print(f"Best score: {test_model_score}")
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)


# def evaluate_models(X_train, y_train, X_test, y_test, models, params):
#     """
#     Simple model evaluation function that trains each model with specified parameters
#     and returns their performance scores.
    
#     Args:
#         X_train, y_train: Training data
#         X_test, y_test: Test data
#         models: Dictionary of models to evaluate
#         params: Dictionary of parameters for each model
    
#     Returns:
#         Dictionary with model names as keys and R2 scores as values
#     """
#     try:
#         report = {}
        
#         for model_name, model in models.items():
#             # Set the parameters directly for the model
#             if params[model_name]:
#                 # Take the first value for each parameter if multiple are provided
#                 best_params = {
#                     param: values[0] if isinstance(values, list) else values
#                     for param, values in params[model_name].items()
#                 }
#                 model.set_params(**best_params)
            
#             # Train the model
#             model.fit(X_train, y_train)
            
#             # Make predictions and calculate score
#             y_test_pred = model.predict(X_test)
#             test_model_score = r2_score(y_test, y_test_pred)
            
#             # Store the score
#             report[model_name] = test_model_score
            
#         return report
        
#     except Exception as e:
#         raise CustomException(e, sys)