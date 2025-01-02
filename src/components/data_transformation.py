# This section is from data features and data cleaning
import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation paths and parameters"""
    preprocessor_obj_file_path: str=os.path.join('artifact', 'preprocessor.pkl')


class DataTransformation:
    """Handles data preprocessing and transformation for model training"""
    
    # Define class-level constants for column groups
    NUMERICAL_COLUMNS = ["writing_score", "reading_score"]
    CATEGORICAL_COLUMNS = [
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course",
    ]
    TARGET_COLUMN = "math_score"

    def __init__(self, data_transformation_config: DataTransformationConfig):
        """
        Initialize data transformation with configuration
        Args:
            data_transformation_config: Configuration object containing paths and parameters
        """
        self.data_transformation_config = data_transformation_config


    def get_data_transformer_object(self):
        """
        Create the main data transformation pipeline
        Returns:
            ColumnTransformer combining numerical and categorical preprocessing
        """
        try:
            logging.info('Creating data transformation object')
            numerical_columns = self.NUMERICAL_COLUMNS
            cagegorical_columns = self.CATEGORICAL_COLUMNS

            numerical_transformer_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer_pipeline, numerical_columns),
                    ('cat', categorical_transformer_pipeline, cagegorical_columns)
                ]
            )

            logging.info('Data transformation object created successfully')
            return preprocessor
        except Exception as e:
            logging.error('Error in creating data transformation object')
            raise CustomException(f"Error in get_data_transformer_object: {str(e)}", sys)


    def initiate_data_transformation(self, train_path, test_path):
        """
        Execute the complete data transformation pipeline
        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV  
        Returns:
            Tuple of (transformed_train_data, transformed_test_data, preprocessor_path)
        """
        logging.info('Data transformation initiated')
        try:
            #Load data
            logging.info('Reading data from csv')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            

            logging.info('obtaining preprocessor object')
            # Split features and target
            traget_column = self.TARGET_COLUMN
            X_train = train_df.drop(traget_column, axis=1)
            y_train = train_df[traget_column]
            X_test = test_df.drop(traget_column, axis=1)
            y_test = test_df[traget_column]

            # Create and fit preprocessor
            logging.info('Applying data transformations')
            preprocessing_obj = self.get_data_transformer_object()
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            # Combine features and target
            train_arr = np.c_[X_train_transformed, y_train]
            test_arr = np.c_[X_test_transformed, y_test]                        

            # Save preprocessor
            logging.info('Saving preprocessor object')
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info('Data transformation completed successfully')
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.error('Error in data transformation pipeline')
            raise CustomException(f"Transformation failed: {str(e)}", sys)      

        
    