import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    logging.info('Initializing PredictPipeline')
    def predict(self, features: pd.DataFrame):
        try:
            model_path = 'artifact/model.pkl'
            preprocessor_path = 'artifact/preprocessor.pkl'
            logging.info('Predicting data')
            # Load the model and preprocessor
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            # Fit the preprocessor
            preprocessor.fit(features)
            # Transform the features
            features_transformed = preprocessor.transform(features)
            # Make predictions
            prediction = model.predict(features_transformed)
            return prediction
        except Exception as e:
            logging.error(f'predict function error: {str(e)}')
            raise CustomException(f"Error in predict pipeline: {e}")

        

class CustomData:
    def __init__(self, gender:str, race_ethnicity:str, parental_level_of_education:str, lunch:str, test_preparation_course:str, reading_score:int, writing_score:int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            logging.error(f'get_data_as_data_frame: {str(e)}')
            raise CustomException(f"CustomData failed: {str(e)}", sys)  