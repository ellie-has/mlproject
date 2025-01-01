import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact', 'train.csv')
    test_data_path: str=os.path.join('artifact', 'test.csv')
    raw_data_path: str=os.path.join('artifact', 'data.csv')

class DataIngestion:
    def __init__(self, ingestion_config: DataIngestionConfig):
        self.ingestion_config = ingestion_config
    
    
    def initiate_data_ingestion(self):
        logging.info('Data ingestion initiated')
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Reading data from csv")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)
            logging.info("Splitting data into train and test")
            train, test = train_test_split(df, test_size=0.2, random_state=42)
            train.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path,index=False, header=True)
            logging.info("Data ingestion completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)