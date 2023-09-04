import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from Data_Transformation import DataTransformation
from model_trainer import ModelTrainer



@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestionConfig = DataIngestionConfig()

    def initiateDataIngestion(self):
        logging.info("Started Data Ingestion")

        try:
            df = pd.read_csv('./notebook/data/stud.csv') 
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path), exist_ok=True)

            df.to_csv(self.ingestionConfig.raw_data_path, header=True, index=False)

            logging.info("Train test split is initiated")

            trainSet, testSet = train_test_split(df, test_size=0.2, random_state=42)

            trainSet.to_csv(self.ingestionConfig.train_data_path, index=False, header=True)
            testSet.to_csv(self.ingestionConfig.test_data_path, index=False, header=True)

            logging.info("Data ingestion complete")

            return self.ingestionConfig.train_data_path, self.ingestionConfig.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj= DataIngestion()
    train_data, test_data = obj.initiateDataIngestion()

    data_transforamtion = DataTransformation()
    train_arr, test_arr, _ = data_transforamtion.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print("Training is initiated")
    print("="*35)
    print("The r2 score of the best model is: {} percent".format(round(modeltrainer.initiate_model_trainer(train_arr, test_arr) *100, 2)))
    print("="*35)


