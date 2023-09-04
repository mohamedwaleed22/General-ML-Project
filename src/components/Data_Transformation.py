import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTrasnformaionConfig:
    preprocessorObjFilePath = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTrasnformaionConfig()

    def get_data_transformer_obj(self):

        """
        This function is responsible for data transformation and preprocessing
        """
        try:
            num_col = ["writing_score", "reading_score"]
            cat_col = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())]
                       )
            
            cat_pipeline = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                       ("encoder", OneHotEncoder()),
                       ("scaler", StandardScaler(with_mean=False))]
                       )
            
            logging.info("Numerical and Categorical features preprocessing completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_col),
                    ("cat_pipeline", cat_pipeline, cat_col)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test df completed")

            preprocessing_obj = self.get_data_transformer_obj()

            target_col = "math_score"

            x_feat_train = train_df.drop(columns=target_col, axis=1)
            y_feat_train = train_df[target_col]

            x_feat_test = test_df.drop(columns=target_col, axis=1)
            y_feat_test = test_df[target_col]

            logging.info("Applying preprocessing object on train and test df")

            x_feat_train_arr = preprocessing_obj.fit_transform(x_feat_train)
            x_feat_test_arr = preprocessing_obj.transform(x_feat_test)

            train_arr = np.c_[x_feat_train_arr, np.array(y_feat_train)]
            test_arr = np.c_[x_feat_test_arr, np.array(y_feat_test)]

            logging.info("Saving preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessorObjFilePath,
                obj = preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessorObjFilePath
 

        except Exception as e:
            raise CustomException(e, sys)

        