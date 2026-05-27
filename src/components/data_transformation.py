import os
import sys
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from src.exception.exception import CustomException
from src.logging.logger import logging

from src.constants.app_constants import DATA_TRANSFORMATION_IMPUTER_PARAMS
from src.constants.app_constants import TARGET_COLUMN

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (DataValidationArtifact,DataTransformationArtifact)

from src.utils.utils import save_object,save_numpy_array

class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig,
                 data_validation_artifact:DataValidationArtifact):
        try:
            self.data_transformation_config=data_transformation_config
            self.data_validation_artifact=data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_data_transformer_object(self)->Pipeline:
        try:
            imputer_params=DATA_TRANSFORMATION_IMPUTER_PARAMS
            knn_imputer=KNNImputer(**imputer_params)
            pipeline=Pipeline(steps=[
                ("knn_imputer",knn_imputer)
            ])
            return pipeline
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            train_file_path=self.data_validation_artifact.valid_train_file_path
            test_file_path=self.data_validation_artifact.valid_test_file_path
            
            train_dataframe=DataTransformation.read_data(train_file_path)
            test_dataframe=DataTransformation.read_data(test_file_path)
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column=TARGET_COLUMN
            
            input_feature_train_df=train_dataframe.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_dataframe[target_column]
            
            input_feature_test_df=test_dataframe.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_dataframe[target_column]
            
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = pd.DataFrame(
                data=input_feature_train_arr, columns=input_feature_train_df.columns
            )
            train_arr[target_column] = target_feature_train_df.values
            
            test_arr = pd.DataFrame(
                data=input_feature_test_arr, columns=input_feature_test_df.columns
            )
            test_arr[target_column] = target_feature_test_df.values
            
            logging.info(f"Saving transformed train array to: {self.data_transformation_config.transformed_train_file_path}")
            save_numpy_array(file_path=self.data_transformation_config.transformed_train_file_path,array=train_arr.values)
            save_numpy_array(file_path=self.data_transformation_config.transformed_test_file_path,array=test_arr.values)
            
            logging.info(f"Saving preprocessor object to: {self.data_transformation_config.preprocessor_object_file_path}")
            save_object(file_path=self.data_transformation_config.preprocessor_object_file_path,obj=preprocessing_obj)
            
            data_transformation_artifact=DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessor_object_file_path=self.data_transformation_config.preprocessor_object_file_path
            )
            logging.info(f"Data transformation completed. Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
             raise CustomException(e,sys)

