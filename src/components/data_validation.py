import pandas as pd
from src.exception.exception import CustomException
from src.utils.utils import read_yaml_file,write_yaml_file
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.constants.app_constants import SCHEMA_FILE_PATH
import os
import sys

from src.logging.logger import logging

from scipy.stats import ks_2samp

class DataValidation:
    def __init__(self,data_validation_config:DataValidationConfig,
                 data_ingestion_artifact:DataIngestionArtifact):
        
        try:
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e,sys)
        
    def validate_number_of_columns(self,dataframe)->bool:
        try:
            number_of_columns=len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {dataframe.shape[1]}")
            if dataframe.shape[1]==number_of_columns:
                return True
            return False
        except Exception as e:
            raise CustomException(e,sys)
        
    def validate_required_columns(self,dataframe)->bool:
        try:
            required_columns=self._schema_config["required_columns"]
            logging.info(f"Required columns: {required_columns}")
            dataframe_columns=dataframe.columns.to_list()
            logging.info(f"Dataframe columns: {dataframe_columns}")
            for column in required_columns:
                if column not in dataframe_columns:
                    logging.info(f"Column: [{column}] is not present in the dataframe")
                    return False
            return True
        except Exception as e:
            raise CustomException(e,sys)
    
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    
                    }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,data=report)
            return status

        except Exception as e:
            raise CustomException(e,sys)
        
    def read_data(file_path)->pd.DataFrame:
        try:
            df=pd.read_csv(file_path)
            return df
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_validation(self)->DataIngestionArtifact:
        try: 
            train_dataframe=DataValidation.read_data(self.data_ingestion_artifact.trained_file_path)
            test_dataframe=DataValidation.read_data(self.data_ingestion_artifact.test_file_path)
            
            logging.info("Validating number of columns")
            if not self.validate_number_of_columns(train_dataframe):
                raise CustomException(f"Number of columns of train dataframe are not matching with the schema file",sys)
            if not self.validate_number_of_columns(test_dataframe):
                raise CustomException(f"Number of columns of test dataframe are not matching with the schema file",sys)
            
            logging.info("Validating required columns")
            if not self.validate_required_columns(train_dataframe):
                raise CustomException(f"Required columns are not present in the train dataframe",sys)
            if not self.validate_required_columns(test_dataframe):
                raise CustomException(f"Required columns are not present in the test dataframe",sys)
            
            logging.info("Detecting data drift")
            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            
            logging.info("Data validation is completed successfully")
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
              
            return data_validation_artifact  
        except Exception as e:
            raise CustomException(e,sys)
            
        
        
        
    