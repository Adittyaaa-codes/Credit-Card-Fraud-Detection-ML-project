import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("DB_URL")

import certifi
ca=certifi.where()

import pandas as pd
import numpy as np
import pymongo
from src.exception.exception import CustomException
from src.logging.logger import logger
from src.constants.app_constants import FILE_NAME

class FraudDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
        
    def csv_to_json(self,file_path):
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True,inplace=True)
            records = list(json.loads(df.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomException(e,sys)
    
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    FILE_PATH=os.path.join("data",FILE_NAME)
    DATABASE="AdiDataBase"
    collection="CC_Fraud_Data"
    obj=FraudDataExtract()
    records=obj.csv_to_json(file_path=FILE_PATH)
    no_of_records=obj.insert_data_mongodb(records,DATABASE,collection)
    print(no_of_records)