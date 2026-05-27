import pickle
import numpy as np
from src.exception.exception import CustomException
import yaml
import os,sys

def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,"r") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys)
    
def write_yaml_file(file_path:str,data:dict):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as yaml_file:
            yaml.dump(data,yaml_file)
    except Exception as e:
        raise CustomException(e, sys)
    
def save_numpy_array(file_path:str,array:np.array):    
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(array,file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_numpy_array(file_path:str)->np.array:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not found")
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def save_object(file_path:str,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path:str):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not found")
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)