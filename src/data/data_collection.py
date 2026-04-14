import pandas as pd
import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def load_params(file_path:str)->float:
    try:
        with open(file_path, "r") as f:
            params = yaml.safe_load(f)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {file_path}: {e}")
# test_size = yaml.safe_load(open("params.yaml"))["data_collection"]["test_size"]

def load_data(file_path:str)->pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")

# data=pd.read_csv(r"C:\Users\DELL\Downloads\water_potability.csv")

def split_data(data:pd.DataFrame,test_size:float):
    try:
        return train_test_split(data, test_size=test_size, random_state=42)
    except Exception as e:
        raise Exception(f"Error splitting data: {e}")

# train_data,test_data=train_test_split(data,test_size=test_size,random_state=42)


def save_data(df:pd.DataFrame,file_path:str)->None:
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {file_path}: {e}")

def main():
    data_filepath=r"C:\Users\DELL\Downloads\water_potability.csv"
    params_filepath="params.yaml"
    raw_data_path=os.path.join("data","raw")

    try:

        data=load_data(data_filepath)
        test_size=load_params(params_filepath)
        train_data,test_data=split_data(data,test_size)
        os.makedirs(raw_data_path)
        save_data(train_data, os.path.join(raw_data_path,"train_data.csv"))
        save_data(test_data, os.path.join(raw_data_path,"test_data.csv"))
    except Exception as e:
        raise Exception(f"Error in data collection process: {e}")

if __name__=="__main__":
    main()

# data_path=os.path.join("data","raw")
# os.makedirs(data_path)
# train_data.to_csv(os.path.join(data_path,"train_data.csv"),index=False)
# test_data.to_csv(os.path.join(data_path,"test_data.csv"),index=False)