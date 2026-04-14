import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
import yaml

def load_params(file_path:str)->float:
    try:
        with open(file_path, "r") as f:
            params = yaml.safe_load(f)
        return params["model_building"]["n_estimators"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {file_path}: {e}")

# n_estimators = yaml.safe_load(open("params.yaml"))["model_building"]["n_estimators"]


def load_data(file_path:str)->pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")

# train_data = pd.read_csv("./data/processed/train_processed_data.csv")

# X_train=train_data.iloc[:,0:-1].values
# y_train=train_data.iloc[:,-1].values

def prepare_data(train_data:pd.DataFrame):
    try:
        X_train = train_data.drop("Potability", axis=1)
        y_train = train_data["Potability"]
        return X_train, y_train
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")

# X_train = train_data.drop("Potability", axis=1)
# y_train = train_data["Potability"]

def train_model(X_train, y_train, n_estimators):
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X_train, y_train)
        return clf
    except Exception as e:
        raise Exception(f"Error training model: {e}")

# clf=RandomForestClassifier(n_estimators=n_estimators)
# clf.fit(X_train,y_train)

def save_model(model, file_path: str):
    try:
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        raise Exception(f"Error saving model to {file_path}: {e}")
# pickle.dump(clf,open("model.pkl","wb"))

def main():
    try:
        params_path="params.yaml"
        data_path="./data/processed/train_processed_data.csv"
        model_name="models/model.pkl"
        n_estimators=load_params(params_path)
        train_data=load_data(data_path)
        X_train,y_train=prepare_data(train_data)
        model=train_model(X_train,y_train,n_estimators)
        save_model(model, model_name)
    except Exception as e:
        raise Exception(f"Error in model building process: {e}")
if __name__=="__main__":
    main()