import pandas as pd
import os
import json
import pickle
import numpy as np
from dvclive import Live
import yaml




from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(file_path:str)->pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}") 


# test_data = pd.read_csv("./data/processed/test_processed_data.csv")

def prepare_data(data:pd.DataFrame):
    try:
        X = data.drop("Potability", axis=1)
        y = data["Potability"]
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")


# X_test = test_data.iloc[:, 0:-1].values
# y_test = test_data.iloc[:, -1].values

def load_model(file_path: str):
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {file_path}: {e}")

# model=pickle.load(open("model.pkl", "rb"))    

def evaluate_model(model, X_test, y_test):
    try:
        params=yaml.safe_load(open("params.yaml", "r"))
        test_size=params["data_collection"]["test_size"]
        n_estimators = params["model_building"]["n_estimators"]
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        with Live(save_dvc_exp=True) as live:
            live.log_metric("accuracy", acc)
            live.log_metric("precision", prec)
            live.log_metric("recall", rec)
            live.log_metric("f1_score", f1)

            live.log_param("test_size", test_size)
            live.log_param("n_estimators", n_estimators)
        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }
        return metrics
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")


# y_pred = model.predict(X_test)

# acc=accuracy_score(y_test, y_pred)
# prec=precision_score(y_test, y_pred)
# rec=recall_score(y_test, y_pred)
# f1=f1_score(y_test, y_pred)

# meterics = {
#     "accuracy": acc,
#     "precision": prec,
#     "recall": rec,
#     "f1_score": f1
# }

def save_metrics(metrics, file_path: str):
    try:
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {file_path}: {e}")
    
# with open("metrics.json", "w") as f:
#     json.dump(meterics, f,indent=4)
    
def main():
    try:
        test_data_path="./data/processed/test_processed_data_mean.csv"
        model_path="models/model.pkl"
        metrics_path="reports/metrics.json"
        test_data=load_data(test_data_path)
        X_test,y_test=prepare_data(test_data)
        model=load_model(model_path)
        metrics=evaluate_model(model, X_test, y_test)
        save_metrics(metrics, metrics_path)
    except Exception as e:
        raise Exception(f"Error in model evaluation process: {e}")
    
if __name__=="__main__":
    main()