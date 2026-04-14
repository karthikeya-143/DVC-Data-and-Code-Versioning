import pandas as pd
import os
import numpy as np

def load_data(file_path:str)->pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")

# train_data = pd.read_csv("./data/raw/train_data.csv")
# test_data = pd.read_csv("./data/raw/test_data.csv")

def fill_missing_with_median(df):
    try:
        for column in df.columns:
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
        return df
    except Exception as e:
        raise Exception(f"Error filling missing values: {e}")

def save_data(df,filepath):
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")


# train_processed_data = fill_missing_with_median(train_data)
# test_processed_data = fill_missing_with_median(test_data)

def main():
    try:
        raw_data_path="./data/raw"
        processed_data_path="./data/processed"
        train_data = load_data(os.path.join(raw_data_path, "train_data.csv"))
        test_data = load_data(os.path.join(raw_data_path, "test_data.csv"))
        train_proceesed_data=fill_missing_with_median(train_data)
        test_processed_data=fill_missing_with_median(test_data)
        os.makedirs(processed_data_path, exist_ok=True)
        save_data(train_proceesed_data, os.path.join(processed_data_path, "train_processed_data.csv"))
        save_data(test_processed_data, os.path.join(processed_data_path, "test_processed_data.csv"))
    except Exception as e:
        raise Exception(f"Error in data preparation process: {e}")
if __name__=="__main__":
    main()
# data_path = os.path.join("data", "processed")
# os.makedirs(data_path, exist_ok=True)

# train_processed_data.to_csv(os.path.join(data_path, "train_processed_data.csv"), index=False)
# test_processed_data.to_csv(os.path.join(data_path, "test_processed_data.csv"), index=False)