import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model import ModelManager
from pathlib import Path

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
        
if __name__ == "__main__":
            
    # 1. Load the configuration
    config = load_config("config.yaml")
    # 2. Use config for data loading
    df = pd.read_csv(config['paths']['data_path'])   
    # 3. Use config for column selection
    X = df.drop(config['features']['target_column'], axis=1)
    y = df[config['features']['target_column']]

    X_train = X.iloc[:5333]
    X_test = X.iloc[5333:]
    y_train = y.iloc[:5333]
    y_test = y.iloc[5333:]

    model = ModelManager(config)
    model.train(X_train,y_train)
    result = model.evaluate(X_test, y_test)

    for key,val in result.items():
        print(f"{key}: {val}")