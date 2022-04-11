import pandas as pd
import argparse
from os import path, getcwd
from typing import Dict
from sklearn.linear_model import LinearRegression
import pandas as pd
import sys
from utilities import load_data, list_files
import yaml
from tqdm import tqdm
import mlflow
from time import time
from gluonts.model.deepar import DeepAREstimator
from gluonts.dataset.common import ListDataset
from gluonts.mx.trainer import Trainer


def train(X: pd.DataFrame, y: pd.Series, config: Dict ={}, run_name: str="", params: Dict = {}):
    """_summary_

    Parameters
    ----------
    X : pd.DataFrame
        _description_
    y : pd.Series
        _description_
    config : Dict, optional
        _description_, by default {}
    run_name : str, optional
        _description_, by default ""
    """
    # mlflow configs
    mlflow.set_tracking_uri("http://localhost:5000")
    try:
        mlflow.create_experiment(name=config["EXPERIMENT"])
    except:
        pass
    
    mlflow.autolog()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        start = time()
        training_data = ListDataset([{"start":y.index[0], "target": y}], freq= "D")
        model = DeepAREstimator(freq="D", prediction_length=30, trainer=Trainer(epochs=5)).train(training_data)
        end = time()

        tr_time = end - start
        mlflow.log_metric("training_time", tr_time)
    mlflow.end_run()

def main(time_series: str, config: dict = {}):
    """_summary_

    Parameters
    ----------
    time_series : str
        _description_
    config : dict, optional
        _description_, by default {}
    """
    # Get train files
    train_files = list_files(time_series, config, pattern="tr_?.csv")
    if len(train_files) == 0:
        print("Error: no files found!")
        sys.exit()
    # Train Gluon models
    target = config["TS"][time_series]["target"]
    for n, file in enumerate(tqdm(train_files)):
        params = {
            'time_series': time_series,
            'target': target,
            'model': "Gluon",
            'iter': n+1
        }
        run_name = f"{time_series}_{target}_Gluon_{n+1}"
        X, y = load_data(file,config["TS"][time_series]["target"])
        train(X, y, config, run_name, params)


if __name__ == "__main__":
    # Read arguments
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Generate CV data files for a time-series dataset, based on config.yaml params')
    parser.add_argument(help='Time-series name.', dest="time_series")
    parser.set_defaults(time_series="porto")
    parser.add_argument('-c', '--config', dest='config', 
                        help='Config yaml file.')
    parser.set_defaults(config="config.yaml")
    args = parser.parse_args()
    # Load configs
    try:
        config =  yaml.safe_load(open(args.config))
    except Exception as e:
        print("Error loading config file: ", e)
        sys.exit()
    # Train LR
    main(args.time_series, config)