# Imports
import argparse
from os import makedirs, path, getcwd
from typing import Dict
from sklearn.linear_model import LinearRegression
import pandas as pd
import sys
from utilities import compute_metrics, load_data, list_files
import yaml
from tqdm import tqdm
import mlflow
from time import time

def train_iteration(X: pd.DataFrame, y: pd.Series, config: Dict = {}, run_name: str= "", params: Dict = {}):
    """Train a Linear Regression model and storing metrics in MLflow.

    Parameters
    ----------
    X : pd.DataFrame
        X data.
    y : pd.Series
        Target values.
    config : Dict, optional
        Configuration dict from config.yaml file, by default {}
    run_name : str, optional
        Run name for MLflow, by default "" (empty)
    """
    # mlflow configs
    mlflow.set_tracking_uri("http://localhost:5000")
    try:
        mlflow.create_experiment(name=config["EXPERIMENT"])
    except:
        pass
    
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        start = time()
        _ = LinearRegression().fit(X, y)
        end = time()

        tr_time = end - start
        mlflow.log_metric("training_time", tr_time)
    mlflow.end_run()

def test_iteration(X: pd.DataFrame, y: pd.Series, config: Dict = {}, run_name: str = "", params: Dict = {}):
    # mlflow configs
    mlflow.set_tracking_uri("http://localhost:5000")

    experiment = dict(mlflow.get_experiment_by_name(config["EXPERIMENT"]))
    runs = mlflow.search_runs([experiment["experiment_id"]])
    run_id = runs[runs['tags.mlflow.runName']==run_name]["run_id"].values[0]
    # Load model
    logged_model = f"runs:/{run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    # Predic and compute metrics
    start = time()
    pred = loaded_model.predict(X)
    end = time()
    inf_time = (end - start) / len(pred)
    metrics = compute_metrics(y, pred, "ALL", "test_")
    # Store predictions and target values
    info = pd.DataFrame([y, pred]).T
    info.columns = ["y_true", "y_pred"]
    FDIR = path.join(config["DATA_PATH"], config["PRED_PATH"], params['time_series'], "LR")
    makedirs(FDIR, exist_ok=True)
    FPATH = path.join(FDIR, f"pred_{str(params['iter'])}.csv")
    info.to_csv(FPATH, index=False)
    # Load new info to mlflow run
    with mlflow.start_run(run_id=run_id) as run:
        mlflow.log_artifact(FPATH)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("test_time", inf_time)
    mlflow.end_run()

def main(time_series: str, config: dict = {}, train: bool = True, test: bool = True):
    """Read all Rolling Window iterarion training files from a given time-series and train a Linear Regression model for each.

    Parameters
    ----------
    time_series : str
        Time-series name.
    config : dict, optional
        Configuration dict from config.yaml file, by default {}
    train: bool, optional
        Whether performs model training or not, by default True (it does)
    test: bool, optional
        Whether performs model testing/evaluation or not, by default True (it does)
    """
    # Get train files
    train_files = list_files(time_series, config, pattern="tr_?_reg.csv")
    test_files = list_files(time_series, config, pattern="ts_?_reg.csv")
    if len(train_files) == 0:
        print("Error: no files found!")
        sys.exit()
    # Train LR models
    target = config["TS"][time_series]["target"]
    for n, (file, file2) in enumerate(tqdm(zip(train_files, test_files))):
        params = {
            'time_series': time_series,
            'target': target,
            'model': "LR",
            'iter': n+1
        }
        run_name = f"{time_series}_{target}_LR_{n+1}"
        if train:
            X, y = load_data(file,config["TS"][time_series]["target"])
            train_iteration(X, y, config, run_name, params)
        if test:
            X, y = load_data(file2,config["TS"][time_series]["target"])
            test_iteration(X, y, config, run_name, params)
        # TODO: remove this for all train/test datasets
        #break


if __name__ == "__main__":
    # Read arguments
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Generate CV data files for a time-series dataset, based on config.yaml params')
    parser.add_argument(help='Time-series name.', dest="time_series")
    parser.set_defaults(time_series="porto")
    parser.add_argument('-c', '--config', dest='config', 
                        help='Config yaml file.')
    parser.set_defaults(config="config.yaml")
    parser.add_argument('-tr', '--train', dest="train",
                        action=argparse.BooleanOptionalAction,
                        help="Performs model training.")
    parser.add_argument('-ts', '--test', dest="test",
                        action=argparse.BooleanOptionalAction,
                        help="Performs model testing (evaluation).")
    args = parser.parse_args()
    # Load configs
    try:
        config =  yaml.safe_load(open(args.config))
    except Exception as e:
        print("Error loading config file: ", e)
        sys.exit()
    # Train/test LR
    main(args.time_series, config, args.train, args.test)
    