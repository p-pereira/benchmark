# Imports
import argparse
from os import makedirs, path
from pickle import dump, load
from typing import Dict
import numpy as np
import pandas as pd
import sys
from utilities import compute_metrics, load_data, list_files, nmae
import yaml
from tqdm import tqdm
import mlflow
from time import time
from sktime.forecasting.model_selection import SlidingWindowSplitter, ForecastingGridSearchCV
from sktime.forecasting.compose import MultiplexForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.naive import NaiveForecaster
import warnings
warnings.filterwarnings("ignore")


def train_iteration(y: pd.Series, config: Dict ={}, run_name: str="", params: Dict = {}):
    
    """AutoML using SKTIME and storing metrics in MLflow.
    Code based on: https://towardsdatascience.com/why-start-using-sktime-for-forecasting-8d6881c0a518

    Parameters
    ----------
    y: pd.Series
        Tim-series values.
    config : Dict, optional
        Configuration dict from config.yaml file, by default {}
    run_name : str, optional
        Run name for MLflow, by default ""
    params : Dict, optional
        Run/model parameters, by default {} (empty)
    """
    # mlflow configs
    mlflow.set_tracking_uri(config["MLFLOW_URI"])
    try:
        mlflow.create_experiment(name=config["EXPERIMENT"])
    except:
        pass
    
    time_series = params["time_series"]
    FDIR = path.join(config["DATA_PATH"], config["MODELS_PATH"],
                     time_series, str(params["iter"]), "SKTIME")
    FDIR2 = path.join(config["DATA_PATH"], config["MODELS_PATH"],
                      time_series, "1", "SKTIME")
    makedirs(FDIR, exist_ok=True)
    FPATH = path.join(FDIR, "MODEL.pkl")
    FPATH2 = path.join(FDIR2, "MODEL.pkl")
    
    H = config["TS"][time_series]["H"]
    if time_series=="madrid":
        des = False
    else:
        des=True

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        
        start = time()
        if params["iter"] == 1:
            forecaster = MultiplexForecaster(
                forecasters=[
                    ("theta", ThetaForecaster(deseasonalize=des)),
                    ("autoets", AutoETS()),
                    ("arima", AutoARIMA(suppress_warnings=True, n_jobs=-1)),
                    ("naive", NaiveForecaster())
                ],
            )
            cv = SlidingWindowSplitter(fh=H, window_length=y.shape[0]-H)
            forecaster_param_grid = {"selected_forecaster": ["theta", 
                                                            "autoets",
                                                            "arima",
                                                            "naive"]}

            gscv = ForecastingGridSearchCV(forecaster, cv=cv,
                                        param_grid=forecaster_param_grid)
            _ = gscv.fit(y)
        else:
            with open(FPATH2, "rb") as f:
                gsvc = load(f)
            gscv = gsvc.update(y, update_params=False)
        
        end = time()
        tr_time = end - start

        with open(FPATH, "wb") as f:
            dump(gscv, f)
        
        mlflow.log_metric("training_time", tr_time)
        mlflow.log_artifact(FPATH)
    mlflow.end_run()

def test_iteration(y: pd.Series, config: Dict = {}, run_name: str = "", params: Dict = {}):
    time_series = params["time_series"]
    H = config["TS"][time_series]["H"]
    # mlflow configs
    mlflow.set_tracking_uri(config["MLFLOW_URI"])
    # Get mlflow run id to load the model.
    experiment = dict(mlflow.get_experiment_by_name(config["EXPERIMENT"]))
    runs = mlflow.search_runs([experiment["experiment_id"]])
    run_id = runs[runs['tags.mlflow.runName']==run_name]["run_id"].values[0]
    # Load model
    FDIR = path.join(config["DATA_PATH"], config["MODELS_PATH"],
                     time_series, str(params["iter"]), "SKTIME")
    FPATH = path.join(FDIR, "MODEL.pkl")
    with open(FPATH, "rb") as f:
        model = load(f)
    # Predic and compute metrics
    start = time()
    pred = [model.predict(fh=n+1).values[0] for n in range(H)]
    end = time()
    inf_time = (end - start) / len(pred)
    metrics = compute_metrics(y, pred, "ALL", "test_")
    min_v = config["TS"][params["time_series"]]["min"]
    max_v = config["TS"][params["time_series"]]["max"]
    nmae_ = nmae(y, pred, min_v, max_v)
    # Store predictions and target values
    info = pd.DataFrame([y, pd.Series(pred)]).T
    info.columns = ["y_true", "y_pred"]
    FDIR2 = path.join(config["DATA_PATH"], config["PRED_PATH"], params['time_series'], "SKTIME")
    makedirs(FDIR2, exist_ok=True)
    FPATH2 = path.join(FDIR2, f"pred_{str(params['iter'])}.csv")
    info.to_csv(FPATH2, index=False)
    # Load new info to mlflow run
    with mlflow.start_run(run_id=run_id) as run:
        mlflow.log_artifact(FPATH2)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("test_time", inf_time)
        mlflow.log_metric("test_nmae", nmae_)
    mlflow.end_run()


def main(time_series: str, config: dict = {}, train: bool = True, test: bool = True):
    """Read all Rolling Window iterarion training files from a given time-series and train a SKTIME model for each.

    Parameters
    ----------
    time_series : str
        _description_
    config : dict, optional
        Configuration dict from config.yaml file, by default {}
    """
    # Get train files
    train_files = list_files(time_series, config, pattern="*_tr.csv")
    test_files = list_files(time_series, config, pattern="*_ts.csv")
    if len(train_files) == 0:
        print("Error: no files found!")
        sys.exit()
    # Define a few parameters
    target = config["TS"][time_series]["target"]
    # Train SKTIME models
    for n, (file, file2) in tqdm(enumerate(zip(train_files, test_files))):
        params = {
            'time_series': time_series,
            'target': target,
            'model': "SKTIME",
            'iter': n+1
        }
        FDIR = path.join(config["DATA_PATH"], config["PRED_PATH"], time_series, params["model"])
        FPATH = path.join(FDIR, f"pred_{str(n+1)}.csv")

        if path.exists(FPATH):
            continue
        run_name = f"{time_series}_{target}_SKTIME_{n+1}"
        
        if train:
            _, y = load_data(file, target)
            train_iteration(y, config, run_name, params)
        if test:
            _, y_ts = load_data(file2, target)
            test_iteration(y_ts, config, run_name, params)
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
    # Train/Test sktime model
    if args.time_series == "ALL":
        for time_series in config["TS"].keys():
            main(time_series, config, args.train, args.test)
    else:
        main(args.time_series, config, args.train, args.test)
    