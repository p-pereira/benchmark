# Imports
import warnings
warnings.filterwarnings("ignore")
import argparse
from os import makedirs, path
from pickle import dump, load
from typing import Dict
import pandas as pd
import sys
from utilities import compute_metrics, list_files, nmae
import yaml
import mlflow
from time import time
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.data.data import InputData
from fedot.api.main import Fedot
from tqdm import tqdm

def train_iteration(data: InputData, task: Task, config: Dict ={}, run_name: str="", params: Dict = {}):
    
    """AutoML using FEDOT and storing metrics in MLflow.
    Code based on: https://github.com/nccr-itmo/FEDOT/blob/master/examples/advanced/time_series_forecasting/multistep.py

    Parameters
    ----------
    data : InputData
        Time-series fedot InputData object.
    task : Task
        Fedot task object.
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
    
    FDIR = path.join(config["DATA_PATH"], config["MODELS_PATH"],
                     params["time_series"], str(params["iter"]), "FEDOT")
    FDIR2 = path.join(config["DATA_PATH"], config["MODELS_PATH"],
                     params["time_series"], "1", "FEDOT")
    makedirs(FDIR, exist_ok=True)
    FPATH = path.join(FDIR, "MODEL.pkl")
    FPATH2 = path.join(FDIR2, "PIPELINE.pkl")
    model_params = config["MODELS"]["fedot"]

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_params(model_params)
        
        start = time()
        # Task selection, initialisation of the framework
        fedot_model = Fedot(problem='ts_forecasting',
                            task_params=task.task_params, 
                            **model_params)
        if params["iter"] > 1:
            with open(FPATH2, "rb") as f:
                pipeline = load(f)
            _ = fedot_model.fit(features=data, predefined_model=pipeline)
        else:
            _ = fedot_model.fit(features=data)
        end = time()
        tr_time = end - start

        if params["iter"] == 1:
            pipeline = fedot_model.current_pipeline
            with open(FPATH2, "wb") as f:
                dump(pipeline, f)
        
        with open(FPATH, "wb") as f:
            dump(fedot_model, f)
        mlflow.log_metric("training_time", tr_time)

        mlflow.log_artifact(FPATH)
        mlflow.log_artifact(FPATH2)
    mlflow.end_run()

def test_iteration(history: InputData, test_data: InputData, config: Dict = {}, run_name: str = "", params: Dict = {}):
    # mlflow configs
    mlflow.set_tracking_uri(config["MLFLOW_URI"])
    # Get mlflow run id to load the model.
    experiment = dict(mlflow.get_experiment_by_name(config["EXPERIMENT"]))
    runs = mlflow.search_runs([experiment["experiment_id"]])
    run_id = runs[runs['tags.mlflow.runName']==run_name]["run_id"].values[0]
    # Load model
    FDIR = path.join(config["DATA_PATH"], config["MODELS_PATH"],
                     params["time_series"], str(params["iter"]), "FEDOT")
    makedirs(FDIR, exist_ok=True)
    FPATH = path.join(FDIR, "MODEL.pkl")
    with open(FPATH, "rb") as f:
        model = load(f)
    # Predic and compute metrics
    start = time()
    pred = model.predict(history)
    end = time()
    inf_time = (end - start) / len(pred)
    y = test_data.target
    metrics = compute_metrics(y, pred, "ALL", "test_")
    min_v = config["TS"][params["time_series"]]["min"]
    max_v = config["TS"][params["time_series"]]["max"]
    nmae_ = nmae(y, pred, min_v, max_v)
    # Store predictions and target values
    info = pd.DataFrame([y, pred]).T
    info.columns = ["y_true", "y_pred"]
    FDIR2 = path.join(config["DATA_PATH"], config["PRED_PATH"], params['time_series'], "FEDOT")
    makedirs(FDIR2, exist_ok=True)
    FPATH2 = path.join(FDIR2, f"pred_{str(params['iter'])}.csv")
    info.to_csv(FPATH2, index=False)
    # Load new info to mlflow run
    
    with mlflow.start_run(run_id=run_id) as run:
        mlflow.log_artifact(FPATH)
        mlflow.log_artifact(FPATH2)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("test_time", inf_time)
        mlflow.log_metric("test_nmae", nmae_)
    mlflow.end_run()


def main(time_series: str, config: dict = {}, train: bool = True, test: bool = True):
    """Read all Rolling Window iterarion training files from a given time-series and train a Linear Regression model for each.

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
    H = config["TS"][time_series]["H"]
    target = config["TS"][time_series]["target"]
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=H))
    # Train FEDOT models
    for n, (file, file2) in tqdm(enumerate(zip(train_files, test_files))):
        params = {
            'time_series': time_series,
            'target': target,
            'model': "FEDOT",
            'iter': n+1
        }
        FDIR = path.join(config["DATA_PATH"], config["PRED_PATH"], time_series, params["model"])
        FPATH = path.join(FDIR, f"pred_{str(n+1)}.csv")

        if path.exists(FPATH):
            continue
        run_name = f"{time_series}_{target}_FEDOT_{n+1}"
        
        tr_data = InputData.from_csv_time_series(task, file, target_column=target)
        if train:
            train_iteration(tr_data, task, config, run_name, params)
        if test:
            ts_data = InputData.from_csv_time_series(task, file2, target_column=target)
            test_iteration(tr_data, ts_data, config, run_name, params)
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
    # Train/Test FEDOT model
    if args.time_series == "ALL":
        for time_series in config["TS"].keys():
            main(time_series, config, args.train, args.test)
    else:
        main(args.time_series, config, args.train, args.test)
    