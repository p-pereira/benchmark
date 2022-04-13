import argparse
from os import path, getcwd, makedirs
from typing import Dict
from autots import AutoTS
import pandas as pd
import sys
from utilities import compute_metrics, load_data, list_files
import yaml
import mlflow
from tqdm import tqdm
from time import time
from pickle import dump, load

def train_iteration(X: pd.DataFrame, y: pd.Series, config: Dict ={}, run_name: str="", params: Dict = {}):
    print(y)
    
    X = pd.concat([X.date_time, y],axis=1)
    X.index = pd.to_datetime(X.index)

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
    
    FDIR = path.join(config["DATA_PATH"], config["MODELS_PATH"], params["time_series"], str(params["iter"]), "AUTOTS")
    makedirs(FDIR, exist_ok=True)
    FPATH = path.join(FDIR, "MODEL.pkl")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        start = time()
        model = AutoTS(
            forecast_length=30,
            frequency='infer',
            ensemble=None,
            model_list="superfast", 
            transformer_list="superfast",
            max_generations=2,
            num_validations=2
        )

        model = model.fit(X)
        end = time()
        tr_time = end - start

        with open(FPATH, "wb") as f:
            dump(model, f)

        mlflow.log_metric("training_time", tr_time)
        mlflow.pmdarima.log_model(model, "model")
        
    mlflow.end_run()

def test_iteration(X: pd.DataFrame, y: pd.Series, config: Dict = {}, run_name: str = "", params: Dict = {}):
    # mlflow configs
    mlflow.set_tracking_uri(config["MLFLOW_URI"])

    experiment = dict(mlflow.get_experiment_by_name(config["EXPERIMENT"]))
    runs = mlflow.search_runs([experiment["experiment_id"]])
    run_id = runs[runs['tags.mlflow.runName']==run_name]["run_id"].values[0]

    # Load model
    FDIR = path.join(config["DATA_PATH"], config["MODELS_PATH"], params["time_series"], str(params["iter"]), "AUTOTS")
    makedirs(FDIR, exist_ok=True)
    FPATH = path.join(FDIR, "MODEL.pkl")
   
    with open(FPATH, "rb") as f:
        model = load(f)
   
    print("------------------------------")
    print(model)
    print("------------------------------")
    # Predic and compute metrics
    start = time()
    pred = model.predict()
    print("------------------------------")
    print(pred)
    #prev = pd.to_datetime(pred.forecast.date_time)
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    pred = pred.forecast.set_index('date_time')
    pred.index = pd.to_datetime(pred.index)
    print(pred)
    print("??????????????????????")
    end = time()
    inf_time = (end - start) / len(pred)
    metrics = compute_metrics(y, pred, "ALL", "test_")
    # Store predictions and target values
    info = pd.DataFrame([y, pred]).T
    info.columns = ["y_true", "y_pred"]
    FDIR = path.join(config["DATA_PATH"], config["PRED_PATH"], params['time_series'], "AUTOTS")
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
    """_summary_

    Parameters
    ----------
    time_series : str
        _description_
    config : dict, optional
        _description_, by default {}
    """
    # Get train files
    train_files = list_files(time_series, config, pattern="?_tr.csv")
    test_files = list_files(time_series, config, pattern="?_ts.csv")
    if len(train_files) == 0:
        print("Error: no files found!")
        sys.exit()
    # Train AUTOTS models
    target = config["TS"][time_series]["target"]
    for n, file in enumerate(tqdm(train_files)):
        params = {
            'time_series': time_series,
            'target': target,
            'model': "AUTOTS",
            'iter': n+1
        }
        run_name = f"{time_series}_{target}_AUTOTS_{n+1}"
        X, y = load_data(file,config["TS"][time_series]["target"])
        #ta martelado, voltar a ver
        if train:
            train_iteration(X, y, config, run_name, params)
        if test:
           X_ts, y_ts = load_data(test_files[n], target)
           test_iteration(X_ts, y_ts, config, run_name, params)
        break

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
    # Train/Test AUTOTS
    main(args.time_series, config, args.train, args.test)