import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import argparse
from os import path, makedirs
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
import pyaf.ForecastEngine as autof


def train_iteration(X: pd.DataFrame, y: pd.Series, config: Dict ={}, run_name: str="", params: Dict = {}):
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

    time_series = params["time_series"]
    target=config["TS"][time_series]["target"]
    date=config["TS"][time_series]["date"]
    ahead=config["TS"][time_series]["H"]

    X = pd.concat([X[date], y],axis=1)
    X[date] = pd.to_datetime(X[date])
    #print(X)
    # mlflow configs
    mlflow.set_tracking_uri(config["MLFLOW_URI"])
    try:
        mlflow.create_experiment(name=config["EXPERIMENT"])
    except:
        pass
    print(path)
    FDIR = path.join(config["DATA_PATH"], config["MODELS_PATH"], params["time_series"], str(params["iter"]), "PYAF")
    makedirs(FDIR, exist_ok=True)
    FPATH = path.join(FDIR, "MODEL.pkl")

    mlflow.autolog(log_models=True, log_model_signatures=False, silent=True)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        start = time()
        model = autof.cForecastEngine()
        #model.mOptions.enable_slow_mode()
        model.mOptions.set_active_autoregressions(['XGB'])
        model.train(X, date, target, ahead)
        end = time()
        tr_time = end - start

        print(model.getModelInfo())
        
        with open(FPATH, "wb") as f:
            dump(model, f)

        mlflow.log_metric("training_time", tr_time)
        #mlflow.pyfunc.log_model(model, "model")
        mlflow.sklearn.log_model(model, "model")
     
    mlflow.end_run()

def test_iteration(Xtrain:pd.DataFrame, ytrain: pd.Series, Xtest: pd.DataFrame, ytest: pd.Series, config: Dict = {}, run_name: str = "", params: Dict = {}):
    
    time_series = params["time_series"]
    target=config["TS"][time_series]["target"]
    date=config["TS"][time_series]["date"]
    #forecast=config["TS"][time_series]["forecast"]
    ahead=config["TS"][time_series]["H"]
    
    Xtrain = pd.concat([Xtrain[date], ytrain],axis=1)
    Xtrain[date] = pd.to_datetime(Xtrain[date], dayfirst=True)
    print(Xtrain.tail())
    Xtest = pd.concat([Xtest[date], ytest],axis=1)
    Xtest[date] = pd.to_datetime(Xtest[date], dayfirst=True)
    print(Xtest.head())
    # mlflow configs
    mlflow.set_tracking_uri(config["MLFLOW_URI"])

    experiment = dict(mlflow.get_experiment_by_name(config["EXPERIMENT"]))
    runs = mlflow.search_runs([experiment["experiment_id"]])
    run_id = runs[runs['tags.mlflow.runName']==run_name]["run_id"].values[0]

    # Load model
    FDIR = path.join(config["DATA_PATH"], config["MODELS_PATH"], params["time_series"], str(params["iter"]), "PYAF")
    makedirs(FDIR, exist_ok=True)
    FPATH = path.join(FDIR, "MODEL.pkl")
   
    with open(FPATH, "rb") as f:
        model = load(f)

    # Predict and compute metrics
    start = time()
    pred = model.forecast(Xtrain, ahead).iloc[-ahead:,]#.tail(ahead)
    pred = pred[f"{target}_Forecast"].values
    end = time()
    
    inf_time = (end - start) / len(pred)
    metrics = compute_metrics(ytest, pred, "ALL", "test_")
    # Store predictions and target values
    info = pd.DataFrame([ytest, pred]).T
    info.columns = ["y_true", "y_pred"]
    FDIR = path.join(config["DATA_PATH"], config["PRED_PATH"], params['time_series'], "PYAF")
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
    pd.set_option('mode.chained_assignment', None)
    # Get train files
    train_files = list_files(time_series, config, pattern="*_tr.csv")
    test_files = list_files(time_series, config, pattern="*_ts.csv")
    if len(train_files) == 0:
        print("Error: no files found!")
        sys.exit()
    # Train PYAF models
    target = config["TS"][time_series]["target"]
    for n, file in enumerate(tqdm(train_files)):
        params = {
            'time_series': time_series,
            'target': target,
            'model': "PYAF",
            'iter': n+1
        }
        FDIR = path.join(config["DATA_PATH"], config["PRED_PATH"], time_series, params["model"])
        FPATH = path.join(FDIR, f"pred_{str(n+1)}.csv")

        if path.exists(FPATH):
            continue
        run_name = f"{time_series}_{target}_PYAF_{n+1}"
        X, y = load_data(file,config["TS"][time_series]["target"])
        #ta martelado, voltar a ver
        if train:
            train_iteration(X, y, config, run_name, params)
        if test:
           X_ts, y_ts = load_data(test_files[n], target)
           test_iteration(X, y, X_ts, y_ts, config, run_name, params)
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
    # Train/Test PYAF
    main(args.time_series, config, args.train, args.test)