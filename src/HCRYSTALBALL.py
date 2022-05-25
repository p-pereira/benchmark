import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import argparse
from os import path, getcwd, makedirs
from typing import Dict
import pandas as pd
import sys
from utilities import compute_metrics, load_data, list_files, nmae
import yaml
import mlflow
from tqdm import tqdm
from time import time
from pickle import dump, load
from hcrystalball.wrappers import ProphetWrapper
from hcrystalball.model_selection import ModelSelector
import numpy as np

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
    date=config["TS"][time_series]["date"] 
    target=config["TS"][time_series]["target"]
    fr=config["TS"][time_series]["freq"]
    print("freq ",fr)
    h=config["TS"][time_series]["H"]
    df = pd.concat([X[date], y],axis=1)
    df = df.set_index(date)
    df = df.asfreq(freq=fr)
    df.index=pd.to_datetime(df.index)
    X = X.set_index(date)
    X.index=pd.to_datetime(X.index)
    # mlflow configs
    mlflow.set_tracking_uri(config["MLFLOW_URI"])
    try:
        mlflow.create_experiment(name=config["EXPERIMENT"])
    except:
        pass
    
    FDIR = path.join(config["DATA_PATH"], config["MODELS_PATH"], params["time_series"], str(params["iter"]), "HCRYSTALBALL")
    makedirs(FDIR, exist_ok=True)
    FPATH = path.join(FDIR, "MODEL.pkl")

    #mlflow.autolog(log_models=False, log_model_signatures=False, silent=True)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        start = time()
        ms = ModelSelector(horizon=h,
                   frequency=fr,
                  )
        ms.create_gridsearch(n_splits=1,
                     sklearn_models=True,
                     sklearn_models_optimize_for_horizon=True,
                     autosarimax_models = True,
                     prophet_models=True,
                     exp_smooth_models = True,
                     #average_ensembles = True,
                     stacking_ensembles = True,
                     theta_models=True,
                     tbats_models=True,
                     hcb_verbose = False
                     )
        ms.select_model(df=df,
                target_col_name=target,
                )
        #X=X.drop(['tempC'], axis=1)
        #print(X)
        model=ms.results[0].best_model.fit(X,y)
        end = time()
        tr_time = end - start
        #print(model)
        with open(FPATH, "wb") as f:
            dump(model, f)

        mlflow.log_metric("training_time", tr_time)
        #mlflow.log_artifact(FPATH)
        mlflow.sklearn.log_model(model, "model")
     
    mlflow.end_run()

def test_iteration(X:pd.DataFrame, y: pd.Series, config: Dict = {}, run_name: str = "", params: Dict = {}):
    
    time_series = params["time_series"]   
    date=config["TS"][time_series]["date"] 
    X = X.set_index(date)
    X.index=pd.to_datetime(X.index)
    

    # mlflow configs
    mlflow.set_tracking_uri(config["MLFLOW_URI"])

    experiment = dict(mlflow.get_experiment_by_name(config["EXPERIMENT"]))
    runs = mlflow.search_runs([experiment["experiment_id"]])
    run_id = runs[runs['tags.mlflow.runName']==run_name]["run_id"].values[0]

    # Load model
    FDIR = path.join(config["DATA_PATH"], config["MODELS_PATH"], params["time_series"], str(params["iter"]), "HCRYSTALBALL")
    makedirs(FDIR, exist_ok=True)
    FPATH = path.join(FDIR, "MODEL.pkl")
   
    with open(FPATH, "rb") as f:
        model = load(f)
    # Predict and compute metrics
    start = time()
    pred = model.predict(X)
    end = time()
    inf_time = (end - start) / len(pred)
    pred = pred[pred.columns[-1]].to_numpy()
    metrics = compute_metrics(y, pred, "ALL", "test_")
    min_v = config["TS"][params["time_series"]]["min"]
    max_v = config["TS"][params["time_series"]]["max"]
    nmae_ = nmae(y, pred, min_v, max_v)
    # Store predictions and target values
    info = pd.DataFrame([y, pred]).T
    info.columns = ["y_true", "y_pred"]
    FDIR = path.join(config["DATA_PATH"], config["PRED_PATH"], params['time_series'], "HCRYSTALBALL")
    makedirs(FDIR, exist_ok=True)
    FPATH = path.join(FDIR, f"pred_{str(params['iter'])}.csv")
    info.to_csv(FPATH, index=False)
    # Load new info to mlflow run
    with mlflow.start_run(run_id=run_id) as run:
        mlflow.log_artifact(FPATH)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("test_time", inf_time)
        mlflow.log_metric("test_nmae", nmae_)
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
            'model': "HCRYSTALBALL",
            'iter': n+1
        }
        run_name = f"{time_series}_{target}_HCRYSTALBALL_{n+1}"
        X, y = load_data(file,config["TS"][time_series]["target"])
        if train:
            train_iteration(X, y, config, run_name, params)
        if test:
           X_ts, y_ts = load_data(test_files[n], target)
           test_iteration(X_ts, y_ts, config, run_name, params)
        

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