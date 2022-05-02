# Imports
import warnings
warnings.filterwarnings("ignore")
import argparse
from os import makedirs, path
from typing import Dict
import pandas as pd
import sys
from utilities import compute_metrics, load_data, list_files, nmae
import yaml
from tqdm import tqdm
import mlflow
from time import time
from ludwig.api import LudwigModel
from ludwig.contribs.mlflow import MlflowCallback


def lforecast(X, model, target):
    PREDS=[]
    H = X.shape[0]
    for H2 in range(H):
        cols=X.columns
        lags = [int(col.replace("lag","")) for col in cols]
        use_lags = [f"lag{x}" for x in range(len(PREDS), -1, -1) 
                    if (x < H) & (x in lags)]
        X2 = pd.DataFrame(X.iloc[H2,].copy()).T
        if len(use_lags) > 0:
            X2[use_lags] = PREDS[-len(use_lags):]
        X2['features']=[' '.join(map(str,vals)) for vals in X2.values]
        res = model.predict(X2)
        pred = res[target+"_predictions"].values[0]
        PREDS.append(pred)
    return pd.Series(PREDS)

def train_iteration(X: pd.DataFrame, y: pd.Series, config: Dict ={}, run_name: str="", params: Dict = {}):
    
    """AutoML using LUDWIG and storing metrics in MLflow.
    Code based on: https://ludwig-ai.github.io/ludwig-docs/0.4/examples/weather/

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
    URI = config["MLFLOW_URI"]
    mlflow.set_tracking_uri(URI)
    try:
        mlflow.create_experiment(name=config["EXPERIMENT"])
    except:
        pass
    mlflow_cb = MlflowCallback(URI)
    mlflow_cb.experiment_id = '0'

    FDIR = path.join(config["DATA_PATH"], config["MODELS_PATH"],
                     params["time_series"], str(params["iter"]), "LUDWIG")
    makedirs(FDIR, exist_ok=True)
    #model_params = config["MODELS"]["ludwig"]
    target = params['target']

    with mlflow.start_run(run_name=run_name) as run:
        mlflow_cb.run = run
        mlflow.log_params(params)
        #mlflow.log_params(model_params)
        
        X['features']=[' '.join(map(str,vals)) for vals in X.values]
        data = pd.concat([X['features'],y], axis=1)
        model_definition = {
            'input_features': [
                {'name': 'features', 'type': 'timeseries'}
                ], 
            'output_features': [{'name': target, 'type': 'numerical'}]
        }

        start = time()
        # Model training
        model = LudwigModel(model_definition, callbacks=[mlflow_cb])
        _ = model.train(data, output_directory=FDIR,
                        experiment_name=config["EXPERIMENT"],
                        skip_save_processed_input=True)
            
        end = time()
        tr_time = end - start
    mlflow.end_run()

    with mlflow.start_run(run_id=mlflow_cb.run.info.run_id):
        mlflow.log_metric("training_time", tr_time)
        res, _ = model.predict(X)
        pred = res[target+"_predictions"].values
        metrics = compute_metrics(y, pred, "ALL", "training_")
        min_v = config["TS"][params["time_series"]]["min"]
        max_v = config["TS"][params["time_series"]]["max"]
        nmae_ = nmae(y, pred, min_v, max_v)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("training_nmae", nmae_)
    mlflow.end_run()

def test_iteration(X:pd.DataFrame, y: pd.Series, config: Dict = {}, run_name: str = "", params: Dict = {}):
    FDIR = path.join(config["DATA_PATH"], config["PRED_PATH"], params['time_series'], "LUDWIG")
    makedirs(FDIR, exist_ok=True)
    FPATH = path.join(FDIR, f"pred_{str(params['iter'])}.csv")
    # mlflow configs
    mlflow.set_tracking_uri(config["MLFLOW_URI"])
    experiment = dict(mlflow.get_experiment_by_name(config["EXPERIMENT"]))
    runs = mlflow.search_runs([experiment["experiment_id"]])
    run_id = runs[runs['tags.mlflow.runName']==run_name]["run_id"].values[0]
    # Load model
    logged_model = f"runs:/{run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    # Predic and compute metrics
    start = time()
    pred = lforecast(X, loaded_model, params["target"])
    end = time()
    inf_time = (end - start) / len(pred)
    metrics = compute_metrics(y, pred, "ALL", "test_")
    min_v = config["TS"][params["time_series"]]["min"]
    max_v = config["TS"][params["time_series"]]["max"]
    nmae_ = nmae(y, pred, min_v, max_v)
    # Store predictions and target values
    info = pd.DataFrame([y, pred]).T
    info.columns = ["y_true", "y_pred"]
    info.to_csv(FPATH, index=False)
    # Load new info to mlflow run
    with mlflow.start_run(run_id=run_id) as run:
        mlflow.log_artifact(FPATH)
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
    train_files = list_files(time_series, config, pattern="*_tr_reg.csv")
    test_files = list_files(time_series, config, pattern="*_ts_reg.csv")
    if len(train_files) == 0:
        print("Error: no files found!")
        sys.exit()
    # Define a few parameters
    target = config["TS"][time_series]["target"]
    
    # Train/Test LUDWIG models
    for n, (file, file2) in enumerate(tqdm(zip(train_files, test_files))):
        params = {
            'time_series': time_series,
            'target': target,
            'model': "LUDWIG",
            'iter': n+1
        }
        FDIR = path.join(config["DATA_PATH"], config["PRED_PATH"], time_series, params["model"])
        FPATH = path.join(FDIR, f"pred_{str(n+1)}.csv")

        if path.exists(FPATH):
            continue
        run_name = f"{time_series}_{target}_LUDWIG_{n+1}"
        
        X, y = load_data(file, target)
        
        if train:
            train_iteration(X, y, config, run_name, params)
        if test:
            X, y_ts = load_data(file2, target)
            test_iteration(X, y_ts, config, run_name, params)
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
    # Train/Test LUDWIG model
    main(args.time_series, config, args.train, args.test)
    