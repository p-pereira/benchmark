from pickle import dump
import pandas as pd
import argparse
from os import makedirs, path, getcwd
from typing import Dict
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
    
    mlflow.gluon.autolog()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        start = time()
        training_data = ListDataset([{"start":y.index[0], "target": y}], freq= "D")
        model = DeepAREstimator(freq="D", prediction_length=30, trainer=Trainer(epochs=5)).train(training_data)
        end = time()

        tr_time = end - start
        mlflow.log_metric("training_time", tr_time)
        FDIR = path.join(config["DATA_PATH"], config["MODELS_PATH"],params["time_series"], str(params["iter"]), "GLUON")
        makedirs(FDIR, exist_ok=True)
        FPATH = path.join(FDIR, "MODEL.pkl")
        with open(FPATH, "wb") as f:
            dump(model, f)
        mlflow.log_artifact(FPATH)
        
    mlflow.end_run()

def test_iteration(y: pd.Series, config: Dict = {}, run_name: str = "", params: Dict = {}):
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
    test_data = ListDataset([{"start": y.index[0], "target": y}], freq= "D")
    pred = loaded_model.predict(test_data)
    end = time()
    inf_time = (end - start) / len(pred)
    metrics = compute_metrics(y, pred, "ALL", "test_")
    # Store predictions and target values
    info = pd.DataFrame([y, pred]).T
    info.columns = ["y_true", "y_pred"]
    FDIR = path.join(config["DATA_PATH"], config["PRED_PATH"], params['time_series'], "Gluon")
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
        #train(X, y, config, run_name, params)
        # TODO: remove next line
        if train:
            train(X, y, config, run_name, params)
        if test:
            #X, y_ts = load_data(file, target)
            test_iteration(X, y, config, run_name, params)
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
    # Train LR
    main(args.time_series, config, args.train, args.test)