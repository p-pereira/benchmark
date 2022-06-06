import warnings
warnings.filterwarnings("ignore")
from pickle import dump, load
import pandas as pd
import argparse
from os import makedirs, path
from typing import Dict
import pandas as pd
import sys
from utilities import load_data, list_files, compute_metrics
import yaml
from tqdm import tqdm
import mlflow
from time import time
from gluonts.model.deepar import DeepAREstimator as DeepAR
from gluonts.dataset.common import ListDataset


def train_iteration(X: pd.DataFrame, y: pd.Series, config: Dict ={}, run_name: str="", params: Dict = {}):
    """Train a DeepAR model and storing metrics in MLflow.

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
    params : Dict, optional
        Run/model parameters, by default {} (empty)
    """
    # mlflow configs
    mlflow.set_tracking_uri(config["MLFLOW_URI"])
    try:
        mlflow.create_experiment(name=config["EXPERIMENT"])
    except:
        pass

    FDIR = path.join(config["DATA_PATH"], config["MODELS_PATH"],params["time_series"], str(params["iter"]), "DEEPAR")
    makedirs(FDIR, exist_ok=True)
    FPATH = path.join(FDIR, "MODEL.pkl")
    
    date_col = config["TS"][params["time_series"]]["date"]
    freq = config["TS"][params["time_series"]]["freq"]
    H = config["TS"][params["time_series"]]["H"]
    X[date_col] = pd.to_datetime(X[date_col])

    mlflow.gluon.autolog()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        start = time()
        training_data = ListDataset([{"start":X[date_col][0], "target": y}], freq=freq)
        model = DeepAR(freq=freq, prediction_length=H).train(training_data)
        end = time()

        tr_time = end - start
        mlflow.log_metric("training_time", tr_time)
        with open(FPATH, "wb") as f:
            dump(model, f)
        mlflow.log_artifact(FPATH)
        
    mlflow.end_run()

def test_iteration(X: pd.DataFrame, y: pd.Series, config: Dict = {}, run_name: str = "", params: Dict = {}):
    """Test a DeepAR model and storing metrics in MLflow.

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
    params : Dict, optional
        Run/model parameters, by default {} (empty)
    """
    # mlflow configs
    mlflow.set_tracking_uri(config["MLFLOW_URI"])

    experiment = dict(mlflow.get_experiment_by_name(config["EXPERIMENT"]))
    runs = mlflow.search_runs([experiment["experiment_id"]])
    run_id = runs[runs['tags.mlflow.runName']==run_name]["run_id"].values[0]

    # Load model
    FDIR = path.join(config["DATA_PATH"], config["MODELS_PATH"],
                     params["time_series"], str(params["iter"]), "DEEPAR")
    makedirs(FDIR, exist_ok=True)
    FPATH = path.join(FDIR, "MODEL.pkl")
    with open(FPATH, "rb") as f:
        model = load(f)
    
    date_col = config["TS"][params["time_series"]]["date"]
    freq = config["TS"][params["time_series"]]["freq"]
    X[date_col] = pd.to_datetime(X[date_col])
    # Predic and compute metrics
    start = time()
    test_data = ListDataset([{"start": X[date_col][0], "target": y}], freq= freq)
    res = model.predict(test_data)
    pred = [x.mean for x in res][0]
    end = time()
    inf_time = (end - start) / len(pred)
    metrics = compute_metrics(y, pred, "ALL", "test_")
    # Store predictions and target values
    info = pd.DataFrame([y, pred]).T
    info.columns = ["y_true", "y_pred"]
    FDIR = path.join(config["DATA_PATH"], config["PRED_PATH"], params['time_series'], "DEEPAR")
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
    """Read all Rolling Window iterarion training files from a given time-series and train a DeepAR model for each.

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
    train_files = list_files(time_series, config, pattern="*_tr.csv")
    test_files = list_files(time_series, config, pattern="*_ts.csv")
    if len(train_files) == 0:
        print("Error: no files found!")
        sys.exit()
    # Train Gluon models
    target = config["TS"][time_series]["target"]
    for n, file in enumerate(tqdm(train_files)):
        params = {
            'time_series': time_series,
            'target': target,
            'model': "DEEPAR",
            'iter': n+1
        }
        FDIR = path.join(config["DATA_PATH"], config["PRED_PATH"], time_series, params["model"])
        FPATH = path.join(FDIR, f"pred_{str(n+1)}.csv")

        if path.exists(FPATH):
            continue
        run_name = f"{time_series}_{target}_DEEPAR_{n+1}"
        X, y = load_data(file,target)
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
    parser.set_defaults(train=False)
    parser.add_argument('-ts', '--test', dest="test",
                        action=argparse.BooleanOptionalAction,
                        help="Performs model testing (evaluation).")
    parser.set_defaults(train=True)
    args = parser.parse_args()
    # Load configs
    try:
        config =  yaml.safe_load(open(args.config))
    except Exception as e:
        print("Error loading config file: ", e)
        sys.exit()
    # Train/Test DeepAR
    main(args.time_series, config, args.train, args.test)