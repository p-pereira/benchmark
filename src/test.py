import argparse
from os import getcwd, makedirs, path
import sys
import pandas as pd
import yaml
import mlflow
from tqdm import tqdm
from utilities import list_files, load_data, compute_metrics

if __name__ == "__main__":
    # Read arguments
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Generate CV data files for a time-series dataset, based on config.yaml params')
    parser.add_argument(help='Time-series name.', dest="time_series")
    parser.set_defaults(time_series="porto")
    parser.add_argument(help='ML model.', dest="model")
    parser.set_defaults(model="LR")
    parser.add_argument('-c', '--config', dest='config', 
                        help='Config yaml file.')
    parser.add_argument('-r', '--reg', dest='make_regression', 
                        help='Convert time-series data in regression task.')
    parser.set_defaults(make_regression=False)
    parser.set_defaults(config="config.yaml")
    args = parser.parse_args()
    # Load configs
    try:
        config =  yaml.safe_load(open(args.config))
    except Exception as e:
        print("Error loading config file: ", e)
        sys.exit()
    
    if args.make_regression:
        pattern = "ts*reg*"
    else:
        pattern = "ts*"
    target = config["TS"][args.time_series]["target"]
    test_files = list_files(args.time_series, config, pattern)

    # mlflow configs
    mlflow.set_tracking_uri("http://localhost:5000")
    
    for n, file in enumerate(tqdm(test_files)):
        # Get mlflow run id to load the model.
        run_name = f"{args.time_series}_{target}_{args.model}_{n+1}"
        X, y = load_data(file,config["TS"][args.time_series]["target"])
        experiment = dict(mlflow.get_experiment_by_name(config["EXPERIMENT"]))
        runs = mlflow.search_runs([experiment["experiment_id"]])
        run_id = runs[runs['tags.mlflow.runName']==run_name]["run_id"].values[0]
        # Load model
        logged_model = f"runs:/{run_id}/model"
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        # Predic and compute metrics
        pred = loaded_model.predict(X)
        metrics = compute_metrics(y, pred, "ALL", "test_")
        # Store predictions and target values
        info = pd.DataFrame([y, pred]).T
        info.columns = ["y_true", "y_pred"]
        FDIR = path.join(config["DATA_PATH"], config["PRED_PATH"], args.time_series, args.model)
        makedirs(FDIR, exist_ok=True)
        FPATH = path.join(FDIR, f"pred_{n+1}.csv")
        info.to_csv(FPATH, index=False)
        # Load new info to mlflow run
        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_artifact(FPATH)
            mlflow.log_metrics(metrics)
