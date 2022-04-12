import argparse
from os import getcwd, makedirs, path
import sys
import pandas as pd
import yaml
import mlflow
from tqdm import tqdm
from utilities import list_files, load_data, compute_metrics
from time import time
from Gluon import main as Gluon


MODELS = {"Gluon": Gluon}

def main(time_series: str= "porto", model: str = "Gluon", config: str = "config.yaml"):
    # Load configs
    try:
        config =  yaml.safe_load(open(config))
    except Exception as e:
        print("Error loading config file: ", e)
        sys.exit()
    
    if model == "ALL":
        for model_ in MODELS.keys():
            MODELS[model_](time_series, config, train=False, test=True)
    elif model in MODELS.keys():
        MODELS[model](time_series, config, train=False, test=True)
    else:
        print(f"Error: unknown model {model}. Options: {MODELS.keys()}")
        sys.exit()

if __name__ == "__main__":
    # Read arguments
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Generate CV data files for a time-series dataset, based on config.yaml params')
    parser.add_argument(help='Time-series name.', dest="time_series")
    parser.set_defaults(time_series="porto")
    parser.add_argument(help='ML model.', dest="model")
    parser.set_defaults(model="Gluon")
    parser.add_argument('-c', '--config', dest='config', 
                        help='Config yaml file.')
    parser.add_argument('-r', '--reg', dest='make_regression', 
                        help='Convert time-series data in regression task.')
    parser.set_defaults(make_regression=False)
    parser.set_defaults(config="config.yaml")
    args = parser.parse_args()
    main(args.time_series, args.model, args.config)