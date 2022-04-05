import argparse
import sys
import yaml
from LR import main as lr
from FEDOT import main as fedot

MODELS = {
    "LR": lr,
    "FEDOT": fedot
    }

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
    parser.set_defaults(config="config.yaml")
    args = parser.parse_args()
    # Load configs
    try:
        config =  yaml.safe_load(open(args.config))
    except Exception as e:
        print("Error loading config file: ", e)
        sys.exit()
    
    if args.model not in MODELS.keys():
        print(f"Error: unknown model {args.model}. Options: {MODELS.keys()}")
        sys.exit()
    
    if args.model == "ALL":
        for model in MODELS.keys():
            MODELS[model](args.time_series, config, train=True, test=False)
    else:
        MODELS[args.model](args.time_series, config, train=True, test=False)
