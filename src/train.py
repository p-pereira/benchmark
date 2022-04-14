import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import sys
import yaml
from LR import main as lr
from FEDOT import main as fedot
from LUDWIG import main as ludwig
from ARIMA import main as arima
from AUTOTS import main as autots
from PYAF import main as pyaf

MODELS = {
    "ARIMA": arima,
    "AUTOTS": autots,
    "PYAF": pyaf,
    "FEDOT": fedot,
    "LUDWIG": ludwig,
    "LR": lr
    }

def main(time_series: str= "porto", model: str = "ARIMA", config: str = "config.yaml"):
    # Load configs
    try:
        config =  yaml.safe_load(open(config))
    except Exception as e:
        print("Error loading config file: ", e)
        sys.exit()
    
    if model == "ALL":
        for model_ in MODELS.keys():
            MODELS[model_](time_series, config, train=True, test=False)
    elif model in MODELS.keys():
        MODELS[model](time_series, config, train=True, test=False)
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
    parser.set_defaults(model="ARIMA")
    parser.add_argument('-c', '--config', dest='config', 
                        help='Config yaml file.')
    parser.set_defaults(config="config.yaml")
    args = parser.parse_args()
    #DESTA FORMA, CORRE O TRAIN, SE ELIMINAR E PUSER
    main(args.time_series, args.model, args.config)
    #, JÁ NÃO FUNCIONA

    # Load configs
    '''
    try:
        config =  yaml.safe_load(open(args.config))
    except Exception as e:
        print("Error loading config file: ", e)
        sys.exit()
    
    if args.model not in MODELS.keys():
        print(f"Error: unkown model {args.model}.")
        sys.exit()

    MODELS[args.model](args.time_series, config)
    '''
