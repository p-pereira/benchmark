from generate_data import main as gen_data
import yaml
from FEDOT import main as fedot
from LUDWIG import main as ludwig
from ARIMA import main as arima
from AUTOTS import main as autots
from PROPHET import main as prophet
from SKTIME import main as sktime
from LSTM import main as lstm
from DEEPAR import main as deepar
from HCRYSTALBALL import main as hcryst
from PYAF import main as pyaf


MODELS = {
    "ARIMA": arima,
    "FEDOT": fedot,
    "LUDWIG": ludwig,
    "PROPHET": prophet,
    "SKTIME": sktime,
    "AUTOTS": autots,
    "LSTM": lstm,
    "DEEPAR": deepar,
    "HCRYSTALBALL": hcryst,
    "PYAF": pyaf
    }

if __name__ == "__main__":
    config_path = "config.yaml"
    config = yaml.safe_load(open(config_path))
    time_series = config["TS"].keys()
    for ts in time_series:
        gen_data(ts, config)
        gen_data(ts, config, make_regression=True)
    
    for model_ in MODELS.keys():
        for ts in time_series:
            
            try:
                MODELS[model_](ts, config, train=True, test=True)
            except Exception as e:
                print(f"Error on model {model_} and dataset {ts}: {e}")