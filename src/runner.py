from generate_data import main as gen_data
from test import main as test
from train import main as train
import yaml

if __name__ == "__main__":
    config_path = "config.yaml"
    config = yaml.safe_load(open(config_path))
    time_series = config["TS"].keys()
    for ts in time_series:
        gen_data(ts)
        gen_data(ts, make_regression=True)
        train(ts, model="ALL", config=config)
        test(ts, model="ALL", config=config)
        break