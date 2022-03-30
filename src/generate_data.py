from typing import Dict, List, Union
import pandas as pd
import sys
import argparse
import os
import yaml
from tqdm import tqdm

def rw(y: Union[pd.Series, List], ratio: Union[float,int], W: int, S: int, iteration: int=1, mode: str="rolling", val_ratio: Union[float, str]=0) -> Dict:
    """Generate train, validation and test indexes for Rolling Window procedure.

    Parameters
    ----------
    y : Union[pd.Series, Array]
        _description_
    ratio : Union[float,int]
        _description_
    W : int
        _description_
    S : int
        _description_
    iteration : int, optional
        _description_, by default 1
    mode : str, optional
        _description_, by default "rolling"
    val_ratio : Union[float, str], optional
        _description_, by default 0

    Returns
    -------
    Dict
        Training, validation and testing indexes.
    """
    L = len(y)
    idx = (iteration-1) * S

    if ratio < 0:
        H = ratio * L
    else:
        H = ratio

    if val_ratio < 0:
        H2 = val_ratio * L
    else:
        H2 = val_ratio

    if H2 == 0:
        if mode == "rolling":
            tr = range(idx, idx+W)
        elif mode == "incremental":
            tr = range(0, idx+W)
        val = []
    else:
        if mode == "rolling":
            tr = range(idx, idx+W-H2)
        elif mode == "incremental":
            tr = range(0, idx+W-H2)
        val = range(0+W-H2, idx+W-H2)
    
    ts = range(idx+W+1, idx+W+1+H)
    
    return {'tr': tr, 'val': val, 'ts': ts}

def create_lags(y: pd.Series, K: int=7, target: str="tempC") -> pd.DataFrame:
    """Create time-lags dataframe from series.

    Parameters
    ----------
    y : pd.Series
        _description_
    H : int, optional
        _description_, by default 7
    target : str, optional
        _description_, by default "tempC"

    Returns
    -------
    pd.DataFrame
        _description_
    """
    d2 = []
    for i in range(K+1):
        d2.append(y.shift(i*-1))
    lags = pd.DataFrame(d2).T.dropna()
    X_cols = [str(i) for i in range(7)]
    X_cols.append(target)
    lags.columns = X_cols
    return lags

def main(time_series: str, config_file: str = "config.yaml"):
    """Generate cross-validation data files for a time-series dataset.

    Parameters
    ----------
    time_series : str
        time series name (same as the file).
    config_file : str, optional
        configuration file path, by default "config.yaml"
    """
    try:
        config =  yaml.safe_load(open(config_file))
    except Exception as e:
        print("Error loading config file: ", e)
        sys.exit()
    
    DATA_PATH = config["DATA_PATH"]
    RAW_PATH = config["RAW_PATH"]
    PREP_PATH = config["PREP_PATH"]
    series_config = config["TS"][time_series]

    lags_path = os.path.join(DATA_PATH, PREP_PATH, time_series)
    os.makedirs(lags_path, exist_ok=True)

    format = series_config["format"]
    fpath = os.path.join(DATA_PATH, RAW_PATH, time_series + format)

    try:
        print(os.getcwd())
        d = pd.read_csv(fpath)
    except Exception as e:
        print("Error loading data: ", e)
        sys.exit()
    
    target = series_config["target"]
    U = series_config["U"]
    K = series_config["K"]
    H = series_config["H"]
    W = series_config["W"]
    L = d.shape[0]
    S = (L-(W+H-1)) // U

    df = create_lags(d[target], K, target)
    
    for it in tqdm(range(1, U+1)):
        res = rw(df[target], ratio=H, W=W, S=S, iteration=it, mode="rolling")
        dtr = df.iloc[res["tr"], ]
        dts = df.iloc[res["ts"], ]
        tr_fpath = os.path.join(lags_path, f"tr_{it}{format}")
        ts_fpath = os.path.join(lags_path, f"ts_{it}{format}")
        dtr.to_csv(tr_fpath, index=None)
        dts.to_csv(ts_fpath, index=None)


if __name__ == "__main__":
    # Read arguments
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Generate CV data files for a time-series dataset, based on config.yaml params')
    parser.add_argument(help='Time-series name.', dest="time_series")
    parser.set_defaults(time_series="porto")
    parser.add_argument('-c', '--config', dest='config', 
                        help='Config yaml file.')
    parser.set_defaults(config="config.yaml")
    args = parser.parse_args()
    # Generate dataset files
    main(args.time_series, args.config)
