from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np
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

def cases_series(t: pd.Series, W: Tuple, target: str = "y", start: int=1, end: int=0) -> pd.DataFrame:
    """Python adaptation of CasesSeries R function from rminer library. Creates lag dataframe from time-series data.

    Parameters
    ----------
    t : pd.Series
        Time-series.
    W : Tuple
        Window.
    target : str, optional
        Target column, by default "y"
    start : int, optional
        Start of time-series (1 means 1st value), by default 1
    end : int, optional
        End of time-series, by default 0

    Returns
    -------
    pd.DataFrame
        Lag dataframe.
    """
    if end == 0:
        end = len(t)
    LW = len(W)
    LL = W[LW-1]
    JL = (end - start + 1) - LL
    I = np.zeros((JL, LW+1))
    S = start - 1
    for j in range(0,JL):
        for i in range(0,LW):
            I[j,i]=t[(S+LL-W[LW-i-1]+j)]
            I[j,(LW)]=t[(S+LL+j)]
    D = pd.DataFrame(I)
    N = list(D.columns)
    LN = len(N)
    for i in range(1, LN):
        N[LN-i-1] = f"lag{W[i-1]}"
    N[LN-1] = target
    D.columns = N
    return D

def main(time_series: str, config_file: str = "config.yaml", make_regression: bool = False):
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
    WR = series_config["WR"]

    if make_regression:
        d = cases_series(d[target], WR, target)
        fn = "_reg"
    else:
        fn = ""
    
    H = series_config["H"]
    W = series_config["W"]
    L = d.shape[0]
    S = (L-(W+H-1)) // U

    for it in tqdm(range(1, U+1)):
        res = rw(d[target], ratio=H, W=W, S=S, iteration=it, mode="rolling")

        dtr = d.iloc[res["tr"], ]
        dts = d.iloc[res["ts"], ]
        tr_fpath = os.path.join(lags_path, f"tr_{it}{fn}{format}")
        ts_fpath = os.path.join(lags_path, f"ts_{it}{fn}{format}")
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
    parser.add_argument('-r', '--reg', dest='make_regression', 
                        help='Convert time-series data in regression task.')
    parser.set_defaults(make_regression=False)
    args = parser.parse_args()
    # Generate dataset files
    main(args.time_series, args.config, args.make_regression)
