import re
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from glob import glob
import os
from sklearn import metrics

def rmse(y_true: Union[pd.Series, np.array], y_pred: Union[pd.Series, np.array]) -> float:
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def nmae(y_true: Union[pd.Series, np.array], y_pred: Union[pd.Series, np.array], min_val: float, max_val: float) -> float:
    mae = METRICS["mae"](y_true, y_pred)
    range_vals = max_val - min_val
    return mae / range_vals

METRICS = {
    "mae": metrics.mean_absolute_error,
    "mse": metrics.mean_squared_error,
    "rmse": rmse,
    "mape": metrics.mean_absolute_percentage_error,
    "r2": metrics.r2_score
}

def load_data(fpath: str, target="tempC", return_Xy: bool = True) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
    """Load data.

    Parameters
    ----------
    fpath : str
        File path.
    target : str, optional
        Target columns, by default "tempC"
    return_Xy : bool, optional
        return tupple (X, y), by default True

    Returns
    -------
    Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]
        data used to train model
    """
    df = pd.read_csv(fpath)
    if return_Xy:
        return (df.drop(target, axis=1), df[target])
    else:
        return df

def list_files(time_series: str, config: Dict, pattern : str="*_tr.csv"):
    DIR = os.path.join(config["DATA_PATH"], config["PREP_PATH"], time_series)
    files = glob(os.path.join(DIR,pattern))
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    return files


def compute_metrics(y_true: Union[pd.Series, np.array], y_pred: Union[pd.Series, np.array], metrics: Union[str, List[str]]="ALL", prefix: str="") -> Dict:
    if isinstance(metrics, list):
        if not all(elem in METRICS.keys() for elem in metrics):
            print("Error: unknown metrics.")
            sys.exit()
    elif metrics == "ALL":
        metrics = METRICS.keys()
    
    return {prefix+key: METRICS[key](y_true, y_pred) for key in metrics}
    