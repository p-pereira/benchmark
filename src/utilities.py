import re
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from glob import glob
import os
from sklearn import metrics

def rmse(y_true: Union[pd.Series, np.array], y_pred: Union[pd.Series, np.array]) -> float:
    """Compute Root Mean Square Error

    Parameters
    ----------
    y_true : Union[pd.Series, np.array]
        Target values
    y_pred : Union[pd.Series, np.array]
        Target predictions

    Returns
    -------
    float
        _description_
    """
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def nmae(y_true: Union[pd.Series, np.array], y_pred: Union[pd.Series, np.array], min_val: float, max_val: float) -> float:
    """Compute Normalized Mean Absolute Error

    Parameters
    ----------
    y_true : Union[pd.Series, np.array]
        Target values
    y_pred : Union[pd.Series, np.array]
        Target predictions
    min_val : float
        Lowest target value
    max_val : float
        Highest target value

    Returns
    -------
    float
        _description_
    """
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
    """_summary_

    Parameters
    ----------
    fpath : str
        Path to folder where data is stored
    target : str, optional
        _description_, by default "tempC"
    return_Xy : bool, optional
        _description_, by default True

    Returns
    -------
    Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]
        _description_
    """
    df = pd.read_csv(fpath)
    if return_Xy:
        return (df.drop(target, axis=1), df[target])
    else:
        return df

def list_files(time_series: str, config: Dict, pattern : str="*_tr.csv"):
    """_summary_

    Parameters
    ----------
    time_series : str
        Time-series name
    config : Dict, optional
        Configuration dict from config.yaml file, by default "config.yaml"
    pattern : str, optional
        _description_, by default "*_tr.csv"

    Returns
    -------
    _type_
        _description_
    """
    DIR = os.path.join(config["DATA_PATH"], config["PREP_PATH"], time_series)
    files = glob(os.path.join(DIR,pattern))
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    return files


def compute_metrics(y_true: Union[pd.Series, np.array], y_pred: Union[pd.Series, np.array], metrics: Union[str, List[str]]="ALL", prefix: str="") -> Dict:
    """Given target values and predictions, compute a set of metrics

    Parameters
    ----------
    y_true : Union[pd.Series, np.array]
        Target values
    y_pred : Union[pd.Series, np.array]
        Target predictions
    metrics : Union[str, List[str]], optional
        If "ALL", computes all metrics defined in the METRICS dict. 
        Otherwise, computes a specific given metric, by default "ALL"
    prefix : str, optional
        _description_, by default ""

    Returns
    -------
    Dict
        _description_
    """
    if isinstance(metrics, list):
        if not all(elem in METRICS.keys() for elem in metrics):
            print("Error: unknown metrics.")
            sys.exit()
    elif metrics == "ALL":
        metrics = METRICS.keys()
    
    return {prefix+key: METRICS[key](y_true, y_pred) for key in metrics}
    