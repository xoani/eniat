import pandas as pd
import numpy as np


def df_to_array(data):
    data = data.copy()
    if isinstance(data, pd.DataFrame):
        return data.values
    else:
        return data
    
def array_to_df(data):
    data = data.copy()
    if isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    else:
        return data

def cut_upperhalf(data):
    data = df_to_array(data)
    if (np.triu(data) == 0).all():
        return array_to_df(data)
    else:
        data = np.tril(data, k=-1)
        return array_to_df(data)
    
def cut_lowerhalf(data):
    data = df_to_array(data)
    if (np.tril(data) == 0).all():
        return array_to_df(data)
    else:
        data = np.triu(data, k=-1)
        return array_to_df(data)

def fill_upperhalf(data):
    data = df_to_array(data)
    if (np.triu(data) == 0).all():
        data += data.T
        data[np.nonzero(np.eye(data.shape[0]))] = 1
        return array_to_df(data)
    else:
        return array_to_df(data)
    
def get_nonzero_idx(data):
    data = df_to_array(data)
    return np.transpose(np.nonzero(data))

def thr_matrix(matrix, stats_matrix, thr):
    """ Threshold matrix """
    matrix = df_to_array(matrix)
    matrix[stats_matrix < thr] = 0
    matrix += matrix.T
    matrix[np.nonzero(np.eye(matrix.shape[0]))] = 1
    return pd.DataFrame(matrix)