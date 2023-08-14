import numpy as np
from typing import Union


def standardize(data: np.ndarray) -> np.ndarray:
    """
    Standardize the input data by removing the mean and scaling to unit variance.
    
    Parameters:
    - data (np.ndarray): 1D or 2D array where the first dimension (V) represents entities (like voxels) 
                         and the second dimension (T) represents time points or observations.

    Returns:
    - np.ndarray: Standardized data with zero mean and unit variance along the time points or observations axis.
    
    Note:
    For 2D data, standardization is performed independently for each entity (row-wise).
    """
    if len(data.shape) == 1:
        return (data - data.mean()) / data.std()
    else:
        # Create a mask to avoid division by zero
        mask = data.std(axis=-1) != 0
        standardized_data = np.zeros(data.shape)
        non_zero_data = data[mask]
        standardized_data[mask] = (non_zero_data - non_zero_data.mean(axis=-1, keepdims=True)) / non_zero_data.std(axis=-1, keepdims=True)
        return standardized_data


def mode_normalization(data: np.ndarray, mode: Union[int, float] = 100) -> np.ndarray:
    """
    Normalize the data based on a given mode.
    
    Parameters:
    - data (np.ndarray): 1D or 2D array where the first dimension (V) represents entities (like voxels)
                         and the second dimension (T) represents time points or observations.
    - mode (int, float): The scaling factor to which the data should be normalized.

    Returns:
    - np.ndarray: Data normalized such that its mean scales to the specified mode.
    
    Note:
    For 2D data, normalization is performed independently for each entity (row-wise).
    """
    if len(data.shape) == 1:
        mean_val = data.mean()
        return ((data - mean_val) * mode / mean_val) + mode
    else:
        mean_vals = data.mean(axis=-1, keepdims=True)
        mask = mean_vals != 0
        normalized_data = np.zeros(data.shape)
        non_zero_data = data[mask.squeeze()]
        normalized_data[mask.squeeze()] = ((non_zero_data - non_zero_data.mean(axis=-1, keepdims=True)) * mode / non_zero_data.mean(axis=-1, keepdims=True)) + mode
        return normalized_data
