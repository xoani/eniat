import numpy as np
import pandas as pd
from scipy.signal import hilbert
from typing import Tuple


def corr(x: np.ndarray) -> float:
    """
    Compute the auto-correlation of a given 1D array.
    
    Parameters:
    - x : np.ndarray
        The input 1D array.

    Returns:
    - r : float
        The auto-correlation value.
    """
    vals = np.zeros(x.shape)
    mask = np.nonzero(x.std(-1))

    try:
        vals[mask] = ((x[mask].T - x[mask].mean(-1)) / x[mask].std(-1)).T
    except:
        vals[mask] = ((x[mask].T - x[mask].mean(-1)) / x[mask].std(-1)).T.astype(np.float32)
    r = np.dot(vals, vals.T) / vals.shape[-1]
    return r


def corr_with(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the cross-correlation between two 1D arrays.
    
    Parameters:
    - x, y : np.ndarray
        The input 1D arrays.

    Returns:
    - r : float
        The cross-correlation value.
    """
    val_x = np.zeros(x.shape)
    val_y = np.zeros(y.shape)

    x_mask = np.nonzero(x.std(-1))
    y_mask = np.nonzero(y.std(-1))
    try:
        val_x[x_mask] = ((x[x_mask].T - x[x_mask].mean(-1)) / x[x_mask].std(-1)).T
        val_y[y_mask] = ((y[y_mask].T - y[y_mask].mean(-1)) / y[y_mask].std(-1)).T
    except:
        val_x[x_mask] = ((x[x_mask].T - x[x_mask].mean(-1)) / x[x_mask].std(-1)).T.astype(np.float32)
        val_y[y_mask] = ((y[y_mask].T - y[y_mask].mean(-1)) / y[y_mask].std(-1)).T.astype(np.float32)
    r = np.dot(val_x, val_y.T) / x.shape[-1]
    return r


def phase_locking_value(x: np.ndaray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute the phase-locking value (PLV) between two 1D arrays.
    
    Parameters:
    - x, y : np.ndarray
        The input 1D arrays representing signal amplitudes.

    Returns:
    - float
        The PLV value between x and y.
    - np.ndarray
        Difference in phase angles between x and y.
    """
    x_phase = np.angle(hilbert(x), deg=False)
    y_phase = np.angle(hilbert(y), deg=False)
    angle_diff = x_phase - y_phase
    return abs(np.exp(1j*angle_diff).mean()), angle_diff


def const_maxcorr(df: pd.DataFrame, dt: float, max_lag: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the maximum cross-correlation within a constrained time lag for time series data in a DataFrame.
    
    Parameters:
    - df : pd.DataFrame
        Time series data where rows are timepoints and columns are different regions of interest (ROIs).
    - dt : float
        Sampling interval in seconds.
    - max_lag : float
        The constrained maximum lag in seconds.

    Returns:
    - pd.DataFrame
        Matrix of maximum correlations for each ROI pair.
    - pd.DataFrame
        Matrix of lags (in timepoints) at which the maximum correlations occur.
    """
    
    max_lag = int(max_lag / dt)
    all_lags = np.arange(-max_lag, max_lag + 1)
    max_corr = pd.DataFrame(np.zeros([df.shape[-1]] * 2), index=df.columns,
                            columns=df.columns)
    max_corr_lag = max_corr.copy()

    for col_id1, ts1 in df.iteritems():
        cross_corr = np.zeros([len(all_lags), len(df.columns)])
        for col_id2, ts2 in df.iteritems():
            for lag_id, lag in enumerate(all_lags):
                cross_corr[lag_id, col_id2] = ts1.corr(ts2.shift(lag))
        max_lag_idxs = abs(cross_corr).argmax(0)
        for col_id2, max_lag_idx in enumerate(max_lag_idxs):
            max_corr.loc[col_id1, col_id2] = cross_corr[max_lag_idx, col_id2]
            max_corr_lag.loc[col_id1, col_id2] = all_lags[max_lag_idx]
    return max_corr, max_corr_lag