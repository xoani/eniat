import numpy as np
from ...error import *
from typing import Union, Tuple


def linear_regression(data: np.ndarray,
                      model: np.ndarray,
                      method: str = 'svd',
                      return_beta: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """ linear regression using linear decomposition algorithm

    Args:
        data: V x T data where V is number of voxels and T is number of time points
        model: T x F data where T is number of time points and F is number of features
        method: 'svd' or 'qr'
        return_beta: return beta coefficient if it is True

    Returns:
        fitted_curve : V x T x F
        beta_coefficient : V x F
    """
    if method == 'svd':
        bs = np.linalg.pinv(model).dot(data.T).T
    elif method == 'qr':
        q, r = np.linalg.qr(model)
        bs = np.linalg.inv(r).dot(q.T).dot(data.T).T
    else:
        raise InvalidApproach('Invalid input for "metrics"')

    try:
        fitted = np.zeros([bs.shape[0], model.shape[0], model.shape[1]])
    except:
        fitted = np.zeros([bs.shape[0], model.shape[0], model.shape[0]], dtype=np.float32)

    for i, b in enumerate(bs):
        fitted[i, ...] = model * b
    if return_beta:
        return fitted, bs
    else:
        return fitted