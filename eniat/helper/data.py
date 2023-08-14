from typing import Union
import numpy as np


def iszero(data: np.ndarray, axis: int=0) -> Union[bool, np.ndarray]:
    """
    Check if all elements of the numpy ndarray along a specified axis are zero.

    Parameters:
    - data : np.ndarray
        Input data to be checked.
    - axis : int, optional (default=0)
        Axis along which elements should be checked. By default, checks along the zeroth axis.

    Returns:
    - bool
        True if all elements along the specified axis are zero, otherwise False.

    Examples:
    --------
    >>> data = np.array([[0, 0, 0], [0, 0, 0]])
    >>> iszero(data)
    True

    >>> data = np.array([[0, 1, 0], [0, 0, 0]])
    >>> iszero(data)
    False

    >>> data = np.array([[0, 0], [0, 1], [0, 0]])
    >>> iszero(data, axis=1)
    [True, False, True]
    """
    return np.all(data == 0, axis=axis)