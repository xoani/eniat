from typing import Tuple
import numpy as np

def r_to_t(r: float, size: int) -> Tuple[float, float]:
    """
    Convert a correlation coefficient to a t-value and compute its corresponding p-value.
    
    Parameters:
    - r : float
        The correlation coefficient.
    - size : int
        The sample size.

    Returns:
    - tval : float
        The t-value corresponding to the given correlation.
    - pval : float
        The p-value corresponding to the t-value.
    """
    from scipy.stats import t
    try:
        tval = r * np.sqrt(size - 2) / np.sqrt(1 - np.square(r))
        pval = 1 - t.cdf(tval, size - 2)
    except:
        r = r.astype(np.float32)
        tval = (r * np.sqrt(size - 2) / np.sqrt(1 - np.square(r))).astype(np.float32)
        pval = (1 - t.cdf(tval, size - 2)).astype(np.float32)
    return tval, pval