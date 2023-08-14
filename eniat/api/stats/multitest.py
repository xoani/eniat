import numpy as np
from .corr import r_to_t

def bonfferoni_correction(r: np.ndarray, size: int, pval: float = 0.05) -> np.ndarray:
    """
    Applies Bonferroni correction to a given set of p-values derived from correlation coefficients.

    The Bonferroni correction is a type of multiple comparison correction used when several 
    dependent or independent statistical tests are being performed simultaneously. 
    This method adjusts the threshold p-value based on the number of tests performed.

    Parameters:
    - r (np.ndarray): A 1D numpy array containing correlation coefficients. It should have a shape (N,), 
                      where N is the number of comparisons.
    - size (int): The size of the data (number of samples) that was used to compute the correlation coefficients.
    - pval (float, optional): The significance threshold value before correction. Defaults to 0.05.

    Returns:
    - np.ndarray: A boolean 1D numpy array with the shape of (N,). True for correlations that remain 
                  significant after Bonferroni correction, and False otherwise.

    Notes:
    - This function uses the `r_to_t` method from the 'corr' module to convert correlation coefficients to t-values and 
      derive the associated p-values.
    - The adjusted threshold is calculated as `pval` divided by the number of tests (length of r).

    Examples:
    --------
    >>> r = np.array([0.1, 0.2, 0.35, 0.5])
    >>> size = 100
    >>> bonfferoni_correction(r, size)
    [False, False, True, True]

    """
    t, p = r_to_t(r, size)
    return p < (pval / r.shape[0])
