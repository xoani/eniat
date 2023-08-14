import numpy as np
from scipy import sparse, linalg as slinalg


def als_fit(data: np.ndarray, l: float, p: float, niter: int) -> np.ndarray:
    """
    Asymmetric Least Squares Smoothing for Baseline or Envelope fitting.

    The function applies the ALS method to fit a baseline or envelope to a given
    data. This method is particularly useful for data where the peaks are positive
    and the baseline might have a complex shape, e.g., spectroscopic data.

    Parameters:
    - data (np.ndarray): 1D array containing the time series data to be fitted.
    - l (float): Lambda (smoothness parameter). Controls the smoothness of the 
      fitted baseline. Higher values lead to a smoother baseline.
    - p (float): Asymmetry parameter. Value should be between 0 and 1. Controls 
      the weight of positive and negative deviations from the baseline in the 
      optimization. A value close to 1 gives more weight to positive deviations.
    - niter (int): Number of iterations for the ALS algorithm.

    Returns:
    - np.ndarray: 1D array of the same size as input data containing the fitted 
      baseline.

    Note:
    The ALS algorithm uses iterative re-weighted least squares optimization with 
    weights updated in each iteration based on the residuals.
    """
    
    L = len(data)
    # Construct a second-difference matrix for the data
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + l * D.dot(D.transpose())
        z = slinalg.spsolve(Z, w * data)
        w = p * (data > z) + (1 - p) * (data < z)
        
    return np.asarray(z)