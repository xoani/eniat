import numpy as np


def rat_hrf(t, b=2.0, p1=7.4, p2=8.9, v=1.5):
    """
    Rat hemodynamic function
    Lambers et al. 2020. doi:10.1016/j.neuroimage.2019.116446
    Args:
        t: time steps
        b: dispersion parameter
        p1: peak parameter 1
        p2: peak parameter 2
        v: ratio parameter

    Returns:
        rat hemodynamic response function
    """
    import math
    peak = (b**p1)/math.gamma(p1)*t**(p1-1)
    under = (b**p2)/(v*math.gamma(p2))*t**(p2-1)
    hrf = np.e**(-1 * b * t) * (peak - under)
    return hrf / hrf.max()