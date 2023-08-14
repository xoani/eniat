import numpy as np

def polynomial_feature(data: np.ndarray,
                       order: int) -> np.ndarray:
    """ Generate polynomial features for input data

    Args:
        data: V x T data where V is voxels and T is time points
        order: order of polynomial

    Returns:
        model: polynomial features with given order
    """
    n_ft = order + 1
    n_dp = data.shape[-1]
    model = np.zeros([n_dp, n_ft])
    for o in range(n_ft):
        if o == 0:
            model[:,0] = np.ones(n_dp)
        else:
            x = np.arange(n_dp)
            model[:,o] = x ** o
            model[:,o] /= model[o,:].max()
    return model