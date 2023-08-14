import numpy as np

def kendall_w(data: np.ndarray) -> float:
    """
    Compute Kendall's coefficient of concordance (W) for the provided data.
    
    The coefficient W is used to assess the agreement among multiple rankers/observers.

    Parameters:
    -----------
    data : np.ndarray
        A 2D numpy array where rows represent rankers or observers and columns represent 
        the items being ranked. Each cell contains the rank given by a particular ranker
        to a particular item.

    Returns:
    --------
    float
        The computed Kendall's W value. A value closer to 1 indicates stronger agreement
        among rankers, while a value closer to 0 indicates weaker agreement.

    Example:
    --------
    >>> data = np.array([[1, 2, 3], [1, 3, 2], [2, 1, 3]])
    >>> kendall_w(data)
    0.5

    """
    m, n = data.shape

    if m != 0:
        # Compute ranks for each ranker/observer
        temp = data.argsort(axis=1)
        ranks = temp.argsort(axis=1).astype(np.float64) + 1
        # Sum ranks for each item
        ranks_sum = ranks.sum(axis=0)
        # Compute mean rank for each item
        mean_ranks = ranks_sum.mean()
        # Calculate the sum of squared deviations from the mean ranks
        ssd = np.square(ranks_sum - mean_ranks).sum()
        # Calculate Kendall's W
        w = 12 * ssd / (m ** 2 * (n ** 3 - n))
        return w

    else:
        return 0
