import numpy as np
from ..regression import linear_regression
from scipy import stats


def ttest(data, model):
    predicted, coef = linear_regression(data, model, method='svd', return_beta=True)

    dof = model.shape[0] - model.shape[1]
    mse = np.square(predicted.sum(-1) - data).sum(-1) / float(dof)
    se = np.sqrt((mse * np.concatenate([np.linalg.inv(np.dot(model.T, model)).diagonal()[:, np.newaxis]], axis=-1)).T)

    t = coef.copy()
    if model.shape[-1] == 1:
        # one sample t-test
        t -= t.mean()  # against population mean
    t[se == 0] = 0
    t[np.nonzero(se)] /= se[np.nonzero(se)]
    p = 2 * (1 - stats.t.cdf(abs(t), df=dof))
    return coef, t, p

def glt(model, contrast, permute=False):
    """
    Performs a general linear test (GLT) using a contrast matrix.

    Args:
        model: The fitted model object.
        contrast: The contrast matrix.
        permute: Whether to perform permutation testing (default: False).

    Returns:
        If permute is True:
            tvals: The t-values.
        If permute is False:
            tvals: The t-values.
            pvals: The p-values.
    """
    cols = model.column_names
    c = np.zeros(len(cols))
    c[:len(contrast)] = contrast 
    X = model._dmat
    covar = np.linalg.inv(X.T.dot(X))
    
    ss_err = model._data['SSerr']
    df = model._attrs['DFerr']
    mse = ss_err / df

    beta = model._data['beta'].dot(c)
    se = np.sqrt(c.dot(covar).dot(c.T) * mse)
    tvals = beta / se
    if permute:
        return tvals
    pvals = 2 * stats.t.sf(abs(tvals), df=df)
    return tvals, pvals

def onesample_ttest_perm(data, pval=0.05, nperm=5000, twosided=False):
    from tqdm.notebook import tqdm
    model = np.ones([data.shape[1], 1])
    b, t_o, p_o = ttest(data, model)

    tmax = np.zeros(nperm)
    tmin = np.zeros(nperm)
    for p in tqdm(range(nperm)):
        model_perm = np.c_[model, np.random.choice([1, -1], data.shape[-1], replace=True)]
        b_p, t_p, p_p = ttest(data, model_perm)
        tmax[p] = t_p[:, 1].max()
        tmin[p] = t_p[:, 1].min()
    if twosided:
        pp_perm = np.zeros(data.shape[0])
        np_perm = pp_perm.copy()
        for i in range(data.shape[0]):
            pp_perm[i] = (tmax >= t_o[i, 0]).astype(int).sum() / nperm
            np_perm[i] = (tmin <= t_o[i, 0]).astype(int).sum() / nperm
        t_o[(pp_perm >= pval / 2) & (np_perm >= pval / 2)] = 0
    else:
        p_perm = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            p_perm[i] = (tmax >= t_o[i, 0]).astype(int).sum() / nperm
        t_o[p_perm >= pval] = 0
    return b