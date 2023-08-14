def anova1_lm_multi(model, meta, data):
    """
    ANOVA for independent multiple sample with type 1 sum of square (SS)
    
    This code is modified from statsmodel.
    (https://github.com/statsmodels/statsmodels/blob/master/statsmodels/stats/anova.py)
    
    Args:
    -----
    model (str): patsy-style formular-like
    meta (dataframe): dataframe 
    data (ndarray): input data (NxM, where N=number of subject(or observation), 
                                           M=number of sample(data points, features))
    
    Returns:
    --------
    fvals: F values for each columns of Y
    pvals: correspond p values
    """
    import numpy as np
    from scipy.stats import f
    from patsy import dmatrix
    
    # Build design matrix
    A = dmatrix(model, data=meta)
    dinfo = A.design_info
    
    N, n_cont = A.shape # N = num of observation
    col_names = A.design_info.column_names
    trm_names = A.design_info.term_names
    num_fact = len(list(dinfo.factor_infos.keys()))
    
    # DOF calculation
    arr = np.zeros([len(trm_names), len(col_names)])
    # 2d array (matrix) with MxN, where M equials number of factors, 
    # and N equials number of columns in design matrix
    # resulted arr will be indicating which columns is the parts of factors. 
    # So that it helps to identify the degree of freedom of the design
    
    for i, sl in enumerate(A.design_info.term_slices.values()):
        arr[i, sl] = 1
    dof = arr.sum(1)[1:]
    if len(dof) == 1:  # One-way anova
        dof_error = N - (dof[:num_fact] + 1)    # N-k
    else:
        dof_error = N - np.multiply(*(dof[:num_fact] + 1))
        
    # QR decomposition
    q, r = np.linalg.qr(A)
    effects = np.dot(q.T, data)
    coefs = np.linalg.solve(r, effects)
    error = np.square(data - np.dot(A, coefs)).sum(0)
        
    # Sum of Square calculation
    sum_sqs = np.dot(arr, effects**2)[1:] # exclude intercept
    
    # mean square calculation
    mean_sqs = (sum_sqs.T / dof).T
    mean_sq_error = error / dof_error
    
    # F test
    fvals = mean_sqs/mean_sq_error
    pvals = np.zeros(fvals.shape)
    for i, fval in enumerate(fvals):
        pvals[i, :] = 1 - f.cdf(fval, dof[i], dof_error) 
    return fvals, pvals