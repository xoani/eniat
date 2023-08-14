import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.stats import chi2, pearsonr
from functools import partial
import statsmodels.formula.api as smf



def test_reliability(data: np.ndarray, subj_ids: List[int]) -> float:
    """
    Test the reliability of subject-level components against a group average.
    
    This function assesses whether subject-level components are correlated 
    with the group average. It employs the linear mixed model to estimate 
    random effect for the Intraclass Correlation Coefficient (ICC) analysis.

    Parameters:
    - data (np.ndarray): 2D array where rows represent subjects and columns represent the components.
    - subj_ids (List[int]): List of subject IDs.

    Returns:
    - float: The calculated Intraclass Correlation Coefficient (ICC) value.

    Notes:
    - This function uses the pearsonr method from scipy.stats for correlation calculations.

    """
    

    avr = data.mean(0)  # group average
    # testing whether the subject level components are correlated with its group average
    corr_to_avr = np.apply_along_axis(partial(pearsonr, y=avr), axis=1, arr=data)
    df = pd.DataFrame(dict(corr=corr_to_avr[:, 0], group=subj_ids))

    # Below code perform linear mixed model to estimate random effect for ICC analysis
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        md = smf.mixedlm("corr ~ 1", df, groups=df["group"])
        mdf = md.fit()

    # The result will only present how the correlation values are consistant across subject.
    return get_icc(mdf)


def get_icc(results) -> float:
    """
    Get the Intraclass Correlation Coefficient (ICC).
    
    The ICC is a measure of the consistency or reproducibility of measurements 
    when different but equivalent units are being measured.

    Parameters:
    - results: The result object of a fitted linear mixed model.

    Returns:
    - float: The calculated ICC value.
    """
    icc = results.cov_re / (results.cov_re + results.scale)
    return icc.values[0, 0]


def lr_test(formula: str, data: pd.DataFrame, groups: str) -> Tuple[float, float]:
    """
    Perform likelihood ratio test of random-effects.
    
    This function carries out a likelihood ratio test to compare 
    a null model (random-effects) with an OLS model.

    Parameters:
    - formula (str): A formula for the model fitting.
    - data (pd.DataFrame): Input data for the models.
    - groups (str): Grouping variable for the mixed linear model.

    Returns:
    - tuple:
      - float: The likelihood ratio test statistic.
      - float: Corresponding p-value.

    Examples:
    --------
    >>> icc = get_icc(mdf)
    >>> lrt, p = lr_test("corr ~ 1", data=df, groups='group')
    >>> print(f'ICC = {icc:.4f}')
    >>> print(f'The LRT statistic: {lrt:.4f} (p = {p:.5})')

    Notes:
    - The function uses statsmodels for model fitting.
    """
    # fit null model in mixed linear model
    null_model = smf.mixedlm(formula, data=data, groups=groups) \
        .fit(reml=False)
    # fit OLS model
    ols_model = smf.ols(formula, data=data) \
        .fit()
    # get the LRT statistic and p-value
    lrt = np.abs(null_model.llf - ols_model.llf) * 2
    p = chi2.sf(lrt, 1)
    return (lrt, p)