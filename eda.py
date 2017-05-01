import numpy as np
import pandas as pd


def woe_frame(xtr, ytr, field):
    '''
    Computes a data frame with weight-of-evidence and information value 
    for one field in the labeled data. Uses Laplace smoothing at the mean 
    label value with a virtual sample size of 1 to avoid 0 or 1 values.

    Args:
        xtr: pandas DataFrame of training data
        ytr: Pandas Series (numpy array ok) of 0-1 labels
        field: the name of the field to compute weight-of-evidence on

    Returns:
        new DataFrame with smoothed distributions for 0 and 1, 
        weight-of-evidence and information value
    '''
    t = pd.crosstab(xtr[field], ytr)
    t.rename(columns={0:'dist0', 1:'dist1'}, inplace=True)
    t.dist1 = t.dist1 + ytr.mean()
    t.dist0 = t.dist0 + (1 - ytr.mean())
    t = t / t.sum()
    t['d1_d0'] = t.dist1 - t.dist0
    t['woe'] = (t.dist1 / t.dist0).map(np.log)
    t['infoval'] = t.woe * t.d1_d0
    return t


def infovalues(xtr, ytr, cols=None):
    '''
    Compute information values for all or select columns in input data.

    Args:
        xtr: pandas DataFrame of training data
        ytr: Pandas Series (numpy array ok) of 0-1 labels
    
    Returns:
        Pandas DataFrame of column names and their (smoothed) information values.
    '''
    if cols is None:
        cols = xtr.columns
    ivals = [woe_frame(xtr, ytr, c).infoval.sum() for c in cols]
    nuniq = xtr.apply(lambda c : c.nunique())
    return pd.DataFrame(index=cols, data={'nuniq':nuniq, 'infoval': ivals})

