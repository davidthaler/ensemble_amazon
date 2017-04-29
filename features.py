import numpy as np
import itertools as it

def make2way(xtr, xte):
    '''
    Makes some features representing 2-way combinations of columns in xtr/xte.
    These are in a dense integer-categorical format.

    Args:
        xtr, xte: training and test features as integer-coded categoricals

    Returns:
        2-tuple of  ndarrays of xtr and xte augmented with 
        columns for 2-way combinations
    '''
    for i, j in it.combinations(range(xtr.shape[1]), 2):
        new_col_tr = xtr[:, i] + xtr[:, j]
        xtr = np.column_stack([xtr, new_col_tr])
        new_col_te = xte[:, i] + xte[:, j]
        xte = np.column_stack([xte, new_col_te])
    return xtr, xte