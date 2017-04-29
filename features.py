import numpy as np
import itertools as it
from sklearn.preprocessing import OneHotEncoder


def make_combo(x, k):
    '''
    Makes dense integer-categorical features representing k-combinations of
    the columns in x.

    Args:
        x: ndarray of train or test data
        k: order of combination features

    Returns:
        ndarray of features representing the k-combinations of features in x
    '''
    out_cols = []
    for t in it.combinations(range(x.shape[1]), k):
        out_cols.append(x[:, t].sum(axis=1))
    return np.column_stack(out_cols)


def range_combos(xtr, xte, kmax):
    '''
    Makes features representing 1...kmax combinations of columns in xtr/xte.
    These are in a dense integer-categorical format.

    Args:
        xtr, xte: ndarrays of train/test features as integer-coded categoricals
        kmax: makes 1...kmax combinations of the columns

    Returns:
        2-tuple of ndarrays of 1...kmax combinations of features in xtr/xte
    '''
    tr = [xtr]
    te = [xte]
    for k in range(2, kmax + 1):
        tr.append(make_combo(xtr, k))
        te.append(make_combo(xte, k))
    xtr = np.hstack(tr)
    xte = np.hstack(te)
    enc = OneHotEncoder()
    enc.fit(np.vstack([xtr, xte]))
    xtr = enc.transform(xtr)
    xte = enc.transform(xte)
    return xtr, xte
