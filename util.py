'''
Module util.py loads data and writes submissions.
File paths are included here.
'''
import numpy as np
import pandas as pd

# float format for output files
FMT = '%.6f'


def load_data(as_pandas=False):
    '''
    Loads training and test data for Amazon Access problem.

    Args:
        as_pandas: Default False. If True, return pandas DataFrame or Series 
                objects instead of numpy arrays

    Returns:
        3-tuple of ndarrays or pandas Series/DataFrame objects 
        with training data, training labels and test data.
    '''
    xtr = pd.read_csv('../data/train.csv').drop('ROLE_CODE', axis=1)
    ytr = xtr['ACTION']
    del xtr['ACTION']
    xte = pd.read_csv('../data/test.csv').drop(['id', 'ROLE_CODE'], axis=1)
    if as_pandas:
        return xtr, ytr, xte
    else:
        return xtr.values, ytr.values, xte.values


def write_submission(preds, tag):
    '''
    Writes a prediction vector in a format suitable for submission.

    Args:
        preds: 1-d ndarray of predictions
        tag: output is submission_<tag>.csv.gz
    '''
    ss = pd.read_csv('../data/sample_submission.csv')
    ss.Action = preds
    path = '../submissions/submission_%s.csv.gz' % tag
    ss.to_csv(path, float_format=FMT, index=False, compression='gzip')
    print('Wrote submission at %s' % path)


def save_cv_preds(cv_preds, tag):
    '''
    Writes CV predictions to artifacts/ for later use

    Args:
        cv_preds: 1-d ndarray of cv predictions
        tag: output is cv_predictions_<tag>.csv
    '''
    cv_preds = pd.Series(cv_preds)
    path = '../artifacts/cv_predictions_%s.csv' % tag
    cv_preds.to_csv(path, float_format=FMT, index=False)
    print('Wrote CV predicitons at %s' % path)
