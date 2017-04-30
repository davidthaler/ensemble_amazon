'''
Module util.py loads data and writes submissions.
File paths are included here.
'''
import numpy as np
import pandas as pd
import os
import re


# float format for output files
FMT = '%.6f'
SUBMIT_TEMPLATE = '../submissions/submission_%s.csv.gz'
CV_PRED_TEMPLATE  = '../artifacts/cv_predictions_%s.csv'

# random number seed for all files
SEED = 42

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
    path = SUBMIT_TEMPLATE % tag
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
    path = CV_PRED_TEMPLATE % tag
    cv_preds.to_csv(path, float_format=FMT, index=False)
    print('Wrote CV predicitons at %s' % path)


def reload_submissions(sub_names=None, as_pandas=False):
    '''
    Reloads the predictions from all of the submission in ../submissions,
    or just a specified set of them. Result is either a Pandas DataFrame 
    or a numpy array.

    Args:
        sub_names: a list of submission names. 
            Default None, which loads everything in ../submissions.
            Uses just the 'tag' part of the name, like 'logreg2way',
            not the path, "submission_" or ".csv.gz".
        as_pandas: Default False. If True, return a DataFrame, 
                otherwise a numpy array.

    Returns:
        Either a pandas DataFrame or numpy array of submission predictions.
    '''
    if sub_names is None:
        l = os.listdir('../submissions')
        pat = re.compile(r'submission_(\w+).csv.gz')
        sub_names = [re.findall(pat, s)[0] for s in l]
    paths = [SUBMIT_TEMPLATE % n for n in sub_names]
    subs = []
    for path in paths:
        subs.append(pd.read_csv(path))
    if as_pandas:
        out = pd.DataFrame(index=subs[0].index)
        for sub_name, sub in zip(sub_names, subs):
            out[sub_name] = sub.Action
        return out
    else:
        preds = [sub.Action.values for sub in subs]
        return np.column_stack(preds)
    

def reload_cv_predictions(pred_names=None, as_pandas=False):
    '''
    Reloads CV predictions from ../artifacts or a subset of them.
    Combines them into either a DataFrame or a numpy array.

    Args:
        pred_names: Optional list of CV prediction file names.
            Just uses the 'tag' part like 'logreg2way', not the 
            prefix, suffix or path. Default is None, which loads
            everything in ../artifacts.
        as_pandas: Default False. If True, return a DataFrame, 
                otherwise a numpy array.

    Returns:
        Either a pandas DataFrame or numpy array of submission predictions.
    '''
    if pred_names is None:
        l = os.listdir('../artifacts')
        pat = re.compile(r'cv_predictions_(\w+).csv')
        pred_names = [re.findall(pat, s)[0] for s in l]
    paths = [CV_PRED_TEMPLATE % n for n in pred_names]
    cv_preds = []
    for name in pred_names:
        cv_preds.append(np.loadtxt(CV_PRED_TEMPLATE % name))
    if as_pandas:
        return pd.DataFrame(data=dict(zip(pred_names, cv_preds)))
    else:
        return np.column_stack(cv_preds)
