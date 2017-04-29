'''
A refactoring of Paul Duan's "bagged_set" code for bagged CV-prediction.
'''
import numpy as np
import pandas as pd
import itertools as it
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score


SEED = 42
FMT = '%.6f'


def bagged_set(xtr, ytr, xval, model, n_runs, seed=7):
    '''
    Repeatedly trains a model on xtr, ytr and predicts on xval.
    Results are averaged.

    Args:
        xtr: numpy array of training data
        ytr: numpy vector (1-D ndarray) of labels
        xval: numpy array of out-of-fold data
        model: the configured sklearn-compatible model to use
        n_runs: number of times to retrain
        seed: starting seed; this is changed on each run

    Returns:
        vector of CV predictions for xval, averaged over the n_runs
    '''
    preds = np.zeros(xval.shape[0])
    for n in range(0, n_runs):
        model.set_params(random_state=seed + n)
        model.fit(xtr, ytr)
        preds += model.predict_proba(xval)[:, 1]
    preds = preds / n_runs
    return preds


def make2way(xtr, xte):
    '''
    Makes some features representing 2-way combinations of columns in xtr/xte

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


def main():
    # Load data
    xtr = pd.read_csv('../data/train.csv').drop('ROLE_CODE', axis=1)
    ytr = xtr['ACTION'].values
    del xtr['ACTION']
    xtr = xtr.values
    xte = pd.read_csv('../data/test.csv').drop(['id', 'ROLE_CODE'], axis=1).values

    # Create one-hot encoded features for 2-way interactions
    xtr, xte = make2way(xtr, xte)
    enc = OneHotEncoder()
    enc.fit(np.vstack([xtr, xte]))
    xtr = enc.transform(xtr)
    xte = enc.transform(xte)

    cv_preds = np.zeros(xtr.shape[0])
    model = LogisticRegression(C=0.7)
    auc = 0.0
    n_runs = 1
    n_folds = 10
    i = 0
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    for tr_ix, te_ix in kfold.split(xtr, ytr):
        x, xval = xtr[tr_ix], xtr[te_ix]
        y, yval = ytr[tr_ix], ytr[te_ix]
        preds = bagged_set(x, y, xval, model, n_runs)
        cv_preds[te_ix] = preds

        roc_score = roc_auc_score(yval, preds)
        print('Fold %d AUC: %.5f' % (i + 1, roc_score))
        auc += roc_score
        i += 1

    print('Mean AUC: %.5f' % (auc / n_folds))
    cv_preds = pd.Series(cv_preds)
    cv_preds.to_csv('../artifacts/LogReg2way.train.csv', 
                    float_format=FMT, index=False)

    # Make final predictions on all train
    preds = bagged_set(xtr, ytr, xte, model, n_runs)
    ss = pd.read_csv('../data/sample_submission.csv')
    ss.Action = preds
    ss.to_csv('../submissions/submitLogReg2Way.csv', float_format=FMT, index=False)
    

if __name__ == '__main__':
    main()
