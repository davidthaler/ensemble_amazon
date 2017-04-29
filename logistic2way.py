'''
A refactoring of "bagged_set" code from KazAnovas github CV-prediction.
'''
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

import features
import util


SEED = 42
TAG = 'logreg2way'      # Part of output file name,

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
    return preds / n_runs


def main(C=0.7, nruns=1, nfolds=5, tag=TAG):
    xtr, ytr, xte = util.load_data()

    # Create one-hot encoded features for 2-way interactions
    xtr, xte = features.make2way(xtr, xte)
    enc = OneHotEncoder()
    enc.fit(np.vstack([xtr, xte]))
    xtr = enc.transform(xtr)
    xte = enc.transform(xte)

    cv_preds = np.zeros(xtr.shape[0])
    model = LogisticRegression(C=C)
    auc = 0.0
    i = 0
    kfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=SEED)

    for tr_ix, te_ix in kfold.split(xtr, ytr):
        x, xval = xtr[tr_ix], xtr[te_ix]
        y, yval = ytr[tr_ix], ytr[te_ix]
        preds = bagged_set(x, y, xval, model, nruns)
        cv_preds[te_ix] = preds

        roc_score = roc_auc_score(yval, preds)
        print('Fold %d AUC: %.5f' % (i + 1, roc_score))
        auc += roc_score
        i += 1

    print('Mean AUC: %.5f' % (auc / nfolds))
    util.save_cv_preds(cv_preds, tag)

    # Fit on all of train, make final predictions on all test
    preds = bagged_set(xtr, ytr, xte, model, nruns)
    util.write_submission(preds, tag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the logistic regression model.')
    parser.add_argument('--C', type=float, default=0.7, 
            help='C parameter of logistic regression (liblinear), default 0.7')
    parser.add_argument('--nfolds', type=int, default=5,
                    help='number of CV folds, default 5.')
    parser.add_argument('--nruns', type=int, default=1, 
            help='number of runs, default 1, which is fine for this model.')
    parser.add_argument('--tag', default=TAG, 
            help='identifying part of output file names')
    args = parser.parse_args()
    main(**args.__dict__)
