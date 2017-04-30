'''
Holds code factored out of the other models for running the 
models and the CV loop.
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import util


def rerun(xtr, ytr, xval, model, n_runs, seed):
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
        if 'random_state' in model.get_params():
            model.random_state = seed + n
        elif 'seed' in model.get_params():
            model.seed = seed + n
        model.fit(xtr, ytr)
        preds += model.predict_proba(xval)[:, 1]
    return preds / n_runs


def cv_loop(xtr, ytr, model, nfolds=5, nruns=1, seed=util.SEED):
    '''
    Runs a CV loop of model on x and y. 
    Uses rerun to do multiple refits of models with different random state.
    Collects scores and predictions.

    Args:
        xtr: training set data as numpy array
        ytr: training set labels as 1-d array
        model: a configured sklearn-compatible model
        nfolds: number of CV folds; default 5
        nruns: number of repeated runs of the model on each fold; default 1
        seed: int - random number seed, default is util.SEED

    Returns:
        CV predictions, also prints out CV results
    '''
    cv_preds = np.zeros(xtr.shape[0])
    auc = 0.0
    i = 0
    kfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)

    for tr_ix, te_ix in kfold.split(xtr, ytr):
        x, xval = xtr[tr_ix], xtr[te_ix]
        y, yval = ytr[tr_ix], ytr[te_ix]
        preds = rerun(x, y, xval, model, nruns, seed)
        cv_preds[te_ix] = preds

        roc_score = roc_auc_score(yval, preds)
        print('Fold %d AUC: %.6f' % (i + 1, roc_score))
        auc += roc_score
        i += 1

    print('Mean AUC: %.6f' % (auc / nfolds))
    return cv_preds
