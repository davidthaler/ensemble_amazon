'''
Holds code factored out of the other models.
'''
import numpy as np
import pandas as pd

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