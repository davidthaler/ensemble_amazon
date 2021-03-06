'''
Logistic regression model with 2-way interactions. 
Based on code in KazAnovas ensemble_amazon repo.
'''
import numpy as np
import pandas as pd
import argparse
from sklearn.linear_model import LogisticRegression

import features
import util
import models

SEED = util.SEED
TAG = 'logreg'      # Part of output file name,


def main(k=2, C=0.7, nruns=1, nfolds=5, tag=TAG):
    # Load data
    xtr, ytr, xte = util.load_data()

    # Create one-hot encoded features for 1..k-way interactions
    xtr, xte = features.range_combos(xtr, xte, k)

    # Create model
    model = LogisticRegression(C=C)

    # Run CV
    cv_preds = models.cv_loop(xtr, ytr, model, nfolds, nruns, SEED)

    # Save CV predictions for stacking later
    util.save_cv_preds(cv_preds, tag)

    # Fit on all of train, make final predictions on all test
    preds = models.rerun(xtr, ytr, xte, model, nruns, SEED)
    util.write_submission(preds, tag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the logistic regression model.')
    parser.add_argument('--k', type=int, default=2,
            help='Make 1..k way combinations of the feature values.')
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
