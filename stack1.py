'''
Implements a 1-level stack model on the Amazon Access data.
'''
import numpy as np
import argparse
from sklearn.ensemble import ExtraTreesClassifier

import util
import models

SEED = util.SEED
TAG = 'stack1'      # Part of output file name
COLS = ['logreg2way', 'logreg3way', 'logreg4way', 
        'xgb01', 'xgb2ct', 'xgb3ct', 'xgbct']

def main(ntree=100, nfolds=5, nruns=1, tag=TAG):
    # Load data
    _, ytr, _ = util.load_data()
    xtr = util.reload_cv_predictions(COLS)
    xte = util.reload_submissions(COLS)

    # Set-up model
    model = ExtraTreesClassifier(n_estimators=ntree,
                                criterion='entropy',
                                max_depth=9,
                                max_features=6,
                                n_jobs=3,
                                random_state=1)
    
    # Run CV
    cv_preds = models.cv_loop(xtr, ytr, model, nfolds, nruns, SEED)

    # Save CV predictions
    util.save_cv_preds(cv_preds, tag)

    # Fit on all of train, make final predictions on all test
    preds = models.rerun(xtr, ytr, xte, model, nruns, SEED)
    util.write_submission(preds, tag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the stack model on CV-predictions.')
    parser.add_argument('--ntree', type=int, default=100, 
        help='number of trees for ExtraTrees model to use in stacking. Default 100.')
    parser.add_argument('--nfolds', type=int, default=5,
                    help='number of CV folds, default 5.')
    parser.add_argument('--nruns', type=int, default=1, 
            help='number of repeated runs. Default 1.')
    parser.add_argument('--tag', default=TAG, 
            help='identifying part of output file names')
    args = parser.parse_args()
    main(**args.__dict__)
