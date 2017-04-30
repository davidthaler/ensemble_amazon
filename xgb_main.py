'''
XGBoost on 0-1 featurea
Based on code in KazAnovas ensemble_amazon repo.
'''
import numpy as np
import pandas as pd
import argparse
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import OneHotEncoder

import util
import models

SEED = util.SEED
TAG = 'xgb01'      # Part of output file name,


def main(ntree=100, nruns=1, nfolds=5, tag=TAG):
    # Load data
    xtr, ytr, xte = util.load_data()

    # Create one-hot encoded features
    enc = OneHotEncoder()
    enc.fit(np.vstack([xtr, xte]))
    xtr = enc.transform(xtr)
    xte = enc.transform(xte)

    # Create model
    model = XGBClassifier(n_estimators=ntree, 
                          learning_rate=0.12,
                          gamma=0.01,
                          max_depth=12,
                          min_child_weight=0.01,
                          subsample=0.6,
                          colsample_bytree=0.7,
                          seed=1)
    
    # Run CV
    cv_preds = models.cv_loop(xtr, ytr, model, nfolds, nruns, SEED)

    # Save CV predictions
    util.save_cv_preds(cv_preds, tag)

    # Fit on all of train, make final predictions on all test
    preds = models.rerun(xtr, ytr, xte, model, nruns, SEED)
    util.write_submission(preds, tag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the xgboost model on 0-1 features')
    parser.add_argument('--ntree', type=int, default=100, 
                    help='number of trees for xgboost to use. Default 100.')
    parser.add_argument('--nfolds', type=int, default=5,
                    help='number of CV folds, default 5.')
    parser.add_argument('--nruns', type=int, default=1, 
            help='number of reruns. Default 1.')
    parser.add_argument('--tag', default=TAG, 
            help='identifying part of output file names')
    args = parser.parse_args()
    main(**args.__dict__)
