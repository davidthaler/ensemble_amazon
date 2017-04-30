'''
XGBoost on combination value-count featurea
Based on code in KazAnovas ensemble_amazon repo.
'''
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost.sklearn import XGBClassifier

import count_features
import util
import models

SEED = 42
TAG = 'xgbct'      # Part of output file name,


def main(kmax=1, ntree=100, nruns=1, nfolds=5, tag=TAG):
    xtr, ytr, xte = util.load_data(as_pandas=True)

    # Create value-count features
    xall = pd.concat([xtr, xte])
    xtr = count_features.range_counts(xall, xtr, kmax)
    xte = count_features.range_counts(xall, xte, kmax)
    xtr = xtr.values
    xte = xte.values

    model = XGBClassifier(n_estimators=ntree, 
                          learning_rate=0.02,
                          gamma=1,
                          max_depth=20,
                          min_child_weight=0.1,
                          subsample=0.9,
                          colsample_bytree=0.5,
                          seed=1)

    cv_preds = np.zeros(xtr.shape[0])
    auc = 0.0
    i = 0
    kfold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=SEED)

    for tr_ix, te_ix in kfold.split(xtr, ytr):
        x, xval = xtr[tr_ix], xtr[te_ix]
        y, yval = ytr[tr_ix], ytr[te_ix]
        preds = models.rerun(x, y, xval, model, nruns, SEED)
        cv_preds[te_ix] = preds

        roc_score = roc_auc_score(yval, preds)
        print('Fold %d AUC: %.6f' % (i + 1, roc_score))
        auc += roc_score
        i += 1

    print('Mean AUC: %.6f' % (auc / nfolds))
    util.save_cv_preds(cv_preds, tag)

    # Fit on all of train, make final predictions on all test
    preds = models.rerun(xtr, ytr, xte, model, nruns, SEED)
    util.write_submission(preds, tag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the xgboost model on value-count features')
    parser.add_argument('--kmax', type=int, default=1,
            help='Make 1..k way combinations of the feature values. Default 1.')
    parser.add_argument('--ntree', type=int, default=100, 
                    help='number of trees for xgboost to use. Default 100.')
    parser.add_argument('--nfolds', type=int, default=5,
                    help='number of CV folds, default 5.')
    parser.add_argument('--nruns', type=int, default=1, 
            help='number of repeated runs. Default 1.')
    parser.add_argument('--tag', default=TAG, 
            help='identifying part of output file names')
    args = parser.parse_args()
    main(**args.__dict__)
