import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold

from clean import clean

if __name__ == '__main__':
    # Data preparation
    logging.basicConfig(level=logging.INFO)
    logging.info('Loading datasets...')
    train_idx = pd.read_csv("../train.csv")
    test_idx = pd.read_csv("../test.csv")

    logging.info('Cleaning train dataset...')
    train_x = clean(train_idx)
    train_y = train_idx.loc[:, "OutcomeType"]

    enc = LabelEncoder()
    enc.fit(train_y)
    train_yt = enc.transform(train_y)

    logging.info('Cleaning test dataset...')
    test_x = clean(test_idx)

    for diff in test_x.columns.difference(train_x.columns):
        test_x[diff] = 0

    for diff in train_x.columns.difference(test_x.columns):
        test_x[diff] = 0

    train_x.sort_index(axis=1, inplace=True)
    test_x.sort_index(axis=1, inplace=True)

    # Model creation
    skf = list(StratifiedKFold(train_yt, 10))

    xgb_params = {
                    'learning_rate': 0.2,
                    'max_depth': 6,
                    'n_estimators': 500,
                    'num_class': 5,
                    'objective': 'multi:softprob',
                    'subsample': 0.8,
                    'num_boost_rounds': 200,
                    'eval_metric': 'logloss'}

    # TODO: logloss eval metric for XGBoost
    clfs = [RandomForestClassifier(n_estimators=300, criterion='gini', n_jobs=8),
           ExtraTreesClassifier(n_estimators=500, n_jobs=-1, criterion='gini'),
           XGBClassifier(n_estimators=500, learning_rate=0.2, max_depth=6, objective='multi:softprob', subsample=0.8)]

    train_blend = np.zeroes(train_x.shape[0], len(clfs))
    test_blend = np.zeroes(test_x.shape[0], len(clfs))

    for (j, clf) in enumerate(clfs):
        test_blend_j = np.zeros(test_x.shape[0], len(skf))

        for i, (train_idx, test_idx) in skf:
            x = train_x[train_idx]
            y = train_y[train_idx]
            xt = test_x[test_idx]

            clf.fit(x, y)
            preds = clf.predict_proba(xt)[:, 1]
            train_blend[]

    # Saving results
    final_results = pd.DataFrame(preds)
    final_results.columns = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
    final_results.index.name = 'ID'
    final_results.index = final_results.index + 1

    final_results = final_results[['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']]

    final_results.to_csv('results_averaging.csv')