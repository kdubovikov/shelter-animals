import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
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
    # 300 500 500
    clfs = [RandomForestClassifier(n_estimators=200, criterion='gini', n_jobs=8),
           ExtraTreesClassifier(n_estimators=200, n_jobs=-1, criterion='gini'),
           XGBClassifier(n_estimators=180, learning_rate=0.2, max_depth=6, objective='multi:softprob', subsample=0.75, colsample_bytree=0.85)]

    # Model creation
    skf = list(StratifiedKFold(train_yt, 10))


    # train_blend = np.zeros((train_x.shape[0], len(clfs)))
    # test_blend = np.zeros((test_x.shape[0], len(clfs)))

    # (data_length, num_of_classifiers, num_of_classes)
    train_blend = np.zeros((train_x.shape[0], len(clfs), 5))
    test_blend = np.zeros((test_x.shape[0], len(clfs), 5))

    for (j, clf) in enumerate(clfs):
        print("Fitting %s" % clf)
        test_blend_j = np.zeros((test_x.shape[0], len(skf), 5))

        for i, (train_idx, test_idx) in enumerate(skf):
            print("Fold %d" % i)
            x = train_x.iloc[train_idx]
            y = train_y.iloc[train_idx]
            xt = train_x.iloc[test_idx]
            yt = train_y.iloc[test_idx]

            clf.fit(x, y)
            preds = clf.predict_proba(xt)
            train_blend[test_idx, j] = preds
            test_blend_j[:, i] = clf.predict_proba(test_x)

            print("Train accuracy - %f" % clf.score(xt, yt))
            # print("Precision - " % metrics.precision_score(preds, yt))
            # print("Recall - " % metrics.recall_score(preds, yt))

        test_blend[:, j] = test_blend_j.mean(1)

    # Start blending!
    bclf = LogisticRegression()
    bclf.fit(train_blend.reshape(26729, 5 * len(clfs)), train_y)

    # Predict now
    final_preds = bclf.predict_proba(test_blend.reshape(test_blend.shape[0], 5 * len(clfs)))

    # Saving results
    final_results = pd.DataFrame(final_preds)
    final_results.columns = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
    final_results.index.name = 'ID'
    final_results.index = final_results.index + 1

    final_results = final_results[['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']]

    final_results.to_csv('results_averaging.csv')