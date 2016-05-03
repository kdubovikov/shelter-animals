from clean import clean
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('Loading datasets...')
    train = pd.read_csv("../train.csv")
    test = pd.read_csv("../test.csv")

    logging.info('Cleaning train dataset...')
    train_x = clean(train)
    train_y = train.loc[:, "OutcomeType"]

    logging.info('Cleaning test dataset...')
    test_x = clean(test)

    enc = LabelEncoder()
    enc.fit(train_y)
    train_yt = enc.transform(train_y)

    logging.info(train_x.head(10))

    xgb_params = {
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'n_estimators': 1000,
                    'num_class': 5,
                    'objective': 'multi:softprob',
                    'subsample': 0.8}

    # bst = xgb.train(xgb_params, xgb.DMatrix(train_x, train_yt))
    # fscores = bst.get_fscore()
    #
    # filtered_fscores_t = {k: v for k, v in fscores.items() if v > 0}
    # filtered_cols_t = list(filtered_fscores_t.keys())
    #
    # train_xf = train_x[filtered_cols_t]
    # # val_xr = val_x[filtered_cols]
    # test_xf = test_x[filtered_cols_t]

    boosters = np.array([])
    predictions = []

    # print(xgb.cv(xgb_params, xgb.DMatrix(train_x, train_yt), nfold=5, num_boost_round=1000, early_stopping_rounds=10, metrics=["mlogloss"], verbose_eval=False))

    for i in range(0, 15):
        print("Fitting XGB model #%d" % i)
        booster = xgb.train(xgb_params, xgb.DMatrix(train_x, train_yt), verbose_eval=False)
        boosters = np.append(boosters, booster)
        predictions.append(booster.predict(xgb.DMatrix(test_x)))
    #
    # rf_models = np.array([])
    #
    # # for i in range(0, 10):
    # #     print("Fitting RF model #%d" % i)
    # #     rf_clf = RandomForestClassifier(n_estimators=300, criterion='gini', n_jobs=8)
    # #     rf_clf.fit(train_x, train_yt)
    # #     rf_models = np.append(rf_models, rf_clf)
    # #     predictions.append(rf_clf.predict_proba(test_x))
    #
    preds = np.mean(predictions, axis=0)
    final_results = pd.DataFrame(preds)
    final_results.columns = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
    final_results.index.name = 'ID'
    final_results.index = final_results.index + 1

    final_results = final_results[['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']]

    final_results.to_csv('results_stacking.csv')