from coffee_csv_loading import process_data
import catboost as cb
import numpy as np
import pandas as pd
import requests
import io
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

par_grid = {
    "learning_rate": [0.1, 0.33],
    "l2_leaf_reg": [0.33, 1.0, 3.0],
    # "bootstrap_type": ["Bayesian"],
    # "bagging_temperature": [0.5],
    # "random_strength": [1.0],
    "depth": [6],
    # "nan_mode": ["Min"],
    "auto_class_weights": ["Balanced"],
    # "boosting_type": ["Plain"],
    # "score_function": ["Cosine"],
    "iterations": [30, 50, 80],
}

best_ps = {
    "auto_class_weights": "Balanced",
    "iterations": 50,
    "l2_leaf_reg": 0.33,
    "learning_rate": 0.1,
}

import warnings

if __name__ == "__main__":
    # csv = io.BytesIO(requests.get("https://bit.ly/gacttCSV").content)
    _df = pd.read_csv("GACTT_RESULTS_ANONYMIZED.csv")
    data, target, ccols, scols = process_data(_df)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.30)
    print(X_train)
    # datapool = cb.Pool(data=data, label=target, cat_features=ccols)

    clf = cb.CatBoostClassifier(metric_period=10, eval_metric="TotalF1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        validator = GridSearchCV(clf, n_jobs=2, param_grid=par_grid, scoring="f1_macro")
        out_val = validator.fit(X_train, y_train, cat_features=ccols)

    # clf = cb.CatBoostClassifier(metric_period=10, eval_metric="TotalF1", **best_ps)
    # pars = clf.grid_search(X=datapool, param_grid=par_grid, plot=True)
    # print(pars)
