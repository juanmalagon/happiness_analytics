#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct  10 00:21:22 2019

@author: juanmalagon
"""

import logging
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier, Pool
from catboost.utils import select_threshold
from sklearn.model_selection import train_test_split
from sklearn import metrics


def create_model(source_df):
    # Remove NAME_COUNTRY columns
    source_df = source_df.drop(columns=source_df.filter(like='NAME_COUNTRY').columns)
    # Data splittting
    X = source_df.drop('HAPPINESS', axis=1)
    y = source_df.HAPPINESS
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, random_state=42, stratify=y)
    X = X_train
    y = y_train
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, train_size=0.75, random_state=42, stratify=y)
    cat_features = np.where((X.dtypes != np.float) & (X.dtypes != np.int))[0]
    # Model training
    model = CatBoostClassifier(
        iterations=1200, random_seed=42, one_hot_max_size=4,
        loss_function='Logloss', custom_loss=['Accuracy', 'Recall', 'F1'])
    model.fit(X_train, y_train, cat_features=cat_features,
              eval_set=(X_validation, y_validation), verbose=50, plot=False)
    # Select decision boundary
    eval_pool = Pool(X_validation, y_validation, cat_features=cat_features)
    cutoff = select_threshold(model=model, data=eval_pool, FNR=0.3)
    # Model evaluation
    probabilities = model.predict_proba(data=X_test)
    probs = [u[1] for u in probabilities]
    y_pred = [1 if u > cutoff else 0 for u in probs]
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    model_stats = {'true_positive': confusion_matrix[1, 1],
                   'false_negative': confusion_matrix[1, 0],
                   'false_positive': confusion_matrix[0, 1],
                   'true_negative': confusion_matrix[0, 0],
                   'precision': metrics.precision_score(y_test, y_pred),
                   'recall': metrics.recall_score(y_test, y_pred),
                   'accuracy': metrics.accuracy_score(y_test, y_pred)}
    logging.info(f"Confusion matrix: {confusion_matrix}")
    logging.info(f"Precision score: {model_stats['precision']}")
    logging.info(f"Recall score: {model_stats['recall']}")
    logging.info(f"Accuracy score: {model_stats['accuracy']}")
    # Create bins
    pd_bins = pd.DataFrame()
    pd_bins['y_test'] = y_test
    pd_bins['probs'] = probs
    pd_bins['bin'] = pd.qcut(pd_bins['probs'], q=int(handler.nr_bins), duplicates='drop')
    HAPPINESSs_per_bin = pd_bins['y_test'].groupby(pd_bins['bin'])
    conversion = pd.DataFrame(HAPPINESSs_per_bin.sum())
    conversion['BIN'] = list(range(1, int(handler.nr_bins) + 1))
    conversion.columns = ['CONVERTED', 'BIN']
    conversion['RANGE'] = conversion.index
    conversion.reset_index(drop=True, inplace=True)
    logging.info('Conversion per bin (training): ', )
    print(conversion)
    # Feature importance
    pool1 = Pool(data=X, label=y, cat_features=cat_features)
    features = model.get_feature_importance(prettified=True)
    shap_values = model.get_feature_importance(pool1, type='ShapValues')    
    return model, conversion, shap_values, features, model_stats


def apply_model(source_df, previous_HAPPINESSs_source_df, model):
    # Remove NAME_COUNTRY columns
    source_df = source_df.drop(columns=source_df.filter(like='NAME_COUNTRY').columns)
    never_HAPPINESS = source_df[~source_df['IDI_COUNTRY'].isin(previous_HAPPINESSs_source_df)]
    X_to_predict = never_HAPPINESS.drop('HAPPINESS', axis=1)
    y_predict_proba = model.predict_proba(data=X_to_predict)
    predictions = never_HAPPINESS.assign(PREDICTION=[u[1] for u in y_predict_proba])
    predictions = predictions[['IDI_COUNTRY', 'PREDICTION']]
    return predictions
