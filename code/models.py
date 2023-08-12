"""
models.py contains baseline & ensemble algorithms:
    1. Baseline Algorithms:
        (1) Linear Regression - OLS + Ridge + Lasso
        (2) Regression Tree - RT
        (3) Support Vector Regression - SVR

    2. Ensemble Algorithm
        (1) Bagging 
        (2) Boosting
        (3) Stacking
"""
from util import *
from sklearn import linear_model, svm, tree
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor, BaggingRegressor

# Baseline Models
def Baseline(model=None, x_train=None, y_train=None, x_test=None, y_test=None):
    """ Fit/predict baseline models and generate evaluation metrics
    
    Args:
        model: str, to specify the regression model
        data: continuous or discrete
        x_train: DataFrame, training dataset of features
        y_train: DataFrame, training dataset of labels
        x_test: DataFrame, testing dataset of features
        y_test: DataFrame, testing dataset of labels

    Returns:
        DataFrame, with evaluation metrics
    """
    if model.lower() == 'ols':
        model = 'Linear Regression'
        clr = linear_model.LinearRegression().fit(x_train, np.ravel(y_train))
    elif model.lower() == 'ridge':
        model = 'Ridge Regression'
        clr = linear_model.Ridge(alpha=.5).fit(x_train, np.ravel(y_train))
    elif model.lower() == 'lasso':
        model = 'Lasso Regression'
        clr = linear_model.Lasso(alpha=.1).fit(x_train, np.ravel(y_train))
    elif model.lower() == 'rt':
        model = 'Regression Tree'
        clr = tree.DecisionTreeRegressor().fit(x_train, np.ravel(y_train))
    elif model.lower() == 'svr':
        model = 'Support Vector Regression'
        clr = svm.SVR().fit(x_train, np.ravel(y_train))

    y_pred = clr.predict(x_test)
    
    return point_eval_metric(y_true=y_test, y_pred=y_pred, model=model)

# Ensemble Algorithm
def Ensemble():
    return