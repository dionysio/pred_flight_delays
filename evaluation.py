#!/usr/bin/env python
#-*- coding: utf-8 -*-

import pickle
import time
import logging
import sys

import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from data_loading import random_state
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, TheilSenRegressor, HuberRegressor, PassiveAggressiveRegressor, SGDRegressor, LassoLars, ElasticNet, Lars, OrthogonalMatchingPursuit, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from xgboost import XGBRegressor


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

regressors = [GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, MLPRegressor, Lars,
              BaggingRegressor, ExtraTreesRegressor, KNeighborsRegressor, ExtraTreeRegressor, LassoLars,
              DecisionTreeRegressor, GaussianProcessRegressor, RadiusNeighborsRegressor, KernelRidge,
              SVR, XGBRegressor, Lasso, Ridge, TheilSenRegressor, HuberRegressor, SGDRegressor,
              PassiveAggressiveRegressor, ElasticNet, OrthogonalMatchingPursuit, BayesianRidge]

def generate_regressors():
    for r in [SGDRegressor]:
        try:
            yield r(random_state=random_state, n_iter=1)
        except TypeError:
            yield r()


def evaluate_batch(data, dump_best_regressor=True):
    #results = evaluate(SGDRegressor(), data, model_params={'alpha': 10.0 ** -np.arange(1, 7)})#, analyze_kwargs={"classif_report": True})
    model_params = {'loss': ['huber'], 'alpha': 10.0 ** -np.arange(1, 7)}

    logger.debug("Got {} training samples and {} test samples.".format(data['train']['X'].shape, data['test']['X'].shape))

    results = []
    lowest_score = {'mean_error': 10000}
    best_performing_regressor = None
    for reg in generate_regressors():
        e = evaluate(reg, data, model_params={'loss': ['huber'], 'alpha': 10.0 ** -np.arange(1, 7)})
        if e[1]['mean_error'] < lowest_score['mean_error']:
            lowest_score = e[1]
            best_performing_regressor = e[0]
        logger.debug(e[1])
        results.append(e)
    logger.info("Best performing regressor was {} with a score of {}".format(best_performing_regressor, lowest_score))

    if dump_best_regressor and best_performing_regressor:
        filename = '_cache/dumps/{}_{}.pickle'.format(lowest_score['mean_error'], best_performing_regressor.__class__.__name__)
        with open(filename, 'wb') as dump_file:
            pickle.dump(best_performing_regressor, dump_file)
        logger.debug('Dumped {} to {}'.format(best_performing_regressor, filename))
    return results


def evaluate(reg, data, analyze_kwargs=None, model_params=None, **kwargs):
    analyze_kwargs = analyze_kwargs or {}
    model_params = model_params or {}

    reg_name = reg.__class__.__name__
    logger.debug("Start fitting '{}'.".format(reg_name))
    kwargs.update({"X": data['train']['X'], "y": data['train']['y']})
    t0 = time.time()
    if model_params:
        reg = GridSearchCV(reg, model_params, cv=5)

    reg.fit(**kwargs)
    t1 = time.time()
    return reg, analyze(reg, data, t1 - t0, **analyze_kwargs)


def sum_absolute_error(y_true, y_pred):
    return sum(abs(y_pred-y_true))/sum(abs(y_true))


def analyze(reg, data, fit_time, classif_report=False):
    results = {}
    t0 = time.time()
    '''predicted = np.array([])
    for i in range(0, len(data['test']['X']), 128):  # go in chunks of size 128
        predicted_single = reg.predict(data['test']['X'][i:(i + 128)])
        predicted = np.append(predicted, predicted_single)
    predicted = np.nan_to_num(predicted)'''
    predicted = reg.predict(data['test']['X'])
    t1 = time.time()
    results['testing_time'] = t1 - t0
    results['training_time'] = fit_time
    metric_data = (data['test']['y'], predicted)
    results['r2_score'] = metrics.r2_score(*metric_data)
    results['absolute_score'] = sum_absolute_error(*metric_data)
    results['mean_error'] = metrics.mean_absolute_error(*metric_data)
    results['classif_report'] = metrics.classification_report(*metric_data) if classif_report else ''
    return results
