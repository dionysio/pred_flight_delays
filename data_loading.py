#!/usr/bin/env python
#-*- coding: utf-8 -*-

import subprocess
import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


from feature_engineering import add_features
import numpy as np
from sklearn.preprocessing import StandardScaler


random_state = np.random.RandomState(666)

def read_csv(filename, **kwargs):
    return pd.read_csv(filename, parse_dates=['scheduled_departure', 'actual_departure'], infer_datetime_format=True,
                       names=['carrier','fltno','dep_apt','arr_apt',
                              'sched_departure_date','scheduled_departure','actual_departure'],
                       usecols=['carrier','dep_apt','arr_apt', 'scheduled_departure','actual_departure'],
                       compression=None, **kwargs)


def chain_shell_commands(commands, stdout=None):
    jobs = []
    for job_num, command in enumerate(commands):
        jobs.append(subprocess.Popen(command, stdin=jobs[job_num-1].stdout if job_num else None,
                                     stdout=stdout or subprocess.PIPE))
    return (jobs[-1].communicate()[0] or b'').decode()


def _get_num_lines(filename):
    return int(chain_shell_commands((('cat', filename), ('wc', '-l'))).strip())


def _preprocess_large_data_to_sample(filename, every_nth_line, output, start=0):
    return chain_shell_commands((('sed', '-n', "{}~{}p".format(start, every_nth_line), filename),),
                                stdout=open(output, 'wb'))


def generate_sample_file(filename, num_samples, output):
    num_lines = _get_num_lines(filename)
    n = num_lines // num_samples
    return _preprocess_large_data_to_sample(filename=filename, every_nth_line=n, output=output)


def get_data_sample(filename, num_samples, **read_csv_kwargs):
    if num_samples == 'all':
        output = filename
    else:
        filepath, output = os.path.split(filename)
        output = os.path.join(filepath, '_cache/samples/sampled_{}_{}'.format(num_samples, output))
        if not os.path.exists(output):
            generate_sample_file(filename, num_samples, output=output)

    return read_csv(output, **read_csv_kwargs)


def get_data(filename, num_samples, add_date_based_features=True, add_aggregates=False, refresh=True):
    df_split = "_cache/split_data/{}_{}_dates_{}_aggregates_{}.pickle".format(filename, num_samples, add_date_based_features,
                                                                       add_aggregates)

    if not refresh and os.path.exists(df_split):
        with open(df_split, 'rb') as pickle_dump:
            data = pickle.load(pickle_dump)
    else:
        data = get_data_sample(filename=filename, num_samples=num_samples)
        for dt in ['scheduled_departure', 'actual_departure']:
            data[dt] = data[dt].astype('datetime64[ns]')
        data = add_features(data, add_date_based_features=add_date_based_features, add_aggregates=add_aggregates)

        y_col = 'delay'
        stratify_col = 'has_delay'
        X = data.drop([y_col, stratify_col], axis=1)
        X = _encode(X)

        X_train, X_test, y_train, y_test = train_test_split(X, data[y_col], test_size=0.2, stratify=data[stratify_col],
                                                            random_state=random_state)
        #X_train, X_test = _standardize(X_train, X_test)
        data = {'train': {'X': X_train, 'y': y_train},
                'test': {'X': X_test, 'y': y_test}
                }

        with open(df_split, 'wb') as pickle_dump:
            pickle.dump(data, pickle_dump)

    return data

def _standardize(X_train, X_test):
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train)  # Don't cheat - fit only on training data
    return (scaler.transform(X_train), scaler.transform(X_test))


def _encode(df):
    encoder = DictVectorizer()
    return encoder.fit_transform(df.to_dict(orient='records'))



