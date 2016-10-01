#!/usr/bin/env python
#-*- coding: utf-8 -*-
from data_loading import get_data
from evaluation import evaluate_batch




evaluate_batch(get_data(filename='delays_dataset.csv', num_samples=1000000, add_date_based_features=False, add_aggregates=False, refresh=True))