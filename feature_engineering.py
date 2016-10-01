#!/usr/bin/env python
#-*- coding: utf-8 -*-
import pandas as pd
from holidays import codes
from functools import partial
import numpy as np


_get_time = lambda date, attr: getattr(date.dt, attr)
_get_hour = lambda date: _get_time(date, 'hour')
_get_date = lambda date: _get_time(date, 'date')
_get_weekday = lambda date: _get_time(date, 'weekday')
_get_month = lambda date: _get_time(date, 'month')
_get_day = lambda date: _get_time(date, 'day')

_get_timedelta = lambda date: date / np.timedelta64(1, 's')
_get_has_delay = lambda date: date > 0

def _days_from_nearest_holiday(data):
    holidays = codes.get(data['dep_apt'])
    if holidays:
        return min(abs((data['scheduled_departure'] - holiday).days) for holiday in holidays)
    else:
        return 365

_calculate_rolling_count = lambda group, window=3, interval='D': group['has_delay'].rolling(window=window).count()



def add_features(data, add_date_based_features=True, add_aggregates=True):
    data['actual_departure'].fillna(data['scheduled_departure'], inplace=True)
    data['delay'] = _get_timedelta(data['actual_departure'] - data['scheduled_departure']).astype(np.int32)
    data['has_delay'] = _get_has_delay(data['delay'])

    if add_date_based_features:
        if add_aggregates:
            data['scheduled_departure_date'] = _get_date(data['scheduled_departure']).astype('datetime64[ns]')
        data['scheduled_departure_weekday'] = _get_weekday(data['scheduled_departure'])
        data['scheduled_departure_hour'] = _get_hour(data['scheduled_departure'])
        data['scheduled_departure_month'] = _get_month(data['scheduled_departure'])
        data['scheduled_departure_day'] = _get_day(data['scheduled_departure'])
        #data['nearest_holiday'] = data.apply(_days_from_nearest_holiday, axis=1)

    if add_aggregates:
        for direction in ('dep_apt', 'arr_apt'):
            avg_inout = data.groupby(direction)[direction].count()
            avg_inout /= (data['scheduled_departure'].max() - data['scheduled_departure'].min()).days
            data = data.join(avg_inout, on=direction, rsuffix='_avg_flights')

            agg = data.copy()
            agg.set_index('scheduled_departure', inplace=True)
            for interval in ('D', 'H'):
                interval_agg = agg.resample(interval).last()
                interval_agg = interval_agg.groupby(direction)#, as_index=False)
                for window in (1,3,7):
                    temp_agg = interval_agg.apply(partial(_calculate_rolling_count, window=window, interval=interval))
                    temp_agg = temp_agg.reset_index()
                    new_column = '_'.join((direction, str(window), interval))
                    temp_agg.columns = [direction, 'scheduled_departure_date', new_column]

                    data = temp_agg.merge(data, how='right', on=['scheduled_departure_date', direction])
                    data[new_column].fillna(0, inplace=True)

        temp_agg = (data.groupby('dep_apt').rolling(window=7)['has_delay'].sum().fillna(0))/7
        temp_agg = temp_agg.reset_index()
        temp_agg.drop('dep_apt', axis=1, inplace=True)
        temp_agg.set_index('level_1', inplace=True)
        temp_agg.columns = ['last_7_delayed_ratio']
        data = data.join(temp_agg, rsuffix='_x')

        data['avg_lateness'] = data.groupby('fltno')['has_delay'].transform(np.mean)

    if add_date_based_features and add_aggregates:
        data.drop('scheduled_departure_date', axis=1, inplace=True)

    data.drop('scheduled_departure', axis=1, inplace=True)
    data.drop('actual_departure', axis=1, inplace=True)
    return data