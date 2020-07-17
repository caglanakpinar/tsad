from pandas import DataFrame, Series
from fbprophet import Prophet
import random
import numpy as np
from itertools import product
import pandas as pd
import threading
from multiprocessing import cpu_count

from functions import *
from utils import *
from logger import LoggerProcess


def get_anomaly(fact, yhat_upper, yhat_lower):
    ad = Series([0, 0])
    if fact > yhat_upper:
        ad = Series([1, abs((fact - yhat_upper) / fact)])
    if fact < yhat_lower:
        ad = Series([1, abs((yhat_lower - fact)/ fact)])
    return ad


def get_anomaly_score(anomaly, fact, yhat_upper, yhat_lower):
    if anomaly == 1:
        return abs((fact - yhat_upper) / fact)
    if anomaly == -1:
        return abs((yhat_lower - fact)/ fact)


def get_tuning_params(parameter_tuning, params, job):
    arrays = []
    for p in params:
        if p not in list(parameter_tuning.keys()):
            arrays.append([params[p]])
        else:
            arrays.append(
                          np.arange(float(parameter_tuning[p].split("*")[0]),
                                    float(parameter_tuning[p].split("*")[1]),
                                    float(parameter_tuning[p].split("*")[0])).tolist()
            )
    comb_arrays = list(product(*arrays))
    if job != 'parameter_tuning':
        return random.sample(comb_arrays, int(len(comb_arrays)*0.5))
    else:
        return comb_arrays


def get_params(params, comb):
    count = 0
    for p in params:
        _p = type(params[p])(comb[count])
        params[p] = _p
        count += 1
    return params


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class TrainProphet:
    def __init__(self,
                 job=None, groups=None, time_indicator=None, feature=None,
                 data_source=None, data_query_path=None, time_period=None):
        self.job = job
        self.params = hyper_conf('prophet')
        self.combination_params = hyper_conf('prophet_cp')
        self.hyper_params = hyper_conf('prophet_pt')
        self.optimized_parameters = {}
        self._p = None
        self.levels_tuning = get_tuning_params(self.hyper_params, self.params, self.job)
        self.query_date = get_query_date(job, period=time_period, dates=None, params=self.params)
        self.data, self.groups = data_manipulation(job=job,
                                                   date=self.query_date,
                                                   time_indicator=time_indicator,
                                                   feature=feature,
                                                   data_source=data_source,
                                                   groups=groups,
                                                   data_query_path=data_query_path)
        self.date = time_indicator
        self.f_w_data = self.data
        self.split_date = get_split_date(period=time_period, dates=list(self.data[self.date]), params=self.params)
        self.feature = feature
        self.anomaly = []
        self.model = None
        self.count = 1
        self.levels = get_levels(self.data, self.groups)
        self.logger = LoggerProcess(job=job,
                                    model='prophet',
                                    total_process=len(self.levels)
                                    if job != 'parameter_tuning' else len(self.levels_tuning))
        self.comb = None
        self.prediction = None

    def get_query(self):
        count = 0
        query = ''
        for c in self.comb:
            if type(c) != str:
                query += self.groups[count] + ' == ' + str(c) + ' and '
            else:
                query += self.groups[count] + " == '" + str(c) + "' and "
            count += 1
        query = query[:-4]
        return query

    def get_related_params(self):
        self._p = self.params if self.combination_params is None else self.combination_params[self.get_param_key()]

    def convert_date_feature_column_for_prophet(self):
        renaming = {self.date: 'ds', self.feature: 'y'}
        self.f_w_data = self.f_w_data.rename(columns=renaming)
        self.f_w_data['ds'] = self.f_w_data['ds'].apply(lambda x: datetime.datetime.strptime(str(x)[0:19], '%Y-%m-%d %H:%M:%S'))
        return self.f_w_data

    def fit_predict_model(self, save_model=True):
        self.f_w_data = self.convert_date_feature_column_for_prophet()
        self.model = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=False,
                             seasonality_mode='multiplicative',
                             interval_width=float(self._p['interval_width']),
                             changepoint_range=float(self._p['changepoint_range']),
                             n_changepoints=int(self._p['n_changepoints'])
                             ).fit(self.f_w_data[['ds', 'y']])
        if save_model:
            model_from_to_pkl(directory=conf('model_main_path'),
                              path=model_path(self.comb, self.groups, 'prophet'),
                              model=self.model, is_writing=True)

    def detect_anomalies(self):
        self.model = model_from_to_pkl(directory=conf('model_main_path'),
                                       path=model_path(self.comb, self.groups, 'prophet'))
        try:
            self.prediction = self.model.predict(self.convert_date_feature_column_for_prophet())
            self.f_w_data = pd.merge(self.f_w_data,
                                     self.prediction.rename(columns={'ds': self.date}),
                                     on=self.date,
                                     how='left')
            self.f_w_data = self.f_w_data[self.f_w_data[self.date] >= self.split_date]
            self.f_w_data[['ad_label_3', 'anomaly_score_3']] = self.f_w_data.apply(lambda row:
                                                                                   get_anomaly(row[self.feature],
                                                                                            row['yhat_upper'],
                                                                                            row['yhat_lower']), axis=1)
            self.anomaly += self.f_w_data[['ad_label_3', self.date, 'anomaly_score_3'] + self.groups].to_dict("results")
            print(self.f_w_data[['ad_label_3', self.date, 'anomaly_score_3'] + self.groups])
        except Exception as e:
            print(e)

    def train_execute(self):
        if not hyper_conf('prophet_has_param_tuning_first_run'):
            self.parameter_tuning()
        for self.comb in self.levels:
            print("*" * 4, "PROPHET - ", self.get_query().replace(" and ", "; ").replace(" == ", " - "), "*" * 4)
            self.f_w_data = self.data.query(self.get_query()).sort_values(by=self.date)
            print("data size :", len(self.f_w_data))
            self.convert_date_feature_column_for_prophet()
            self.get_related_params()
            self.fit_predict_model()
            self.logger.counter()
            if not check_request_stoped(self.job):
                break

    def prediction_execute(self):
        for self.comb in self.levels:
            print("*" * 4, "PROPHET - ", self.get_query().replace(" and ", "; ").replace(" == ", " - "), "*" * 4)
            if check_model_exists(model_path(self.comb, self.groups, 'prophet'), conf('model_main_path')):
                self.f_w_data = self.data.query(self.get_query()).sort_values(by=self.date)
                print("prediction size :", len(self.f_w_data))
                self.detect_anomalies()
            self.logger.counter()
            if not check_request_stoped(self.job):
                break
        self.anomaly = DataFrame(self.anomaly)

    def process_execute(self, pr, count):
        self.get_related_params()
        self._p = get_params(self._p, pr)
        print("hyper parameters : ", self._p)
        self.convert_date_feature_column_for_prophet()
        self.fit_predict_model(save_model=False)
        self.prediction = self.model.predict(self.convert_date_feature_column_for_prophet())
        error[count] = mean_absolute_percentage_error(self.f_w_data['y'], abs(self.prediction['yhat']))

    def parameter_tuning_threading(self, has_comb=True):
        global error
        error = {}
        _optimized_parameters = None
        self.f_w_data = self.data.query(self.get_query()).sort_values(by=self.date) if has_comb else self.f_w_data
        self.f_w_data = self.f_w_data[-int(0.1 * len(self.f_w_data)):]
        err = 100000000
        for iter in range(int(len(self.levels_tuning) / cpu_count())):
            _levels = self.levels_tuning[(iter * cpu_count()):((iter + 1) * cpu_count())]
            for i in range(len(_levels)):
                process = threading.Thread(target=self.process_execute, daemon=True, args=(_levels[i], i, ))
                process.start()
            process.join()
            print(error)
            for i in error:
                if i in list(error.keys()):
                    if error[i] < err:
                        err = error[i]
                        _optimized_parameters = get_params(self.params, _levels[i])
                self.logger.counter()
        return _optimized_parameters

    def get_param_key(self):
        return "_".join([str(i[0]) + "*" + str(i[1]) for i in zip(self.groups, self.comb)])

    def parameter_tuning(self):
        if len(self.levels) == 0:
            self.optimized_parameters = self.parameter_tuning_threading(has_comb=False)
        else:
            for self.comb in self.levels:
                self.optimized_parameters[self.get_param_key()] = self.parameter_tuning_threading()
        print("updating model parameters")
        pt_config = read_yaml(conf('docs_main_path'), 'parameter_tunning.yaml')
        pt_config['has_param_tuning_first_run']['prophet'] = True
        _key = 'hyper_parameters' if len(self.levels) == 0 else 'combination_params'
        pt_config[_key]['prophet'] = self.optimized_parameters
        write_yaml(conf('docs_main_path'), "parameter_tunning.yaml", pt_config)
        self.params = hyper_conf('prophet')
        self.combination_params = hyper_conf('prophet_cp')








