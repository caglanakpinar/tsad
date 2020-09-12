from pandas import DataFrame
from multiprocessing import Pool
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
import random
from numpy import arange, isnan, array
import threading
from multiprocessing import cpu_count

from functions import *
from configs import conf
from data_access import *
from logger import LoggerProcess


def get_params(params, comb):
    count = 0
    for p in params:
        _p = type(params[p])(comb[count])
        params[p] = _p
        count += 1
    print("params :::", params)
    return params


def get_tuning_params(parameter_tuning, params, job):
    arrays = []
    for p in params:
        if p not in list(parameter_tuning.keys()):
            arrays.append([params[p]])
        else:
            arrays.append(
                          arange(float(parameter_tuning[p].split("*")[0]),
                                 float(parameter_tuning[p].split("*")[1]),
                                 float(parameter_tuning[p].split("*")[0])).tolist()
            )
    comb_arrays = list(product(*arrays))
    if job != 'parameter_tuning':
        return random.sample(comb_arrays, int(len(comb_arrays)*0.5))
    else:
        return comb_arrays


class TrainIForest:
    def __init__(self, job=None, groups=None, time_indicator=None, feature=None,
                 data_source=None, data_query_path=None, time_period=None):
        self.job = job
        self.params = hyper_conf('iso_f')
        self.combination_params = hyper_conf('iso_f_cp')
        self.hyper_params = hyper_conf('iso_f_pt')
        self.optimized_parameters = {}
        self._p = None
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
        self.optimized_parameters = self.params
        self.feature = feature
        self.anomaly = []
        self.model = None
        self.count = 1
        self.levels = get_levels(self.data, self.groups)
        self.levels_tuning = get_tuning_params(self.hyper_params, self.params, self.job)
        self.logger = LoggerProcess(job=job,
                                    model='iso_f',
                                    total_process=len(self.levels)
                                    if job != 'parameter_tuning' else len(self.levels_tuning))
        self.comb = None
        self.train, self.prediction = None, None
        self.model = None
        self.anomaly = []
        self.outliers = []

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
        print("_p :::::", self._p)
        print("params :::::", self.params)

    def split_data(self, is_prediction=False):
        if not is_prediction:
            self.train = self.f_w_data[self.f_w_data[self.date] < self.split_date]
        else:
            self.prediction = self.f_w_data[self.f_w_data[self.date] >= self.split_date]

    def train_model(self, save_model=True):
        self.model = IsolationForest(n_estimators=self._p['num_of_trees'],
                                     max_samples='auto',
                                     contamination=self._p['contamination'],
                                     bootstrap=False, n_jobs=-1, random_state=42, verbose=1
                                     ).fit(self.train[[self.feature]].values)
        if save_model:
            model_from_to_pkl(directory=conf('model_main_path'),
                              path=model_path(self.comb, self.groups, 'iso_f'),
                              model=self.model, is_writing=True)

    def detect_anomalies(self):
        try:
            self.model = model_from_to_pkl(directory=conf('model_main_path'),
                                           path=model_path(self.comb, self.groups, 'iso_f'))
            self.prediction['ad_label_2'] = self.model.fit_predict(self.prediction[[self.feature]].fillna(0).values)
            self.prediction['anomaly_score_2'] = self.model.score_samples(self.prediction[[self.feature]].fillna(0).values)
            self.prediction['ad_label_2'] = self.prediction['ad_label_2'].apply(lambda x: 1 if x == -1 else 0)
            self.anomaly += self.prediction[['ad_label_2', self.date, 'anomaly_score_2'] + self.groups].to_dict("results")
            print(self.prediction[['ad_label_2', self.date, 'anomaly_score_2'] + self.groups].head())
        except Exception as e:
            print(e)

    def train_execute(self):
        if not hyper_conf('iso_f_has_param_tuning_first_run'):
            self.parameter_tuning()
        for self.comb in self.levels:
            print("*" * 4, "ISO FOREST - ", self.get_query().replace(" and ", "; ").replace(" == ", " - "), "*" * 4)
            self.f_w_data = self.data.query(self.get_query()).sort_values(by=self.date)
            print("data size :", len(self.f_w_data))
            self.get_related_params()
            self.split_data()
            self.train_model()
            self.logger.counter()
            if not check_request_stoped(self.job):
                break

    def prediction_execute(self):
        for self.comb in self.levels:
            print("*" * 4, "ISO FOREST - ", self.get_query().replace(" and ", "; ").replace(" == ", " - "), "*" * 4)
            if check_model_exists(model_path(self.comb, self.groups, 'iso_f'), conf('model_main_path')):
                self.f_w_data = self.data.query(self.get_query()).sort_values(by=self.date)
                self.split_data(is_prediction=True)
                print("prediction size :", len(self.prediction))
                self.detect_anomalies()
            self.logger.counter()
            if not check_request_stoped(self.job):
                break
        self.anomaly = DataFrame(self.anomaly)

    def get_outliers_for_tuning_iso_f(self):
        for self.comb in self.levels:
            self.f_w_data = self.data.query(self.get_query()).sort_values(by=self.date)
            self.f_w_data['outliers'] = get_outliers(list(self.f_w_data[self.feature]))
            self.outliers += self.f_w_data[['outliers', self.date, self.feature] + self.groups].to_dict("results")
        self.f_w_data = DataFrame(self.outliers)

    def process_execute(self, pr, count):
        self.get_related_params()
        print("data size :", len(self.f_w_data))
        self._p = get_params(self._p, pr)
        self.split_data()
        self.train_model(save_model=False)
        self.f_w_data['ad_label_2'] = self.model.predict(self.f_w_data[[self.feature]].fillna(0).values)
        self.f_w_data['ad_label_2'] = self.f_w_data['ad_label_2'].apply(lambda x: 1 if x != 0 else 0)
        print(self.f_w_data[['ad_label_2', 'outliers']].head(10))
        error[count] = isnan(array(1-f1_score(self.f_w_data['outliers'], self.f_w_data['ad_label_2'], average='macro')))

    def parameter_tuning_threading(self, has_comb=True):
        global error
        error = {}
        _optimized_parameters = None
        err = 100000000
        self.f_w_data = self.data.query(self.get_query()).sort_values(by=self.date) if has_comb else self.f_w_data
        self.get_outliers_for_tuning_iso_f()
        self.f_w_data = self.f_w_data[-int(0.3 * len(self.f_w_data)):]
        for iter in range(int(len(self.levels_tuning) / cpu_count())):
            _levels = self.levels_tuning[(iter * cpu_count()):((iter + 1) * cpu_count())]
            for i in range(len(_levels)):
                self.logger.counter()
                process = threading.Thread(target=self.process_execute, daemon=True, args=(_levels[i], i, ))
                process.start()
            process.join()
            for i in error:
                if i in list(error.keys()):
                    if error[i] < err:
                        err = error[i]
                        _optimized_parameters = get_params(self.params, _levels[i])
        return _optimized_parameters

    def get_param_key(self):
        return "_".join([str(i[0]) + "*" + str(i[1]) for i in zip(self.groups, self.comb)])

    def parameter_tuning(self):
        if len(self.levels) == 0:
            self.optimized_parameters = self.parameter_tuning_threading(has_comb=False)
        else:
            for self.comb in self.levels:
                self.optimized_parameters[self.get_param_key()] = self.parameter_tuning_threading()
                if not check_request_stoped(self.job):
                    break
        print("updating model parameters")
        print(self.optimized_parameters)
        pt_config = read_yaml(conf('docs_main_path'), 'parameter_tunning.yaml')
        pt_config['has_param_tuning_first_run']['iso_f'] = True
        _key = 'hyper_parameters' if len(self.levels) == 0 else 'combination_params'
        pt_config[_key]['iso_f'] = self.optimized_parameters
        write_yaml(conf('docs_main_path'), "parameter_tunning.yaml", pt_config, ignoring_aliases=True)
        self.params = hyper_conf('iso_f')
        self.combination_params = hyper_conf('iso_f_cp')








