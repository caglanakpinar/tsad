from pandas import DataFrame
from multiprocessing import Pool
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

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
    return params


def get_tuning_params(parameter_tuning, params):
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
    return arrays


class TrainIForest:
    def __init__(self, job=None, groups=None, date=None, time_indicator=None, feature=None, data_source=None, data_query_path=None, time_period=None):
        self.job = job
        self.params = conf('parameters_2')
        self.query_date = get_query_date(job, period=time_period, dates=None, params=self.params)
        self.data, self.groups = data_manipulation(job, self.query_date, time_indicator, feature, data_source, groups, data_query_path)
        self.date = time_indicator
        self.f_w_data = self.data
        self.split_date = get_split_date(period=time_period, dates=list(self.data[self.date]), params=self.params)
        self.optimized_parameters = self.params
        self.hyper_params = conf('parameter_tuning')['iso_f']
        self.feature = feature
        self.anomaly = []
        self.model = None
        self.count = 1
        self.levels = list(product(*[list(self.data[self.data[g] == self.data[g]][g].unique()) for g in self.groups if g not in [None, '', 'none', 'null', 'None']]))
        self.levels_tuning = list(product(*get_tuning_params(self.hyper_params, self.params)))
        self.logger = LoggerProcess(job=job,
                                    model='iso_f',
                                    total_process=len(self.levels) if job != 'parameter_tuning' else len(self.levels_tuning))
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

    def split_data(self, is_prediction=False):
        if not is_prediction:
            self.train = self.f_w_data[self.f_w_data[self.date] < self.split_date]
        else:
            self.prediction = self.f_w_data[self.f_w_data[self.date] >= self.split_date]

    def train_model(self, save_model=True):
        self.model = IsolationForest(n_estimators=self.params['num_of_trees'],
                                     max_samples='auto',
                                     contamination=self.params['contamination'],
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
            self.prediction['ad_label_2'] = self.model.predict(self.prediction[[self.feature]].fillna(0).values)
            self.prediction['anomaly_score_2'] = self.model.score_samples(self.prediction[[self.feature]].fillna(0).values)
            self.prediction['ad_label_2'] = self.prediction['ad_label_2'].apply(lambda x: 1 if x != 0 else 0)
            self.anomaly += self.prediction[['ad_label_2', self.date, 'anomaly_score_2'] + self.groups].to_dict("results")
            print(self.prediction[['ad_label_2', self.date, 'anomaly_score_2'] + self.groups].head())
        except Exception as e:
            print(e)

    def train_execute(self):
        if not conf('has_param_tuning_first_run'):
            self.parameter_tuning()
        for self.comb in self.levels:
            print("*" * 4, "ISO FOREST - ", self.get_query().replace(" and ", "; ").replace(" == ", " - "), "*" * 4)
            self.f_w_data = self.data.query(self.get_query()).sort_values(by=self.date)
            print("data size :", len(self.f_w_data))
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

    def parameter_tuning(self):
        error = 1000000
        self.f_w_data = self.data.pivot_table(index=self.date,
                                              aggfunc={self.feature: 'mean'}
                                              ).reset_index().sort_values(by=self.date, ascending=True)
        self.get_outliers_for_tuning_iso_f()
        for pr in list(product(*get_tuning_params(self.hyper_params, self.params))):
            print("*" * 4, "ISO FOREST - ", self.get_query().replace(" and ", "; ").replace(" == ", " - "), "*" * 4)
            print("data size :", len(self.f_w_data))
            self.params = get_params(self.params, pr)
            self.split_data()
            self.train_model(save_model=False)
            self.f_w_data['ad_label_2'] = self.model.predict(self.f_w_data[[self.feature]].fillna(0).values)
            self.f_w_data['ad_label_2'] = self.f_w_data['ad_label_2'].apply(lambda x: 1 if x != 0 else 0)
            score = f1_score(self.f_w_data['outliers'], self.f_w_data['ad_label_2'], average='macro')
            print("F1 Score : ", score)
            if score < error:
                error = score
                self.optimized_parameters = self.params
            self.logger.counter()
            if not check_request_stoped(self.job):
                break
        print("updating model parameters")
        config = read_yaml(conf('docs_main_path'), 'configs.yaml')
        config['hyper_parameters']['prophet'] = self.optimized_parameters
        config['has_param_tuning_first_run'] = True
        write_yaml(conf('docs_main_path'), "configs.yaml", config)








