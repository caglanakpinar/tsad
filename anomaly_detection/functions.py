import pandas as pd
import numpy as np
from math import sqrt
from scipy import stats
from itertools import product, combinations

from data_access import GetData
from configs import conf, time_dimensions, alpha, day_of_year, time_indicator_accept_threshold, s_size_ratio, hyper_conf
from utils import *


def data_manipulation(job, date, time_indicator, feature, data_source, groups, data_query_path):
    data_process = GetData(data_source=data_source,
                           data_query_path=data_query_path,
                           time_indicator=time_indicator,
                           feature=feature,
                           date=date)
    data_process.data_execute()
    print("data size :", len(data_process.data))

    print("Checking for null values ..")
    nulls = AssignNullValues(data=data_process.data,
                             groups=split_groups(groups),
                             feature=feature,
                             time_indicator=time_indicator)
    nulls.null_value_assign_from_prev_time_steps()
    print("check for time part data ..")
    date_features = TimePartFeatures(job=job,
                                     data=nulls.data,
                                     time_indicator=time_indicator,
                                     groups=split_groups(groups),
                                     feature=feature)
    date_features.date_dimension_deciding()
    return date_features.data.sort_values(by=time_indicator, ascending=True), date_features.groups


class AssignNullValues:
    def __init__(self, data, groups, feature, time_indicator):
        self.data_raw = data
        self.groups = groups
        self.feature = feature
        self.time_indicator = time_indicator
        self.comb = None
        self.timestamps = list(range(4))
        self.columns = list(self.data_raw.columns)
        self.levels = list(
            product(*[list(self.data_raw[self.data_raw[g] == self.data_raw[g]][g].unique()) for g in self.groups]))
        self.data = []

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

    def last_timestamps_of_mean(self, row, mean):
        if row[self.feature] != row[self.feature]:
            result = np.mean([row[i] for i in self.timestamps if row[i] == row[i]])
            if result != result:
                result = mean
        else:
            result = row[self.feature]
        return result

    def null_value_assign_from_prev_timestamps(self, _data):
        mean = np.mean(_data[_data[self.feature] == _data[self.feature]][self.feature])
        for i in self.timestamps:
            _data[i] = _data[[self.feature]].shift(i)
        _data[self.feature] = _data.apply(lambda row: self.last_timestamps_of_mean(row, mean), axis=1)
        self.data += _data[self.columns].to_dict('results')

    def check_date_column_is_unique_on_each_group(self):
        if self.groups:
            self.data = self.data.pivot_table(index=self.groups,
                                              aggfunc={self.time_indicator: 'count',
                                                       self.time_indicator + '_unique': lambda x: len(x.unique())}
                                              ).reset_index()
        else:
            if len(self.data[self.time_indicator]) != len(self.data[self.time_indicator].unique()):
                self.data = self.data.pivot_table(index=self.time_indicator,
                                                  aggfunc={self.feature: 'mean'}
                                                  ).reset_index()

    def null_value_assign_from_prev_time_steps(self):
        for self.comb in self.levels:
            _data = self.data_raw.query(self.get_query()).sort_values(by=self.time_indicator)
            if len(_data) != 0:
                _ratio = len(_data[_data[self.feature] != _data[self.feature]]) / len(_data)
                if _ratio < 0.5:
                    self.null_value_assign_from_prev_timestamps(_data)
        self.data = pd.DataFrame(self.data)


def calculate_t_test(mean1, mean2, var1, var2, n1, n2, alpha):
    """
    It Test according to T- Distribution of calculations
    There are two main Hypotheses Test are able to run. Two Sample Two Tail Student-T Test, One Sample Student-T-Test
    In order to test two main parameters are needed Mean and Variance, T~(M1, Var)
    :param mean1: Mean of Sample 1
    :param mean2: Mean of Sample 2. If one sample T-Test assign None
    :param var1: Variance of Sample 1.
    :param var2: Variance of Sample 2. If one sample T-Test assign None
    :param n1: sample 1 size
    :param n2: sample 2 size. If one sample T-Test assign None
    :param alpha: Confidence level
    :param two_sample: Boolean; True - False
    :return: returns p-value of test, confidence interval of test, H0 Accepted!! or H0 REjected!!
    """
    # Two Sample T Test (M0 == M1) (Two Tails)
    t = (mean1 - mean2) / sqrt((var1 / n1) + (var2 / n2))  # t statistic calculation for two sample
    df = n1 + n2 - 2  # degree of freedom for two sample t - set
    pval = 1 - stats.t.sf(np.abs(t), df) * 2  # two-sided pvalue = Prob(abs(t)>tt) # p - value
    cv = stats.t.ppf(1 - (alpha / 2), df)
    standart_error = cv * sqrt((var1 / n1) + (var2 / n2))
    confidence_intervals = [abs(mean1 - mean2) - standart_error, abs(mean1 - mean2) + standart_error, standart_error]
    acception = 'HO REJECTED!' if pval < (alpha / 2) else 'HO ACCEPTED!'  # left tail
    acception = 'HO REJECTED!' if pval > 1 - (alpha / 2) else 'HO ACCEPTED!'  # right tail
    return pval, confidence_intervals, acception


def sampling(sample, sample_size):
    if len(sample) <= sample_size:
        return sample
    else:
        return random.sample(sample, sample_size)


def boostraping_calculation(sample1, sample2, iteration, sample_size, alpha):
    """
    Randomly selecting samples from two population on each iteration.
    Iteratively independent test are applied each randomly selected samples
    :param sample1: list of values related to sample 1
    :param sample2: list of values related to sample 2
    :param iteration: numeber of iteration. It is better to asssign higher values.
    :param sample_size: number of sample when randomly sampled from each data set.
                        Make sure this parameters bigger than both sample of size
    :param alpha: Confidence level
    :param two_sample: Is it related to Two Sample Test or not.
    :return: HO_accept ratio: num of accepted testes / iteration
             result data set: each iteration of test outputs of pandas data frame
    """
    pval_list, h0_accept_count, test_parameters_list = [], 0, []
    for i in range(iteration):
        random1 = sampling(sample=sample1,
                           sample_size=sample_size)  # random.sample(sample1, sample_size)  # randomly picking samples from sample 1
        random2 = sampling(sample=sample2,
                           sample_size=sample_size)  # random.sample(sample2, sample_size)  # randomly picking samples from sample 2
        mean1, mean2 = np.mean(random1), np.mean(random2)
        var1, var2 = np.var(random1), np.var(random2)
        pval, confidence_intervals, hypotheses_accept = calculate_t_test(mean1, mean2, var1, var2, sample_size,
                                                                         sample_size, alpha)
        h0_accept_count += 1 if hypotheses_accept == 'HO ACCEPTED!' else 0
    return h0_accept_count / iteration, pd.DataFrame(test_parameters_list)


def smallest_time_part(dates):
    sample_dates = random.sample(dates, int(len(dates) * 0.8))
    smallest = False
    t_dimensions = list(reversed(time_dimensions))
    count = 0
    while not smallest:
        (unique, counts) = np.unique([date_part(d, t_dimensions[count]) for d in sample_dates], return_counts=True)
        if len(unique) >= 2:
            smallest = True
            smallest_td = t_dimensions[count]
            break
        count += 1
    accepted_t_dimensions = list(reversed(t_dimensions[count + 1:]))
    return accepted_t_dimensions, smallest_td  # smallest time indicator not included to time_dimensions


def get_time_difference(dates):
    return (max(dates) - min(dates)).total_seconds()


class TimePartFeatures:
    def __init__(self, job=None, data=None, time_indicator=None, groups=None, feature=None):
        self.job = job
        self.data = data
        self._data = data
        self.time_indicator = time_indicator
        self.groups = groups
        self.time_groups = None
        self.date = time_indicator
        self.feature = feature
        self.time_diff = get_time_difference(list(self.data[self.time_indicator]))
        self.time_dimensions, self.smallest_time_indicator = smallest_time_part(list(self.data[self.time_indicator]))
        self.time_dimensions_accept = {d: False for d in self.time_dimensions}
        self.threshold = time_indicator_accept_threshold['threshold']
        self.accept_ratio_value = time_indicator_accept_threshold['accept_ratio_value']

    def remove_similar_time_dimensions(self, part):
        accept = False
        if part == 'year':
            if self.smallest_time_indicator not in ['week', 'week_part', 'week_day', 'day']:
                accept = True
        if part == 'quarter':
            if self.smallest_time_indicator not in ['week', 'week_part', 'week_day', 'day']:
                if not self.time_dimensions_accept['year']:
                    accept = True
        if part == 'month':
            if self.smallest_time_indicator not in ['week', 'week_part', 'week_day', 'day']:
                if not self.time_dimensions_accept['quarter']:
                    accept = True
        if part == 'week':
            if self.smallest_time_indicator in ['hour', 'min', 'second']:
                if len([1 for p in ['year', 'quarter', 'month'] if self.time_dimensions_accept[p]]) == 0:
                    accept = True
        if part == 'week_part':
            if self.smallest_time_indicator in ['day', 'hour']:
                accept = True
        if part == 'week_day':
            if self.smallest_time_indicator in ['hour', 'min', 'second']:
                if not self.time_dimensions_accept['week_part']:
                    accept = True
        if part == 'day_part':
            if self.smallest_time_indicator in ['min', 'second']:
                accept = True
        if part == 'hour':
            if self.smallest_time_indicator == 'second':
                if not self.time_dimensions_accept['day_part']:
                    accept = True
        return accept

    def iteration_count(self, s1, s2):
        iter = int(min(len(s1), len(s2)) * 0.0001)
        if 1000 < min(len(s1), len(s2)) < 10000:
            iter = int(min(len(s1), len(s2)) * 0.01)
        if min(len(s1), len(s2)) < 1000:
            iter = int(min(len(s1), len(s2)) * 0.1)
        return iter

    def get_threshold(self, part):
        update_values = False
        if part == 'quarter':
            if self.time_dimensions_accept['year']:
                update_values = True
        if part == 'month':
            if self.time_dimensions_accept['year'] or self.time_dimensions_accept['quarter']:
                update_values = True
        if part == 'week':
            if len([1 for p in ['year', 'quarter', 'month'] if self.time_dimensions_accept[p]]) != 0:
                update_values = True

        if update_values:
            self.threshold = time_indicator_accept_threshold['threshold'] - 0.2
            self.accept_ratio_value = time_indicator_accept_threshold['accept_ratio_value'] + 0.2

    def time_dimension_decision(self, part):
        if self.remove_similar_time_dimensions(part):
            self.get_threshold(part=part)
            accept_count = 0
            combs = list(combinations(list(self.data[part].unique()), 2))
            for comb in combs:
                sample_1 = sampling(list(self._data[self._data[part] == comb[0]][self.feature]), sample_size=100000)
                sample_2 = sampling(list(self._data[self._data[part] == comb[1]][self.feature]), sample_size=100000)
                iter = self.iteration_count(sample_1, sample_2)
                h0_accept_ratio, params = boostraping_calculation(sample1=sample_1,
                                                                  sample2=sample_2,
                                                                  iteration=iter,
                                                                  sample_size=int(
                                                                      min(len(sample_1), len(sample_2)) * s_size_ratio),
                                                                  alpha=0.05)
                accept_count += 1 if h0_accept_ratio < self.threshold else 0
            accept_ratio = len(combs) * self.accept_ratio_value  # 50%
            print("Time Part :", part, "Accept Treshold :", accept_ratio, "Accepted Count :", accept_count)
            return True if accept_count > accept_ratio else False

    def day_decision(self):
        return True if self.smallest_time_indicator in ['min', 'second'] else False

    def year_decision(self):
        return True if int(self.time_diff / 60 / 60 / 24) >= (day_of_year * 2) else False

    def quarter_decision(self):
        return True if int(self.time_diff / 60 / 60 / 24) >= (day_of_year * 1) else False

    def check_for_time_difference_ranges_for_accepting_time_part(self, part):
        decision = False
        if part == 'year':
            decision = self.year_decision()
        if part == 'quarter':
            decision = self.quarter_decision()
        if part == 'week_day':
            decision = self.day_decision()
        return decision

    def decide_timepart_of_group(self, part):
        print("*" * 5, "decision for time part :", part, "*" * 5)
        result = False
        (unique, counts) = np.unique(list(self.data[part]), return_counts=True)
        if len(unique) >= 2:
            if 1 not in counts:
                if part not in ['week_day', 'hour', 'min', 'second']:
                    if self.check_for_time_difference_ranges_for_accepting_time_part(part):
                        result = self.time_dimension_decision(part)
                else:
                    if part == 'week_day':
                        results = self.check_for_time_difference_ranges_for_accepting_time_part(part)
                    if self.smallest_time_indicator == 'second' and part == 'hour':
                        result = self.time_dimension_decision(part)
        print("result :", " INCLUDING" if result else "EXCLUDING")
        self.time_dimensions_accept[part] = result
        return result

    def calculate_date_parts(self):
        accepted_date_parts = []
        for t_dimension in self.time_dimensions:
            if t_dimension not in self.groups:
                self.data[t_dimension] = self.data[self.date].apply(lambda x: date_part(x, t_dimension))
                if self.decide_timepart_of_group(part=t_dimension):
                    accepted_date_parts.append(t_dimension)
        self.time_groups = accepted_date_parts
        self.groups += accepted_date_parts

    def date_dimension_deciding(self):
        if self.job != 'prediction':
            self.calculate_date_parts()
            info = {'infos': {'min_date': str(min(list(self.data[self.time_indicator])))[0:19],
                              'max_date': str(max(list(self.data[self.time_indicator])))[0:19],
                              'time_groups': "*".join(self.time_groups)}
                    }
            write_yaml(conf('model_main_path'), 'model_configuration.yaml', info)
        else:
            self.time_groups = read_yaml(conf('model_main_path'), "model_configuration.yaml")['infos'][
                'time_groups'].split("*")
            for t_dimension in self.time_groups:
                if t_dimension not in self.groups:
                    self.data[t_dimension] = self.data[self.date].apply(lambda x: date_part(x, t_dimension))
            if self.time_groups != ['']:
                self.groups += self.time_groups
        print("time parts : ", self.time_groups)


class DimensionsCorrelation:
    def __init__(self, job=None, data=None, time_indicator=None, groups=None, feature=None):
        self.job = job
        self.data = data
        self.time_indicator = time_indicator
        self.groups = groups
        self.time_groups = None
        self.date = time_indicator
        self.feature = feature
        self.time_dimensions, self.smallest_time_indicator = smallest_time_part(list(self.data[self.time_indicator]))
        self.time_dimensions_accept = {d: False for d in self.time_dimensions}


reshape_3 = lambda x: x.reshape((x.shape[0], x.shape[1], 1))
reshape_2 = lambda x: x.reshape((x.shape[0], 1))


def drop_calculation(df, parameters, is_prediction=False):
    data_count = len(df)
    to_drop = max((parameters['tsteps'] - 1), (parameters['lahead'] - 1))
    df = df[to_drop:]
    if not is_prediction:
        to_drop = df.shape[0] % parameters['batch_size']
        if to_drop > 0:
            df = df[:-1 * to_drop]
    return df


def data_preparation(df, f, parameters, is_prediction):
    y = df[f].rolling(window=parameters['tsteps'], center=False).mean()
    x = pd.DataFrame(np.repeat(df[f].values, repeats=parameters['lag'], axis=1))
    shift_day = int(parameters['lahead'] / parameters['lag'])
    if parameters['lahead'] > 1:
        for i, c in enumerate(x.columns):
            x[c] = x[c].shift(i * shift_day)  # every each same days of shifted
    x = drop_calculation(x, parameters, is_prediction=is_prediction)
    y = drop_calculation(y, parameters, is_prediction=is_prediction)
    return split_data(y, x, parameters) if not is_prediction else reshape_3(x.values)


def split_data(Y, X, params):
    x_train = reshape_3(X[: -int(params['batch_count'] * params['batch_size'])].values)
    y_train = reshape_2(Y[: -int(params['batch_count'] * params['batch_size'])].values)
    x_test = reshape_3(X[-int(params['batch_count'] * params['batch_size']):].values)
    y_test = reshape_2(Y[-int(params['batch_count'] * params['batch_size']):].values)
    return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}


def get_randoms(sample, s_size):
    try:
        if len(sample) != 0:
            return random.sample(sample, s_size)
    except Exception as e:
        print(e)
        return sample


class AnomalyScore:
    """
    Randomly selecting samples from two population on each iteration.
    Iteratively calculating p-value by using Normal Distribution Probability Density Function (PDF)
    Then, use Bayesian Approach Posterior = Prior * P(Q), Q ~ ß(a, b), Q = p-value
    a = number of (+), b number of (-)
    Ex: if p-value = 0.4 in sample size 2000, a = 800, b = 1200 (a+b = 2000)
    Update Beta Distribution of a, b paramteres by sum them up
    Final calcuate p-value on Beta Distribution PDF
    :param sample: list of values related to sample 1
    :param iteration: numeber of iteration. It is better to asssign higher values.
    :param s_size: number of sample when randomly sampled from each data set.
                        Make sure this parameters bigger than both sample of size

    :return: HO_accept ratio: num of accepted testes / iteration
             result data set: each iteration of test outputs of pandas data frame
    """

    def __init__(self, data, feature, iteration, s_size):
        self.data = list(data['residuals'].fillna(0))  # assign 0 which indicates mean(residuals) ~= 0
        self.sample_mean = np.median(data[data[feature] == data[feature]]['residuals'])
        self.sample = list(data[data[feature] == data[feature]]['residuals'].fillna(self.sample_mean))
        self.iteration = iteration
        self.s_size = s_size if len(self.sample) > s_size else len(self.sample) - 2
        self.mean = None
        self.scaled_values = {}
        self.a, self.b = 1, 1
        self.iter_samples = {}
        self.min_max_norm_vals = []
        print("sample size :", s_size)

    def create_mean_p_value(self):
        count = 0
        _min, _max = min(self.data), max(self.data)
        for i in self.data:
            self.scaled_values[count] = min_max_scaling(i, _min, (_max - _min))
            count += 1
        self.mean = np.mean(list(self.scaled_values.values()))

    def create_iterations(self):
        self.create_mean_p_value()
        for iter in range(self.iteration):
            _r_values = get_randoms(self.sample, self.s_size)
            if len(_r_values) == 0:
                _r_values = self.sample
            _min, _max = min(_r_values), max(_r_values)
            self.iter_samples[iter] = {
                'mean': np.mean([min_max_scaling(i, _min, (_max - _min)) for i in _r_values]),
                'min': _min,
                'max': _max,
                'range': _max - _min
            }

    def calculate_beta_pdf(self, value, idx):
        p_value = self.scaled_values[idx]
        p_value = (0.5 + p_value) / 1 if value < self.mean else p_value
        _p_value_2 = 1
        self.a, self.b = 1, 1
        for i in range(self.iteration):
            if self.iter_samples[i]['max'] > value > self.iter_samples[i]['min']:
                _p_value_1 = abs(value - self.iter_samples[i]['min']) / self.iter_samples[i]['range']
                _p_value_2 = (0.5 + _p_value_1) / 1 if _p_value_1 < self.iter_samples[i]['mean'] else _p_value_1
            self.b += int(self.s_size * _p_value_2) if _p_value_2 == _p_value_2 else 0
            self.a += int(self.s_size * (1 - _p_value_2)) if _p_value_2 == _p_value_2 else self.s_size
        return stats.beta.ppf(p_value, self.a, self.b)

    def outlier_detection(self):
        cal_t_value = lambda x: abs(x - np.mean(self.data)) / sqrt(np.var(self.data) / len(self.data))
        return [stats.t.sf(np.abs(cal_t_value(val)), len(self.data) - 1) * 2 for val in self.data]


def calculation_as(sample, iteration, s_size, value):
    """
    Randomly selecting samples from two population on each iteration.
    Iteratively calculating p-value by using Normal Distribution Probability Density Function (PDF)
    Then, use Bayesian Approach Posterior = Prior * P(Q), Q ~ ß(a, b), Q = p-value
    a = number of (+), b number of (-)
    Ex: if p-value = 0.4 in sample size 2000, a = 800, b = 1200 (a+b = 2000)
    Update Beta Distribution of a, b paramteres by sum them up
    Final calcuate p-value on Beta Distribution PDF
    :param sample: list of values related to sample 1
    :param iteration: numeber of iteration. It is better to asssign higher values.
    :param s_size: number of sample when randomly sampled from each data set.
                        Make sure this parameters bigger than both sample of size

    :return: HO_accept ratio: num of accepted testes / iteration
             result data set: each iteration of test outputs of pandas data frame
    """
    start = datetime.datetime.now()
    mean = np.mean([min_max_scaling(i, min(sample), (max(sample) - min(sample))) for i in sample])
    p_value = min_max_scaling(value, min(sample), (max(sample) - min(sample)))

    p_value = (0.5 + p_value) / 1 if value < mean else p_value
    a, b = 1, 1  # Beta Distribution parameters
    for i in range(iteration):
        try:
            _r_values = random.sample(sample, s_size)  # randomly picking samples from sample 1
            _mean = np.mean([min_max_scaling(i, min(_r_values), (max(_r_values) - min(_r_values))) for i in _r_values])
            _p_value_2 = 1
            if max(_r_values) > value > min(_r_values):
                _p_value_1 = min_max_scaling(value, min(_r_values),
                                             (max(_r_values) - min(_r_values)))  # Normal Distribution P-Value
                _p_value_2 = (0.5 + _p_value_1) / 1 if _p_value_1 < _mean else _p_value_1
            b += int(s_size * _p_value_2) if _p_value_2 == _p_value_2 else 0
            a += int(s_size * (1 - _p_value_2)) if _p_value_2 == _p_value_2 else s_size
        except Exception as e:
            print(e)
    print("residuals startes :", (datetime.datetime.now() - start).total_seconds())
    print(datetime.datetime.now())
    result = stats.beta.ppf(p_value, a, b)
    return result


def min_max_scaling(val, min_val, range_val):
    return (val - min_val) / range_val if range_val != 0 else 0


def calculate_residuals(actual, predict):
    return abs(actual - predict) if actual != 0 else 1000000


def calculate_intersect_days(parameters):
    return datetime.timedelta(days=max(parameters['tsteps'], parameters['lahead']))


def check_model_exists(model_path, path=None):
    if model_path in listdir(dirname(join(path, ""))):
        return True
    else:
        return False


def calculate_predict_label(row, columns):
    labels = [row['ad_label_' + str(i)] for i in range(1, 4) if 'ad_label_' + str(i) in columns]
    if len(labels) == 0:
        return 2
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return 1 if sum(labels) in [1, 2] else 0
    if len(labels) == 3:
        return 1 if sum(labels) in [2, 3] else 0


def final_label_anomaly(data):
    cols = list(data.columns)
    data['predicted_label'] = data.apply(lambda row:
                                         calculate_predict_label(row, cols), axis=1)
    return data


def merged_models(model_1, model_2, model_3, date, data_source, data_query_path, time_indicator, groups, time_period,
                  feature):
    groups = split_groups(groups)
    split_date = get_split_date(period=time_period)
    data_process = GetData(data_source=data_source,
                           data_query_path=data_query_path,
                           time_indicator=time_indicator,
                           feature=feature,
                           date=split_date)
    data_process.data_execute()
    raw_data = data_process.data
    models = {1: model_1, 2: model_2, 3: model_3}
    pv_columns = groups + [time_indicator]
    for m in models:
        label, score = 'ad_label_' + str(m), 'anomaly_score_' + str(m)
        if len(models[m]) != 0:
            _columns = pv_columns + [label, score]
            if 'predict' in list(models[m].columns):
                _columns = pv_columns + ['predict', label, score]
            raw_data = pd.merge(raw_data, models[m][_columns], on=pv_columns, how='left').fillna(0)
    data = final_label_anomaly(raw_data)
    raw_data.to_csv(conf('merged_results'), index=False, encoding='utf-8')
    return data


def get_outliers(values):
    n, var, mean = len(values), np.var(values), np.mean(values)
    df = n - 1  # degree of freedom for two sample t - set
    cv = stats.t.ppf(1 - (alpha / 2), df)
    standart_error = cv * sqrt(var / n)
    intervals = [mean - standart_error, mean + standart_error]
    return list(map(lambda x: 1 if x > intervals[0] and x < intervals[1] else 0, values))


def save_model_configurations(job, data, time_indicator, time_groups):
    info = {}
    if job == 'prediction':
        info = {'infos': {'min_date': min(list(data[time_indicator])), 'max_date': max(list(data[time_indicator])),
                          'time_groups': time_groups}
                }
        with open(join(conf('model_main_path'), "model_configuration.yaml"), 'w') as file:
            yaml.dump(info, file)


def check_request_stoped(job):
    return True if read_yaml(conf('docs_main_path'), 'ml_execute.yaml')[job]['active'] is True else False


def check_for_normalization(data):
    return True if 1 > min(data) >= 0 and 1 >= max(data) > 0 else False


def get_levels(data, groups):
    groups = [g for g in groups if g not in [None, '', 'none', 'null', 'None']]
    return list(product(*[list(data[data[g] == data[g]][g].unique()) for g in groups]))


















