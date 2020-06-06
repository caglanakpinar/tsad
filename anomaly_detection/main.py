import sys
from inspect import getargspec
from functions import merged_models
from train_v2 import TrainLSTM
from prophet_model import TrainProphet
from iforest_model import TrainIForest
from logger import *
from utils import kill_process_with_name, url_string


def get_model(model):
    if model == 'lstm':
        return TrainLSTM
    if model == 'iso_f':
        return TrainIForest
    if model == 'prophet':
        return TrainProphet


def model_execute(model, groups, feature, time_indicator, data_source, data_query_path, time_period):
    train = get_model(model)(job='prediction',
                             groups=groups,
                             time_indicator=time_indicator,
                             feature=feature,
                             data_source=data_source,
                             data_query_path=data_query_path, time_period=time_period)
    train.prediction_execute()
    return train.anomaly


def main(job=None,
         model=None,
         groups=None,
         date=None,
         time_indicator=None,
         feature=None,
         data_source=None,
         data_query_path=None,
         time_period=None):

    if job != 'stop':
        sys.stdout = Logger(job)
    else:
        print("job is stoped!")

    data_query_path = url_string(data_query_path)
    print("received :",
                        {'job': job, 'model': model, 'groups': groups,
                         'date': date, 'time_indicator': time_indicator,
                         'feature': feature, 'data_source': data_source,
                         'data_query_path': data_query_path, 'time_period': time_period},
          " time :", get_time()
          )
    if job == 'train':
        train = get_model(model)(job=job,
                                 groups=groups,
                                 time_indicator=time_indicator,
                                 feature=feature,
                                 data_source=data_source,
                                 data_query_path=data_query_path)
        train.train_execute()
        train.logger.check_total_process_and_regenerate()
    if job == 'prediction':
        outputs = {m: model_execute(m, groups, feature, time_indicator, data_source, data_query_path, time_period) for m in ['lstm', 'iso_f', 'prophet']}
        result = merged_models(model_1=outputs['lstm'],
                               model_2=outputs['iso_f'],
                               model_3=outputs['prophet'],
                               date=date,
                               data_source=data_source,
                               data_query_path=data_query_path,
                               time_indicator=time_indicator,
                               groups=groups,
                               time_period=time_period,
                               feature=feature)
        logger = LoggerProcess(job=job)
        logger.check_total_process_and_regenerate()
    if job == 'parameter_tuning':
        p_tunning = get_model(model)(job=job,
                                     groups=groups,
                                     time_indicator=time_indicator,
                                     feature=feature,
                                     data_source=data_source,
                                     data_query_path=data_query_path)
        p_tunning.parameter_tuning()
        logger = LoggerProcess(job=job)
        logger.check_total_process_and_regenerate()
    if job == 'stop':
        kill_process_with_name(process_name="main.py", argument=model)
        logger = LoggerProcess(job=job)
        logger.regenerate_file()
    get_time()
    print("Done!!")


if __name__ == '__main__':
    args = {arg: None for arg in getargspec(main)[0]}
    counter = 1
    for k in args:
        args[k] = sys.argv[counter] if sys.argv[counter] not in ['-', None] else None
        counter += 1
    main(**args)
