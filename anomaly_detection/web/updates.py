from numpy import mean
from os import listdir
from os.path import dirname, join
import requests
import datetime
import yaml

from utils import read_yaml, convert_date, write_yaml
from configs import conf
from data_access import GetData
from ml_executor import CreateJobs, start_job_and_update_job_active


def db_connection_update(**args):
    configs = read_yaml(conf('docs_main_path'), 'configs.yaml')
    configs['db_connection']['data_source'] = args['data_source']
    configs['db_connection']['is_from_db'] = False if args['data_source'] in ['csv', 'json', 'pickle'] else True
    infos = {'db': args.get('db_name', None), 'password': args.get('pw', None),
             'port': args.get('port', None), 'server': args.get('host', None), 'user': args.get('user', None)}
    print(infos)
    for i in infos:
        configs['db_connection'][i] = infos[i]
    write_yaml(conf('docs_main_path'), "configs.yaml", configs)


def db_connection_reset(configs):
    for arg in configs['db_connection']:
        configs['db_connection'][arg] = None
    print("reset db connection information")
    write_yaml(conf('docs_main_path'), "configs.yaml", configs)


def models_reset(model_configuration):
    model_configuration2 = model_configuration
    for m in model_configuration['infos']:
        model_configuration2['infos'][m] = None
    write_yaml(conf('model_main_path'),  "model_configuration.yaml", model_configuration2)


def logs_reset(process):
    process2 = process
    for j in process:
        for m in process[j]:
            process2[j][m] = 0
    write_yaml(conf('log_main_path'), "process.yaml", process2)


def ml_execute_update(**update_dict):
    keys = ['jobs', 'description', 'data_query_path', 'data_source', 'groups', 'dates', 'data_query_path',
            'data_source', 'groups', 'dates', 'time_indicator', 'feature', 'description', 'days']
    infos = {k: update_dict.get(k, None) for k in keys}
    print("yeesssss")
    jobs = infos['jobs']
    for j in jobs:
        jobs[j]['description'] = infos['description']
        jobs[j]['day'] = infos['days'][j] if infos['days'] else None
        jobs[j]['job_start_date'] = str(infos['dates'][j][0])[0:16] if infos['dates'] else None
        jobs[j]['job_end_date'] = None if not infos['dates'] else None if infos['dates'][j][1] else str(infos['dates'][j][1])[0:16]
        e2 = []
        for e in jobs[j]['execute']:
            for p in e['params']:
                if p in infos.keys():
                    e['params'][p] = str(infos[p])
                if p == 'time_period':
                    e['params'][p] = infos['days'][j] if infos['days'] else None
            e2.append(e)
        jobs[j]['execute'] = e2
    print("yesssadadadadadadasdadda")
    print("update :")
    write_yaml(conf('docs_main_path'), "ml_execute.yaml", jobs)


def ml_execute_reset(jobs):
    for j in jobs:
        jobs[j]['description'] = None
        jobs[j]['day'] = None
        jobs[j]['job_start_date'] = None
        jobs[j]['job_end_date'] = None
        e2 = []
        for e in jobs[j]['execute']:
            for p in e['params']:
                if p not in ['model', 'job']:
                    e['params'][p] = None
            e2.append(e)
        jobs[j]['execute'] = e2
    print("reset ml-execute")
    write_yaml(conf('docs_main_path'), "ml_execute.yaml", jobs)


def get_dates(job, req):
    date1 = convert_date(req['date1_' + job], req.get('time1_' + job, None))
    date2 = convert_date(req.get('date2_' + job, None), req.get('time2_' + job, None))
    return [date1, date2]


def calculate_percent(models):
    return int(mean(list(map(lambda x: float(x), models.values()))) * 100)


def get_logs(job_names, dates, current_active_status, process):
    log_infos = {jn: {'status': 'init',
                      'process': 'stop',
                      'active': current_active_status[jn],
                      'start_time': datetime.datetime.strptime(dates[jn], '%Y-%m-%d %H:%M') if dates[jn] else '',
                      'percent': calculate_percent(process[jn])} for jn in job_names}
    print("*****")
    for jn in log_infos:
        if log_infos[jn]['active'] is True:
            log_infos[jn]['status'] = 'waiting' if log_infos[jn]['percent'] == 0 else 'in_progress'
    return log_infos


def update_logs(jobs, log_infos, req):
    for job_name in jobs:
        print(list(req.keys()))
        if job_name in list(req.keys()):
            print("fact :")
            print("Job was initialized before.." if jobs[job_name]['active'] is True else "job was stopped or even was not started before..")
            print("request :", req[job_name])
            j = CreateJobs(jobs, job_name)
            if req[job_name] == 'start':
                log_infos[job_name]['process'] = 'start'
                if jobs[job_name]['active'] is not True:
                    log_infos[job_name]['status'] = 'waiting'
                    print("stop if there is existed running task")
                    j.stop_job()
                    print("request for start !!!")
                    start_job_and_update_job_active(jobs=jobs, job=job_name)
                else:
                    print("job has already been started!!!")
            if req[job_name] == 'stop':
                log_infos[job_name]['process'], log_infos[job_name]['status'] = 'stop', 'init'
                if jobs[job_name]['active'] is True:
                    print("request for stop !!!")
                    j.stop_job()
                else:
                    print("job has already been stopped!!!")
        else:
            log_infos[job_name]['process'] = 'start' if jobs[job_name]['active'] is True else 'stop'
            log_infos[job_name]['status'] = log_infos[job_name]['status'] if jobs[job_name]['active'] is True else 'init'
    print(log_infos)
    return log_infos


def get_request_values(job, params, request):
    req = dict(request.form)
    list_value_none = lambda x, multiple=False: None if len(x) == 0 else x[0] if not multiple else x
    feature = list_value_none(request.form.getlist('feature'))
    time_indicator = list_value_none(request.form.getlist('time_indicator'))
    groups = list_value_none(request.form.getlist('dimensions'), multiple=True)
    groups = "+".join(groups) if groups is not None else None
    days = {j: list_value_none(request.form.getlist('day_' + j)) for j in ['train', 'parameter_tuning', 'prediction']}
    dates = {j: get_dates(j, req) for j in ['train', 'prediction']}
    dates['parameter_tuning'] = [dates['train'][0] + datetime.timedelta(days=1), None]
    ml_dict = {'jobs': job,
               'data_query_path': params['data_query_path'],
               'data_source': params['data_source'],
               'groups': groups,
               'dates': dates,
               'time_indicator': time_indicator,
               'feature': feature, 'description': req['description'], 'days': days}
    return ml_dict


def get_sample_data(params, connection, create_sample_data=True):
    data, cols = None, None
    #try:
    sample_size = 10 if not create_sample_data else 1000
    d = GetData(data_query_path=params['data_query_path'],
                data_source=params['data_source'],
                test=sample_size)
    print("yessssssssss")
    d.query_data_source()
    cols = d.data.columns.values
    # data = d.data.to_html(classes=["table table-bordered table-striped table-hover table-sm"])
    if create_sample_data:
        d.data.to_csv(join(conf('data_main_path'), 'sample_data.csv'))
    #except Exception as e:
    #    print(e)
    #    connection = False
    print(connection)
    return data, cols, connection


def get_model_arguments(job):
    return job[list(job.keys())[0]]['execute'][0]['params']


def get_reset_script():
    jobs = read_yaml(conf('docs_main_path'), 'ml_execute.yaml')
    configs = read_yaml(conf('docs_main_path'), 'configs.yaml')
    reset_script = ""
    if configs['db_connection']['data_source'] not in ['', None]:
        reset_script += "You have a data source connction form " + configs['db_connection']['data_source'] + "."
        active_jobs = []
        for j in jobs:
            if jobs[j]['active'] is True:
                active_jobs.append([j, jobs[j]['day']])
        if len(active_jobs) == 1:
                reset_script += " You also have active job which is " + j[0] +" running " + j[1] + "."
        if len(active_jobs) > 2:
            reset_script += " You also have active job which are " + ", ".join([j[0] for j in active_jobs]) + " running " + ", ".join([j[1] for j in active_jobs]) + ". "
    reset_script += " Would you like to reset them all?"
    return reset_script


reset_script = get_reset_script()


def check_available_data_for_dashboard(jobs):
    params = jobs[list(jobs.keys())[0]]['execute'][0]['params']
    connection = 'Done'
    for p in params:
        if p in ['data_source', 'feature', 'time_indicator']:
            if params[p] in [None, 'None']:
                connection = False
                break

    return connection

