import datetime
import sys
import importlib
import os
import yaml
import json
import pickle
import datetime
import subprocess
import os
from os.path import join
import signal
import math
import random

from configs import weekdays, conf, boostrap_ratio
import socket
import urllib
import time


def callfunc(my_file):
    pathname, filename = os.path.split(my_file)
    sys.path.append(os.path.abspath(pathname))
    modname = os.path.splitext(filename)[0]
    my_mod = importlib.import_module(modname)
    return my_mod


def get_running_pids(process_name, argument=None):
    pids = []
    p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
    out, err = p.communicate()
    for line in out.splitlines():
        if process_name in line.decode('utf-8'):
            pid = int(line.decode('utf-8').split(None, 1)[0])
            if argument:
                if len(line.decode('utf-8').split(process_name)) != 0:
                    args = line.decode('utf-8').split(process_name)[1].split(None, 1)
                    if str(argument) in args:
                        print("running job :", line.decode('utf-8'))
                        pids.append(pid)
            else:
                print("running job :", line.decode('utf-8'))
                pids.append(pid)
        else:
            print("no initialized ", argument, " is detected")
    return pids


def kill_process_with_name(process_name, argument=None):
    pids = get_running_pids(process_name, argument=argument)
    if len(pids) != 0:
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
    else:
        print("no running jobs")


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def read_yaml(directory, filename):
    with open(join(directory, "", filename)) as file:
        docs = yaml.full_load(file)
    return docs


def write_yaml(directory, filename, data):
    with open(join(directory, "", filename), 'w') as file:
        yaml.dump(data, file)


def read_write_to_json(directory, filename, data, is_writing):
    if is_writing:
        with open(join(directory, "", filename), 'w') as file:
            json.dump(data, file)
    else:
        with open(join(directory, "", filename), "r") as file:
            data = json.loads(file.read())
        return data


def model_from_to_pkl(directory=None, path=None, model=None, is_writing=False):
    if is_writing:
        with open(join(directory, "", path), "wb") as f:
            pickle.dump(model, f)
    else:
        with open(join(directory, "", path), 'rb') as f:
            model = pickle.load(f)
        return model


def get_col(c1, c2):
    if c2 == 1:
        return c1
    else:
        return c1 + '_' + str(c2)


def split_groups(groups):
    s_groups = []
    if groups not in [None, 'None']:
        if "+" not in groups:
            s_groups = [groups]
        else:
            s_groups = groups.split("+")
    return s_groups


def date_part(date, part):
    if part == 'year':
        return date.year
    if part == 'quarter':
        return get_quarter(date)
    if part == 'month':
        return date.month
    if part == 'week':
        return date.isocalendar()[1]
    if part == 'week_part':
        return 1 if date.isoweekday() in [6, 7] else 0
    if part == 'week_day':
        return date.isoweekday()
    if part == 'day':
        return datetime.datetime.strftime(date, "%Y-%m-%d")
    if part == 'hour':
        return date.hour
    if part == 'min':
        return date.min
    if part == 'second':
        return date.second


def get_quarter(d):
    return "Q%d_%d" % (math.ceil(d.month/3), d.year)


def model_path(comb, group, model):
    return "_".join(["_".join([str(i[0]) + "*" + str(i[1]) for i in zip(group, comb)]), model]) + ".json"


def convert_date(date=None, time=None, str=False):
    if date not in ['', None]:
        date_merged = date + ' ' + time if time != '' else date
        if len(date_merged) == 0:
            format_str = '%Y-%m-%d'
        if len(date_merged) == 16:
            format_str = '%Y-%m-%d %H:%M'
        if len(date_merged) > 16:
            format_str = '%Y-%m-%d %H:%M:%S.%f'
        if not str:
            date_merged = datetime.datetime.strptime(date_merged, format_str)
    else:
        date_merged = datetime.datetime.now() + datetime.timedelta(minutes=2)
    return date_merged


def get_split_date(period=None, dates=None, params=None):
    convert_date = lambda x:  datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S")
    model_infos = read_yaml(conf('model_main_path'), 'model_configuration.yaml')['infos']
    if period is not None:
        today = max(dates) if dates is not None else convert_date(model_infos['max_date'])
        isoweekday = {weekdays[i]: i for i in range(7)}
        day_diffs = {"Daily": 1, 'Weekly': 7, 'Every 2 Weeks': 14, 'Monthly': 30}
        hour_diffs = {"hourly": datetime.timedelta(hours=1), 'Every Min': datetime.timedelta(minutes=1), 'Every Second': datetime.timedelta(seconds=1)}
        if period in weekdays:
            last_isoweekday = lambda x: today + datetime.timedelta(days=-today.weekday()) + datetime.timedelta(days=x)
            split_date = last_isoweekday(isoweekday[period])
        if period in list(day_diffs.keys()):
            print()
            split_date = today - datetime.timedelta(days=day_diffs[period])
        if period in list(hour_diffs.keys()):
            split_date = today - hour_diffs[period]
    else:
        if dates is not None:
            split_date = get_ratio_of_date(max(dates), min(dates), params['split_ratio'])
        else:
            max_date, min_date = convert_date(model_infos['max_date']), convert_date(model_infos['min_date'])
            split_date = get_ratio_of_date(max_date, min_date, params['split_ratio'])
    return split_date


def get_ratio_of_date(max_date, min_date, ratio):
    range_dates = (max_date - min_date).total_seconds()
    return min_date + datetime.timedelta(seconds=int(range_dates * ratio))


def get_query_date(job, period=None, dates=None, params=None):
    date = None
    if job == 'prediction':
        date = get_split_date(period=period, dates=dates, params=params)
    return date


def url_string(value, res=False):
    if value is not None:
        if res:
            return value.replace("\r", " ").replace("\n", " ").replace(" ", "+")
        else:
            return value.replace("+", " ")
    else:
        return None


def request_url(url, params):
    url += '?'
    for p in params:

        url += p + '=' + url_string(str(params[p]), res=True) + '&'
    response = 404
    while response != 200:
        try:
            res = urllib.request.urlopen(url)
            response = res.code
        except Exception as e:
            print(e)
        time.sleep(2)


def convert_feature(value):
    try:
        if value == value:
            return float(value)
        else:
            return None
    except Exception as e:
        print(e)
        return None


def get_sample_size(size):
    if size > 30:
        if size >= 2000:
            return 1000
        else:
            return int(size * boostrap_ratio)
    else:
        return 5


def get_residuals(residuals, ratio):
    if len(residuals) > 10000:
        return random.sample(residuals, 10000)
    else:
        return random.sample(residuals, len(residuals) * ratio)







