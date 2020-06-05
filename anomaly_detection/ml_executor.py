import schedule
import time

from configs import conf
from logger import *
from create_api import api_executor
from utils import write_yaml, request_url


def read_yaml(directory, filename):
    with open(join(directory, "", filename)) as file:
        docs = yaml.full_load(file)
    return docs


def get_api_url(host, port, api_name):
    host = '0.0.0.0' if host is None else host
    return "http://" + host + ":" + str(port) + "/" + api_name


def ml_execute_times(j):
    browser, start = j['browser_time'], j['job_start_date']
    browser = datetime.datetime.strptime(browser, '%Y-%m-%d %H')
    diff = datetime.datetime.now().hour - browser.hour
    start_time = datetime.datetime.strptime(str(start), '%Y-%m-%d %H:%M')
    if diff != 0:
        start_time = start_time + datetime.timedelta(hours=diff)
    time = str(start_time)[11:16]
    print("browser_time :", browser, " || "
          "start_time :", start_time, " || "
          "diff :", diff, " || ",
          "time :", time,
          "current time :", datetime.datetime.now()
          )
    return browser, diff, start_time, time


class CreateJobs:
    def __init__(self, jobs, job_name):
        self.job_name = job_name
        self.jobs_yaml = jobs
        self.job = self.jobs_yaml[job_name]
        self.api_infos = read_yaml(conf('docs_main_path'), 'apis.yaml')
        self.api_info = None
        self.url = None
        self.logger = LoggerProcess(job=self.job_name)
        self.browser_time, self.diff, self.start_time, self.time = ml_execute_times(self.job)
        self.schedule = None
        self.total_minutes_in_month = 30 * 24 * 60
        self.total_minutes_in_2_weeks = 15 * 24 * 60

    def jobs(self):
        for j in self.job['execute']:
            self.api_info = self.api_infos['model_' + j['params']['model']]
            self.url = get_api_url(host=self.api_info['host'],
                                   port=self.api_info['port'],
                                   api_name=self.api_info['api_name'])
            request_url(self.url, j['params'])
        print("requests are sent!!!")

    def stop_job(self, request=True):
        if self.job['active'] is True:  # if there is active job update ml_execute.yaml
            self.logger.regenerate_file()
            self.jobs_yaml[self.job_name]['active'] = False
            write_yaml(conf('docs_main_path'), "ml_execute.yaml", self.jobs_yaml)
            for j in self.job['execute']:
                self.api_info = self.api_infos['model_' + j['params']['model']]
                self.url = get_api_url(host=self.api_info['host'],
                                       port=self.api_info['port'],
                                       api_name=self.api_info['api_name'])
                if request:
                    request_url(self.url, self.job['stop_job'])

    def job_that_executes_once(self):
        while True:
            if self.start_time < datetime.datetime.now():
                self.jobs()
                time.sleep(300)
                break

    def job_that_executes_monthly_weekly(self):
        mode_value = self.total_minutes_in_month if self.job['day'] == 'Monthly' else self.total_minutes_in_2_weeks
        if self.start_time < datetime.datetime.now():
            now = datetime.datetime.now()
            print(mode_value)
            print(int((self.start_time - now).total_seconds() / 60) % mode_value)
            if int((self.start_time - now).total_seconds() / 60) % mode_value == 0:
                self.jobs()

    def timer(self):
        print("job will run first at  2020-05-25 20:42:00")
        time.sleep(abs((datetime.datetime.now() - self.start_time).total_seconds()))

    def job_schedule(self):
        print("job is initialized!!!")
        print(self.time)
        if self.job['day'] not in ['Mondays', 'Every Min', 'Every Second', 'Hourly',
                                   'only once', 'Weekly']:
            {'Mondays': schedule.every().monday,
             'Tuesdays': schedule.every().tuesday,
             'Wednesdays': schedule.every().wednesday,
             'Thursdays': schedule.every().thursday,
             'Fridays': schedule.every().friday,
             'Saturdays': schedule.every().saturday,
             'Sundays': schedule.every().sunday,
             'Daily': schedule.every().day
             }[self.job['day']].at(self.time).do(self.jobs)
        if self.job['day'] == 'Weekly':
            self.timer()
            schedule.every().week.do(self.jobs)
        if self.job['day'] == 'Every Min':
            schedule.every(1).minutes.at().do(self.jobs)
        if self.job['day'] == 'Every Second':
            schedule.every(1).seconds.at().do(self.jobs)
        if self.job['day'] == 'Hourly':
            schedule.every(1).hours.at().do(self.jobs)


def start_job(job):
    Logger('ml_execute_' + job)
    print("received :", {'job': job, 'process': 'start'}, " time :", get_time())
    j = CreateJobs(read_yaml(conf('docs_main_path'), 'ml_execute.yaml'), job)
    if j.job['day'] == 'only once':
        j.job_that_executes_once()
        return 'done!!!!'
    if j.job['day'] in ['Monthly', 'Every 2 Weeks']:
        while True:
            jobs = read_yaml(conf('docs_main_path'), 'ml_execute.yaml')
            j.job_that_executes_monthly_weekly()
            print("job is working - ", job)
            if jobs[job]['active'] is False:
                print("job is stopped !!")
                break
            time.sleep(60)
    elif j.job['day'] in ['only once', 'Monthly', 'Every 2 Weeks']:
        j.job_schedule()
        while True:
            schedule.run_pending()
            jobs = read_yaml(conf('docs_main_path'), 'ml_execute.yaml')
            print("job is working - ", job)
            if jobs[job]['active'] is False:
                print("job is stopped !!")
                break
            time.sleep(10)


def start_job_and_update_job_active(jobs, job):
    jobs[job]['active'] = True
    write_yaml(conf('docs_main_path'), "ml_execute.yaml", jobs)
    ml_execute_api = read_yaml(conf('docs_main_path'), 'apis.yaml')['ml_execute']
    url = get_api_url(ml_execute_api['host'], ml_execute_api['port'], ml_execute_api['api_name'])
    request_url(url, {'job': job})


if __name__ == '__main__':
    api = read_yaml(conf('docs_main_path'), 'apis.yaml')['ml_execute']
    api_executor(api)


# if __name__ == '__main__':
#    print("Creating Jobs")
#    jobs = read_yaml(directory + 'ml_execute.yaml')
#    jobs = [jobs[_j] for _j in jobs if jobs[_j]['active']]
#    for _j in jobs:
#        j = CreateJobs(_j)
#        j.job_schedule()
#    while True:
#        schedule.run_pending()
#        time.sleep(1)
#
