import sys
import datetime
import yaml
from numpy import arange
from os.path import join

from configs import conf
from utils import read_yaml, write_yaml


class Logger(object):
    def __init__(self, job):
        self.job = job
        self.terminal = sys.stdout
        self.log = open(join(conf('log_main_path'), self.job + "_logfile_" + "".join(filter(lambda x: x not in ['-', ' ', ':', '.'], str(datetime.datetime.now())[0:16])) + ".log"), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class LoggerProcess:
    def __init__(self, job=None, model=None, total_process=None):
        self.log = "process.yaml"
        self.job = job
        self.model = model
        self.total_process = total_process
        print(total_process)
        self.ratios = [int(i * total_process) for i in arange(0.1, 1.1, 0.1)] if total_process else None
        self.count = 0

    def write(self):
        file = read_yaml(conf('log_main_path'), self.log)
        file[self.job][self.model] = self.count / self.total_process
        write_yaml(conf('log_main_path'), self.log, file)

    def counter(self):
        self.count += 1
        if self.count in self.ratios:
            self.write()
        print("number of process done :", self.count)

    def regenerate_file(self):
        file = read_yaml(conf('log_main_path'), self.log)
        for j in file:
            if j == self.job:
                for m in file[j]:
                    file[j][m] = 0
        with open(join(conf('log_main_path'), "", self.log), 'w') as f:
            yaml.dump(file, f)

    def check_total_process_and_regenerate(self):
        file = read_yaml(conf('log_main_path'), self.log)
        for j in file:
            if j == self.job:
                if sum([0 for m in file[j] if file[j][m] >= 1]) == 0:
                    self.regenerate_file()
                    print("all processes are done. Process.yaml file is regenerated!!!")


def get_time():
    return str(datetime.datetime.now())
