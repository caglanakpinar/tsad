from os.path import join, abspath, exists
from os import popen
import socket
from shutil import copytree, rmtree, ignore_patterns
import threading
import requests
from numpy import random
from datetime import datetime
import time
import subprocess

from utils import is_port_in_use, read_yaml, write_yaml, show_chart, get_results
from configs import init_directory, folder_name, web_port_default


class CreateDirectory:
    """
    With Given directory it creates models, docs, data folders with required documents which are
    'ml_execute.yaml', 'configs.yaml', 'apis.yaml.

    1. First, it checks if there is existed folder.
    2. Then, imports files by deleting if it is already existed.
    3. You may reset folcder to default

    ** Notes **
        - Pls. make sure before stop and rerun the process copy the models, data in a safe folder.
        - If the platform is running on docker containers it copies also the .py file.
        - *directory* represents the path for copying folders
        - *remove_exits* if you want to clear the data on the path assign True. Default; False
        - *host* 'local' or 'docker'. Default; False
    """

    def __init__(self, directory=None, remove_existed=False):
        """
        :param directory: represents the path for copying folders
        :param remove_existed: if you want to clear the data on the path assign True. Default; False
        """
        self.directory = directory
        self.init_directory = init_directory
        self.remove_existed = remove_existed
        self.folder = join(self.directory, folder_name)

    def check_for_directory(self):
        """
        checks directory is existed. Otherwise files can not be copied.
        """
        if self.directory:
            return True
        else:
            print("please enter a directory")

    def import_files(self, host=None):
        """
        copies files to given path on directory.
        It copies files in given directory in to the default folder name which is 'anomaly_detection_framework'
        :param host: local or docker
        """
        if exists(self.folder):
            if self.remove_existed:
                rmtree(self.folder)
        if not exists(self.folder):
            if host == 'docker':
                copytree(join(abspath("")), self.folder,
                         ignore=ignore_patterns('*.ipynb', 'tmp*', "anomaly_detection", '__pycache__'))
            else:
                copytree(join(abspath("")), self.folder,
                         ignore=ignore_patterns('*.ipynb', 'tmp*', "anomaly_detection", "*.py", "__pycache__"))

    def reset_update_config_directory(self, env, reset=False):
        """
            - It allows us to run platform on given folder, otherwise directory will be the local path.
            - It updates 'directory' variable at "configs.yaml".
            - When it checks for the 'directory', it won`t be 'None' after all.

        ** Additional Information **
        When platform is closed or shutdown,
        directory variable at 'configs.yaml' is updated back to 'None' in order to get default 'configs.yaml' file.
        If directory variable == None directory will be 'init_directory'.

        :param env: docker, local. if docker == True, no need to to update directory be
                    cause whole .py file are also copied to new folder
        :param reset: reset updates back to normal 'None' value to derectory variable at configs.yaml
        """
        if env != 'docker':
            instances = read_yaml(init_directory, "instance.yaml")
            ins_updated = []
            for ins in instances['instances']:
                if ins['directory'] == self.folder:
                    if reset:
                        ins['active'] = False
                        print("instance is closed!!!")
                        print("is active ? ", False, "\n",
                              "directory :", ins['directory'], "\n",
                              "start date :", ins['start_date'], "\n")
                    else:
                        ins['active'] = True
                    ins_updated.append(ins)
            if len(ins_updated) == 0 and not reset:
                instances['instances'] += [{'directory': self.folder,
                                            'id': str(random.random()).replace(".", ""),
                                            'active': True,
                                            'start_date': datetime.now()}]
            write_yaml(init_directory, "instance.yaml", instances)

            #folder = self.folder
            #self.configs['directory'] = self.folder if not reset else None
            #for f in [self.folder, init_directory]:
            #    write_yaml(join(f, "docs"), "configs.yaml", self.configs)


class Configurations:
    """
    This handles the configurations of the Machine Learning Platform. Where to run?
    Which folder are related to?
    There are 3 model and 1 Machine Learning Execution services on the platform.
    You may also run each platform on nodes or workers.

    ****
    Unless you are runnning on all services on same machine,
    you have no specific parameters to assign servers to nodes or worker,
    model services; model_iso_f, model_lstm, model_prophet
    Machine Learning Service; ml_exeture
    ***

    Best Features
    1. Docker
    2. Local (Server):
        - runs on server with given path
        - Automatically finds unused ports between 6000 - 7000 and assigns api ports
        - You may also assign given ports
        - You can run apis in in different nodes which can communicates to master node.
        - If you are running you apis on nodes, you have to specify your 'master_node' by assigning it True.
        - For each nodes or workers you have to assign the configurations.
        - **Example How to apply Configuration for Workers And Nodes:**
            1. For Instance, you have 3 worker nodes and 1 master node
            2. you will assign model_iso_f, model_lstm, model_prophet to each node.
            3. ml_execute service an web application must be run at master node.
            4. use
    """
    def __init__(self, path, host='local', remove_existed=False, master_node=True):
        """

        :param path:
        :param host:
        :param remove_existed:
        :param master_node:
        """
        self.path = path
        self.host = host
        self.cd = CreateDirectory(directory=path, remove_existed=remove_existed)
        self.folder = self.cd.folder
        self.compose_file = None
        self.api_file =  None
        self.ports = []
        self.host = host
        self.master_node = master_node

    def create_directory(self):
        """
        It works with class Configuration
        Checks the directory is existed. Then, creates or updates the directories and 'configs.yaml'
        """
        if self.cd.check_for_directory():
            self.cd.import_files(host=self.host)
            self.cd.reset_update_config_directory(env=self.host)

    def check_for_ports(self, service_count):
        """
        checks for available ports. It picks prosts from the range between 6000 - 7000.
        :param service_count: number of service which cpecifically assigned for this configuration.
                              By defaults finds available port for each services.
        """
        from configs import conf
        if self.cd.check_for_directory():
            count = 0
            available_ports = conf('available_ports')
            while len(self.ports) != service_count:
                if not is_port_in_use(available_ports[count]):
                    self.ports.append(available_ports[count])
                count += 1

    def check_for_host(self, api_name, count):

        if self.cd.check_for_directory():
            if type(self.host) == 'list':
                return self.host[count]
            if type(self.host) == 'set':
                return self.host[api_name]
            elif type(self.host) != 'list' and type(self.host) != 'set':
                return self.host

    def create_docker_compose_file(self):
        if self.cd.check_for_directory():
            self.compose_file = read_yaml(self.folder, "docker-compose.yml")
            services = self.compose_file['services']
            self.check_for_ports(service_count=len(services))
            count = 0
            for s in services:
                services[s]['ports'] = [str(self.ports[count]) + ":" + str(self.ports[count])]
                print("available port for service :", s, " - ", str(self.ports[count]))
                ## TODO: volumes change to data, models, logs, docs
                services[s]['volumes'] = [join(self.folder, "") + "/:/app"]
                count += 1
            if not self.master_node:
                services = {s: services[s] for s in list(services.keys()) if s != 'ml_executor - services'}
            self.compose_file['services'] = services
            write_yaml(self.folder, "docker-compose.yml", self.compose_file)

    def check_for_api_and_host(self, apis=None):
        if apis is None and self.host not in ['docker', 'local']:
            if len(apis) == len(self.host):
                return True
            else:
                return False
        else:
            return  True

    def update_api_file(self, apis=None):
        from configs import conf
        self.api_file = read_yaml(conf('docs_main_path'), "apis.yaml")
        if apis is not None:
            if not self.master_node:
                self.api_file = {a: self.api_file[a] for a in apis}
            if type(apis) == 'dict':
                for a in apis:
                    for p in apis[a]:
                        self.api_file[a][p] = apis[a][p]
            else:
                self.api_file = self.api_file[apis]

    def update_api_yaml_file(self):
        if not self.master_node:
            self.api_file = {a: self.api_file[a] for a in list(self.api_file.keys()) if a != 'ml_execute'}
        write_yaml(join(self.folder, "docs"), "apis.yaml", self.api_file)

    def create_api_file(self, environment):
        print(self.host)
        if self.cd.check_for_directory():
            self.check_for_ports(service_count=len(self.api_file))
            count = 0
            for s in self.api_file:
                print("available port for service :", s, " - ", str(self.ports[count]))
                self.api_file[s]['port'] = self.ports[count]
                self.api_file[s]['host_create'] = socket.gethostname() if self.host != 'docker' else '0.0.0.0'
                self.api_file[s]['host'] = socket.gethostname() if self.host != 'docker' else '0.0.0.0'
                if self.host != 'local' and environment != 'docker':
                    self.api_file[s]['host'] = self.check_for_host(s, count)
                count += 1
            self.update_api_yaml_file()


class BuildPlatform:
    def __init__(self, conf, environment=None, master_node=True):
        self.conf = conf
        self.env = environment
        self.api_file = self.conf.api_file
        self.master_node = master_node
        self.jobs = None

    def create_web(self):
        from web.main import web_service_run

        if self.master_node:
            thr = threading.Thread(target=web_service_run)
            thr.start()

    def docker_env(self):
        print(join(self.conf.folder, "docker-compose.yml") + " up")
        # stream = popen((join(self.conf.folder, "docker-compose.yml") + " up"))
        result = subprocess.run(["docker-compose", "-f", join(self.conf.folder, "docker-compose.yml"), 'up'], stdout=subprocess.PIPE)
        print(result.stdout)
        # create web
        self.create_web()

    def local_env(self):
        #  create services on local
        from create_api import api_executor
        for api in self.api_file:
            thr = threading.Thread(target=api_executor, args=(self.api_file[api], ))
            thr.daemon = True
            thr.start()  # Will run "api_executor"
        # create web
        self.create_web()

    def initialize(self):
        if self.env == 'local':
            self.local_env()
        if self.env == 'docker':
            self.docker_env()

    def down_web(self):
        from configs import conf
        requests.post(url='http://0.0.0.0:' + str(web_port) + '/shutdown')

    def down(self):
        self.down_web()
        if self.env == 'docker':
            popen(join(self.conf.folder, "docker-compose.yml") + " down")
        else:
            for s in self.conf.api_file:
                api = self.conf.api_file[s]
                requests.post(url='http://' + api['host'] + ':' + str(api['port']) + '/shutdown')
                time.sleep(2)

    def create_data_source(self, **args):
        from web.updates import db_connection_update, ml_execute_update, get_sample_data
        db_connection_update(**args)
        if get_sample_data(args, connection=True)[-1]:
            self.jobs = read_yaml(join(self.conf.folder, "docs"), 'ml_execute.yaml')
            args['jobs'] = self.jobs
            ml_execute_update(**args)
            return True
        else:
            return False

    def create_job(self, job, **args):
        self.jobs = read_yaml(join(self.conf.folder, "docs"), 'ml_execute.yaml')
        for j in self.jobs:
            if j == job:
                self.jobs[j]['browser_time'] = str(datetime.now())[:13]
                self.jobs[j]['description'] = args['description']
                self.jobs[j]['day'] = args['days']
                self.jobs[j]['job_start_date'] = str(args['dates'])[0:16]
                e2 = []
                for e in self.jobs[j]['execute']:
                    for p in e['params']:
                        if p in args.keys():
                            e['params'][p] = str(args[p])
                    e2.append(e)
                self.jobs[j]['execute'] = e2
        write_yaml(join(self.conf.folder, "docs"), "ml_execute.yaml", self.jobs)

    def run_stop_jobs(self, job, stop=False):
        from web.updates import start_job_and_update_job_active
        from ml_executor import CreateJobs
        self.jobs = self.jobs if self.jobs is not None else read_yaml(join(self.conf.folder, "docs"), 'ml_execute.yaml')
        if stop:
            j = CreateJobs(self.jobs, job)
            j.stop_job()
        else:
            start_job_and_update_job_active(jobs=self.jobs, job=job)


class AnomalyDetection:
    def __init__(self, path=None, environment=None, host='local', remove_existed=False, master_node=True):
        self.path = path
        self.conf = Configurations(path=path, host=host.lower(), remove_existed=remove_existed, master_node=master_node)
        self.env = environment.lower() if environment is not None else None
        self.platform = None
        self.web_port = None
        self.info_dict = {}
        self.jobs = None

        global recent_directory, web_port
        recent_directory = path
        web_port = web_port_default
        while is_port_in_use(web_port):
            web_port += 1

    def init(self, apis=None):
        if self.conf.check_for_api_and_host(apis=apis):
            self.conf.create_directory()
            if self.env == 'docker':
                self.conf.create_docker_compose_file()
            self.conf.update_api_file(apis=apis)
            self.conf.create_api_file(environment=self.env)
            print("Configuration process is completed!!!")
            print("platform is initialized!!!")
        else:
            print("given host and services are not the same number")

    def run_platform(self):
        self.platform = BuildPlatform(conf=self.conf, environment=self.env, master_node=self.conf.master_node)
        self.platform.initialize()
        if self.conf.master_node:
            from configs import conf
            self.web_port = conf('web_port')
            print("platform is up!!!")
            print("*"*5, " WEB APPLICATION ", "*"*5)
            print('Running on ', 'http://0.0.0.0:' + str(self.web_port) + '/')
        else:
            print("platform is up!!!")
            print("Running Services:")
            for api in self.platform.api_file:
                print(api, " :", 'http://'+api['host']+':' +api['port']+'/'+api['api_name']+'/')

    def stop_platform(self, reset_docs=False):
        if reset_docs:
            self.conf.cd.reset_update_config_directory(env=self.env, reset=True)
        self.platform.down()

    def reset_web_app(self):
        self.platform.down_web()
        self.platform.create_web()

    def api_configuration(self, api_conf):
        if self.conf.master_node:
            if type(api_conf) == 'dict':
                self.conf.update_api_file(apis=api_conf)
                self.conf.update_api_yaml_file()
            else:
                example_apis = {}
                print("Api Configurations must be dictionary.")
                print("Here example 'api_conf' dictionary :")
                for api in self.conf.api_file:
                    if api != 'ml_execute':
                        example_apis[api] = {'host': self.conf.api_file[api]['host'],
                                             'port': self.conf.api_file[api]['port']}
                print(example_apis)
                print("*** "*5)
                print("If you don`t know the port you have assigned to apis,")
                print("You can check the 'check_available_apis' method to gather running apis on the server")

            print("services are located at:")
        else:
            print("Api configuration must be implemented when the master node is applied on a server.")
            print("This is not the master node which ml_execute service and web application are running.")

    def check_available_apis(self):
        print(self.conf.api_file)
        return self.conf.api_file

    def create_data_source(self,
                           data_source_type,
                           data_query_path,
                           db=None,
                           host=None,
                           port=None,
                           user=None,
                           pw=None):

        self.info_dict = {'data_source': data_source_type.lower(),
                          'data_query_path': data_query_path,
                          'db_name': db, 'host': host, 'port': port, 'user': user, 'pw': pw}

        sources = ['csv', 'xslx', 'json', 'yaml', 'mysql', 'googlebigquery', 'awsredshift', 'postgresql']
        if data_source_type in sources:
            if data_source_type in ['awsredshift', 'postgresql', 'mysql']:
                if None not in [db, host, port, user, pw]:
                    if len(set(data_query_path.split()) & set('SELECT', 'FROM', 'WHERE', 'WITH')) != 0:
                        connection = self.platform.create_data_source(**self.info_dict)
                        print("Data source is created!!!") if connection else print("Connection is refused!!!")
                    else:
                        print(data_source_type, "data source type needs right SQL statement.")
                        print("Please assign write SQL statement to 'data_query_path'.")
                else:
                    print(data_source_type, "connection needs db name, host, port, user and password information")
            elif data_source_type in ['json', 'yaml', 'xslx', 'csv']:
                if data_query_path.split(".")[-1] == data_source_type:
                    connection = self.platform.create_data_source(**self.info_dict)
                    print("Data source is created!!!") if connection else print("Connection is refused!!!")
                    if not connection:
                        print("Please make sure  '", data_query_path, "' at ", self.conf.folder)
                else:
                    print("please makse sure file format is .", data_source_type, " but ", data_query_path, " is not.")
            elif data_source_type == 'googlebigquery':
                if db.split(".")[-1] == 'json':
                    print("yess")
                    connection = self.platform.create_data_source(**self.info_dict)
                    print("Data source is created!!!") if connection else print("Connection is refused!!!")
                else:
                    print("pls create from google cloud console a .json format conection file, api.")
        else:
            print("Here are the available data connections: ", ", ".join(sources))

    def get_ml_dictionary(self, job={}):
        print(job)
        asd =  {'description': job.get('description', None),
                "dates": job.get('dates', None),
                'groups': job.get('groups', None),
                'time_indicator': job.get('time_indicator', None),
                'feature': job.get('feature', None),
                'days': job.get('days', None)}
        print(asd)
        return asd

    def create_jobs(self, jobs):
        if len(jobs.keys()) > 1:
            for job in jobs:
                print(jobs[job])
                if None not in [self.get_ml_dictionary(job=jobs[job])[p] for p in ['dates', 'time_indicator', 'feature', 'days']]:
                    self.platform.create_job(job=job, **self.get_ml_dictionary(job=jobs[job]))
                else:
                    print("Here is the example dictionary :")
                    print({'train': self.get_ml_dictionary({}), 'prediction': self.get_ml_dictionary({})})
                    print("*** "*5)
                    print("Rules of Creating Job / Jobs :")
                    print("**1. 'time_indicator', 'feature', 'days' are monditory fields.")
                    print("**2. Jobs are 'train', 'prediction', 'paramter_tuning'.")
                    print("**3. If you don`t assign any date to train it will directly start assignin as 2 min after the current time.")
                    print("**4. Prediction 'date' is monditory.")
                    print("**5. If you don`t assign any date to paramter_tuning it will directly start assignin as 1 day after the  train job date.")
        else:
            self.platform.create_job(job=list(jobs.keys())[0], **self.get_ml_dictionary(job=jobs[list(jobs.keys())[0]]))

        print("job is created")

    def manage_train(self, stop=False):
        self.platform.run_stop_jobs(job='train', stop=stop)

        print("job is created")

    def manage_prediction(self, run=True):
        self.platform.run_stop_jobs(job='prediction', stop=run)
        print("job is created")

    def manage_parameter_tuning(self, run=True):
        self.platform.run_stop_jobs(job='prediction', stop=run)
        print("job is created")

    def create_dashboard(self):
        print("Dashboard is created!!")
        print("You can run on ...")

    def show_lstm_prediction(self, filter=None):
        """
        enable to filter data then shows the prediction and actula values
        :param filter: "column_name == value" or "column_name in (value_1, value_2)"
        :return:
        """
        if self.jobs is None:
            from configs import conf
            self.jobs = read_yaml(conf('docs_main_path'), "ml_execute.yaml")
        time_indicator = self.jobs['prediction']['execute'][0]['params']['time_indicator']
        feature = self.jobs['prediction']['execute'][0]['params']['feature']
        results = get_results(time_indicator)
        results = results if filter is None else results.query(filter)
        show_chart(results, time_indicator, 'predict', feature, is_bar_chart=False)

    def show_anomaly_detection(self):
        print("")











