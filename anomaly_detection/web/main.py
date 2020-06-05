from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import time
import datetime
from os.path import join

from dashboard import create_dashboard
from web.updates import *
from configs import conf
from utils import read_yaml, write_yaml


def web_service_run():
    app = Flask(__name__, instance_relative_config=True)
    dashboard = create_dashboard(app)

    @app.route('/')
    @app.route('/home', methods=['POST', 'GET'])
    def home():
        jobs = read_yaml(conf('docs_main_path'), 'ml_execute.yaml')
        configs = read_yaml(conf('docs_main_path'), 'configs.yaml')
        model_configuration = read_yaml(conf('model_main_path'), 'model_configuration.yaml')
        process = read_yaml(conf('log_main_path'), 'process.yaml')
        req = dict(request.form)
        if 'save' in req.keys():
            if bool(req['save']):
                ml_execute_reset(jobs)
                db_connection_reset(configs)
                models_reset(model_configuration)
                logs_reset(process)
        return render_template("home.html", reset_script=reset_script)

    @app.route('/configs', methods=['GET'])
    def configs():
        return render_template("configs2.html", exception="")

    @app.route('/configs', methods=['POST'])
    def get_data():
        info_dict = dict(request.form)
        if info_dict != {}:
            db_connection_update(**info_dict)
            info_dict['jobs'] = read_yaml(conf('docs_main_path'), 'ml_execute.yaml')
            ml_execute_update(**info_dict)
            data, cols, connection = get_sample_data(info_dict, connection=True)
            if connection:
                return redirect(url_for('create_task', messages=True))
            else:
                return render_template("configs2.html", connect="Connection is failed!!",
                                       exception="failed to connect !!")
        else:
            render_template("configs2.html", exception="")

    @app.route('/query', methods=['POST', 'GET'])
    def create_task():
        exception = ""
        job = read_yaml(conf('docs_main_path'), 'ml_execute.yaml')
        params = get_model_arguments(job)
        data, cols, connection = get_sample_data(params, connection=True, create_sample_data=False)
        if bool(request.args['messages']) and params['data_source'] and connection:
            if dict(request.form) != {}:
                update_dict = get_request_values(job, params, request)
                if update_dict['feature'] and update_dict['time_indicator'] and dict(request.form)[
                    'date1_prediction'] != '':
                    ml_execute_update(**update_dict)
                else:
                    exception = "Pls make sure you have entered Anomaly Feature and Date Indicator!!!!"
                if get_dash(request):
                    return redirect(url_for('show_dash'))
                else:
                    return render_template("configs_data2.html", cols=cols, exception=exception)
            else:
                return render_template("configs_data2.html", cols=cols, exception=exception)
        else:
            return redirect(url_for('get_data'))

    def get_dash(request):
        req = dict(request.form)
        return True if 'dash' in list(req.keys()) else False

    @app.route('/ml_execute', methods=['GET', 'POST'])
    def job_init():
        exception = ''
        req = dict(request.form)
        jobs = read_yaml(conf('docs_main_path'), 'ml_execute.yaml')
        process = read_yaml(conf('log_main_path'), 'process.yaml')
        print(process)
        job_names = list(jobs.keys())
        dates = {j: jobs[j]['job_start_date'] for j in job_names}
        current_active_status = {j: jobs[j]['active'] for j in job_names}
        log_infos = get_logs(job_names, dates, current_active_status, process)
        log_infos = update_logs(jobs, log_infos, req)
        return render_template("ml_execute.html",
                               train_process=log_infos['train']['process'],
                               prediction_process=log_infos['prediction']['process'],
                               tuning_process=log_infos['parameter_tuning']['process'],
                               train_status=log_infos['train']['status'],
                               prediction_status=log_infos['prediction']['status'],
                               tuning_status=log_infos['parameter_tuning']['status'],
                               train_precent=str(log_infos['train']['percent']),
                               prediction_precent=str(log_infos['prediction']['percent']),
                               tuning_percent=str(log_infos['parameter_tuning']['percent']),
                               train_date=str(log_infos['train']['start_time'])[0:16],
                               prediction_date=str(log_infos['prediction']['start_time'])[0:16],
                               tuning_date=str(log_infos['parameter_tuning']['start_time'])[0:16],
                               exception=exception)

    @app.route('/timer', methods=['GET'])
    def get_time():
        try:
            print("browser time: ", request.args['time'])
            print("server time : ", time.strftime('%A %B, %d %Y %H:%M:%S'))
            jobs = read_yaml(conf('docs_main_path'), 'ml_execute.yaml')
            print(" ".join(request.args['time'].split()[0:5]))
            print(datetime.datetime.strptime(" ".join(request.args['time'].split()[0:5]), "%a %b %d %Y %H:%M:%S"))
            for j in jobs:
                jobs[j]['browser_time'] = str(
                    datetime.datetime.strptime(" ".join(request.args['time'].split()[0:5]), "%a %b %d %Y %H:%M:%S"))[
                                          0:13]
            write_yaml(conf('docs_main_path'), "ml_execute.yaml", jobs)
        except Exception as e:
            print(e)
        return "Done"

    @app.route("/dash")
    def my_dash_app():
        return dashboard.index()

    @app.route("/dashboard")
    def show_dash():
        jobs = read_yaml(conf('docs_main_path'), 'ml_execute.yaml')
        connection = check_available_data_for_dashboard(jobs)
        return render_template('dashboard.html',
                               connection=connection,
                               dash_url="http://127.0.0.1:" + str(conf('config')['web_port']) + "/dash")

    @app.route('/shutdown', methods=['POST'])
    def shutdown():
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()
        return 'Server shutting down...'
    app.run(threaded=False, port=int(conf('web_port')), debug=False, host='127.0.0.1')


if __name__ == '__main__':
    web_service_run()
