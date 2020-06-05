from flask import Flask, request, Response
from multiprocessing import Process
from os.path import join, dirname
from inspect import getmembers, getargspec
import socket

from utils import callfunc


class CreateApi:
    def __init__(self, host=None, port=None, function=None, api_name=None, parameters=None):
        self.function = function
        self.parameters = parameters
        self.api_name = api_name
        self.host = socket.gethostname() if host is None else host
        self.port = port

    def init_api(self):
        app = Flask(__name__)
        api = self.api_name
        function = self.function
        params = {p: None for p in self.parameters}

        @app.route('/' + api)
        def render_script():
            # Create a daemonic process with heavy "my_func"
            heavy_process = Process(
                                    target=run_ml_executor,
                                    daemon=True
                                    )
            heavy_process.start()
            return Response(mimetype='application/json', status=200)

        def run_ml_executor():
            for p in params:
                if p in request.args.keys():
                    params[p] = request.args[p]

            print("sent :", params)
            return {api: function(**params)}

        @app.route('/shutdown', methods=['POST'])
        def shutdown():
            shutdown_server()
            return 'Server shutting down...'

        def shutdown_server():
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                raise RuntimeError('Not running with the Werkzeug Server')
            func()
        print("port :", self.port)
        print("host :", self.host)
        return app.run(threaded=False, debug=False, port=self.port, host=self.host)


#if __name__ == '__main__':
#    api_configs = read_yaml('apis.yaml')
#
#    def api_executor(api_info):
#        _file_path = join(dirname(__file__), api_info['py_file'])
#        _py = callfunc(_file_path)
#        _func = [o[1] for o in getmembers(_py) if o[0] == api_info['function']][0]
#        api = CreateApi(host=api_info['host'],
#                        port=api_info['port'],
#                        function=_func,
#                        api_name=api_info['api_name'],
#                        parameters=list(getargspec(_func)[0]))
#        api.init_api()
#
#    pool = Pool(4)
#    pool.map(api_executor, api_configs)
#    pool.close()
#

# api_info = read_yaml(directory + 'apis.yaml')[0]


def api_executor(api_info):
    _file_path = join(dirname(__file__), api_info['py_file'])
    _py = callfunc(_file_path)
    _func = [o[1] for o in getmembers(_py) if o[0] == api_info['function']][0]
    api = CreateApi(host=api_info['host'],
                    port=api_info['port'],
                    function=_func,
                    api_name=api_info['api_name'],
                    parameters=list(getargspec(_func)[0]))
    api.init_api()



