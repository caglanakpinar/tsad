import sys
from create_api import *
from utils import read_yaml
from configs import conf


if __name__ == '__main__':
    print("arguments :", sys.argv)
    api = read_yaml(conf('docs_main_path'), 'apis.yaml')[sys.argv[1]]
    api_executor(api)