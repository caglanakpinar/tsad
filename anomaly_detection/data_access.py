import pandas as pd
from dateutil.parser import parse
from os.path import join
import os


from configs import conf
from utils import read_yaml, read_write_to_json, convert_feature


class GetData:
    def __init__(self, data_source=None, date=None, data_query_path=None, time_indicator=None, feature=None, test=None):
        self.data_source = data_source if data_source is not None else 'csv'
        self.data_query_path = data_query_path
        self.time_indicator = time_indicator
        self.feature = feature
        self.conn = None
        self.data = pd.DataFrame()
        self.nrows = test
        self.date = date
        self.query = data_query_path

    def get_connection(self):
        config = conf('config')
        if self.data_source in ['postgresql', 'awsredshift', 'mysql']:
            server, db, user, pw, port = str(config['db_connection']['server']), str(config['db_connection']['db']), \
                                         str(config['db_connection']['user']), str(config['db_connection']['password']),\
                                         int(config['db_connection']['port'])
        if self.data_source == 'mysql':
            from mysql import connector
            self.conn = connector.connect(host=server, database=db, user=user, password=pw)
        if self.data_source in ['postgresql', 'awsredshift']:
            import psycopg2
            self.conn = psycopg2.connect(user=user, password=pw, host=server, port=port, database=db)
        if self.data_source == 'googlebigquery':
            from google.cloud.bigquery.client import Client
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = join(conf('data_main_path'), "", config['db_connection']['db'])
            self.conn = Client()
        print("db connection is done!")

    def convert_date(self):
        try:
            self.data[self.time_indicator] = self.data[self.time_indicator].apply(lambda x: parse(str(x)))
        except Exception as e:
            print(e)

    def convert_feature(self):
        try:
            self.data[self.feature] = self.data[self.feature].apply(lambda x: convert_feature(x))
        except Exception as e:
            print(e)

    def check_data_with_filtering(self):
        if self.data_source in ('json', 'yaml', 'csv'):
            self.query = lambda x: x.query(self.time_indicator + " > '" + str(self.date) + "'") if self.date is not None else x
        if self.data_source in ['mysql', 'postgresql', 'awsredshift', 'googlebigquery']:
            if self.date:
                self.query = "SELECT * FROM (" + self.data_query_path + ") AS T WHERE T." + self.time_indicator + " >= '" + str(self.date) + "'   "

    def query_data_source(self):
        self.check_data_with_filtering()

        # import data via pandas
        if self.data_source in ['mysql', 'postgresql', 'awsredshift']:
            self.get_connection()

            self.data = pd.read_sql(self.query + " LIMIT " + str(self.nrows) if self.nrows else self.query, self.conn)

        # import data via google
        if self.data_source == 'googlebigquery':
            self.get_connection()
            self.data = self.conn.query(self.query + " LIMIT " + str(self.nrows) if self.nrows else self.query).to_dataframe()

        # import via pandas
        if self.data_source == 'csv':
            try:
                for sep in [',', ';', ':']:

                    self.data = pd.read_csv(filepath_or_buffer=join(conf('data_main_path'), self.data_query_path),
                                            error_bad_lines=False,
                                            encoding="ISO-8859-1",
                                            sep=sep,
                                            nrows=self.nrows)
                    if len(self.data.columns) > 1:
                        break
            except Exception as e:
                print(e)

        if self.data_source == 'json':
            self.data = read_write_to_json(conf('directory'), self.data_query_path, None, is_writing=False)

        if self.data_source == 'yaml':
            self.data = read_yaml(conf('data_main_path'), self.data_query_path)

        if self.data_source in ('json', 'yaml', 'csv'):
            self.data = self.query(self.data)

    def data_execute(self):
        self.query_data_source()
        self.convert_date()
        self.convert_feature()











