import datetime

import pandas as pd

import modules.general as gen
from kucoin.client import Market
import requests


class KC:

    def __init__(self, api_params: dict):
        self.__key = api_params['key']
        self.__secret = api_params['secret'],
        self.__pp = api_params['passphrase']
        self.__client = Market(url='https://api.kucoin.com')

    def get_historical_data(self, symbol, interval, start_at, end_at):
        return self.__client.get_kline(symbol, interval, startAt=start_at, endAt=end_at)

    def historical_data_frame(self, symbol, interval, start_at, end_at, df_parameters: dict):

        data = self.get_historical_data(symbol, interval, start_at, end_at)

        df = pd.DataFrame(data, columns=df_parameters['kline']['columns'])

        for column in df_parameters['kline']['data_types']:
            df[column] = pd.to_numeric(df[column], errors='coerce')
            if df_parameters['kline']['data_types'][column] == 'float':
                df[column] = df[column].astype(float)
            elif df_parameters['kline']['data_types'][column] == 'int':
                df[column] = df[column].astype(int)

        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

