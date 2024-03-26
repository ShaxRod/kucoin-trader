import numpy as np
import pandas as pd
import numpy
import sklearn
import scipy
import datetime
import time

from modules.kucoin_helpers import KC
import modules.general as gen


class Optimise:

    def __init__(self, portfolio_params: dict, api_params: dict, df_params: dict):
        self.__portfolio = portfolio_params
        self.__kucoin = KC(api_params)
        self.__dfp = df_params
        self.assets = pd.DataFrame()

    def get_assets(self):
        ndays = round(1 * 365.3)
        now = datetime.datetime.now()
        delta_t = now - datetime.timedelta(days=ndays)
        for asset in self.__portfolio:
            asset_df = self.__kucoin.historical_data_frame(self.__portfolio[asset],
                                                           '1day',
                                                           gen.to_unix_time(delta_t),
                                                           gen.to_unix_time(now),
                                                           self.__dfp
                                                           )
            asset_df['asset'] = asset
            print(f'{asset} data retrieved')
            self.assets = pd.concat([self.assets, asset_df], ignore_index=True)

    def pivot_close(self):
        return pd.pivot_table(self.assets, values=['close'], index=['time'], columns=['asset'], aggfunc='sum')

    def log_returns(self):
        data = self.pivot_close()
        log_returns = np.log(1 + data.pct_change().dropna())
        return log_returns

    def sharpe_ratio(self, rfr: float):
        _log_returns = self.log_returns()

        weights_i = np.array(np.random.random(len(self.__portfolio)))
        weights_i = weights_i / np.sum(weights_i)

        exp_returns = np.sum(_log_returns.mean() * weights_i * 252)  # Annualised returns
        exp_volatility = np.sqrt(np.dot(weights_i.T,
                                        np.dot(_log_returns.cov() * 252,
                                               weights_i)
                                        )
                                 )
        sharpe = (exp_returns - rfr) / exp_volatility
        return {'sharpe_ratio': sharpe,
                'expected_returns': exp_returns,
                'expected_volatility': exp_volatility,
                'weights': weights_i}

    def monte_carlo(self, n: int, rfr: float):
        simulations = dict()
        for i in range(0, n):
            simulations[i] = self.sharpe_ratio(rfr=rfr)
        return simulations


