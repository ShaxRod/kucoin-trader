import datetime
import time
from matplotlib import pyplot as plt
import pandas as pd

import modules.general as gen
from modules.kucoin_helpers import KC
from modules.portfolio import Optimise
from modules.forecast import Forecast

api_params = gen.read_json('authentication/shax_kucoin.json')
df_params = gen.read_json('paramaters/pandas_dataframes.json')
portfolio_params = gen.read_json('paramaters/portfolio.json')


def portfolio_optimisation():
    portfolio = Optimise(portfolio_params=portfolio_params, api_params=api_params, df_params=df_params)
    portfolio.get_assets()
    simulations = portfolio.monte_carlo(n=10000, rfr=0.6)
    frame = pd.DataFrame(simulations).T
    optimised_weights = frame[frame['sharpe_ratio'] == frame['sharpe_ratio'].max()]['weights']


def price_forecasting():
    backtest = 120
    models = dict()
    now = datetime.datetime.now()
    delta_t = now - datetime.timedelta(days=int(365.3*3))
    for asset in portfolio_params:
        data = KC(api_params).historical_data_frame(portfolio_params[asset],
                                                    '1day',
                                                    gen.to_unix_time(delta_t),
                                                    gen.to_unix_time(now),
                                                    df_params)
        lstm_model = Forecast(data, 'close', time_step=60)
        lstm_model.lstm()
        predictions = []
        for i in range(0, backtest):
            if lstm_model.rolling_window > lstm_model.lag_count:
                padding = lstm_model.rolling_window
            else:
                padding = lstm_model.lag_count
            index = 60 + padding + i
            index = index * - 1
            if i == 0:
                input_t = data[index:]
                input_object = Forecast(input_t, 'close', time_step=60)
                input_object.pipeline()
                p = lstm_model.predict(input_data=input_object.data.drop(columns=['close']))
            else:
                input_t = data[index:- i]
                input_object = Forecast(input_t, 'close', time_step=60)
                input_object.pipeline()
                p = lstm_model.predict(input_data=input_object.data.drop(columns=['close']))
            predictions.append(p)
        actual = data[['time', 'close']][-backtest + 1:]
        actual['predictions'] = predictions[:-1]

        models[asset] = actual

    for asset in models:
        plt.figure(figsize=(20, 8))
        plt.plot(pd.to_datetime(models[asset]['time']), models[asset]['close'],
                 pd.to_datetime(models[asset]['time']), models[asset]['predictions'])
        plt.legend(['actual closing price', 'predicted closing price'])
        plt.title(f'LSTM forward looking outcomes: {asset}')
        plt.xlabel('date')
        plt.ylabel('closing price')
        plt.savefig(f'C:/Users/User/OneDrive/Documents/python outputs/stock forecasts/plots/{asset}.png')
        plt.close()


if __name__ == '__main__':
    price_forecasting()
