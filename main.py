import pandas as pd

import modules.general as gen
from modules.kucoin_helpers import KC
import datetime
import time
from matplotlib import pyplot as plt


def main():
    api_params = gen.read_json('authentication/shax_kucoin.json')
    df_params = gen.read_json('paramaters/pandas_dataframes.json')
    kc = KC(api_params)

    # Example Usage
    symbol_btc = 'BTC-USDT'
    symbol_sol = 'SOL-USDT'

    now = datetime.datetime.now()
    delta_t = now - datetime.timedelta(days=30)

    data_btc = kc.historical_data_frame(symbol_btc,
                                        '1min',
                                        gen.to_unix_time(delta_t),
                                        gen.to_unix_time(now),
                                        df_params
                                        )

    data_sol = kc.historical_data_frame(symbol_sol,
                                        '1min',
                                        gen.to_unix_time(delta_t),
                                        gen.to_unix_time(now),
                                        df_params)



if __name__ == '__main__':
    main()
