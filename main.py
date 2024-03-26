import datetime
import time
from matplotlib import pyplot as plt
import pandas as pd

import modules.general as gen
from modules.kucoin_helpers import KC
from modules.portfolio import Optimise


def main():
    api_params = gen.read_json('authentication/shax_kucoin.json')
    df_params = gen.read_json('paramaters/pandas_dataframes.json')
    portfolio_params = gen.read_json('paramaters/portfolio.json')

    portfolio = Optimise(portfolio_params=portfolio_params, api_params=api_params, df_params=df_params)
    portfolio.get_assets()
    simulations = portfolio.monte_carlo(n=10000, rfr=0.6)
    frame = pd.DataFrame(simulations).T
    optimised_weights = frame[frame['sharpe_ratio'] == frame['sharpe_ratio'].max()]['weights']


if __name__ == '__main__':
    main()
