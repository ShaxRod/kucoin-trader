import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class Forecast:

    def __init__(self, data: pd.DataFrame, kline: str, time_step: int):

        self.__X = None
        self.__Y = data[[kline]]
        self.__X_scaler = MinMaxScaler()
        self.__Y_scaler = MinMaxScaler()

        self.data = data[[kline]]
        self.time = time_step
        self.kline = kline
        self.model = None
        self.lag_count = 0
        self.rolling_window = 0

    def pipeline(self):

        if len(self.data.columns) > 1:
            self.data = self.__Y.copy(deep=True)

        self.lags()
        self.volatility()
        self.data = self.data.dropna()
        self.__X = self.data.drop(columns=self.kline)
        # Drop NaN values that are a result of lagging and rolling calculations

    def lags(self, lags: int = 12):
        for i in range(1, lags + 1):
            self.data[f'close_lag_{i}'] = self.__Y.shift(i)
        self.lag_count = lags

    def volatility(self, rolling_window: int = 20):
        self.data['daily_returns'] = self.__Y.pct_change()
        self.data['volatility'] = self.data['daily_returns'].rolling(window=rolling_window).std()
        self.rolling_window = rolling_window

    def create_dataset(self):
        self.__X = self.__X_scaler.fit_transform(self.__X)
        self.__Y = self.__Y_scaler.fit_transform(self.__Y)

        X_data, Y_data = [], []
        for i in range(len(self.__X) - self.time - 1):
            X_data.append(self.__X[i:(i + self.time), :])
            Y_data.append(self.__Y[i + self.time, 0])

        self.__X = np.array(X_data)
        self.__Y = np.array(Y_data)

    def lstm(self):

        self.pipeline()
        self.create_dataset()
        # Reshape X for LSTM [samples, time steps, features]
        self.__X = self.__X.reshape(self.__X.shape[0],
                                    self.time,
                                    -1)

        # LSTM Model
        lstm = Sequential()
        lstm.add(LSTM(50,
                      return_sequences=True,
                      input_shape=(self.time,
                                   self.__X.shape[2]
                                   )
                      )
                 )

        lstm.add(LSTM(50,
                      return_sequences=False
                      )
                 )
        lstm.add(Dense(25))
        lstm.add(Dropout(0.2))
        lstm.add(Dense(1))

        lstm.compile(optimizer='adam',
                     loss='mean_squared_error')

        # Train the model
        lstm.fit(self.__X, self.__Y, batch_size=32, epochs=100, verbose=1)

        self.model = lstm

    def predict(self, input_data):
        scaled_input = self.__X_scaler.transform(input_data)
        prediction_scaled = self.model.predict(np.reshape(scaled_input,
                                                          (1,
                                                           input_data.shape[0],
                                                           -1
                                                           )
                                                          )
                                               )
        prediction = self.__Y_scaler.inverse_transform(prediction_scaled)
        return prediction[0, 0]
