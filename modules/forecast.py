import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class Forecast:

    def __init__(self, data: pd.DataFrame, time_step: int):
        self.__X = np.array([])

        self._X = np.array([])
        self._Y = np.array([])

        self.asset = data
        self.time = time_step
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def create_dataset(self):
        scaled_data = self.scaler.fit_transform(self.asset)
        X, Y = [], []
        for i in range(len(scaled_data) - self.time - 1):
            X.append(scaled_data[i:(i + self.time), 0])
            Y.append(scaled_data[i + self.time, 0])

        self._X = np.array(X)
        self._Y = np.array(Y)

    def lstm(self):
        self.create_dataset()
        self.__X = self._X.reshape(self._X.shape[0], self._X.shape[1], 1)

        # LSTM Model
        lstm = Sequential()
        lstm.add(LSTM(50, return_sequences=True, input_shape=(self.time, 1)))
        lstm.add(LSTM(50, return_sequences=False))
        lstm.add(Dense(25))
        lstm.add(Dense(1))

        lstm.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        lstm.fit(self.__X, self._Y, batch_size=32, epochs=100, verbose=1)

        self.model = lstm

    def predict(self, input_data):
        scaled_input = self.scaler.transform(input_data)
        prediction = self.model.predict(np.reshape(scaled_input,
                                                   (1, len(input_data), 1)
                                                   )
                                        )
        return self.scaler.inverse_transform(prediction)
