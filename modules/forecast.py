import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class Forecast:

    def __init__(self, data: pd.DataFrame, time_step: int):
        self.__X = np.array([])
        self.__Y = np.array([])

        self.asset = data
        self.time = time_step

        self.lstm = None
        self.arima = None

    def create_dataset(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.asset)
        X, Y = [], []
        for i in range(len(scaled_data) - self.time - 1):
            X.append(scaled_data[i:(i + self.time), 0])
            Y.append(scaled_data[i + self.time, 0])

        self.__X = np.array(X)
        self.__Y = np.array(Y)

    def lstm(self):

        self.create_dataset()
        self.__X = self.__X.reshape(self.__X.shape[0], self.__X.shape[1], 1)

        # LSTM Model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.time, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(self.__X, self.__Y, batch_size=32, epochs=100, verbose=1)

        self.lstm = model

    def arima(self):
        model = ARIMA(self.asset, order=(self.time, 1, 0))  # These parameters can be optimized
        model_fit = model.fit()

        # Summary of the model
        print(model_fit.summary())

        # Forecast
        forecast = model_fit.forecast(steps=self.time)  # Forecasting next 5 days
        print(forecast)



