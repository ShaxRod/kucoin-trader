import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import CuDNNLSTM

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class Forecast:

    def __init__(self, data: pd.DataFrame, target: str, time_step: int):

        self.__X = None
        self.__Y = data[[target]].copy(deep=True)
        self.__X_scaler = MinMaxScaler()
        self.__Y_scaler = MinMaxScaler()

        self.data = data.copy(deep=True)
        self.time = time_step
        self.kline = target
        self.model = None
        self.lag_count = 0
        self.rolling_window = 0

    def pipeline(self):
        if len(self.data.columns) > 1:
            self.data = self.__Y.copy(deep=True)

        self.target_lags()
        self.volatility()
        self.add_technical_indicators()
        self.add_bollinger_bands()
        self.add_momentum_features()

        self.data = self.data.dropna()
        self.__X = self.data.drop(columns=self.kline)
        self.__X = self.__X_scaler.fit_transform(self.__X).astype(np.float32)

    def target_lags(self, lags: int = 12):
        for i in range(1, lags + 1):
            self.data[f'{self.kline}_lag_{i}'] = self.__Y.shift(i)
        self.lag_count = lags

    def volatility(self, rolling_window: int = 20):
        self.data['daily_returns'] = self.__Y.pct_change()
        self.data['volatility'] = self.data['daily_returns'].rolling(window=rolling_window).std()
        self.rolling_window = rolling_window

    def add_technical_indicators(self):
        self.data['SMA_10'] = self.__Y.rolling(window=10).mean()
        self.data['SMA_50'] = self.__Y.rolling(window=50).mean()
        self.data['RSI'] = self.compute_rsi(self.__Y)
        self.data['MACD'] = self.compute_macd(self.__Y)

    def add_bollinger_bands(self, window=20):
        sma = self.__Y.rolling(window=window).mean()
        std = self.__Y.rolling(window=window).std()
        self.data['BB_Mid'] = sma
        self.data['BB_Upper'] = sma + (2 * std)
        self.data['BB_Lower'] = sma - (2 * std)

    def add_momentum_features(self, period=14):
        self.data['ROC'] = (self.__Y - self.__Y.shift(period)) / self.__Y.shift(period) * 100
        self.data['Momentum'] = self.__Y - self.__Y.shift(period)

    def compute_rsi(self, series, period=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def compute_macd(self, series, short_window=12, long_window=26, signal_window=9):
        short_ema = series.ewm(span=short_window, adjust=False).mean()
        long_ema = series.ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd - signal

    def create_dataset(self, batch_size=32):
        self.__Y = self.__Y.to_numpy().reshape(-1, 1)
        self.__Y = self.__Y_scaler.fit_transform(self.__Y).astype(np.float32)
        num_samples = len(self.__X) - self.time
        for i in range(0, num_samples, batch_size):
            X_batch = [self.__X[j:j + self.time, :] for j in range(i, min(i + batch_size, num_samples))]
            y_batch = [self.__Y[j + self.time, 0] for j in range(i, min(i + batch_size, num_samples))]
            yield np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)

    def lstm(self, batch):
        self.pipeline()

        # Create dataset generator instead of storing full dataset
        train_gen = self.create_dataset(batch_size=batch)

        # Define LSTM Model
        lstm = Sequential()
        lstm.add(Input(shape=(self.time, self.__X.shape[1])))
        lstm.add(LSTM(50, return_sequences=True, activation=None))  # CuDNN equivalent
        lstm.add(LSTM(50, return_sequences=False, activation=None))  # Second LSTM
        lstm.add(Dense(25))
        lstm.add(Dropout(0.2))
        lstm.add(Dense(1))

        lstm.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mean_absolute_percentage_error'
                     )

        # Train using generator (no need for full dataset in memory)
        steps_per_epoch = len(self.__X) // batch  # Number of batches per epoch
        lstm.fit(train_gen, epochs=100, steps_per_epoch=steps_per_epoch, verbose=1)

        self.model = lstm

    def predict(self, input_data):
        scaled_input = self.__X_scaler.transform(input_data)
        prediction_scaled = self.model.predict(np.reshape(scaled_input, (1, input_data.shape[0], -1)))
        prediction = self.__Y_scaler.inverse_transform(prediction_scaled)
        return prediction[0, 0]
