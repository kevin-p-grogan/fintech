from typing import Optional

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

from src.data_munger import DataMunger


class FinancialModel:
    _num_days_prediction_period: float
    _interest_rates: Optional[pd.Series]
    _covariances: Optional[pd.DataFrame]

    def __init__(self, num_days_prediction_period: float = 30):
        self._num_days_prediction_period = num_days_prediction_period
        self._interest_rates = None
        self._covariances = None

    def predict_expected_value(self, data: pd.DataFrame) -> pd.Series:
        pass

    def predict_covariance(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def train(self, data: pd.DataFrame):
        data = data.copy()
        self._compute_interest_rates(data)
        self._compute_covariances(data)

    def _compute_interest_rates(self, data: pd.DataFrame):
        lr = LinearRegression(fit_intercept=False)
        times = DataMunger.get_times_from_index(data)
        interest_rates = []
        symbols = []
        for symbol in data:
            prices = data[symbol]
            price_array = prices.to_numpy()
            time_array = times.to_numpy().reshape(-1, 1)
            lr.fit(time_array, np.log(price_array))
            interest_rate = lr.coef_[0]
            interest_rates.append(interest_rate)
            symbols.append(symbol)

        self._interest_rates = pd.Series(interest_rates, symbols)

    def _compute_covariances(self, data: pd.DataFrame):
        data = data.copy()
        times = DataMunger.get_times_from_index(data)
        predicted_data = self.predict(times)
        noise = data / predicted_data - 1.0
        self._covariances = noise.cov()

    def predict(self, times: pd.Series) -> pd.DataFrame:
        predicted_data = {}
        for symbol in self._interest_rates.index:
            interest_rate = self._interest_rates[symbol]
            predicted_data[symbol] = np.exp(interest_rate * times)

        return pd.DataFrame(predicted_data)

    def get_covariance(self, symbol1: str, symbol2: str) -> float:
        if self._covariances is None:
            raise AttributeError("Covariances are not available. Make sure to train model first.")
        return self._covariances[symbol1][symbol2]

    def get_interest_rate(self, symbol: str) -> float:
        if self._interest_rates is None:
            raise AttributeError("Interest rates are not available. Make sure to train model first.")
        return self._interest_rates[symbol]
