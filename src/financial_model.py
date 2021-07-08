from typing import Optional, List

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

    def predict_yearly_return(self, portfolio_weights: np.array) -> float:
        interest_rates_array = np.array(self.interest_rates).reshape((-1, 1))
        weights_array = portfolio_weights.reshape((1, -1))
        yearly_return = weights_array @ interest_rates_array
        assert len(yearly_return) == 1
        return yearly_return[0, 0]

    def predict_apr(self, portfolio_weights: np.array) -> float:
        yearly_return = self.predict_yearly_return(portfolio_weights)
        apr = np.exp(yearly_return) - 1.0
        return apr

    def predict_yearly_return_jacobian(self) -> np.ndarray:
        return np.array(self.interest_rates)

    def predict_risk(self, portfolio_weights: np.ndarray) -> float:
        covariance_matrix = np.array(self.covariances)
        weights_array = portfolio_weights.reshape((1, -1))
        risk = weights_array @ covariance_matrix @ weights_array.T
        assert len(risk) == 1
        return risk[0, 0]

    @staticmethod
    def volatility_to_risk(volatility) -> float:
        return volatility ** 2.0

    def predict_risk_jacobian(self, portfolio_weights: np.array) -> np.ndarray:
        covariance_matrix = np.array(self.covariances)
        weights_array = portfolio_weights.reshape((1, -1))
        risk_gradient = 2 * covariance_matrix @ weights_array.T
        return np.squeeze(risk_gradient)

    @property
    def num_assets(self) -> int:
        return len(self.interest_rates)

    @property
    def asset_names(self) -> List[str]:
        return list(self.interest_rates.index)

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

    @property
    def covariances(self) -> pd.DataFrame:
        if self._covariances is None:
            raise AttributeError("Covariances are not available. Make sure to train model first.")
        return self._covariances

    @property
    def interest_rates(self) -> pd.Series:
        if self._interest_rates is None:
            raise AttributeError("Interest rates are not available. Make sure to train model first.")
        return self._interest_rates
