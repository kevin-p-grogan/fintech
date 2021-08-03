from typing import Optional, List

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

from src.data_munger import DataMunger


class FinancialModel:
    _interest_rates: Optional[pd.Series] = None
    _covariances: Optional[pd.DataFrame] = None
    _current_portfolio_weights: Optional[np.ndarray] = None
    _minimum_risk: Optional[float] = None

    PORTFOLIO_VALUE_COLUMN: str = "Equity"

    def predict_yearly_return(self, portfolio_weights: np.array) -> float:
        """Computes the yearly return compounded continuously.
        Includes current portfolio weights if set during training."""
        interest_rates_array = np.array(self.interest_rates).reshape((-1, 1))
        weights_array = self.get_weights_array(portfolio_weights)
        yearly_return = weights_array @ interest_rates_array
        assert len(yearly_return) == 1
        return yearly_return[0, 0]

    def get_weights_array(self, portfolio_weights: np.ndarray) -> np.ndarray:
        """Gets the normalized weights array. Includes the current portfolio weights."""
        weights_array = (portfolio_weights + self.current_portfolio_weights).reshape((1, -1))
        weights_array = weights_array / self.weights_scale
        return weights_array

    @property
    def weights_scale(self) -> float:
        portfolio_weights_scale = 1.0  # due to optimization constraint
        current_weights_scale = self.current_portfolio_weights.sum()
        return portfolio_weights_scale + current_weights_scale

    def predict_apr(self, portfolio_weights: np.array) -> float:
        yearly_return = self.predict_yearly_return(portfolio_weights)
        apr = np.exp(yearly_return) - 1.0
        return apr

    def predict_yearly_return_jacobian(self) -> np.ndarray:
        return np.array(self.interest_rates) / self.weights_scale

    def predict_risk(self, portfolio_weights: np.ndarray) -> float:
        covariance_matrix = np.array(self.covariances)
        weights_array = self.get_weights_array(portfolio_weights)
        risk = weights_array @ covariance_matrix @ weights_array.T
        assert len(risk) == 1
        return risk[0, 0]

    @staticmethod
    def volatility_to_risk(volatility) -> float:
        return volatility ** 2.0

    @staticmethod
    def risk_to_volatility(risk) -> float:
        return risk ** 0.5

    def predict_risk_jacobian(self, portfolio_weights: np.array) -> np.ndarray:
        covariance_matrix = np.array(self.covariances)
        weights_array = self.get_weights_array(portfolio_weights)
        risk_gradient = 2 * covariance_matrix @ weights_array.T / self.weights_scale
        return np.squeeze(risk_gradient)

    @property
    def num_assets(self) -> int:
        return len(self.interest_rates)

    @property
    def asset_names(self) -> List[str]:
        return list(self.interest_rates.index)

    def train(self, data: pd.DataFrame, portfolio_data: Optional[pd.DataFrame] = None,
              investment_amount: Optional[float] = None):
        data = data.copy()
        self._compute_interest_rates(data)
        self._compute_covariances(data)
        self._current_portfolio_weights = np.zeros_like(self.interest_rates)
        if portfolio_data is not None and investment_amount is not None:
            print(f"Incorporating current portfolio into calculations.")
            self._current_portfolio_weights = self._compute_current_portfolio_weights(portfolio_data, investment_amount)
        elif portfolio_data is None and investment_amount is None:
            print("No current portfolio data inputted. Not incorporating current investments.")
        else:
            raise ValueError("Portfolio data and investment amount must be defined to incorporate current portfolio.")

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
        self._minimum_risk = np.min(np.linalg.eigvals(self._covariances))

    def _compute_current_portfolio_weights(self, portfolio_data: pd.DataFrame, investment_amount: float) -> np.ndarray:
        symbols = self._get_current_portfolio_symbols(portfolio_data)
        values = portfolio_data.loc[symbols, self.PORTFOLIO_VALUE_COLUMN]
        values /= investment_amount
        current_portfolio_weights = pd.Series(np.zeros_like(self.asset_names, dtype=np.float), index=self.asset_names)
        current_portfolio_weights[symbols] = values
        return current_portfolio_weights.to_numpy()

    def _get_current_portfolio_symbols(self, portfolio_data: pd.DataFrame) -> set:
        trained_symbols = set(self.asset_names)
        portfolio_symbols = set(portfolio_data.index)
        removed_symbols = set()
        for portfolio_symbol in portfolio_symbols:
            if portfolio_symbol not in trained_symbols:
                print(f"WARNING: {portfolio_symbol} symbol not found in the training data."
                      f" Will not be included in analysis.")
                removed_symbols.add(portfolio_symbol)

        symbols = portfolio_symbols.difference(removed_symbols)
        return symbols

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

    @property
    def current_portfolio_weights(self) -> np.ndarray:
        if self._current_portfolio_weights is None:
            raise AttributeError("Current portfolio weights are not available. Make sure to train model first.")
        return self._current_portfolio_weights

    def remove_current_portfolio_weights(self, portfolio_weights: pd.Series) -> pd.Series:
        self._check_portfolio(portfolio_weights)
        current_weights = pd.Series(self.current_portfolio_weights, index=self.asset_names)
        scaled_current_weights = current_weights / self.weights_scale
        portfolio_update = portfolio_weights - scaled_current_weights
        portfolio_update /= portfolio_update.sum()
        return portfolio_update

    @staticmethod
    def _check_portfolio(portfolio: pd.Series):
        eps = 1.e-6
        portfolio_doesnt_sum_to_one = not np.isclose(portfolio.sum(), 1.0, atol=eps)
        negative_assets_exist = np.any(portfolio < -eps)  # assume no negative assets for now
        if portfolio_doesnt_sum_to_one or negative_assets_exist:
            raise ValueError(f"Invalid portfolio found. "
                             f"Ensure that the portfolio contains only positive assets and that all values sum to one")

    @property
    def maximum_yearly_return(self) -> float:
        max_portfolio_weights = np.zeros_like(self.interest_rates)
        max_interest_index = np.argmax(self.interest_rates)
        max_portfolio_weights[max_interest_index] = 1.0
        return self.predict_yearly_return(max_portfolio_weights)

    @property
    def minimum_risk(self) -> float:
        if self._minimum_risk is None:
            raise AttributeError("Minimum risk of portfolio are not available. Make sure to train model first.")
        return self._minimum_risk
