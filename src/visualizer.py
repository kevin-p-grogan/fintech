import os
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from src.financial_model import FinancialModel
from src.portfolio_optimizer import PortfolioOptimizer
from src.data_munger import DataMunger


class Visualizer:
    _folder: str

    def __init__(self, folder: str):
        self._folder = folder

    def make_financial_model_plots(self, model: FinancialModel, data: pd.DataFrame):
        data = data.copy()
        times = DataMunger.get_times_from_index(data)
        predicted_data = model.predict(times)
        for symbol in data:
            plt.plot(times, data[symbol], 'ks', label="Actual")
            plt.plot(times, predicted_data[symbol], 'r', label="Predictions")
            plt.xlabel("Time in Fraction of a Year")
            plt.ylabel("Prices")
            plt.title(f"Interest Rate={model.interest_rates[symbol]}. "
                      f"Variance={model.covariances[symbol][symbol]}.")
            filename = f"financial_model_{symbol}.png"
            path = os.path.join(self._folder, filename)
            plt.savefig(path)
            plt.clf()

    def make_portfolio_optimizer_plots(
            self,
            optimizer: PortfolioOptimizer,
            data: pd.DataFrame,
            num_volatilities: int = 10,
            price_limits: Optional[list[float]] = None):
        data = data.copy()
        self._plot_portfolio_per_volatility(optimizer, data, num_volatilities, price_limits)
        self._plot_volatility_apr_tradeoff(optimizer, num_volatilities)

    def _plot_volatility_apr_tradeoff(self, optimizer: PortfolioOptimizer, num_volatilities: int = 10):
        volatilities = np.linspace(optimizer.min_volatility, optimizer.max_volatility, num_volatilities)
        aprs = [optimizer.compute_apr_from_volatility(vol) for vol in volatilities]
        plt.figure()
        plt.plot(volatilities, aprs, "ks-")
        plt.xlabel("Portfolio Volatility")
        plt.ylabel("APR")
        filename = f"volatility_apr_tradeoff.png"
        path = os.path.join(self._folder, filename)
        plt.savefig(path)
        plt.clf()

    def _plot_portfolio_per_volatility(
            self,
            optimizer: PortfolioOptimizer,
            data: pd.DataFrame,
            num_volatilities: int = 10,
            price_limits: Optional[list[float]] = None):
        times = DataMunger.get_times_from_index(data)
        volatilities = np.linspace(optimizer.min_volatility, optimizer.max_volatility, num_volatilities)
        for i, volatility in enumerate(volatilities):
            risk = optimizer.financial_model.volatility_to_risk(volatility)
            portfolio_weights = optimizer.get_portfolio_weights(risk)
            portfolio = data @ portfolio_weights
            plt.plot(times, data)
            plt.plot(times, portfolio, 'ks-', label="Portfolio")
            plt.xlabel("Time in Fraction of a Year")
            plt.ylabel("Prices")
            apr = optimizer.compute_apr_from_volatility(volatility)
            plt.title(f"Portfolio Volatility = {volatility:1.3f}. Portfolio APR = {apr:1.3f}.")
            plt.legend(loc="best")
            filename = f"portfolio_volatility_{volatility:1.3f}_apr_{apr:1.3f}.png"
            plt.ylim(price_limits) if price_limits else None
            path = os.path.join(self._folder, filename)
            plt.savefig(path)
            plt.clf()