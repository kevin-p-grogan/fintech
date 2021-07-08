import os
from typing import Sequence

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
            volatilities: Sequence[float]):
        data = data.copy()
        times = DataMunger.get_times_from_index(data)
        aprs = []
        for i, volatility in enumerate(volatilities):
            risk = optimizer.financial_model.volatility_to_risk(volatility)
            portfolio_weights = optimizer.get_portfolio_weights(risk)
            portfolio = data @ portfolio_weights
            apr = optimizer.financial_model.predict_apr(np.array(portfolio_weights))
            aprs.append(apr)
            plt.figure(i)
            plt.plot(times, data)
            plt.plot(times, portfolio, 'ks-', label="Portfolio")
            plt.xlabel("Time in Fraction of a Year")
            plt.ylabel("Prices")
            plt.title(f"Portfolio Volatility = {volatility}. Portfolio APR = {apr}.")
            plt.legend(loc="best")
            filename = f"portfolio_volatility_{volatility}.png"
            path = os.path.join(self._folder, filename)
            plt.savefig(path)
            plt.clf()

        plt.figure()
        plt.plot(volatilities, aprs, "ks")
        plt.xlabel("Portfolio Volatility")
        plt.ylabel("APR")
        filename = f"volatility_apr_tradeoff.png"
        path = os.path.join(self._folder, filename)
        plt.savefig(path)
        plt.clf()
