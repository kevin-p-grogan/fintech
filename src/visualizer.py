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
            risks: Sequence[float]):
        data = data.copy()
        times = DataMunger.get_times_from_index(data)
        yearly_returns = []
        for i, risk in enumerate(risks):
            portfolio_weights = optimizer.get_portfolio_weights(risk)
            portfolio = data @ portfolio_weights
            yearly_return = optimizer.financial_model.predict_yearly_return(np.array(portfolio_weights))
            yearly_returns.append(yearly_return)
            plt.figure(i)
            plt.plot(times, data)
            plt.plot(times, portfolio, 'ks-', label="Portfolio")
            plt.xlabel("Time in Fraction of a Year")
            plt.ylabel("Prices")
            plt.title(f"Portfolio Risk = {risk}. Portfolio Yearly Return = {yearly_return}.")
            plt.legend(loc="best")
            filename = f"portfolio_risk_{risk}.png"
            path = os.path.join(self._folder, filename)
            plt.savefig(path)
            plt.clf()

        plt.figure()
        plt.plot(risks, yearly_returns, "ks")
        plt.xlabel("Portfolio Risk")
        plt.ylabel("Yearly Return")
        filename = f"risk_return_tradeoff.png"
        path = os.path.join(self._folder, filename)
        plt.savefig(path)
        plt.clf()
