import os

import pandas as pd
from matplotlib import pyplot as plt

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
            plt.title(f"Interest Rate={model.get_interest_rate(symbol)}. "
                      f"Variance={model.get_covariance(symbol, symbol)}.")
            filename = f"financial_model_{symbol}.png"
            path = os.path.join(self._folder, filename)
            plt.savefig(path)
            plt.clf()

    def make_portfolio_optimizer_plots(self, optimizer: PortfolioOptimizer, data: pd.DataFrame):
        pass
