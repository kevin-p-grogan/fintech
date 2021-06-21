import pandas as pd

from src.financial_model import FinancialModel
from src.portfolio_optimizer import PortfolioOptimizer


class Visualizer:
    _folder: str

    def __init__(self, folder: str):
        self._folder = folder

    def make_financial_model_plots(self, model: FinancialModel, data: pd.DataFrame):
        pass

    def make_portfolio_optimizer_plots(self, optimizer: PortfolioOptimizer, data: pd.DataFrame):
        pass
