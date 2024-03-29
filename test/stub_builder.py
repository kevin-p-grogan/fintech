from collections import namedtuple
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

from src.financial_model import FinancialModel
from src.portfolio_optimizer import PortfolioOptimizer

DataParameters = namedtuple("TestParameters", ["name", "interest_rate", "variance"])


class StubBuilder:
    """Class that handles the building of test stubs"""
    @staticmethod
    def create_portfolio_optimzer(params: List[DataParameters]) -> PortfolioOptimizer:
        financial_model = StubBuilder.create_financial_model(params)
        portfolio_optimizer = PortfolioOptimizer(financial_model)
        return portfolio_optimizer

    @staticmethod
    def create_financial_model(params: List[DataParameters]) -> FinancialModel:
        data = StubBuilder.create_test_data(params)
        financial_model = FinancialModel()
        financial_model.train(data)
        return financial_model

    @staticmethod
    def create_financial_model_with_portfolio(
            params: List[DataParameters],
            portfolio_data: pd.DataFrame) -> FinancialModel:
        data = StubBuilder.create_test_data(params)
        financial_model = FinancialModel()
        financial_model.train(data, portfolio_data=portfolio_data, investment_amount=100)
        return financial_model

    @staticmethod
    def create_test_data(params: List[DataParameters]) -> pd.DataFrame:
        """Create data according to geometric Brownian motion model."""
        num_times = 10000
        times = np.linspace(0, 1, num_times)
        dates = StubBuilder.make_dates(times)
        data = pd.DataFrame({
            p.name: np.exp(p.interest_rate * times + (p.variance*times)**0.5 * np.random.randn(num_times))
            for p in params
        }, index=dates)
        return data

    @staticmethod
    def make_dates(times: np.array) -> np.array:
        days_in_year = 365.0
        deltas = (timedelta(days=int(days_in_year * t)) for t in times)
        start_date = datetime.now()
        dates = [start_date + delta for delta in deltas]
        return np.array(dates)
