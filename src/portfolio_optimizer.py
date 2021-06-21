from typing import List

import pandas as pd

from src.financial_model import FinancialModel


class PortfolioOptimizer:
    _model: FinancialModel

    def __init__(self, model: FinancialModel):
        self._model = model

    def optimize(self, data: pd.DataFrame, risk_tolerances: List[float]):
        pass

    def save_portfolio_weights(self, path: str):
        pass
