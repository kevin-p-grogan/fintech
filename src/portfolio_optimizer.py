from typing import Optional

from scipy.optimize import minimize, LinearConstraint
import pandas as pd
import numpy as np

from src.financial_model import FinancialModel


class PortfolioOptimizer:
    _model: FinancialModel
    _optimal_weights = Optional[pd.DataFrame]

    LOWER_TRADEOFF_BOUND: float = 0.0
    UPPER_TRADEOFF_BOUND: float = 1.0

    def __init__(self, model: FinancialModel):
        self._model = model
        self._optimal_weights = None

    def optimize(self, num_evaluations: int = 11, verbose: bool = True):
        num_weights = self._model.num_assets
        initial_weights = np.ones((num_weights, 1)) / num_weights
        constraints = [self._portfolio_weights_sum_to_one, self._portfolio_weights_non_negative]
        tradeoff_params = np.linspace(self.LOWER_TRADEOFF_BOUND, self.UPPER_TRADEOFF_BOUND, num_evaluations)
        optimal_weights = []
        for i, tradeoff_param in enumerate(tradeoff_params):
            if verbose:
                print(f"Beginning optimization for tradeoff parameter = {tradeoff_param}. {i+1} of {num_evaluations}")

            result = minimize(
                self._objective,
                initial_weights,
                jac=self._objective_jacobian,
                args=(tradeoff_param,),
                constraints=constraints)
            if verbose:
                print("Finished optimization")

            optimal_weights.append(result.x)

        data = np.stack(optimal_weights)
        asset_names = self._model.asset_names
        self._optimal_weights = pd.DataFrame(data, index=tradeoff_params, columns=asset_names)

    @property
    def _portfolio_weights_sum_to_one(self) -> LinearConstraint:
        num_weights = self._model.num_assets
        lower_bound = 1.0
        upper_bound = 1.0
        constraint_matrix = np.ones((1, num_weights))
        return LinearConstraint(constraint_matrix, lower_bound, upper_bound)

    @property
    def _portfolio_weights_non_negative(self) -> LinearConstraint:
        num_weights = self._model.num_assets
        lower_bound = 0.0
        upper_bound = np.inf
        constraint_matrix = np.identity(num_weights)
        return LinearConstraint(constraint_matrix, lower_bound, upper_bound)

    def _objective(self, portfolio_weights: np.ndarray, tradeoff_parameter: float) -> float:
        yearly_return = self._model.predict_yearly_return(portfolio_weights)
        risk = self._model.predict_risk(portfolio_weights)
        obj = -tradeoff_parameter*yearly_return + (1.-tradeoff_parameter)*risk
        return obj

    def _objective_jacobian(self, portfolio_weights: np.ndarray, tradeoff_parameter: float) -> np.ndarray:
        yearly_return_jacobian = self._model.predict_yearly_return_jacobian()
        risk_jacobian = self._model.predict_risk_jacobian(portfolio_weights)
        obj = -tradeoff_parameter*yearly_return_jacobian + (1.-tradeoff_parameter)*risk_jacobian
        return obj

    def save_portfolio_weights(self, path: str):
        pass
