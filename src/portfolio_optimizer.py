from typing import Optional
import os

from scipy.optimize import minimize, LinearConstraint
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np

from src.financial_model import FinancialModel


class PortfolioOptimizer:
    financial_model: FinancialModel
    _sparsity_importance: float
    _optimal_weights: Optional[pd.DataFrame]
    _optimal_results: Optional[pd.DataFrame]
    _risk_interpolator: Optional[interp1d]

    LOWER_TRADEOFF_BOUND: float = 0.0
    UPPER_TRADEOFF_BOUND: float = 1.0
    EPS: float = 1.0e-6

    def __init__(self, model: FinancialModel, sparsity_importance: float = 0.1):
        self.financial_model = model
        self._sparsity_importance = sparsity_importance
        self._optimal_weights = None
        self._optimal_results = None
        self._risk_interpolator = None

    def optimize(self, num_evaluations: int = 11, verbose: bool = True):
        num_weights = self.financial_model.num_assets
        initial_weights = np.ones((num_weights, 1)) / num_weights
        constraints = [self._portfolio_weights_sum_to_one, self._portfolio_weights_non_negative]
        tradeoff_params = np.linspace(self.LOWER_TRADEOFF_BOUND, self.UPPER_TRADEOFF_BOUND, num_evaluations)
        optimal_weights = []
        optimal_results = []
        for i, tradeoff_param in enumerate(tradeoff_params):
            if verbose:
                print(f"Beginning optimization for tradeoff parameter = {tradeoff_param}. {i + 1} of {num_evaluations}")

            result = minimize(self._objective, initial_weights,
                              jac=self._objective_jacobian, args=(tradeoff_param,), constraints=constraints)
            if verbose:
                print("Finished optimization")

            optimal_weights.append(list(result.x))
            risk = self.financial_model.predict_risk(result.x)
            yearly_return = self.financial_model.predict_yearly_return(result.x)
            optimal_results.append([yearly_return, risk])

        optimal_weights = np.array(optimal_weights)
        self._optimal_weights = pd.DataFrame(optimal_weights, index=tradeoff_params,
                                             columns=self.financial_model.asset_names)
        optimal_results = np.array(optimal_results)
        self._optimal_results = pd.DataFrame(optimal_results, index=tradeoff_params, columns=["Yearly Return", "Risk"])
        optimal_risks = self.optimal_results["Risk"]
        self._risk_interpolator = interp1d(optimal_risks, self.optimal_weights, axis=0)

    @property
    def _portfolio_weights_sum_to_one(self) -> LinearConstraint:
        num_weights = self.financial_model.num_assets
        lower_bound = 1.0
        upper_bound = 1.0
        constraint_matrix = np.ones((1, num_weights))
        return LinearConstraint(constraint_matrix, lower_bound, upper_bound)

    @property
    def _portfolio_weights_non_negative(self) -> LinearConstraint:
        num_weights = self.financial_model.num_assets
        lower_bound = 0.0
        upper_bound = np.inf
        constraint_matrix = np.identity(num_weights)
        return LinearConstraint(constraint_matrix, lower_bound, upper_bound)

    def _objective(self, portfolio_weights: np.ndarray, tradeoff_parameter: float) -> float:
        yearly_return = self.financial_model.predict_yearly_return(portfolio_weights)
        risk = self.financial_model.predict_risk(portfolio_weights)
        obj = -tradeoff_parameter * yearly_return + (
                    1. - tradeoff_parameter) * risk + self.sparsity_weight * self._entropy(portfolio_weights)
        return obj

    @property
    def sparsity_weight(self) -> float:
        """Weighs the sparsity of the profile in consideration of the number of assets and the current returns"""
        max_objective = max(self.financial_model.maximum_yearly_return, self.financial_model.minimum_risk)
        min_objective = min(self.financial_model.maximum_yearly_return, self.financial_model.minimum_risk)
        objective_scale = max_objective - min_objective
        entropy_scale = np.log(self.financial_model.num_assets)
        return self._sparsity_importance * objective_scale / entropy_scale

    @staticmethod
    def _entropy(probabilities: np.ndarray) -> float:
        smoothed_probabilities = np.maximum(probabilities, PortfolioOptimizer.EPS)
        return -float(np.sum(smoothed_probabilities * np.log(smoothed_probabilities)))

    def _objective_jacobian(self, portfolio_weights: np.ndarray, tradeoff_parameter: float) -> np.ndarray:
        yearly_return_jacobian = self.financial_model.predict_yearly_return_jacobian()
        risk_jacobian = self.financial_model.predict_risk_jacobian(portfolio_weights)
        obj = -tradeoff_parameter * yearly_return_jacobian + (
                    1. - tradeoff_parameter) * risk_jacobian + self.sparsity_weight * self._entropy_jacobian(
            portfolio_weights)
        return obj

    @staticmethod
    def _entropy_jacobian(probabilities: np.ndarray) -> np.ndarray:
        smoothed_probabilities = np.maximum(probabilities, PortfolioOptimizer.EPS)
        return -1.0 - np.log(smoothed_probabilities)

    @property
    def optimal_weights(self) -> pd.DataFrame:
        self._check_property("optimal_weights")
        return self._optimal_weights

    @property
    def optimal_results(self) -> pd.DataFrame:
        self._check_property("optimal_results")
        return self._optimal_results

    @property
    def risk_interpolator(self) -> interp1d:
        self._check_property("risk_interpolator")
        return self._risk_interpolator

    def _check_property(self, property_name: str):
        private_property_name = "_" + property_name
        assert private_property_name in self.__dict__.keys()
        if self.__dict__[private_property_name] is None:
            raise AttributeError(f"{property_name} was not found. "
                                 "Ensure that the PortfolioOptimizer has been optimized.")

    def save_portfolio_update(self, path: str, portfolio_update: pd.Series, metadata_path: Optional[str] = None):
        portfolio = portfolio_update
        if metadata_path is not None:
            metadata_path = self._check_filepath(metadata_path)
            portfolio = pd.read_csv(metadata_path)
            portfolio = portfolio.set_index("Symbol")
            portfolio["Weights"] = portfolio_update

        path = self._check_filepath(path)
        portfolio = portfolio.sort_values(by="Weights", ascending=False)
        portfolio.to_csv(path)

    def get_portfolio_update(self, volatility) -> pd.Series:
        risk = self.financial_model.volatility_to_risk(volatility)
        portfolio_weights = self.get_portfolio_weights(risk)
        portfolio_weights = self.financial_model.remove_current_portfolio_weights(portfolio_weights)
        return portfolio_weights

    @staticmethod
    def _check_filepath(path: str):
        prefix, ext = os.path.splitext(path)
        if ext.lower() != ".csv":
            raise ValueError(f"File extension must be 'csv' or not defined. Found {ext}.")
        elif not ext:
            path += ".csv"
            print(f"No extension found. New path with extension will be '{path}'.")
        return path

    def get_portfolio_weights(self, risk: float) -> pd.Series:
        self._check_risk(risk)
        interpolated_weights = self.risk_interpolator(risk)
        weights_array = self.financial_model.get_weights_array(interpolated_weights)
        weights_array = np.squeeze(weights_array)
        portfolio_weights = pd.Series(weights_array, index=self.financial_model.asset_names)
        return portfolio_weights

    def _check_risk(self, risk: float):
        min_risk = self.optimal_results["Risk"].min()
        max_risk = self.optimal_results["Risk"].max()
        in_bounds = min_risk <= risk <= max_risk
        if not in_bounds:
            raise ValueError(f"Inputted risk, {risk}, is not in bounds. "
                             f"It must be between {min_risk} and {max_risk}.")

    @property
    def min_volatility(self) -> float:
        min_risk = self.optimal_results["Risk"].min()
        min_volatility = self.financial_model.risk_to_volatility(min_risk)
        stable_min_volatility = min_volatility + self.EPS
        return stable_min_volatility

    @property
    def max_volatility(self) -> float:
        max_risk = self.optimal_results["Risk"].max()
        max_volatility = self.financial_model.risk_to_volatility(max_risk)
        stable_max_volatility = max_volatility - self.EPS
        return stable_max_volatility

    def compute_apr_from_volatility(self, volatility: float) -> float:
        risk = self.financial_model.volatility_to_risk(volatility)
        portfolio_weights = self.get_portfolio_weights(risk)
        apr = self.financial_model.predict_apr(np.array(portfolio_weights))
        return apr