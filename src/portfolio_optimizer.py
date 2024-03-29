from typing import Optional
import os
from warnings import warn

from scipy.optimize import minimize, LinearConstraint
import pandas as pd
import numpy as np

from src.financial_model import FinancialModel
from src.risk_interpolator import RiskInterpolator


class PortfolioOptimizer:
    financial_model: FinancialModel
    _sparsity_importance: float
    _max_portfolio_weight: float
    _disable_selling: bool
    _optimal_weights: Optional[pd.DataFrame]
    _optimal_results: Optional[pd.DataFrame]
    _risk_interpolator: Optional[RiskInterpolator]
    _active_assets: Optional[list[str]]
    _active_asset_indices: Optional[np.ndarray] = None
    _sell_only_losers: bool

    LOWER_TRADEOFF_BOUND: float = 0.0
    UPPER_TRADEOFF_BOUND: float = 1.0
    EPS: float = 1.0e-6
    SEED: int = 1337

    def __init__(self, model: FinancialModel,
                 sparsity_importance: float = 0.1,
                 max_portfolio_weight: float = 1.0,
                 disable_selling: bool = False,
                 active_assets: Optional[list[str]] = None,
                 sell_only_losers: bool = False):
        self.financial_model = model
        self._sparsity_importance = sparsity_importance
        self._disable_selling = disable_selling
        self._set_active_assets(active_assets)
        self.max_portfolio_weight = max_portfolio_weight
        self._optimal_weights = None
        self._optimal_results = None
        self._risk_interpolator = None
        self._sell_only_losers = sell_only_losers
        np.random.seed(self.SEED)

    def _set_active_assets(self, active_assets) -> None:
        self._active_assets = active_assets if active_assets is not None else self.financial_model.asset_names
        unmodeled = [aa for aa in self._active_assets if aa not in self.financial_model.asset_names]
        if unmodeled:
            raise ValueError(f"Active assets need to be a subset of the modeled assets. "
                             f"The following assets are not modeled: {', '.join(unmodeled)}.")

        self._active_asset_indices = np.array([a in self._active_assets for a in self.financial_model.asset_names])
        self._num_weights = len(self._active_assets)

    @property
    def max_portfolio_weight(self) -> float:
        return self._max_portfolio_weight

    @max_portfolio_weight.setter
    def max_portfolio_weight(self, value) -> None:
        lower_bound = 1.0 / self._num_weights
        if value < lower_bound:
            raise AttributeError(f"Invalid 'max_portfolio_weight' found. "
                                 f"Found {value} while the lower bound is {lower_bound}.")

        self._max_portfolio_weight = value

    def optimize(self, num_evaluations: int = 11, verbose: bool = True):
        initial_weights = self._make_initial_weights()
        constraints = self._make_constraints()
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

            weights = self._add_inactive_assets(result.x)
            optimal_weights.append(list(weights))
            risk = self.financial_model.predict_risk(weights)
            yearly_return = self.financial_model.predict_yearly_return(weights)
            optimal_results.append([yearly_return, risk])

        self._store_results(optimal_results, optimal_weights, tradeoff_params)
        self._risk_interpolator = RiskInterpolator(
            self.optimal_results["Risk"], self.optimal_weights, self.financial_model.covariances
        )

    def _add_inactive_assets(self, weights: np.ndarray) -> np.ndarray:
        """Returns a weight array with inactive asset weights set to zero."""
        all_weights = np.zeros(self.financial_model.num_assets)
        all_weights[self._active_asset_indices] = weights
        return all_weights

    def _remove_inactive_assets(self, assets: np.ndarray) -> np.ndarray:
        return assets[self._active_asset_indices]

    def _store_results(self, optimal_results: list[list[float]], optimal_weights, tradeoff_params) -> None:
        optimal_weights = np.array(optimal_weights)
        self._optimal_weights = pd.DataFrame(optimal_weights, index=tradeoff_params,
                                             columns=self.financial_model.asset_names)
        optimal_results = np.array(optimal_results)
        self._optimal_results = pd.DataFrame(optimal_results, index=tradeoff_params, columns=["Yearly Return", "Risk"])

    def _make_initial_weights(self) -> np.ndarray:
        initial_weights = np.random.random(self._num_weights)
        initial_weights /= initial_weights.sum()
        return initial_weights

    def _make_constraints(self) -> list[LinearConstraint]:
        selling_constraint = self._no_shorting
        if self._disable_selling:
            selling_constraint = self._no_selling
        elif self._sell_only_losers:
            selling_constraint = self._only_sell_losers

        constraints = [self._weights_sum_to_one, selling_constraint]
        maximal_weights = self._compute_maximal_weights()
        constraints.append(self._weights_less_than_max)
        if self._disable_selling and np.any(maximal_weights < 0):
            warn(f"max_portfolio_weight constraint is not satisfiable. Ignoring.")
        else:
            constraints.append(self._weights_less_than_max)

        return constraints

    @property
    def _weights_sum_to_one(self) -> LinearConstraint:
        lower_bound = 1.0
        upper_bound = 1.0
        constraint_matrix = np.ones((1, self._num_weights))
        return LinearConstraint(constraint_matrix, lower_bound, upper_bound)

    @property
    def _no_shorting(self) -> LinearConstraint:
        """Disables shorting of assets but allows selling of current active assets."""
        lower_bound = -self._remove_inactive_assets(self.financial_model.current_portfolio_weights)
        upper_bound = np.inf
        constraint_matrix = np.identity(self._num_weights)
        return LinearConstraint(constraint_matrix, lower_bound, upper_bound)

    @property
    def _only_sell_losers(self) -> LinearConstraint:
        """Disables shorting and only sell loser assets."""
        lower_bound = -self._remove_inactive_assets(self.financial_model.current_portfolio_weights)
        is_winner = ~self._remove_inactive_assets(self.financial_model.is_loser.to_numpy())
        lower_bound[is_winner] = 0.0
        upper_bound = np.inf
        constraint_matrix = np.identity(self._num_weights)
        return LinearConstraint(constraint_matrix, lower_bound, upper_bound)

    @property
    def _no_selling(self) -> LinearConstraint:
        """Disables the selling of assets."""
        lower_bound = 0.0
        upper_bound = np.inf
        constraint_matrix = np.identity(self._num_weights)
        return LinearConstraint(constraint_matrix, lower_bound, upper_bound)

    @property
    def _weights_less_than_max(self) -> LinearConstraint:
        """Ensures that all assets are less than a prescribed weight."""
        lower_bound = -np.inf
        upper_bound = self._compute_maximal_weights()
        constraint_matrix = np.identity(self._num_weights)
        return LinearConstraint(constraint_matrix, lower_bound, upper_bound)

    def _compute_maximal_weights(self):
        current_weights = self._remove_inactive_assets(self.financial_model.current_portfolio_weights)
        normalizing_factor = current_weights.sum() + 1  # assumes _weights_sum_to_one is applied
        maximal_weights = self.max_portfolio_weight * normalizing_factor - current_weights
        return maximal_weights

    def _objective(self, portfolio_weights: np.ndarray, tradeoff_parameter: float) -> float:
        all_portfolio_weights = self._add_inactive_assets(portfolio_weights)
        yearly_return = self.financial_model.predict_yearly_return(all_portfolio_weights)
        risk = self.financial_model.predict_risk(all_portfolio_weights)
        obj = -tradeoff_parameter * yearly_return + (
                    1. - tradeoff_parameter) * risk + self.sparsity_weight * self._entropy(portfolio_weights)
        return obj

    @property
    def sparsity_weight(self) -> float:
        """Weighs the sparsity of the profile in consideration of the number of assets and the current returns"""
        max_objective = max(self.financial_model.maximum_yearly_return, self.financial_model.minimum_risk)
        min_objective = min(self.financial_model.maximum_yearly_return, self.financial_model.minimum_risk)
        objective_scale = max_objective - min_objective
        entropy_scale = np.log(self._num_weights)
        return self._sparsity_importance * objective_scale / entropy_scale

    def _entropy(self, portfolio_weights: np.ndarray) -> float:
        """Computes the entropy of the portfolio. Assumes that the _no_shorting constraint is enforced."""
        current_weights = self._remove_inactive_assets(self.financial_model.current_portfolio_weights)
        normalizing_factor = current_weights.sum() + 1  # assumes _weights_sum_to_one is applied
        probabilities = (portfolio_weights + current_weights) / normalizing_factor
        smoothed_probabilities = np.maximum(probabilities, PortfolioOptimizer.EPS)
        return -float(np.sum(smoothed_probabilities * np.log(smoothed_probabilities)))

    def _objective_jacobian(self, portfolio_weights: np.ndarray, tradeoff_parameter: float) -> np.ndarray:
        all_portfolio_weights = self._add_inactive_assets(portfolio_weights)
        yearly_return_jacobian = self._remove_inactive_assets(
            self.financial_model.predict_yearly_return_jacobian(all_portfolio_weights)
        )
        risk_jacobian = self._remove_inactive_assets(
            self.financial_model.predict_risk_jacobian(all_portfolio_weights)
        )
        sparsity_jacobian = self._entropy_jacobian(portfolio_weights)
        obj = -tradeoff_parameter * yearly_return_jacobian + (
                    1. - tradeoff_parameter) * risk_jacobian + self.sparsity_weight * sparsity_jacobian
        return obj

    def _entropy_jacobian(self, portfolio_weights: np.ndarray) -> np.ndarray:
        current_weights = self._remove_inactive_assets(self.financial_model.current_portfolio_weights)
        normalizing_factor = current_weights.sum() + 1  # assumes _weights_sum_to_one is applied
        probabilities = (portfolio_weights + current_weights) / normalizing_factor
        smoothed_probabilities = np.maximum(probabilities, PortfolioOptimizer.EPS)
        smoothed_probabilities_jacobian = 1.0 / normalizing_factor
        entropy_jacobian = -(1.0 + np.log(smoothed_probabilities)) * smoothed_probabilities_jacobian
        return entropy_jacobian

    @property
    def optimal_weights(self) -> pd.DataFrame:
        self._check_property("optimal_weights")
        return self._optimal_weights

    @property
    def optimal_results(self) -> pd.DataFrame:
        self._check_property("optimal_results")
        return self._optimal_results

    @property
    def risk_interpolator(self) -> RiskInterpolator:
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
        risk = self._check_risk(risk)
        interpolated_weights = self.risk_interpolator(risk)
        weights_array = self.financial_model.get_weights_array(interpolated_weights.to_numpy())
        portfolio_weights = pd.Series(weights_array, index=self.financial_model.asset_names)
        return portfolio_weights

    def _check_risk(self, unchecked_risk: float) -> float:
        min_risk = self.optimal_results["Risk"].min()
        max_risk = self.optimal_results["Risk"].max()
        in_bounds = min_risk <= unchecked_risk <= max_risk
        risk = unchecked_risk
        if not in_bounds:
            risk = (max_risk + min_risk) / 2.0
            warn(f"Inputted risk, {unchecked_risk}, is not in bounds. "
                 f"It must be between {min_risk} and {max_risk}. "
                 f"Using middle value of {risk} (volatility={self.financial_model.risk_to_volatility(risk)}.")

        return risk

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

    def compute_exceedance_from_volatility(
            self,
            volatility: float,
            time: float,
            exceedance_probability: float) -> float:
        risk = self.financial_model.volatility_to_risk(volatility)
        portfolio_weights = self.get_portfolio_weights(risk)
        ev = self.financial_model.predict_exceedance_value(np.array(portfolio_weights), time, exceedance_probability)
        return ev
