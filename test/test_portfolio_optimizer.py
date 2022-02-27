import unittest

import numpy as np
import pandas as pd

from stub_builder import StubBuilder, DataParameters
from src.data import PortfolioRecord
from src.financial_model import FinancialModel
from src.portfolio_optimizer import PortfolioOptimizer


class TestPortfolioOptimizer(unittest.TestCase):
    EPS: float = 1.e-6

    def test_optimize(self):
        params = [DataParameters("test1", 1.0, 0.01), DataParameters("test2", -1.0, 0.05)]
        portfolio_optimizer = StubBuilder.create_portfolio_optimzer(params)
        portfolio_optimizer.optimize()
        optimal_weights = portfolio_optimizer._optimal_weights
        self.assertIsNotNone(optimal_weights)
        first_asset_is_best = np.all(optimal_weights["test1"] >= optimal_weights["test1"])
        self.assertTrue(first_asset_is_best)
        optimal_results = portfolio_optimizer._optimal_results
        self.assertIsNotNone(optimal_results)

    def test_optimizer_jacobian(self):
        params = [DataParameters("test1", 0.1, 0.02), DataParameters("test2", -0.1, 0.03)]
        portfolio_optimizer = StubBuilder.create_portfolio_optimzer(params)
        portfolio_optimizer.financial_model._current_portfolio_weights = np.array([0.6, 0.0])
        portfolio_weights = np.array([-0.5, 1.5])
        obj1 = portfolio_optimizer._objective(portfolio_weights, tradeoff_parameter=0.5)
        obj2 = portfolio_optimizer._objective(portfolio_weights + self.EPS, tradeoff_parameter=0.5)
        finite_difference = (obj2 - obj1) / self.EPS
        predicted_jacobian = portfolio_optimizer._objective_jacobian(portfolio_weights, tradeoff_parameter=0.5)
        predicted = predicted_jacobian.sum()
        self.assertTrue(np.isclose(finite_difference, predicted, rtol=1.e-3))

    def test_entropy_jacobian(self):
        params = [DataParameters("test1", 0.15, 0.04), DataParameters("test2", -0.05, 0.02)]
        portfolio_optimizer = StubBuilder.create_portfolio_optimzer(params)
        portfolio_optimizer.financial_model._current_portfolio_weights = np.array([0.4, 0.0])
        portfolio_weights = np.array([-0.2, 1.2])
        entropy1 = portfolio_optimizer._entropy(portfolio_weights)
        entropy2 = portfolio_optimizer._entropy(portfolio_weights + self.EPS)
        finite_difference = (entropy2 - entropy1) / self.EPS
        predicted_jacobian = portfolio_optimizer._entropy_jacobian(portfolio_weights)
        predicted = predicted_jacobian.sum()
        self.assertTrue(np.isclose(finite_difference, predicted, rtol=1.e-3))

    def test_sparsity_importance_increases_sparsity(self):
        params = [DataParameters(f"test{i}", 0.1, 0.01) for i in range(10)]
        portfolio_optimizer = StubBuilder.create_portfolio_optimzer(params)
        portfolio_optimizer._sparsity_importance = 0.0
        portfolio_optimizer.optimize()
        non_sparse_weights = portfolio_optimizer._optimal_weights
        portfolio_optimizer._sparsity_importance = 1.0
        portfolio_optimizer.optimize()
        sparse_weights = portfolio_optimizer._optimal_weights
        num_sparse_zeros = int((sparse_weights.abs() < self.EPS).to_numpy().sum())
        num_non_sparse_zeros = int((non_sparse_weights.abs() < self.EPS).to_numpy().sum())
        self.assertGreater(num_sparse_zeros, num_non_sparse_zeros)

    def test_get_portfolio_weights(self):
        params = [DataParameters("test1", 0.1, 0.01), DataParameters("test2", 0.3, 0.05)]
        portfolio_optimizer = StubBuilder.create_portfolio_optimzer(params)
        portfolio_optimizer.optimize()
        portfolio_weights = portfolio_optimizer.get_portfolio_weights(0.015)
        self.assertIsInstance(portfolio_weights, pd.Series)
        self.assertGreater(portfolio_weights["test1"], portfolio_weights["test2"])

    def test_bad_assets_are_sold(self):
        params = [DataParameters("test1", -0.3, 0.1), DataParameters("test2", 0.3, 0.01)]
        data = StubBuilder.create_test_data(params)
        records = [
            PortfolioRecord(
                symbol="test1",
                name="test1",
                shares=1.0,
                price=1.0,
                average_cost=2.0,
                last_transaction='2021-12-30T03:34:10.6153228-02:00'
            )
        ]
        df = PortfolioRecord.convert_records_to_dataframe(records)
        financial_model = FinancialModel()
        financial_model.train(data, portfolio_data=df, investment_amount=1.0)
        portfolio_optimizer = PortfolioOptimizer(financial_model)
        portfolio_optimizer.optimize()
        bad_assets_are_sold = np.all(portfolio_optimizer.optimal_weights["test1"] < 0)
        self.assertTrue(bad_assets_are_sold)
        totals = sum(portfolio_optimizer.optimal_weights[sym] for sym in ("test1", "test2"))
        weights_sum_to_one = np.allclose(totals, 1.0, atol=self.EPS)
        self.assertTrue(weights_sum_to_one)

    def test_portfolio_weights_below_max(self):
        params = [DataParameters("test1", 0.1, 0.1), DataParameters("test2", 1.0, 0.01)]
        portfolio_optimizer = StubBuilder.create_portfolio_optimzer(params)
        portfolio_optimizer.max_portfolio_weight = 0.5
        portfolio_optimizer.optimize()
        weights = portfolio_optimizer.optimal_weights
        self.assertTrue(np.allclose(weights["test1"], weights["test2"]))

    def test_invalid_max_portfolio_weight_raises_exception(self):
        num_assets = 5
        params = [DataParameters(f"test{i}", 0.1, 0.01) for i in range(num_assets)]
        financial_model = StubBuilder.create_financial_model(params)
        invalid_max_portfolio_weight = 1.0 / (2.0 * num_assets)
        with self.assertRaises(AttributeError):
            PortfolioOptimizer(financial_model, max_portfolio_weight=invalid_max_portfolio_weight)

        portfolio_optimizer = PortfolioOptimizer(financial_model)
        with self.assertRaises(AttributeError):
            portfolio_optimizer.max_portfolio_weight = invalid_max_portfolio_weight


if __name__ == '__main__':
    unittest.main()
