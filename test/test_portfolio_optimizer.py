import unittest

import numpy as np
import pandas as pd

from stub_builder import StubBuilder, DataParameters


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
        portfolio_weights = np.array([0.5, 0.5])
        obj1 = portfolio_optimizer._objective(portfolio_weights, tradeoff_parameter=0.5)
        obj2 = portfolio_optimizer._objective(portfolio_weights + self.EPS, tradeoff_parameter=0.5)
        finite_difference = (obj2 - obj1) / self.EPS
        predicted_jacobian = portfolio_optimizer._objective_jacobian(portfolio_weights, tradeoff_parameter=0.5)
        predicted = predicted_jacobian.sum()
        self.assertTrue(np.isclose(finite_difference, predicted, rtol=1.e-3))

    def test_entropy_jacobian(self):
        params = [DataParameters("test1", 0.15, 0.04), DataParameters("test2", -0.05, 0.02)]
        portfolio_optimizer = StubBuilder.create_portfolio_optimzer(params)
        portfolio_weights = np.array([0.3, 0.7])
        entropy1 = portfolio_optimizer._entropy(portfolio_weights)
        entropy2 = portfolio_optimizer._entropy(portfolio_weights + self.EPS)
        finite_difference = (entropy2 - entropy1) / self.EPS
        predicted_jacobian = portfolio_optimizer._entropy_jacobian(portfolio_weights)
        predicted = predicted_jacobian.sum()
        self.assertTrue(np.isclose(finite_difference, predicted, rtol=1.e-3))

    def test_get_portfolio_weights(self):
        params = [DataParameters("test1", 0.1, 0.01), DataParameters("test2", 0.3, 0.05)]
        portfolio_optimizer = StubBuilder.create_portfolio_optimzer(params)
        portfolio_optimizer.optimize()
        portfolio_weights = portfolio_optimizer.get_portfolio_weights(0.015)
        self.assertIsInstance(portfolio_weights, pd.Series)
        self.assertGreater(portfolio_weights["test1"], portfolio_weights["test2"])


if __name__ == '__main__':
    unittest.main()
