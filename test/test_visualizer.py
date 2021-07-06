import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from src.visualizer import Visualizer
from stub_builder import StubBuilder, DataParameters


class TestVisualizer(unittest.TestCase):
    def test_make_financial_model_plots(self):
        params = [DataParameters("test1", 0.1, 0.05), DataParameters("test2", -.1, 0.01)]
        data = StubBuilder.create_test_data(params)
        financial_model = StubBuilder.create_financial_model(params)
        with TemporaryDirectory(dir="resources") as tmp_dir:
            visualizer = Visualizer(folder=tmp_dir)
            visualizer.make_financial_model_plots(financial_model, data)
            num_images = len(os.listdir(tmp_dir))
            self.assertGreater(num_images, 0)

    def test_make_portfolio_optimizer_plots(self):
        num_assets = 20
        interest_rates = 0.1 * np.random.randn(num_assets)
        variances = 0.01 * np.exp(np.random.randn(num_assets))/np.exp(0.5)
        param_generator = zip(range(num_assets), interest_rates, variances)
        params = [DataParameters(f"test{i}", rate, var) for i, rate, var in param_generator]
        data = StubBuilder.create_test_data(params)
        portfolio_optimizer = StubBuilder.create_portfolio_optimzer(params)
        portfolio_optimizer.optimize()
        max_risk = portfolio_optimizer.optimal_results["Risk"].max()
        min_risk = portfolio_optimizer.optimal_results["Risk"].min()
        risks = list(np.linspace(min_risk, max_risk, 3))
        with TemporaryDirectory(dir="resources") as tmp_dir:
            visualizer = Visualizer(folder=tmp_dir)
            visualizer.make_portfolio_optimizer_plots(portfolio_optimizer, data, risks)
            num_images = len(os.listdir(tmp_dir))
            self.assertGreater(num_images, 0)


if __name__ == '__main__':
    unittest.main()
