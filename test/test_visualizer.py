import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

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
        with TemporaryDirectory(dir="resources") as tmp_dir:
            visualizer = Visualizer(folder=tmp_dir)
            visualizer.make_portfolio_optimizer_plots(portfolio_optimizer, data)
            num_images = len(os.listdir(tmp_dir))
            self.assertGreater(num_images, 0)

    def test_make_portfolio_update_plot(self):
        num_assets = 100
        x = np.linspace(0, 1, num_assets)
        decay = 10
        update_array = np.exp(-decay*x)
        update_array /= np.sum(update_array)
        index = [f"test{i}" for i in range(num_assets)]
        portfolio_update = pd.Series(update_array, index)
        with TemporaryDirectory(dir="resources") as tmp_dir:
            visualizer = Visualizer(folder=tmp_dir)
            visualizer.make_portfolio_update_plot(portfolio_update)
            num_images = len(os.listdir(tmp_dir))
            self.assertGreater(num_images, 0)



if __name__ == '__main__':
    unittest.main()
