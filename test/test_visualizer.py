import os
import unittest
from tempfile import TemporaryDirectory

from src.visualizer import Visualizer
from src.financial_model import FinancialModel
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


if __name__ == '__main__':
    unittest.main()
