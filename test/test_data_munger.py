import unittest

import pandas as pd
import numpy as np

from src.data_munger import DataMunger


class TestDataMunger(unittest.TestCase):
    _data_filepath = "resources/test_data.pkl"
    _portfolio_data_filepath = "resources/test_portfolio.txt"

    def test_preprocess(self):
        data = pd.read_pickle(self._data_filepath)
        data_munger = DataMunger()
        preprocessed_data = data_munger.preprocess(data)
        self.assertIsInstance(preprocessed_data, pd.DataFrame)

    def test_load_portfolio_data(self):
        data_munger = DataMunger()
        portfolio_data = data_munger.load_portfolio_data(self._portfolio_data_filepath)
        self.assertIsInstance(portfolio_data, pd.DataFrame)
        num_test_assets = 2
        self.assertEqual(len(portfolio_data), num_test_assets)
        num_columns = 6
        self.assertEqual(len(portfolio_data.columns), num_columns)
        text_column = "Name"
        numeric_columns = [col for col in portfolio_data.columns if col != text_column]
        all_numeric_columns_are_floats = np.all(
            portfolio_data[numeric_columns].applymap(lambda x: isinstance(x, float)))
        self.assertTrue(all_numeric_columns_are_floats)


if __name__ == '__main__':
    unittest.main()
