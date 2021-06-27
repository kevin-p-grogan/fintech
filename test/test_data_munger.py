import unittest

import pandas as pd
import numpy as np

from src.data_munger import DataMunger


class TestDataMunger(unittest.TestCase):
    _data_filepath = "resources/test_data.pkl"

    def test_preprocess(self):
        data = pd.read_pickle(self._data_filepath)
        data_munger = DataMunger()
        preprocessed_data = data_munger.preprocess(data)
        self.assertIsInstance(preprocessed_data, pd.DataFrame)
        data_is_the_same = np.all(preprocessed_data.to_numpy() == data.to_numpy())
        self.assertFalse(data_is_the_same)


if __name__ == '__main__':
    unittest.main()
