import unittest
from datetime import datetime, timedelta
from collections import namedtuple

import pandas as pd
import numpy as np

from src.financial_model import FinancialModel

TestParameters = namedtuple("TestParameters", ["name", "interest_rate", "variance"])


class TestFinancialModel(unittest.TestCase):

    def test_train_interest_rates(self):
        interest_rate1 = 1.0
        interest_rate2 = -1.0
        params = [TestParameters("test1", interest_rate1, 0.0), TestParameters("test2", interest_rate2, 0.0)]
        data = self.create_test_data(params)
        financial_model = FinancialModel()
        financial_model.train(data)
        predicted_interest_rate1, predicted_interest_rate2 = financial_model._interest_rates
        self.assertTrue(np.isclose(predicted_interest_rate1, interest_rate1, atol=1.0e-2))
        self.assertTrue(np.isclose(predicted_interest_rate2, interest_rate2, atol=1.0e-2))

    @staticmethod
    def create_test_data(params: list[TestParameters]) -> pd.DataFrame:
        num_times = 10000
        times = np.linspace(0, 1, num_times)
        dates = TestFinancialModel._make_dates(times)
        data = pd.DataFrame({
            p.name: np.exp(p.interest_rate * times + p.variance**0.5 * np.random.randn(num_times)) for p in params
        }, index=dates)
        return data

    def test_train_covariances(self):
        variance = 0.01
        params = [TestParameters("test", 0.0, variance)]
        data = self.create_test_data(params)
        financial_model = FinancialModel()
        financial_model.train(data)
        predicted_variance = financial_model._covariances.to_numpy()[0, 0]
        self.assertTrue(np.isclose(predicted_variance, variance, rtol=0.1))

    @staticmethod
    def _make_dates(times: np.array) -> np.array:
        days_in_year = 365.0
        deltas = (timedelta(days=int(days_in_year * t)) for t in times)
        start_date = datetime.now()
        dates = [start_date + delta for delta in deltas]
        return np.array(dates)


if __name__ == '__main__':
    unittest.main()
