import unittest
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src.financial_model import FinancialModel


class TestFinancialModel(unittest.TestCase):
    DAYS_IN_YEAR: int = 365

    def test_train_interest_rates(self):
        interest_rate1 = 1.0
        interest_rate2 = -1.0
        times = np.linspace(0, 1)
        dates = self._make_dates(times)
        data = pd.DataFrame({
            "test1": np.exp(interest_rate1*times),
            "test2": np.exp(interest_rate2*times),
        }, index=dates)
        financial_model = FinancialModel()
        financial_model.train(data)
        predicted_interest_rate1, predicted_interest_rate2 = financial_model._interest_rates
        self.assertTrue(np.isclose(predicted_interest_rate1, interest_rate1, atol=1.0e-2))
        self.assertTrue(np.isclose(predicted_interest_rate2, interest_rate2, atol=1.0e-2))

    def test_train_covariances(self):
        num_times = 100000
        times = np.linspace(0, 1, num_times)
        dates = self._make_dates(times)
        variance = 0.01
        data = pd.DataFrame({
            "test1": 1.0 + variance**0.5*np.random.randn(num_times),
        }, index=dates)
        financial_model = FinancialModel()
        financial_model.train(data)
        predicted_variance = financial_model._covariances.to_numpy()[0, 0]
        self.assertTrue(np.isclose(predicted_variance, variance, rtol=0.1))

    def _make_dates(self, times: np.array) -> np.array:
        deltas = (timedelta(days=int(self.DAYS_IN_YEAR * t)) for t in times)
        start_date = datetime.now()
        dates = [start_date + delta for delta in deltas]
        return np.array(dates)


if __name__ == '__main__':
    unittest.main()
