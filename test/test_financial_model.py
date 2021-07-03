import unittest

import numpy as np

from stub_builder import StubBuilder, DataParameters

class TestFinancialModel(unittest.TestCase):
    def test_train_interest_rates(self):
        interest_rate1 = 1.0
        interest_rate2 = -1.0
        params = [DataParameters("test1", interest_rate1, 0.0), DataParameters("test2", interest_rate2, 0.0)]
        financial_model = StubBuilder.create_financial_model(params)
        predicted_interest_rate1, predicted_interest_rate2 = financial_model._interest_rates
        self.assertTrue(np.isclose(predicted_interest_rate1, interest_rate1, atol=1.0e-2))
        self.assertTrue(np.isclose(predicted_interest_rate2, interest_rate2, atol=1.0e-2))

    def test_train_covariances(self):
        variance = 0.01
        params = [DataParameters("test", 0.0, variance)]
        financial_model = StubBuilder.create_financial_model(params)
        predicted_variance = financial_model._covariances.to_numpy()[0, 0]
        self.assertTrue(np.isclose(predicted_variance, variance, rtol=0.1))


if __name__ == '__main__':
    unittest.main()
