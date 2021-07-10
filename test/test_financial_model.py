import unittest

import numpy as np

from stub_builder import StubBuilder, DataParameters
from src.data_munger import DataMunger


class TestFinancialModel(unittest.TestCase):
    _portfolio_data_filepath = "resources/test_portfolio.txt"

    EPS: float = 1.e-6

    def test_train_interest_rates(self):
        interest_rate1 = 1.0
        interest_rate2 = -1.0
        params = [DataParameters("test1", interest_rate1, 0.0), DataParameters("test2", interest_rate2, 0.0)]
        financial_model = StubBuilder.create_financial_model(params)
        predicted_interest_rate1, predicted_interest_rate2 = financial_model.interest_rates
        self.assertTrue(np.isclose(predicted_interest_rate1, interest_rate1, atol=1.0e-2))
        self.assertTrue(np.isclose(predicted_interest_rate2, interest_rate2, atol=1.0e-2))

    def test_train_covariances(self):
        variance = 0.01
        params = [DataParameters("test", 0.0, variance)]
        financial_model = StubBuilder.create_financial_model(params)
        predicted_variance = financial_model._covariances.to_numpy()[0, 0]
        self.assertTrue(np.isclose(predicted_variance, variance, rtol=0.1))

    def test_predict_yearly_return(self):
        params = [DataParameters("test1", 1.0, 0.0), DataParameters("test2", 0.0, 0.0)]
        financial_model = StubBuilder.create_financial_model(params)

        # balanced
        portfolio_weights = 0.5 * np.ones(len(params))
        yearly_return = financial_model.predict_yearly_return(portfolio_weights)
        self.assertTrue(np.isclose(yearly_return, 0.5, rtol=0.05))

        # Unbalanced
        portfolio_weights = np.array([1.0, 0])
        yearly_return = financial_model.predict_yearly_return(portfolio_weights)
        self.assertTrue(np.isclose(yearly_return, 1.0, rtol=0.05))

    def test_predict_yearly_return_jacobian(self):
        params = [DataParameters("test1", 1.2, 0.02)]
        financial_model = StubBuilder.create_financial_model(params)
        portfolio_weights = np.array([1])
        yearly_return1 = financial_model.predict_yearly_return(portfolio_weights)
        yearly_return2 = financial_model.predict_yearly_return(portfolio_weights + self.EPS)
        finite_difference_jacobian = (yearly_return2 - yearly_return1) / self.EPS
        predicted_jacobian = financial_model.predict_yearly_return_jacobian()
        self.assertTrue(np.isclose(finite_difference_jacobian, predicted_jacobian, rtol=1.e-3))

    def test_predict_risk(self):
        variance1 = 0.01
        variance2 = 0.05
        params = [DataParameters("test1", 0.0, variance1), DataParameters("test2", 0.0, variance2)]
        financial_model = StubBuilder.create_financial_model(params)

        # Unbalanced1
        portfolio_weights = np.array([1.0, 0])
        risk = financial_model.predict_risk(portfolio_weights)
        self.assertTrue(np.isclose(risk, variance1, rtol=0.1))

        # Unbalanced2
        portfolio_weights = np.array([0., 1.])
        risk = financial_model.predict_risk(portfolio_weights)
        self.assertTrue(np.isclose(risk, variance2, rtol=0.1))

    def test_predict_risk_jacobian(self):
        params = [DataParameters("test1", 1.2, 0.02)]
        financial_model = StubBuilder.create_financial_model(params)
        portfolio_weights = np.array([1])
        risk1 = financial_model.predict_risk(portfolio_weights)
        risk2 = financial_model.predict_risk(portfolio_weights + self.EPS)
        finite_difference_jacobian = (risk2 - risk1) / self.EPS
        predicted_jacobian = financial_model.predict_risk_jacobian(portfolio_weights)
        self.assertTrue(np.isclose(finite_difference_jacobian, predicted_jacobian, rtol=1.e-3))

    def test_compute_current_portfolio_weights(self):
        params = [
            DataParameters("TEST1", 0.1, 1.e-3),
            DataParameters("TEST3", 0.2, 2.e-3)
        ]
        financial_model = StubBuilder.create_financial_model(params)
        data_munger = DataMunger()
        portfolio_data = data_munger.load_portfolio_data(self._portfolio_data_filepath)
        current_portfolio_weights = financial_model._compute_current_portfolio_weights(portfolio_data, 10.0)
        self.assertIsInstance(current_portfolio_weights, np.ndarray)
        self.assertGreater(current_portfolio_weights.sum(), 1.0)
        self.assertEqual(len(current_portfolio_weights), len(financial_model._interest_rates))


if __name__ == '__main__':
    unittest.main()
