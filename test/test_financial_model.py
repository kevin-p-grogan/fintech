import unittest

import numpy as np

from stub_builder import StubBuilder, DataParameters
from src.data import Munger


class TestFinancialModel(unittest.TestCase):
    _portfolio_data_filepath = "resources/test_portfolio.pkl"

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

    def test_minimum_risk(self):
        minimum_variance = 0.01
        params = [DataParameters("test1", 0.0, minimum_variance), DataParameters("test2", 0.0, 2.0*minimum_variance)]
        financial_model = StubBuilder.create_financial_model(params)
        predicted_minimum_risk = financial_model.minimum_risk
        self.assertTrue(np.isclose(predicted_minimum_risk, minimum_variance, rtol=0.1))

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

    def test_yearly_return_is_lower_with_taxes(self):
        params = [DataParameters("test1", 1.0, 0.0), DataParameters("test2", 0.0, 0.0)]
        financial_model = StubBuilder.create_financial_model(params)
        portfolio_weights = np.array([0.5, 0.5])
        financial_model.tax_rate = 0.0
        return_without_taxes = financial_model.predict_yearly_return(portfolio_weights)
        financial_model.tax_rate = 1.0
        return_with_taxes = financial_model.predict_yearly_return(portfolio_weights)
        self.assertGreater(return_without_taxes, return_with_taxes)

    def test_predict_yearly_return_jacobian_without_taxes(self):
        params = [DataParameters("test1", 1.2, 0.02), DataParameters("test2", .7, 0.05)]
        financial_model = StubBuilder.create_financial_model(params)
        financial_model.tax_rate = 0.0
        portfolio_weights = np.array([0.5, 0.5])
        financial_model._current_portfolio_weights = np.array([1.0, 0.0])
        yearly_return1 = financial_model.predict_yearly_return(portfolio_weights)
        yearly_return2 = financial_model.predict_yearly_return(portfolio_weights + self.EPS)
        finite_difference_jacobian = (yearly_return2 - yearly_return1) / self.EPS
        predicted_jacobian = financial_model.predict_yearly_return_jacobian(portfolio_weights)
        self.assertTrue(np.isclose(finite_difference_jacobian, predicted_jacobian.sum(), rtol=1.e-3))

    def test_predict_yearly_return_jacobian_with_taxes(self):
        params = [DataParameters("test1", 0.5, 0.02), DataParameters("test2", 0.3, 0.01)]
        financial_model = StubBuilder.create_financial_model(params)
        financial_model.tax_rate = 0.5
        portfolio_weights = np.array([0.75, 0.25])
        financial_model._current_portfolio_weights = np.array([0.0, 0.0])
        yearly_return1 = financial_model.predict_yearly_return(portfolio_weights)
        yearly_return2 = financial_model.predict_yearly_return(portfolio_weights + self.EPS)
        finite_difference_jacobian = (yearly_return2 - yearly_return1) / self.EPS
        predicted_jacobian = financial_model.predict_yearly_return_jacobian(portfolio_weights)
        self.assertTrue(np.isclose(finite_difference_jacobian, predicted_jacobian.sum(), rtol=1.e-3))

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
        financial_model._current_portfolio_weights = np.array([10.0])
        risk1 = financial_model.predict_risk(portfolio_weights)
        risk2 = financial_model.predict_risk(portfolio_weights + self.EPS)
        finite_difference_jacobian = (risk2 - risk1) / self.EPS
        predicted_jacobian = financial_model.predict_risk_jacobian(portfolio_weights)
        self.assertTrue(np.isclose(finite_difference_jacobian, predicted_jacobian, rtol=1.e-3))

    def test_compute_current_portfolio_weights(self):
        portfolio_data = Munger().load_portfolio_data(self._portfolio_data_filepath)
        params = [DataParameters(symbol, 0.1, 1e-3) for symbol in portfolio_data.index]
        financial_model = StubBuilder.create_financial_model(params)
        total_equity = portfolio_data["Equity"].sum()
        current_portfolio_weights = financial_model._compute_current_portfolio_weights(portfolio_data)
        self.assertIsInstance(current_portfolio_weights, np.ndarray)
        self.assertEqual(current_portfolio_weights.sum(), total_equity)
        self.assertEqual(len(current_portfolio_weights), len(financial_model._interest_rates))

    def test_predict_exceedance_value(self):
        interest_rate = 0.1
        variance = 0.01
        params = [DataParameters("test", interest_rate, variance)]
        financial_model = StubBuilder.create_financial_model(params)
        portfolio_weights = np.array([1.0])

        # 100% probability that you do not lose everything
        ev = financial_model.predict_exceedance_value(portfolio_weights, time=1, exceedance_probability=1)
        self.assertEqual(ev, -1.0)

        # 0% probability that you do not make an infinite amount
        ev = financial_model.predict_exceedance_value(portfolio_weights, time=1, exceedance_probability=0)
        self.assertTrue(np.isinf(ev))

        # The exceedance value should be less than the interest rate for a high exceedance probability
        ev = financial_model.predict_exceedance_value(portfolio_weights, time=1, exceedance_probability=0.95)
        self.assertGreater(interest_rate, ev)

        # Exceedance value should be close to interest rate at 50% probaility
        ev = financial_model.predict_exceedance_value(portfolio_weights, time=1, exceedance_probability=0.5)
        self.assertTrue(np.isclose(interest_rate, ev, atol=0.01))


if __name__ == '__main__':
    unittest.main()
