import unittest

import numpy as np
import pandas as pd

from src.risk_interpolator import RiskInterpolator


class TestRiskInterpolator(unittest.TestCase):
    def test_interpolator_finds_correct_risk(self):
        weights = pd.DataFrame(
            {"asset1": [1.0, 0.5, 0.0], "asset2": [0.0, 0.5, 1.0]},
            index=[0.0, 0.5,  1.0]
        )
        covariances = pd.DataFrame({"asset1": [2.0, 0.0], "asset2": [0.0, 1.0]}, index=["asset1", "asset2"])
        risks = weights @ covariances @ weights.transpose()
        risks = pd.Series(np.diag(risks), weights.index)
        expected_risk = (risks[0] + risks[1]) / 2.0
        risk_interpolator = RiskInterpolator(risks, weights, covariances)
        interpolated_weights = risk_interpolator(expected_risk)
        interpolated_risk = interpolated_weights @ covariances @ interpolated_weights.transpose()
        self.assertAlmostEqual(interpolated_risk, expected_risk)

    def test_interpolator_handles_single_asset(self):
        weights = pd.DataFrame({"asset1": [1.0] * 3, }, index=[0.0, 0.5,  1.0])
        covariances = pd.DataFrame({"asset1": [2.0]}, index=["asset1"])
        risks = weights @ covariances @ weights.transpose()
        risks = pd.Series(np.diag(risks), weights.index)
        expected_risk = risks[0]
        risk_interpolator = RiskInterpolator(risks, weights, covariances)
        interpolated_weights = risk_interpolator(expected_risk)
        interpolated_risk = interpolated_weights @ covariances @ interpolated_weights.transpose()
        self.assertAlmostEqual(interpolated_risk, expected_risk)

    def test_extrapolation_uses_end_values(self):
        weights = pd.DataFrame(
            {"asset1": [1.0, 0.0], "asset2": [0.0, 1.0]},
            index=[0.0, 1.0]
        )
        covariances = pd.DataFrame({"asset1": [10.0, -1.0], "asset2": [-1.0, 4.0]}, index=["asset1", "asset2"])
        risks = weights @ covariances @ weights.transpose()
        risks = pd.Series(np.diag(risks), weights.index)
        risk_interpolator = RiskInterpolator(risks, weights, covariances)

        max_risk = risks.max()
        interpolated_weights = risk_interpolator(2.0 * max_risk)
        interpolated_risk = interpolated_weights @ covariances @ interpolated_weights.transpose()
        self.assertAlmostEqual(interpolated_risk, max_risk)

        min_risk = risks.min()
        interpolated_weights = risk_interpolator(min_risk / 2.0)
        interpolated_risk = interpolated_weights @ covariances @ interpolated_weights.transpose()
        self.assertAlmostEqual(interpolated_risk, min_risk)


if __name__ == '__main__':
    unittest.main()
