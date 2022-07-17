import pandas as pd
from warnings import warn


class RiskInterpolator:
    """Determines the interpolating weights for the quadratic risk function."""
    _sorted_risks: pd.Series
    _sorted_weights: pd.DataFrame
    _cross_risks: pd.Series

    def __init__(self, risks: pd.Series, weights: pd.DataFrame, covariances: pd.DataFrame):
        self._sorted_risks = risks.sort_values()
        self._sorted_weights = weights.loc[self._sorted_risks.index]
        self._cross_risks = self._compute_cross_risks(self._sorted_weights, covariances)

    @staticmethod
    def _compute_cross_risks(weights: pd.DataFrame, covariances: pd.DataFrame) -> pd.Series:
        """Returns the cross risks for a given sequence of weights."""
        num_weights = len(weights)
        cross_risks = []
        for i in range(num_weights - 1):
            w0 = weights.iloc[i, :]
            w1 = weights.iloc[i + 1, :]
            cross_risk = w0.transpose() @ covariances @ w1
            cross_risks.append(cross_risk)

        cross_risks = pd.Series(cross_risks, index=weights.index[:-1])
        return cross_risks

    def __call__(self, risk: float) -> pd.Series:
        idx = self._sorted_risks.searchsorted(risk)
        num_risks = len(self._sorted_risks)
        if idx in (0, num_risks):  # use end values for extrapolation
            idx = num_risks-1 if idx == num_risks else idx
            warn(
                f"Inputted risk, {risk}, not within bounds. Using weights at end value {self._sorted_risks.iloc[idx]}."
            )
            return self._sorted_weights.iloc[idx]

        # compute the convex coefficient for the quadratic risk
        r0 = self._sorted_risks.iloc[idx-1]
        r1 = self._sorted_risks.iloc[idx]
        r01 = self._cross_risks[idx-1]
        a = r0 + r1 - 2.0*r01
        b = 2.0*r01 - 2.0*r0
        c = r0-risk
        alpha = (-b + (b**2.0-4.0*a*c)**0.5) / (2.0*a)

        w0 = self._sorted_weights.iloc[idx-1]
        w1 = self._sorted_weights.iloc[idx]
        w = (1.0-alpha)*w0 + alpha*w1
        return w
