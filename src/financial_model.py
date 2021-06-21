import pandas as pd


class FinancialModel:
    def predict_expected_value(self, data: pd.DataFrame) -> pd.Series:
        pass

    def predict_covariance(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def train(self, data: pd.DataFrame):
        pass
