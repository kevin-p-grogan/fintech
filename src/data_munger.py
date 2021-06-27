from datetime import datetime, timedelta

import pandas as pd


class DataMunger:
    _num_days_time_horizon: float
    _analysis_column: str

    def __init__(self, num_days_time_horizon: float = 365., anaysis_column: str = "Adj Close"):
        self._num_days_time_horizon = num_days_time_horizon
        self._analysis_column = anaysis_column

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        start_date = self._get_start_date(data.index)
        data = data[start_date <= data.index]
        daily_prices = pd.DataFrame()
        for symbol, subcolumn in data:
            if subcolumn == self._analysis_column:
                daily_prices[symbol] = data[symbol][subcolumn]

        scaled_prices = daily_prices / daily_prices.iloc[0]
        return scaled_prices

    def _get_start_date(self, dates: pd.Index) -> datetime:
        end_date = dates.max()
        time_horizon = timedelta(days=self._num_days_time_horizon)
        start_date = end_date - time_horizon
        earliest_start_date = dates.min()
        if start_date < earliest_start_date:
            raise ValueError(
                f"Time horizon exceeds data: {start_date} < {earliest_start_date}. "
                f"Reduce the time horizon or ensure that data is sufficiently far out.")

        return start_date

