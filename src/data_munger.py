from datetime import datetime, timedelta
from typing import Any
import locale

import numpy as np
import pandas as pd

locale.setlocale(locale.LC_ALL, 'en_us')


class DataMunger:
    _num_days_time_horizon: float
    _analysis_column: str

    NUM_PORTFOLIO_DATA_COLS = 7

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

    @staticmethod
    def get_times_from_index(data: pd.DataFrame) -> pd.Series:
        """Converts the dates to a fraction of a year starting at the first date."""
        data = data.copy()
        dates = data.index.to_series()
        start_date = dates.min()
        deltas = (date - start_date for date in dates)
        days_in_year = 365.0
        times = pd.Series([delta.days / days_in_year for delta in deltas], index=dates)
        return times

    @staticmethod
    def load_portfolio_data(file_path: str, index_col: str = "Symbol") -> pd.DataFrame:
        raw_sequence = pd.read_csv(file_path, header=None, comment="#")
        raw_sequence = np.squeeze(raw_sequence.to_numpy())
        header = raw_sequence[:DataMunger.NUM_PORTFOLIO_DATA_COLS]
        data = raw_sequence[DataMunger.NUM_PORTFOLIO_DATA_COLS:]
        DataMunger._check_data_format(data, header)
        data = data.reshape((-1, DataMunger.NUM_PORTFOLIO_DATA_COLS))
        df = pd.DataFrame(data, columns=header)
        df = df.set_index(index_col)
        df = df.applymap(DataMunger._convert_numerics)
        return df

    @staticmethod
    def _check_data_format(data: np.ndarray, header: np.ndarray):
        data_doesnt_match_header = len(data) % len(header) != 0
        if data_doesnt_match_header:
            raise ValueError(f"Found {len(header)} header rows and {len(data)} data rows. "
                             f"Ensure that the data has been properly copied or that formatting hasn't changed.")

    @staticmethod
    def _convert_numerics(datum: Any) -> Any:
        """Converts numeric and currency strings to floats."""
        if isinstance(datum, str):
            datum = datum.strip("$")
            try:
                datum = locale.atof(datum)
            except ValueError:
                pass

        return datum

