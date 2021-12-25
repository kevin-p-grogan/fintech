from datetime import datetime, timedelta
from typing import Any, Optional
import locale
import pickle as pkl
from collections import OrderedDict

import numpy as np
import pandas as pd
import yfinance as yf
from robin_stocks import robinhood as r
from dotenv import dotenv_values
from pydantic import BaseModel

locale.setlocale(locale.LC_ALL, 'en_us')


class PortfolioRecord(BaseModel):
    """Pydantic class to handle the validation and transformation."""

    symbol: str
    name: str
    shares: float
    price: float
    average_cost: float
    last_transaction: datetime

    @property
    def total_return(self) -> float:
        return (self.price - self.average_cost) * self.shares

    @property
    def equity(self) -> float:
        return self.price * self.shares

    @staticmethod
    def make_from_stock(stock: dict) -> 'PortfolioRecord':
        return PortfolioRecord(
            symbol=stock["symbol"],
            name=stock["name"],
            shares=stock['quantity'],
            price=stock['price'],
            average_cost=stock['average_buy_price'],
            last_transaction=stock['updated_at']
        )

    @staticmethod
    def make_from_crypto(crypto: dict) -> 'PortfolioRecord':
        assert len(crypto["cost_bases"]) == 1  # Assume only one cost basis to pull average cost
        cost_basis = crypto['cost_bases'][0]
        average_cost = float(cost_basis["direct_cost_basis"]) / float(crypto['quantity'])
        return PortfolioRecord(
            symbol=crypto["currency"]["code"] + "-USD",
            name=crypto["currency"]["name"],
            shares=crypto['quantity'],
            price=crypto['price'],
            average_cost=average_cost,
            last_transaction=crypto['updated_at']
        )

    @staticmethod
    def convert_records_to_dataframe(records: list['PortfolioRecord']) -> pd.DataFrame:
        attribute_to_column = OrderedDict({
            "symbol": "Symbol",
            "name": "Name",
            "shares": "Shares",
            "price": "Price",
            "average_cost": "Average Cost",
            "total_return": "Total Return",
            "equity": "Equity",
            "last_transaction": "Last Transaction"
        })
        data = [tuple(getattr(record, attr) for attr in attribute_to_column.keys()) for record in records]
        df = pd.DataFrame.from_records(data, columns=list(attribute_to_column.values()))
        df = df.set_index("Symbol")
        return df


class Munger:
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

        interpolated_prices = daily_prices.interpolate(limit=None, limit_direction="both")
        scaled_prices = interpolated_prices / interpolated_prices.iloc[0]
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
        times = pd.Series([Munger.convert_days_to_time(delta.days) for delta in deltas], index=dates)
        return times

    @staticmethod
    def convert_days_to_time(days: float) -> float:
        """Normalizes the days to a standard time"""
        days_in_year = 365.0
        return days / days_in_year

    @staticmethod
    def load_portfolio_data(file_path: str) -> pd.DataFrame:
        with open(file_path, "rb") as f:
            portfolio_data = pkl.load(f)

        records = [PortfolioRecord.make_from_stock(stock) for stock in portfolio_data["stocks"]]
        records += [PortfolioRecord.make_from_crypto(crypto) for crypto in portfolio_data["cryptos"]]
        df = PortfolioRecord.convert_records_to_dataframe(records)
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


class Fetcher:
    _metadata: pd.DataFrame
    _login_data: Optional[dict] = None

    TICKER_SYMBOL_COLUMN = "Symbol"
    ENV_FILE = "../.env"

    def __init__(self, metadata_filepath: str):
        self._metadata = pd.read_csv(metadata_filepath, comment="#")

    def __enter__(self) -> 'Fetcher':
        self.login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logout()

    def login(self) -> None:
        login_info = dotenv_values(self.ENV_FILE)
        self._login_data = r.login(username=login_info["USERNAME"], password=login_info["PASSWORD"])

    def logout(self) -> None:
        r.logout()
        self._login_data = None

    def fetch_financial_data(self, period: str = '5y', interval: str = '1d') -> pd.DataFrame:
        ticker_symbols = " ".join(self._metadata[self.TICKER_SYMBOL_COLUMN])
        data = yf.download(tickers=ticker_symbols, period=period, interval=interval, group_by="ticker")
        return data

    @staticmethod
    def fetch_portfolio_data() -> dict:
        portfolio_data = {
            "stocks": Fetcher._fetch_stocks(),
            "cryptos": Fetcher._fetch_cryptos()
        }
        return portfolio_data

    @staticmethod
    def _fetch_stocks() -> list[dict]:
        stocks = r.get_open_stock_positions()
        for stock in stocks:
            stock["symbol"] = r.get_symbol_by_url(stock["instrument"])
            stock["name"] = r.get_name_by_url(stock["instrument"])

        prices = r.get_latest_price([s["symbol"] for s in stocks])
        for stock, price in zip(stocks, prices):
            stock["price"] = price

        return stocks

    @staticmethod
    def _fetch_cryptos() -> list[dict]:
        cryptos = r.get_crypto_positions()
        for crypto in cryptos:
            quote = r.crypto.get_crypto_quote(crypto["currency"]["code"])
            crypto["price"] = quote["mark_price"]

        return cryptos
