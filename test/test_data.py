import unittest
import pickle as pkl
from typing import Optional

import pandas as pd
import numpy as np

from src.data import Munger, PortfolioRecord


class TestMunger(unittest.TestCase):
    _data_filepath = "resources/test_data.pkl"
    _portfolio_data_filepath = "resources/test_portfolio.pkl"

    def test_preprocess(self):
        data = pd.read_pickle(self._data_filepath)
        munger = Munger()
        preprocessed_data = munger.preprocess(data)
        self.assertIsInstance(preprocessed_data, pd.DataFrame)

    def test_load_portfolio_data(self):
        munger = Munger()
        portfolio_data = munger.load_portfolio_data(self._portfolio_data_filepath)
        self.assertIsInstance(portfolio_data, pd.DataFrame)
        self.assertGreater(len(portfolio_data), 0)
        non_numeric_columns = ("Name", "Last Transaction")
        numeric_columns = [col for col in portfolio_data.columns if col not in non_numeric_columns]
        all_numeric_columns_are_floats = np.all(
            portfolio_data[numeric_columns].applymap(lambda x: isinstance(x, float)))
        self.assertTrue(all_numeric_columns_are_floats)


class TestPortfolioRecord(unittest.TestCase):
    _portfolio_data_filepath = "resources/test_portfolio.pkl"
    _portfolio_data: Optional[dict] = None

    @property
    def portfolio_data(self) -> dict:
        data = self._portfolio_data
        if data is None:
            with open(self._portfolio_data_filepath, 'rb') as f:
                data = pkl.load(f)

        return data

    def test_stock_can_be_converted_to_record(self):
        stock = self.portfolio_data["stocks"][0]
        portfolio_record = PortfolioRecord.make_from_stock(stock)
        self.assertIsInstance(portfolio_record, PortfolioRecord)

    def test_equity_is_greater_than_return(self):
        """This will be true when the cost is greater than zero."""
        portfolio_record = PortfolioRecord(
            symbol="TST",
            name="test",
            shares=10,
            price='3.3',
            average_cost='2e0',
            last_transaction='2021-12-10T13:34:10.615898-02:00'
        )
        self.assertGreater(portfolio_record.equity, portfolio_record.total_return)

    def test_crypto_can_be_converted_to_record(self):
        crypto = self.portfolio_data["cryptos"][0]
        portfolio_record = PortfolioRecord.make_from_crypto(crypto)
        self.assertIsInstance(portfolio_record, PortfolioRecord)

    def test_records_can_be_converted_to_dataframe(self):
        records = [
            PortfolioRecord(
                symbol="test",
                name="TST",
                shares=1,
                price='2.3',
                average_cost='2e0',
                last_transaction='2021-12-30T03:34:10.6153228-02:00'
            )
        ]
        df = PortfolioRecord.convert_records_to_dataframe(records)
        self.assertIsInstance(df, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
