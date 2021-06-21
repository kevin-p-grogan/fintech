import pandas as pd
import yfinance as yf

METADATA_FILEPATH = "data/ETF_metadata.csv"
OUTPUT_FILEPATH = "data/ETF_data.pkl"
TICKER_SYMBOL_COLUMN = "Symbol"
PERIOD = "5y"  # Grab data for the last 5 years
INTERVAL = "1d"  # Grab daily data


def main():
    metadata = pd.read_csv(METADATA_FILEPATH)
    ticker_symbols = " ".join(metadata[TICKER_SYMBOL_COLUMN])
    data = yf.download(tickers=ticker_symbols, period=PERIOD, interval=INTERVAL, group_by="ticker")
    data.to_pickle(OUTPUT_FILEPATH)


if __name__ == "__main__":
    main()


