import pandas as pd
import yfinance as yf
from src.config import Config


def main(cfg: Config):
    metadata = pd.read_csv(cfg.METADATA_FILEPATH, comment="#")
    ticker_symbols = " ".join(metadata[cfg.TICKER_SYMBOL_COLUMN])
    data = yf.download(tickers=ticker_symbols, period=cfg.PERIOD, interval=cfg.INTERVAL, group_by="ticker")
    data.to_pickle(cfg.OUTPUT_FILEPATH)


if __name__ == "__main__":
    config = Config.load(__file__)
    main(config)


