import pickle as pkl

from src.config import Config
from src.data import Fetcher


def main(cfg: Config) -> None:
    with Fetcher(metadata_filepath=cfg.METADATA_FILEPATH) as fetcher:
        financial_data = fetcher.fetch_financial_data(cfg.PERIOD, cfg.INTERVAL)
        financial_data.to_pickle(cfg.FINANCIAL_DATA_FILEPATH)
        portfolio_data = fetcher.fetch_portfolio_data()
        with open(cfg.PORTFOLIO_DATA_FILEPATH, "wb") as f:
            pkl.dump(portfolio_data, f)


if __name__ == "__main__":
    config = Config.load(__file__)
    main(config)


