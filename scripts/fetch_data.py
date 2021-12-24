from src.config import Config
from src.data import Fetcher


def main(cfg: Config):
    with Fetcher(metadata_filepath=cfg.METADATA_FILEPATH) as fetcher:
        data = fetcher.fetch(cfg.PERIOD, cfg.INTERVAL)
        data.to_pickle(cfg.OUTPUT_FILEPATH)


if __name__ == "__main__":
    config = Config.load(__file__)
    main(config)


