import os.path
from typing import TypeVar

import yaml

C = TypeVar('C', bound='Config')


class Config(dict):
    """Configuration object, which is a dictionary using dot notation"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    CONFIG_FILE = "../config.yaml"

    @classmethod
    def load(cls, filename: str) -> C:
        """Creates a configuration for a particular script.
        The filename should contain the key in the config.yaml file."""
        with open(Config.CONFIG_FILE, "r") as f:
            full_config = yaml.safe_load(f)

        file_config = next(v for k, v in full_config.items() if k in filename)
        return Config(file_config)

    def save(self, filepath: str) -> None:
        data = {k: v for k, v in self.items()}
        with open(filepath, "w") as f:
            yaml.dump(data, f)

